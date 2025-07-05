from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import uuid
import re
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Simple in-memory storage for testing
conversations = []
sessions = []

def analyze_text(text):
    """Analyze text for various metrics"""
    words = len(text.split())
    characters = len(text)
    sentences = len(re.split(r'[.!?]+', text.strip())) - 1 if text.strip() else 0
    return {
        'word_count': words,
        'character_count': characters,
        'sentence_count': max(sentences, 1) if text.strip() else 0
    }

def log_conversation_memory(session_id, message_type, content, watch_model=None):
    """Log conversation data to memory"""
    try:
        text_analysis = analyze_text(content)
        
        log_entry = {
            'id': str(uuid.uuid4()),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': message_type,
            'content': content,
            'watch_model': watch_model,
            'word_count': text_analysis['word_count'],
            'character_count': text_analysis['character_count'],
            'sentence_count': text_analysis['sentence_count'],
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        conversations.append(log_entry)
        print(f"Logged conversation: {message_type} - {watch_model} - {text_analysis['word_count']} words")
    except Exception as e:
        print(f"Error logging conversation: {e}")

def log_session_start_memory(session_id, config_id):
    """Log session start to memory"""
    try:
        session_entry = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_messages': 0,
            'total_words': 0,
            'duration_seconds': 0,
            'watch_models_shown': [],
            'user_messages': 0,
            'assistant_messages': 0,
            'config_id': config_id,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        sessions.append(session_entry)
        print(f"Logged session start: {session_entry}")
    except Exception as e:
        print(f"Error logging session start: {e}")

def update_session_stats(session_id):
    """Update session statistics"""
    try:
        session = next((s for s in sessions if s['session_id'] == session_id), None)
        if not session:
            return
        
        session_conversations = [c for c in conversations if c['session_id'] == session_id]
        
        # Count messages by type
        user_messages = [c for c in session_conversations if c['message_type'] == 'user']
        assistant_messages = [c for c in session_conversations if c['message_type'] == 'assistant']
        
        # Count unique watch models actually shown (only when watch_model is not None)
        watch_models_shown = list(set([c['watch_model'] for c in session_conversations if c['watch_model']]))
        
        # Calculate total words
        total_words = sum(c['word_count'] for c in session_conversations)
        
        # Update session
        session.update({
            'total_messages': len(session_conversations),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'total_words': total_words,
            'watch_models_shown': watch_models_shown
        })
        
        print(f"Updated session stats: {len(session_conversations)} messages, {len(watch_models_shown)} watches shown")
        
    except Exception as e:
        print(f"Error updating session stats: {e}")

class WatchImageMatcher:
    def __init__(self, images_folder="watch_images"):
        self.images_folder = Path(images_folder)
        if not self.images_folder.exists():
            self.images_folder.mkdir(exist_ok=True)
            
        self.watch_models = {
            # Classic Collection
            "Linea": ["linea"],
            "Serene": ["serene"],
            "Winchester": ["winchester"],
            "Sheffield": ["sheffield"],

            # Contemporary Collection
            "Ophelia": ["ophelia"],
            "Eterna": ["eterna"],
            "Lunaire Noir": ["lunaire noir", "lunaire-noir", "Lunaire-noir", "Lunaire-Noir"],
            "Lunaire Rose": ["lunaire rose", "lunaire-rose",  "Lunaire-rose", "Lunaire-Rose"],

            # Sport Collection
            "Explorer": ["explorer"],
            "Dive Master": ["dive master", "Dive-master", "dive-master", "Dive-Master"],
            "Field Ranger": ["field ranger", "Field-ranger", "field-ranger", "Field-Ranger"],
            "Nightfall": ["nightfall"],

            # Special / Extravagant Collection
            "Luna": ["luna"],
            "Commander": ["commander"],
            "Volt": ["volt"],
            "Dynastia": ["dynastia"]
        }
        
        # Collection mapping
        self.model_to_collection = {
            "Linea": "Classic",
            "Serene": "Classic",
            "Winchester": "Classic",
            "Sheffield": "Classic",
            "Ophelia": "Contemporary",
            "Eterna": "Contemporary",
            "Lunaire Noir": "Contemporary",
            "Lunaire Rose": "Contemporary",
            "Explorer": "Sport",
            "Dive Master": "Sport",
            "Field Ranger": "Sport",
            "Nightfall": "Sport",
            "Luna": "Special",
            "Commander": "Special",
            "Volt": "Special",
            "Dynastia": "Special"
        }
        
        self.variation_to_model = {}
        for model, variations in self.watch_models.items():
            for variation in variations:
                self.variation_to_model[variation.lower()] = model
    
    def find_watch_model(self, text):
        text_lower = text.lower()
        
        for variation, model in self.variation_to_model.items():
            if variation in text_lower:
                return model
        
        for model in self.watch_models.keys():
            model_name = model.rsplit(' ', 1)[0].lower()
            if model_name in text_lower:
                return model
        
        return None
    
    def get_collection(self, model_name):
        return self.model_to_collection.get(model_name, "Unknown")
    
    def get_image_path(self, model_name):
        if not model_name:
            return None
            
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_path = self.images_folder / f"{model_name}{ext}"
            if image_path.exists():
                return str(image_path)
        
        return None
    
    def get_image_base64(self, model_name):
        image_path = self.get_image_path(model_name)
        if not image_path:
            return None
        
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None

watch_matcher = WatchImageMatcher()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        html_path = Path(__file__).parent.parent / 'static' / 'index.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth', methods=['GET'])
def get_auth_token():
    """Generate authentication token for Hume API"""
    try:
        print("Auth endpoint called")
        
        hume_api_key = os.environ.get('HUME_API_KEY')
        hume_secret_key = os.environ.get('HUME_SECRET_KEY')
        
        config_id = request.args.get('config_id')
        if not config_id:
            config_id = os.environ.get('HUME_CONFIG_ID')
        
        if not all([hume_api_key, hume_secret_key, config_id]):
            missing = []
            if not hume_api_key:
                missing.append('HUME_API_KEY')
            if not hume_secret_key:
                missing.append('HUME_SECRET_KEY')
            if not config_id:
                missing.append('HUME_CONFIG_ID (provide via ?config_id=... URL parameter)')
            error_msg = f'Missing: {", ".join(missing)}'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Generate session ID for this conversation
        session_id = str(uuid.uuid4())
        log_session_start_memory(session_id, config_id)
        
        response_data = {
            'apiKey': hume_api_key,
            'secretKey': hume_secret_key,
            'configId': config_id,
            'sessionId': session_id
        }
        
        print(f"Returning auth data with session: {session_id}")
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Auth error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch-image', methods=['POST'])
def get_watch_image():
    """Get watch image based on text content"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        session_id = data.get('session_id', 'unknown')
        
        watch_model = watch_matcher.find_watch_model(text)
        
        # Only log if this is actually showing a watch (not just any assistant message)
        if watch_model:
            print(f"Watch model found: {watch_model}")
            # The assistant message itself will be logged separately via /api/log-message
            # We only want to count actual watch displays here
            
        if not watch_model:
            return jsonify({'watchModel': None, 'watchImage': None}), 200
        
        watch_image = watch_matcher.get_image_base64(watch_model)
        
        return jsonify({
            'watchModel': watch_model,
            'watchImage': watch_image,
            'collection': watch_matcher.get_collection(watch_model)
        }), 200
    except Exception as e:
        print(f"Watch image error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/log-message', methods=['POST'])
def log_message():
    """Log user/assistant messages"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        message_type = data.get('message_type', 'unknown')
        content = data.get('content', '')
        watch_model = data.get('watch_model', None)
        
        log_conversation_memory(session_id, message_type, content, watch_model)
        update_session_stats(session_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Log message error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End a conversation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        
        # Find and update session in memory
        session = next((s for s in sessions if s['session_id'] == session_id), None)
        if session:
            session['end_time'] = datetime.now().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(session['start_time'])
            end_time = datetime.fromisoformat(session['end_time'])
            session['duration_seconds'] = int((end_time - start_time).total_seconds())
            
        update_session_stats(session_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"End session error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get conversation data for analysis"""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 100))
        
        filtered_conversations = conversations
        
        if session_id:
            filtered_conversations = [c for c in conversations if c['session_id'] == session_id]
        
        # Sort by timestamp (newest first) and limit
        sorted_conversations = sorted(filtered_conversations, key=lambda x: x['timestamp'], reverse=True)
        limited_conversations = sorted_conversations[:limit]
        
        return jsonify({'conversations': limited_conversations}), 200
    except Exception as e:
        print(f"Get conversations error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get session data for analysis"""
    try:
        # Update all session stats before returning
        for session in sessions:
            update_session_stats(session['session_id'])
        
        # Sort sessions by start time (newest first)
        sorted_sessions = sorted(sessions, key=lambda x: x['start_time'], reverse=True)
        return jsonify({'sessions': sorted_sessions}), 200
    except Exception as e:
        print(f"Get sessions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get advanced analytics"""
    try:
        # Update all session stats
        for session in sessions:
            update_session_stats(session['session_id'])
        
        # Calculate advanced metrics
        total_sessions = len(sessions)
        total_conversations = len(conversations)
        
        if total_sessions == 0:
            return jsonify({
                'total_sessions': 0,
                'total_messages': 0,
                'total_words': 0,
                'avg_messages_per_session': 0,
                'avg_words_per_session': 0,
                'avg_session_duration': 0,
                'unique_watches_shown': 0,
                'total_watch_displays': 0,
                'collection_breakdown': {},
                'popular_watches': []
            }), 200
        
        # User vs Assistant messages
        user_messages = [c for c in conversations if c['message_type'] == 'user']
        assistant_messages = [c for c in conversations if c['message_type'] == 'assistant']
        
        # Watch analytics
        watch_displays = [c for c in conversations if c['watch_model']]
        unique_watches = list(set([c['watch_model'] for c in watch_displays]))
        
        # Collection breakdown
        collection_counts = {}
        for conv in watch_displays:
            collection = watch_matcher.get_collection(conv['watch_model'])
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
        # Popular watches
        watch_counts = {}
        for conv in watch_displays:
            watch_counts[conv['watch_model']] = watch_counts.get(conv['watch_model'], 0) + 1
        popular_watches = sorted(watch_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Word statistics
        total_words = sum(c['word_count'] for c in conversations)
        avg_words_per_message = total_words / total_conversations if total_conversations > 0 else 0
        
        # Session statistics
        completed_sessions = [s for s in sessions if s.get('end_time')]
        avg_duration = 0
        if completed_sessions:
            total_duration = sum(s.get('duration_seconds', 0) for s in completed_sessions)
            avg_duration = total_duration / len(completed_sessions)
        
        analytics = {
            'total_sessions': total_sessions,
            'total_messages': total_conversations,
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'total_words': total_words,
            'avg_messages_per_session': round(total_conversations / total_sessions, 1),
            'avg_words_per_session': round(total_words / total_sessions, 1),
            'avg_words_per_message': round(avg_words_per_message, 1),
            'avg_session_duration': round(avg_duration, 1),
            'unique_watches_shown': len(unique_watches),
            'total_watch_displays': len(watch_displays),
            'collection_breakdown': collection_counts,
            'popular_watches': [{'model': model, 'count': count} for model, count in popular_watches]
        }
        
        return jsonify(analytics), 200
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch-models', methods=['GET'])
def get_watch_models():
    """Get all available watch models"""
    try:
        return jsonify({'models': list(watch_matcher.watch_models.keys())}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'sessions': len(sessions),
            'conversations': len(conversations)
        },
        'environment_vars': {
            'HUME_API_KEY': bool(os.environ.get('HUME_API_KEY')),
            'HUME_SECRET_KEY': bool(os.environ.get('HUME_SECRET_KEY')),
            'HUME_CONFIG_ID': bool(os.environ.get('HUME_CONFIG_ID'))
        }
    }), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# This is required for Vercel
if __name__ == "__main__":
    app.run(debug=True)
