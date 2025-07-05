from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Simple in-memory storage for testing
conversations = []
sessions = []

def log_conversation_memory(session_id, message_type, content, watch_model=None):
    """Log conversation data to memory"""
    try:
        log_entry = {
            'id': str(uuid.uuid4()),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': message_type,
            'content': content,
            'watch_model': watch_model,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        conversations.append(log_entry)
        print(f"Logged conversation: {log_entry}")
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
            'config_id': config_id,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        sessions.append(session_entry)
        print(f"Logged session start: {session_entry}")
    except Exception as e:
        print(f"Error logging session start: {e}")

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
        
        print(f"API Key exists: {bool(hume_api_key)}")
        print(f"Secret Key exists: {bool(hume_secret_key)}")
        
        # Get config_id from query parameter, fallback to env variable
        config_id = request.args.get('config_id')
        if not config_id:
            config_id = os.environ.get('HUME_CONFIG_ID')
        
        print(f"Config ID: {config_id}")
        
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
        
        print(f"Returning auth data: {response_data}")
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
        
        # Log the assistant message
        log_conversation_memory(session_id, 'assistant', text, watch_model)
        
        if not watch_model:
            return jsonify({'watchModel': None, 'watchImage': None}), 200
        
        watch_image = watch_matcher.get_image_base64(watch_model)
        
        return jsonify({
            'watchModel': watch_model,
            'watchImage': watch_image
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
        for session in sessions:
            if session['session_id'] == session_id:
                session['end_time'] = datetime.now().isoformat()
                session['total_messages'] = len([c for c in conversations if c['session_id'] == session_id])
                break
        
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
        # Sort sessions by start time (newest first)
        sorted_sessions = sorted(sessions, key=lambda x: x['start_time'], reverse=True)
        return jsonify({'sessions': sorted_sessions}), 200
    except Exception as e:
        print(f"Get sessions error: {str(e)}")
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
            
