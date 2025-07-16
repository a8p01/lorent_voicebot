from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import uuid
import re
import asyncio
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from hume import AsyncHumeClient
from hume.empathic_voice.types import ReturnChatEvent
from typing import List, Dict, Optional, cast

app = Flask(__name__)
CORS(app)

# MongoDB connection
def get_mongodb_client():
    """Get MongoDB client with connection string from environment"""
    try:
        connection_string = os.environ.get('MONGODB_CONNECTION_STRING')
        if not connection_string:
            print("ERROR: MONGODB_CONNECTION_STRING environment variable not set")
            return None
        
        client = MongoClient(connection_string)
        # Test connection
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return client
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return None
    except Exception as e:
        print(f"MongoDB error: {e}")
        return None

def get_database():
    """Get database instance"""
    client = get_mongodb_client()
    if client is None:
        return None
    return client['voicebot_analytics']

def test_database_connection():
    """Test if database connection works"""
    try:
        db = get_database()
        if db is None:
            return False
        # Try a simple operation to test connection
        db.test_collection.find_one()
        return True
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

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

def log_conversation_mongodb(session_id, message_type, content, watch_model=None, emotions=None, chat_id=None):
    """Log conversation data to MongoDB with emotion data"""
    try:
        db = get_database()
        if db is None:
            print("Failed to connect to database")
            return
        
        text_analysis = analyze_text(content)
        
        conversation_doc = {
            '_id': str(uuid.uuid4()),
            'session_id': session_id,
            'chat_id': chat_id,  # Add Hume chat ID
            'timestamp': datetime.now(),
            'message_type': message_type,
            'content': content,
            'watch_model': watch_model,
            'emotions': emotions,  # Store emotion data
            'word_count': text_analysis['word_count'],
            'character_count': text_analysis['character_count'],
            'sentence_count': text_analysis['sentence_count'],
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        
        result = db.conversations.insert_one(conversation_doc)
        print(f"Logged conversation: {message_type} - {watch_model} - {text_analysis['word_count']} words - ID: {result.inserted_id}")
        
    except Exception as e:
        print(f"Error logging conversation to MongoDB: {e}")

def log_session_start_mongodb(session_id, config_id):
    """Log session start to MongoDB"""
    try:
        db = get_database()
        if db is None:
            print("Failed to connect to database")
            return
        
        session_doc = {
            '_id': session_id,
            'session_id': session_id,
            'chat_id': None,  # Will be updated when chat_metadata is received
            'chat_group_id': None,  # Will be updated when chat_metadata is received
            'start_time': datetime.now(),
            'end_time': None,
            'total_messages': 0,
            'total_words': 0,
            'duration_seconds': 0,
            'watch_models_shown': [],
            'user_messages': 0,
            'assistant_messages': 0,
            'config_id': config_id,
            'emotions_processed': False,  # Track if emotions have been extracted
            'top_emotions': None,  # Store top emotions for the session
            'emotion_summary': None,  # Store emotion analytics
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        }
        
        result = db.sessions.insert_one(session_doc)
        print(f"Logged session start: {result.inserted_id}")
        
    except Exception as e:
        print(f"Error logging session start to MongoDB: {e}")

def update_session_metadata_mongodb(session_id, chat_id, chat_group_id):
    """Update session with Hume chat metadata"""
    try:
        db = get_database()
        if db is None:
            return
        
        update_data = {
            'chat_id': chat_id,
            'chat_group_id': chat_group_id
        }
        
        result = db.sessions.update_one(
            {'session_id': session_id},
            {'$set': update_data}
        )
        
        print(f"Updated session metadata: chat_id={chat_id}, chat_group_id={chat_group_id}")
        
    except Exception as e:
        print(f"Error updating session metadata in MongoDB: {e}")

def update_session_stats_mongodb(session_id):
    """Update session statistics in MongoDB"""
    try:
        db = get_database()
        if db is None:
            return
        
        # Get all conversations for this session
        conversations = list(db.conversations.find({'session_id': session_id}))
        
        # Calculate statistics
        user_messages = [c for c in conversations if c['message_type'] == 'user']
        assistant_messages = [c for c in conversations if c['message_type'] == 'assistant']
        watch_models_shown = list(set([c['watch_model'] for c in conversations if c.get('watch_model')]))
        total_words = sum(c.get('word_count', 0) for c in conversations)
        
        # Calculate emotion summary for user messages with emotions
        user_messages_with_emotions = [c for c in user_messages if c.get('emotions')]
        emotion_summary = calculate_emotion_summary(user_messages_with_emotions)
        
        # Update session document
        update_data = {
            'total_messages': len(conversations),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'total_words': total_words,
            'watch_models_shown': watch_models_shown,
            'emotion_summary': emotion_summary
        }
        
        result = db.sessions.update_one(
            {'session_id': session_id},
            {'$set': update_data}
        )
        
        print(f"Updated session stats: {len(conversations)} messages, {len(watch_models_shown)} watches shown")
        
    except Exception as e:
        print(f"Error updating session stats in MongoDB: {e}")

def calculate_emotion_summary(user_messages_with_emotions):
    """Calculate emotion summary from user messages"""
    if not user_messages_with_emotions:
        return None
    
    try:
        emotion_sums = {}
        total_messages = len(user_messages_with_emotions)
        
        # Sum up all emotions across messages
        for message in user_messages_with_emotions:
            emotions = message.get('emotions', {})
            for emotion, score in emotions.items():
                if emotion in emotion_sums:
                    emotion_sums[emotion] += score
                else:
                    emotion_sums[emotion] = score
        
        # Calculate averages
        emotion_averages = {
            emotion: score / total_messages 
            for emotion, score in emotion_sums.items()
        }
        
        # Get top 5 emotions
        sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
        top_emotions = dict(sorted_emotions[:5])
        
        return {
            'total_emotional_messages': total_messages,
            'average_emotions': emotion_averages,
            'top_emotions': top_emotions
        }
        
    except Exception as e:
        print(f"Error calculating emotion summary: {e}")
        return None

# Async function to fetch chat events from Hume
async def fetch_chat_events(chat_id: str) -> List[ReturnChatEvent]:
    """Fetch all chat events for a specific chat ID"""
    try:
        api_key = os.environ.get('HUME_API_KEY')
        if not api_key:
            print("HUME_API_KEY not found")
            return []
        
        client = AsyncHumeClient(api_key=api_key)
        all_chat_events: List[ReturnChatEvent] = []
        
        # Fetch events with pagination
        page_number = 0
        while True:
            try:
                response = await client.empathic_voice.chats.list_chat_events(
                    id=chat_id, 
                    page_number=page_number,
                    page_size=100  # Max page size
                )
                
                events_found = False
                async for event in response:
                    all_chat_events.append(event)
                    events_found = True
                
                if not events_found:
                    break
                    
                page_number += 1
                
            except Exception as e:
                print(f"Error fetching page {page_number}: {e}")
                break
        
        print(f"Fetched {len(all_chat_events)} events for chat {chat_id}")
        return all_chat_events
        
    except Exception as e:
        print(f"Error fetching chat events: {e}")
        return []

def extract_emotions_from_events(chat_events: List[ReturnChatEvent]) -> Dict[str, any]:
    """Extract emotion data from chat events"""
    try:
        # Filter user messages that have emotion features
        user_messages = [
            e for e in chat_events 
            if e.type == "USER_MESSAGE" and hasattr(e, 'emotion_features') and e.emotion_features
        ]
        
        if not user_messages:
            return {
                'total_emotional_messages': 0,
                'emotions_by_message': [],
                'average_emotions': {},
                'top_emotions': {}
            }
        
        total_messages = len(user_messages)
        emotions_by_message = []
        
        # Parse the emotion features of the first user message to determine emotion keys
        first_message_emotions = json.loads(cast(str, user_messages[0].emotion_features))
        emotion_keys: List[str] = list(first_message_emotions.keys())
        
        # Initialize sums for all emotions to 0
        emotion_sums = {key: 0.0 for key in emotion_keys}
        
        # Accumulate emotion scores from each user message
        for event in user_messages:
            emotions = json.loads(cast(str, event.emotion_features))
            emotions_by_message.append({
                'timestamp': event.timestamp.isoformat() if hasattr(event, 'timestamp') and event.timestamp else None,
                'emotions': emotions
            })
            
            for key in emotion_keys:
                emotion_sums[key] += emotions.get(key, 0)
        
        # Compute average scores for each emotion
        average_emotions = {key: emotion_sums[key] / total_messages for key in emotion_keys}
        
        # Sort by average score (descending) and get top 10
        sorted_emotions = sorted(average_emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = dict(sorted_emotions[:10])
        
        return {
            'total_emotional_messages': total_messages,
            'emotions_by_message': emotions_by_message,
            'average_emotions': average_emotions,
            'top_emotions': top_emotions
        }
        
    except Exception as e:
        print(f"Error extracting emotions from events: {e}")
        return {
            'total_emotional_messages': 0,
            'emotions_by_message': [],
            'average_emotions': {},
            'top_emotions': {}
        }

def run_async_function(func, *args):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(func(*args))

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
        log_session_start_mongodb(session_id, config_id)
        
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
        
        if watch_model:
            print(f"Watch model found: {watch_model}")
            
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
        emotions = data.get('emotions', None)
        chat_id = data.get('chat_id', None)
        
        log_conversation_mongodb(session_id, message_type, content, watch_model, emotions, chat_id)
        update_session_stats_mongodb(session_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Log message error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/log-chat-metadata', methods=['POST'])
def log_chat_metadata():
    """Log Hume chat metadata"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        chat_id = data.get('chat_id')
        chat_group_id = data.get('chat_group_id')
        
        if chat_id and chat_group_id:
            update_session_metadata_mongodb(session_id, chat_id, chat_group_id)
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Missing chat_id or chat_group_id'}), 400
            
    except Exception as e:
        print(f"Log chat metadata error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract-emotions', methods=['POST'])
def extract_emotions():
    """Extract emotions for a specific session using Hume API"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        
        db = get_database()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get session data to find chat_id
        session = db.sessions.find_one({'session_id': session_id})
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_id = session.get('chat_id')
        if not chat_id:
            return jsonify({'error': 'No chat_id found for this session'}), 400
        
        # Fetch chat events from Hume
        chat_events = run_async_function(fetch_chat_events, chat_id)
        
        if not chat_events:
            return jsonify({'error': 'No chat events found'}), 404
        
        # Extract emotions
        emotion_data = extract_emotions_from_events(chat_events)
        
        # Update session with emotion data
        db.sessions.update_one(
            {'session_id': session_id},
            {'$set': {
                'emotions_processed': True,
                'top_emotions': emotion_data['top_emotions'],
                'emotion_summary': emotion_data
            }}
        )
        
        # Update conversations with individual emotion data
        for i, emotion_entry in enumerate(emotion_data['emotions_by_message']):
            # Find corresponding conversation entry
            conversations = list(db.conversations.find({
                'session_id': session_id,
                'message_type': 'user'
            }).sort('timestamp', 1))
            
            if i < len(conversations):
                db.conversations.update_one(
                    {'_id': conversations[i]['_id']},
                    {'$set': {'emotions': emotion_entry['emotions']}}
                )
        
        print(f"Extracted emotions for session {session_id}: {len(emotion_data['emotions_by_message'])} emotional messages")
        
        return jsonify({
            'success': True,
            'emotion_data': emotion_data
        }), 200
        
    except Exception as e:
        print(f"Extract emotions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End a conversation session and extract emotions"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        
        db = get_database()
        if db is not None:
            end_time = datetime.now()
            
            # Get session start time to calculate duration
            session = db.sessions.find_one({'session_id': session_id})
            if session:
                start_time = session['start_time']
                duration_seconds = int((end_time - start_time).total_seconds())
                
                # Update session with end time and duration
                db.sessions.update_one(
                    {'session_id': session_id},
                    {'$set': {
                        'end_time': end_time,
                        'duration_seconds': duration_seconds
                    }}
                )
                
                # Try to extract emotions if we have a chat_id
                chat_id = session.get('chat_id')
                if chat_id and not session.get('emotions_processed', False):
                    try:
                        # Add a small delay to ensure all events are available
                        import time
                        time.sleep(2)
                        
                        # Extract emotions
                        chat_events = run_async_function(fetch_chat_events, chat_id)
                        if chat_events:
                            emotion_data = extract_emotions_from_events(chat_events)
                            
                            # Update session with emotion data
                            db.sessions.update_one(
                                {'session_id': session_id},
                                {'$set': {
                                    'emotions_processed': True,
                                    'top_emotions': emotion_data['top_emotions'],
                                    'emotion_summary': emotion_data
                                }}
                            )
                            
                            print(f"Auto-extracted emotions for ended session {session_id}")
                            
                    except Exception as e:
                        print(f"Failed to auto-extract emotions: {e}")
        
        update_session_stats_mongodb(session_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"End session error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get conversation data for analysis"""
    try:
        db = get_database()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 100))
        include_emotions = request.args.get('include_emotions', 'false').lower() == 'true'
        
        # Build query
        query = {}
        if session_id:
            query['session_id'] = session_id
        
        # Get conversations from MongoDB
        conversations = list(db.conversations.find(query)
                           .sort('timestamp', -1)  # Newest first
                           .limit(limit))
        
        # Convert ObjectId and datetime to strings for JSON serialization
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
            conv['timestamp'] = conv['timestamp'].isoformat()
            
            # Optionally exclude emotions data if not requested
            if not include_emotions and 'emotions' in conv:
                del conv['emotions']
        
        return jsonify({'conversations': conversations}), 200
    except Exception as e:
        print(f"Get conversations error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get session data for analysis"""
    try:
        db = get_database()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Update all session stats before returning
        for session in db.sessions.find():
            update_session_stats_mongodb(session['session_id'])
        
        # Get sessions from MongoDB
        sessions = list(db.sessions.find().sort('start_time', -1))
        
        # Convert datetime to strings for JSON serialization
        for session in sessions:
            session['_id'] = str(session['_id'])
            session['start_time'] = session['start_time'].isoformat()
            if session.get('end_time'):
                session['end_time'] = session['end_time'].isoformat()
        
        return jsonify({'sessions': sessions}), 200
    except Exception as e:
        print(f"Get sessions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get advanced analytics including emotion data"""
    try:
        db = get_database()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Update all session stats
        for session in db.sessions.find():
            update_session_stats_mongodb(session['session_id'])
        
        # Get all data
        sessions = list(db.sessions.find())
        conversations = list(db.conversations.find())
        
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
                'popular_watches': [],
                'emotion_analytics': {}
            }), 200
        
        # Calculate analytics
        user_messages = [c for c in conversations if c['message_type'] == 'user']
        assistant_messages = [c for c in conversations if c['message_type'] == 'assistant']
        
        # Watch analytics
        watch_displays = [c for c in conversations if c.get('watch_model')]
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
        total_words = sum(c.get('word_count', 0) for c in conversations)
        avg_words_per_message = total_words / total_conversations if total_conversations > 0 else 0
        
        # Session statistics
        completed_sessions = [s for s in sessions if s.get('end_time')]
        avg_duration = 0
        if completed_sessions:
            total_duration = sum(s.get('duration_seconds', 0) for s in completed_sessions)
            avg_duration = total_duration / len(completed_sessions)
        
        # Emotion analytics
        emotion_analytics = calculate_global_emotion_analytics(sessions, conversations)
        
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
            'popular_watches': [{'model': model, 'count': count} for model, count in popular_watches],
            'emotion_analytics': emotion_analytics
        }
        
        return jsonify(analytics), 200
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_global_emotion_analytics(sessions, conversations):
    """Calculate global emotion analytics across all sessions"""
    try:
        # Sessions with emotion data
        sessions_with_emotions = [s for s in sessions if s.get('emotions_processed') and s.get('top_emotions')]
        
        # Conversations with emotion data
        conversations_with_emotions = [c for c in conversations if c.get('emotions') and c.get('message_type') == 'user']
        
        if not sessions_with_emotions and not conversations_with_emotions:
            return {
                'sessions_with_emotions': 0,
                'messages_with_emotions': 0,
                'global_top_emotions': {},
                'emotion_trends': [],
                'average_emotional_intensity': 0
            }
        
        # Global emotion aggregation
        global_emotion_sums = {}
        total_emotional_sessions = len(sessions_with_emotions)
        
        # Aggregate from session summaries
        for session in sessions_with_emotions:
            top_emotions = session.get('top_emotions', {})
            for emotion, score in top_emotions.items():
                if emotion in global_emotion_sums:
                    global_emotion_sums[emotion] += score
                else:
                    global_emotion_sums[emotion] = score
        
        # Calculate global averages
        global_avg_emotions = {}
        if total_emotional_sessions > 0:
            global_avg_emotions = {
                emotion: score / total_emotional_sessions 
                for emotion, score in global_emotion_sums.items()
            }
        
        # Get top 10 global emotions
        sorted_global_emotions = sorted(global_avg_emotions.items(), key=lambda x: x[1], reverse=True)
        global_top_emotions = dict(sorted_global_emotions[:10])
        
        # Calculate average emotional intensity (sum of top 3 emotions per session)
        emotional_intensities = []
        for session in sessions_with_emotions:
            top_emotions = session.get('top_emotions', {})
            top_3_sum = sum(list(top_emotions.values())[:3])
            emotional_intensities.append(top_3_sum)
        
        avg_emotional_intensity = sum(emotional_intensities) / len(emotional_intensities) if emotional_intensities else 0
        
        # Emotion trends over time (simplified)
        emotion_trends = []
        if conversations_with_emotions:
            # Group conversations by date
            from collections import defaultdict
            emotions_by_date = defaultdict(list)
            
            for conv in conversations_with_emotions:
                try:
                    date_str = conv['timestamp'][:10]  # Extract YYYY-MM-DD
                    emotions = conv.get('emotions', {})
                    emotions_by_date[date_str].append(emotions)
                except:
                    continue
            
            # Calculate daily emotion averages
            for date, daily_emotions in emotions_by_date.items():
                daily_sums = {}
                for emotions in daily_emotions:
                    for emotion, score in emotions.items():
                        if emotion in daily_sums:
                            daily_sums[emotion] += score
                        else:
                            daily_sums[emotion] = score
                
                daily_averages = {
                    emotion: score / len(daily_emotions) 
                    for emotion, score in daily_sums.items()
                }
                
                # Get top emotion for the day
                if daily_averages:
                    top_emotion = max(daily_averages.items(), key=lambda x: x[1])
                    emotion_trends.append({
                        'date': date,
                        'top_emotion': top_emotion[0],
                        'score': top_emotion[1],
                        'total_messages': len(daily_emotions)
                    })
        
        return {
            'sessions_with_emotions': len(sessions_with_emotions),
            'messages_with_emotions': len(conversations_with_emotions),
            'global_top_emotions': global_top_emotions,
            'emotion_trends': sorted(emotion_trends, key=lambda x: x['date']),
            'average_emotional_intensity': round(avg_emotional_intensity, 3)
        }
        
    except Exception as e:
        print(f"Error calculating global emotion analytics: {e}")
        return {
            'sessions_with_emotions': 0,
            'messages_with_emotions': 0,
            'global_top_emotions': {},
            'emotion_trends': [],
            'average_emotional_intensity': 0
        }

@app.route('/api/emotions/<session_id>', methods=['GET'])
def get_session_emotions(session_id):
    """Get emotion data for a specific session"""
    try:
        db = get_database()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get session
        session = db.sessions.find_one({'session_id': session_id})
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get conversations with emotions
        conversations = list(db.conversations.find({
            'session_id': session_id,
            'message_type': 'user',
            'emotions': {'$exists': True, '$ne': None}
        }).sort('timestamp', 1))
        
        # Convert timestamps for JSON
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
            conv['timestamp'] = conv['timestamp'].isoformat()
        
        response_data = {
            'session_id': session_id,
            'emotions_processed': session.get('emotions_processed', False),
            'top_emotions': session.get('top_emotions', {}),
            'emotion_summary': session.get('emotion_summary', {}),
            'emotional_messages': conversations
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Get session emotions error: {str(e)}")
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
    try:
        # Test database connection properly
        db_connected = test_database_connection()
        db_status = "Connected" if db_connected else "Failed"
        
        response_data = {
            'message': 'API is working!',
            'timestamp': datetime.now().isoformat(),
            'database_status': db_status,
            'environment_vars': {
                'HUME_API_KEY': bool(os.environ.get('HUME_API_KEY')),
                'HUME_SECRET_KEY': bool(os.environ.get('HUME_SECRET_KEY')),
                'HUME_CONFIG_ID': bool(os.environ.get('HUME_CONFIG_ID')),
                'MONGODB_CONNECTION_STRING': bool(os.environ.get('MONGODB_CONNECTION_STRING'))
            }
        }
        
        # Add more details if database is connected
        if db_connected:
            try:
                db = get_database()
                conversation_count = db.conversations.count_documents({})
                session_count = db.sessions.count_documents({})
                emotions_processed_count = db.sessions.count_documents({'emotions_processed': True})
                
                response_data['database_stats'] = {
                    'conversations': conversation_count,
                    'sessions': session_count,
                    'sessions_with_emotions': emotions_processed_count
                }
            except Exception as e:
                response_data['database_error'] = str(e)
        
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# This is required for Vercel
if __name__ == "__main__":
    app.run(debug=True)
