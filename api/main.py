
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from pathlib import Path
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.conversations_collection = None
        self.sessions_collection = None
        self.connect()
    
    def connect(self):
        """Initialize MongoDB connection"""
        try:
            mongo_uri = os.environ.get('MONGODB_URI')
            if not mongo_uri:
                logger.warning("MONGODB_URI not found - database features disabled")
                return
                
            self.client = MongoClient(mongo_uri)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Initialize database and collections
            db_name = os.environ.get('MONGODB_DB_NAME', 'lorent_voicebot')
            self.db = self.client[db_name]
            
            self.conversations_collection = self.db.conversations
            self.sessions_collection = self.db.sessions
            
            # Create indexes for better performance
            self.conversations_collection.create_index("session_id")
            self.conversations_collection.create_index("timestamp")
            self.sessions_collection.create_index("session_id")
            self.sessions_collection.create_index("start_time")
            
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
    
    def create_session(self, session_id, user_agent=None, ip_address=None):
        """Create a new conversation session"""
        if not self.client:
            return None
            
        try:
            session_data = {
                "session_id": session_id,
                "start_time": datetime.now(timezone.utc),
                "end_time": None,
                "user_agent": user_agent,
                "ip_address": ip_address,
                "message_count": 0,
                "watch_recommendations": [],
                "session_duration": None,
                "status": "active"
            }
            
            result = self.sessions_collection.insert_one(session_data)
            logger.info(f"Created session: {session_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def end_session(self, session_id):
        """End a conversation session"""
        if not self.client:
            return False
            
        try:
            end_time = datetime.now(timezone.utc)
            
            # Get session start time to calculate duration
            session = self.sessions_collection.find_one({"session_id": session_id})
            if session:
                duration = (end_time - session['start_time']).total_seconds()
                
                self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "end_time": end_time,
                            "session_duration": duration,
                            "status": "completed"
                        }
                    }
                )
                
                logger.info(f"Ended session: {session_id}, Duration: {duration}s")
                return True
                
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            
        return False
    
    def log_conversation(self, session_id, message_type, content, watch_model=None, 
                        emotions=None, audio_duration=None, metadata=None):
        """Log a conversation message"""
        if not self.client:
            return None
            
        try:
            conversation_data = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "message_type": message_type,  # 'user' or 'assistant'
                "content": content,
                "watch_model": watch_model,
                "emotions": emotions,
                "audio_duration": audio_duration,
                "metadata": metadata or {}
            }
            
            result = self.conversations_collection.insert_one(conversation_data)
            
            # Update session message count
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$inc": {"message_count": 1}}
            )
            
            # If watch model was recommended, add to session
            if watch_model:
                self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$addToSet": {"watch_recommendations": watch_model}}
                )
            
            logger.info(f"Logged conversation: {session_id}, Type: {message_type}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
            return None

class WatchImageMatcher:
    def __init__(self, images_folder="watch_images"):
        self.images_folder = Path(images_folder)
        # Ensure the images folder exists
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
            logger.error(f"Image not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return None

# Initialize components
watch_matcher = WatchImageMatcher()
db_manager = DatabaseManager()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Path relative to api directory
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
        hume_api_key = os.environ.get('HUME_API_KEY')
        hume_secret_key = os.environ.get('HUME_SECRET_KEY')
        
        # Get config_id from query parameter, fallback to env variable
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
            return jsonify({'error': f'Missing: {", ".join(missing)}'}), 500
        
        return jsonify({
            'apiKey': hume_api_key,
            'secretKey': hume_secret_key,
            'configId': config_id
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch-image', methods=['POST'])
def get_watch_image():
    """Get watch image based on text content"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        watch_model = watch_matcher.find_watch_model(text)
        
        if not watch_model:
            return jsonify({'watchModel': None, 'watchImage': None}), 200
        
        watch_image = watch_matcher.get_image_base64(watch_model)
        
        return jsonify({
            'watchModel': watch_model,
            'watchImage': watch_image
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch-models', methods=['GET'])
def get_watch_models():
    """Get all available watch models"""
    try:
        return jsonify({'models': list(watch_matcher.watch_models.keys())}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoints for conversation tracking
@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new conversation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_agent = request.headers.get('User-Agent')
        ip_address = request.remote_addr
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        session_doc_id = db_manager.create_session(session_id, user_agent, ip_address)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'session_doc_id': session_doc_id
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/end', methods=['POST'])
def end_session():
    """End a conversation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        success = db_manager.end_session(session_id)
        
        return jsonify({
            'success': success,
            'session_id': session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/log', methods=['POST'])
def log_conversation():
    """Log a conversation message"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        message_type = data.get('message_type')  # 'user' or 'assistant'
        content = data.get('content')
        watch_model = data.get('watch_model')
        emotions = data.get('emotions')
        audio_duration = data.get('audio_duration')
        metadata = data.get('metadata', {})
        
        if not all([session_id, message_type, content]):
            return jsonify({'error': 'session_id, message_type, and content are required'}), 400
        
        conversation_id = db_manager.log_conversation(
            session_id=session_id,
            message_type=message_type,
            content=content,
            watch_model=watch_model,
            emotions=emotions,
            audio_duration=audio_duration,
            metadata=metadata
        )
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")
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
