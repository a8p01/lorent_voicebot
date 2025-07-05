from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import uuid

app = Flask(__name__)
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp DATETIME,
            message_type TEXT,
            content TEXT,
            watch_model TEXT,
            user_agent TEXT,
            ip_address TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time DATETIME,
            end_time DATETIME,
            total_messages INTEGER,
            config_id TEXT,
            user_agent TEXT,
            ip_address TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_conversation(session_id, message_type, content, watch_model=None):
    """Log conversation data to database"""
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (id, session_id, timestamp, message_type, content, watch_model, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            session_id,
            datetime.now(),
            message_type,
            content,
            watch_model,
            request.headers.get('User-Agent'),
            request.remote_addr
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging conversation: {e}")

def log_session_start(session_id, config_id):
    """Log session start"""
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions 
            (session_id, start_time, config_id, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.now(),
            config_id,
            request.headers.get('User-Agent'),
            request.remote_addr
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging session start: {e}")

def log_session_end(session_id):
    """Log session end"""
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Count total messages in this session
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE session_id = ?', (session_id,))
        total_messages = cursor.fetchone()[0]
        
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, total_messages = ?
            WHERE session_id = ?
        ''', (datetime.now(), total_messages, session_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging session end: {e}")

# Initialize database
init_db()

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
        
        # Generate session ID for this conversation
        session_id = str(uuid.uuid4())
        log_session_start(session_id, config_id)
        
        return jsonify({
            'apiKey': hume_api_key,
            'secretKey': hume_secret_key,
            'configId': config_id,
            'sessionId': session_id
        }), 200
    except Exception as e:
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
        log_conversation(session_id, 'assistant', text, watch_model)
        
        if not watch_model:
            return jsonify({'watchModel': None, 'watchImage': None}), 200
        
        watch_image = watch_matcher.get_image_base64(watch_model)
        
        return jsonify({
            'watchModel': watch_model,
            'watchImage': watch_image
        }), 200
    except Exception as e:
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
        
        log_conversation(session_id, message_type, content, watch_model)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End a conversation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        
        log_session_end(session_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get conversation data for analysis"""
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Get query parameters
        session_id = request.args.get('session_id')
        limit = request.args.get('limit', 100)
        
        if session_id:
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp DESC
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT * FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        conversations = cursor.fetchall()
        
        # Convert to list of dictionaries
        columns = [description[0] for description in cursor.description]
        conversations_list = [dict(zip(columns, row)) for row in conversations]
        
        conn.close()
        
        return jsonify({'conversations': conversations_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get session data for analysis"""
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions ORDER BY start_time DESC')
        sessions = cursor.fetchall()
        
        # Convert to list of dictionaries
        columns = [description[0] for description in cursor.description]
        sessions_list = [dict(zip(columns, row)) for row in sessions]
        
        conn.close()
        
        return jsonify({'sessions': sessions_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch-models', methods=['GET'])
def get_watch_models():
    """Get all available watch models"""
    try:
        return jsonify({'models': list(watch_matcher.watch_models.keys())}), 200
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
    app.run()
