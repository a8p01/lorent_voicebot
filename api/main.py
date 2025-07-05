from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

class WatchImageMatcher:
    def __init__(self, images_folder="watch_images"):
        self.images_folder = Path(images_folder)
        if not self.images_folder.exists():
            self.images_folder.mkdir(exist_ok=True)
            
        self.watch_models = {
            "Linea": ["linea"],
            "Serene": ["serene"],
            "Winchester": ["winchester"],
            "Sheffield": ["sheffield"],
            "Ophelia": ["ophelia"],
            "Eterna": ["eterna"],
            "Lunaire Noir": ["lunaire noir", "lunaire-noir", "Lunaire-noir", "Lunaire-Noir"],
            "Lunaire Rose": ["lunaire rose", "lunaire-rose",  "Lunaire-rose", "Lunaire-Rose"],
            "Explorer": ["explorer"],
            "Dive Master": ["dive master", "Dive-master", "dive-master", "Dive-Master"],
            "Field Ranger": ["field ranger", "Field-ranger", "field-ranger", "Field-Ranger"],
            "Nightfall": ["nightfall"],
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
        return None
    
    def get_image_base64(self, model_name):
        if not model_name:
            return None
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_path = self.images_folder / f"{model_name}{ext}"
            if image_path.exists():
                try:
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        return base64.b64encode(image_data).decode('utf-8')
                except Exception as e:
                    print(f"Error reading image: {e}")
                    return None
        return None

watch_matcher = WatchImageMatcher()

@app.route('/')
def index():
    try:
        html_path = Path(__file__).parent.parent / 'static' / 'index.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth', methods=['GET'])
def get_auth_token():
    try:
        hume_api_key = os.environ.get('HUME_API_KEY')
        hume_secret_key = os.environ.get('HUME_SECRET_KEY')
        config_id = request.args.get('config_id') or os.environ.get('HUME_CONFIG_ID')
        
        if not all([hume_api_key, hume_secret_key, config_id]):
            missing = []
            if not hume_api_key: missing.append('HUME_API_KEY')
            if not hume_secret_key: missing.append('HUME_SECRET_KEY')
            if not config_id: missing.append('HUME_CONFIG_ID')
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
    try:
        data = request.get_json()
        text = data.get('text', '')
        watch_model = watch_matcher.find_watch_model(text)
        
        if not watch_model:
            return jsonify({'watchModel': None, 'watchImage': None}), 200
        
        watch_image = watch_matcher.get_image_base64(watch_model)
        return jsonify({'watchModel': watch_model, 'watchImage': watch_image}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'status': 'working', 'env_vars_set': {
        'HUME_API_KEY': bool(os.environ.get('HUME_API_KEY')),
        'HUME_SECRET_KEY': bool(os.environ.get('HUME_SECRET_KEY')),
        'HUME_CONFIG_ID': bool(os.environ.get('HUME_CONFIG_ID'))
    }}), 200

if __name__ == "__main__":
    app.run()
