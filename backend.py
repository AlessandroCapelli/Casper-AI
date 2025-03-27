import os
import logging
from datetime import datetime, timezone
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import LLaMAConfig, LLaMAModel

# Configuration
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'model.pth')
SAVE_MODEL = os.getenv('SAVE_MODEL', 'False').lower() in ('true', '1', 'yes')
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:4200')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model — ora usiamo distilgpt2
config = LLaMAConfig()
model = LLaMAModel(config)

# Aggiorniamo tokenizer per gestire correttamente il pad_token
model.tokenizer.pad_token = model.tokenizer.eos_token

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load(MODEL_SAVE_PATH)
        logger.info(f"Successfully loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        logger.info("Starting with fresh model")
else:
    try:
        model.save(MODEL_SAVE_PATH)
        logger.info(f"Initial model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Error saving initial model: {e}", exc_info=True)

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

@app.route('/chat', methods=['POST'])
def chat() -> tuple:
    try:
        payload = request.get_json(force=True)
        user_msg = payload.get('message', '').strip()
        history: List[Dict[str, str]] = payload.get('history', [])

        if not user_msg:
            return jsonify({'error': 'Empty message'}), 400

        prompt = "".join(f"{m['sender']}: {m['text']}\n" for m in history if m['sender'] != 'Error') + f"{user_msg}\nAssistant:"
        response = model.generate_text(prompt)

        if SAVE_MODEL:
            try:
                model.save(MODEL_SAVE_PATH)
                logger.info(f"Model saved after chat to {MODEL_SAVE_PATH}")
            except Exception as e:
                logger.error(f"Error saving model after chat: {e}", exc_info=True)

        return jsonify({
            'reply': response,
            'timestamp': datetime.now(timezone.utc)
        }), 200

    except Exception as e:
        logger.error('Unhandled exception in /chat', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
