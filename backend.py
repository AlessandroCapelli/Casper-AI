import os
import logging
from datetime import datetime, timezone
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import LLMConfig, LLMModel

MODEL_PATH = 'Model'
CORS_ORIGINS = 'http://localhost:4200'
PORT = 5000
DEBUG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LLMModel(LLMConfig())

if os.path.exists(MODEL_PATH):
    try:
        model.load(MODEL_PATH)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
else:
    try:
        model.save(MODEL_PATH)
        logger.info(f"Initial model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error saving initial model: {e}", exc_info=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

@app.route('/chat', methods=['POST'])
def chat() -> tuple:
    try:
        history: List[Dict[str, str]] = request.get_json(force=True).get('history', [])

        if not history or len(history) == 0:
            return jsonify({'error': 'Empty message'}), 400

        prompt = ''.join(f"{m['sender']}: {m['text'].strip()}\n" for m in history if m['sender'] != 'Error') + f"Assistant:"
        response = model.generate_text(prompt)

        return jsonify({
            'reply': response,
            'timestamp': datetime.now(timezone.utc)
        }), 200

    except Exception as e:
        logger.error('Unhandled exception in /chat', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)