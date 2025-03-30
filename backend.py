import os
import logging
from datetime import datetime, timezone
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import LLMConfig, LLMModel, load_json_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = load_json_config("config.json").get("llm_config")
backend_config = load_json_config("config.json").get("backend_config")

model = LLMModel(LLMConfig(llm_config))

if os.path.exists(llm_config.get('model_path')):
    try:
        model.load(llm_config.get('model_path'))
    except Exception as e:
        logger.error(f'Error loading model: {e}', exc_info=True)
else:
    try:
        model.save(llm_config.get('model_path'))
    except Exception as e:
        logger.error(f'Error saving model: {e}', exc_info=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": backend_config.get("cors_origins")}})


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
    app.run(host="0.0.0.0", port=backend_config.get("port"), debug=backend_config.get("debug"))