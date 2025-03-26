import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import LLaMAConfig, LLaMAModel
import torch

config = LLaMAConfig(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
model = LLaMAModel(config)

# If you have a checkpoint, load it:
# state_dict = torch.load("llama_model.pth", map_location="cpu")
# model.load_pretrained(state_dict)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint to receive a user message and return the AI response."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_msg = data.get('message')
        history = data.get('history', [])
        
        if not user_msg:
            return jsonify({'error': 'Empty message'}), 400

        if history:
            prompt = "".join([f"{msg['sender']}: {msg['text']}\n" for msg in history])
            prompt += f"User: {user_msg}\nAssistant:"
        else:
            prompt = f"User: {user_msg}\nAssistant:"

        ai_response = model.generate_text(prompt)

        return jsonify({
            'reply': ai_response,
            'timestamp': torch.datetime.now().isoformat()
        })

    except Exception as e:
        print("Error: ", traceback.format_exc())
        return jsonify({'error': 'Error'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)