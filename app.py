from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import traceback

app = Flask(__name__)
CORS(app)

# Lấy API key từ biến môi trường (không commit key vào git)
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/')
def home():
    # Không khẳng định "kết nối OpenAI thành công" nếu chưa test key
    if openai.api_key:
        return "✅ Backend đang chạy - OPENAI_API_KEY được thiết lập"
    else:
        return "⚠️ Backend chạy nhưng OPENAI_API_KEY chưa được thiết lập"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150
        )
        bot_reply = response.choices[0].message.content
        return jsonify({'reply': bot_reply})
    except Exception as e:
        # trả traceback trong logs, trả message ngắn cho client
        traceback_str = traceback.format_exc()
        app.logger.error(traceback_str)
        return jsonify({'reply': 'Lỗi nội bộ. Kiểm tra logs.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
