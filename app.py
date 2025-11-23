# app.py
import os
import json
import math
import string
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# ============================
# 1) Tải dữ liệu knowledge.json
# ============================
try:
    with open("knowledge.json", "r", encoding="utf-8") as f:
        KNOWLEDGE = json.load(f)
    print("✅ Loaded knowledge.json")
except Exception as e:
    KNOWLEDGE = {}
    print("⚠️ Không thể load knowledge.json:", e)


# ============================
# 2) OpenRouter API
# ============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = "https://api.openai.com/v1"


if not OPENROUTER_API_KEY:
    print("⚠️ CẢNH BÁO: OPENROUTER_API_KEY chưa được thiết lập!")


# ============================
# 3) Bộ công cụ NLP thông minh
# ============================

STOPWORDS = {
    "là", "và", "hoặc", "các", "những", "khi", "nào", "ở", "đi", "đến", "với",
    "gì", "có", "bao", "nhiêu", "cho", "tôi", "bạn", "thế", "nào"
}


def normalize(text):
    """Chuẩn hóa văn bản để so khớp chính xác hơn."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_keywords(text):
    """Tách từ khóa có ý nghĩa."""
    text = normalize(text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def semantic_score(query_keywords, text):
    """
    Chấm điểm mức độ liên quan theo ý nghĩa.
    - Điểm dựa trên tần suất từ khóa
    - Cộng thêm khi có cụm từ liên quan
    """
    if not isinstance(text, str):
        return 0

    base = normalize(text)
    score = 0

    for w in query_keywords:
        if w in base:
            score += 3  # từ khóa khớp

        # Bonus khi có dạng gần nghĩa
        if w[:-1] in base:
            score += 1

    return score


def flatten_knowledge(obj):
    """Trích toàn bộ text từ knowledge.json thành list."""
    results = []

    def scan(o):
        if isinstance(o, dict):
            for v in o.values():
                scan(v)
        elif isinstance(o, list):
            for x in o:
                scan(x)
        elif isinstance(o, str):
            results.append(o)

    scan(obj)
    return results


FLATTENED_KNOWLEDGE = flatten_knowledge(KNOWLEDGE)


def smart_search(query):
    """Tìm thông tin liên quan thông minh (semantic search)."""
    if not FLATTENED_KNOWLEDGE:
        return "Không có dữ liệu."

    keywords = extract_keywords(query)
    if not keywords:
        return "Không tìm thấy dữ liệu."

    scored = []

    for text in FLATTENED_KNOWLEDGE:
        s = semantic_score(keywords, text)
        if s > 0:
            scored.append((s, text))

    if not scored:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    # Sắp xếp theo điểm giảm dần
    scored.sort(reverse=True, key=lambda x: x[0])

    # Chỉ lấy 5 đoạn hay nhất
    top_texts = [t for _, t in scored[:5]]
    return "\n- ".join(top_texts)


# ============================
# 4) Home
# ============================
@app.route("/")
def home():
    if OPENROUTER_API_KEY:
        return "✅ Ruby Wings Backend Online — AI Ready — Knowledge Loaded"
    return "⚠️ Backend chạy nhưng thiếu OPENROUTER_API_KEY"


# ============================
# 5) Chat API
# ============================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập nội dung nào."})

        # 👉 Tìm kiến thức liên quan bằng tìm kiếm thông minh
        related_info = smart_search(user_message)

        # 👉 System Prompt tối tân (AI Ruby Wings)
        system_prompt = f"""
Bạn là trợ lý AI của công ty du lịch trải nghiệm Ruby Wings.
Trách nhiệm:
- Tư vấn chính xác, thân thiện, súc tích.
- Ưu tiên thông tin tìm thấy trong cơ sở dữ liệu nội bộ.
- Nếu knowledge.json không có thông tin, sử dụng kiến thức tổng hợp của AI du lịch.
- Tuyệt đối không bịa giá, lịch trình hoặc thông tin nội bộ nếu không có trong dữ liệu.

Dữ liệu nội bộ liên quan (tối đa 5 đoạn):
{related_info}

"""

        payload = {
            "model": "gpt-4o-mini",  # model thông minh - có thể đổi
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.6,
            "max_tokens": 700
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )

        if response.status_code not in (200, 201):
            try:
                msg = response.json().get("error") or response.text
            except:
                msg = response.text
            return jsonify({"reply": f"Lỗi OpenRouter {response.status_code}: {msg}"}), 500

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"Lỗi server: {e}"}), 500


# ============================
# 6) Run app
# ============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
