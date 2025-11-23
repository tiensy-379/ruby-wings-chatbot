# app.py
import os
import json
import string
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================
# Config: API keys & endpoints
# ============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

if not OPENAI_API_KEY:
    print("⚠️ CẢNH BÁO: OPENAI_API_KEY chưa được thiết lập! (Set env var OPENAI_API_KEY).")

# ============================
# 1) Load knowledge.json robustly
# ============================
KNOWLEDGE = {}
KNOWLEDGE_PATH = "knowledge.json"

def try_load_json(path):
    """Thử đọc file JSON, nếu lỗi 'Extra data' cố gắng ghép nhiều object thành list."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Thử heuristic: nếu file chứa nhiều JSON objects liên tiếp, chuyển thành list
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            # heuristic: nếu file có nhiều '}\n{' hoặc '}\r\n{', chuyển thành array
            if "}\n{" in txt or "}\r\n{" in txt:
                repaired = "[" + txt.replace("}\r\n{", "},{").replace("}\n{", "},{") + "]"
                return json.loads(repaired)
            # heuristic: nếu file có nhiều JSON per-line (ndjson), parse each line
            lines = [line.strip() for line in txt.splitlines() if line.strip()]
            if len(lines) > 1:
                objs = []
                for ln in lines:
                    try:
                        objs.append(json.loads(ln))
                    except:
                        # skip problematic line
                        pass
                if objs:
                    return objs
        except Exception as e2:
            print("⚠️ Lỗi khi cố gắng phục hồi knowledge.json:", e2)
        # nếu không phục hồi được, ném ra để thông báo
        raise e

try:
    KNOWLEDGE = try_load_json(KNOWLEDGE_PATH)
    print("✅ Loaded knowledge.json")
except Exception as e:
    KNOWLEDGE = {}
    print("⚠️ Không thể load knowledge.json:", e)

# ============================
# 2) NLP helpers & semantic search (lightweight)
# ============================
STOPWORDS = {
    "là", "và", "hoặc", "các", "những", "khi", "nào", "ở", "đi", "đến", "với",
    "gì", "có", "bao", "nhiêu", "cho", "tôi", "bạn", "thế", "nào", "the", "a", "an"
}

def normalize(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return t

def extract_keywords(text):
    txt = normalize(text)
    words = txt.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def semantic_score(query_keywords, text):
    if not isinstance(text, str):
        return 0
    base = normalize(text)
    score = 0
    for w in query_keywords:
        if w in base:
            score += 3
        if len(w) > 3 and w[:-1] in base:
            score += 1
    return score

def flatten_knowledge(obj):
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
    if not FLATTENED_KNOWLEDGE:
        return []
    keywords = extract_keywords(query)
    if not keywords:
        return []
    scored = []
    for text in FLATTENED_KNOWLEDGE:
        s = semantic_score(keywords, text)
        if s > 0:
            scored.append((s, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    # trả về tối đa 5 đoạn
    return [t for _, t in scored[:5]]

# ============================
# 3) OpenAI call helper
# ============================
def call_openai_chat(system_prompt, user_message, model="gpt-4o-mini", temperature=0.6, max_tokens=800, timeout=60):
    """
    Gọi OpenAI Chat Completions (sử dụng OpenAI_BASE_URL).
    Trả về text hoặc raise Exception.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY chưa cấu hình.")
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code not in (200, 201):
        # try extract message
        try:
            j = resp.json()
            err = j.get("error") or j
        except:
            err = resp.text
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {err}")
    data = resp.json()
    # robust extraction
    try:
        # new responses format may differ; try common keys
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content")
            if content:
                return content
            # fallback to 'text' key
            return data["choices"][0].get("text", "")
        # fallback single 'output' style
        if "output" in data:
            return data["output"]
        # default
        return json.dumps(data)
    except Exception as e:
        raise RuntimeError(f"Không thể trích nội dung trả lời từ OpenAI: {e}")

# ============================
# 4) Routes
# ============================
@app.route("/")
def home():
    ok = bool(OPENAI_API_KEY)
    status = "✅ Ruby Wings Backend Online — AI Ready" if ok else "⚠️ Backend chạy nhưng thiếu OPENAI_API_KEY"
    return status

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập nội dung nào."})

        # 1) tìm nội dung liên quan trong knowledge
        related = smart_search(user_message)  # list
        related_text = "\n\n".join([f"- {r}" for r in related]) if related else "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        # 2) build system prompt
        system_prompt = f"""
Bạn là trợ lý AI của Ruby Wings - một đơn vị du lịch trải nghiệm & chữa lành.
Yêu cầu:
- Trả lời thân thiện, chính xác, súc tích.
- Ưu tiên thông tin từ cơ sở dữ liệu nội bộ nếu có (đã cung cấp dưới đây).
- Nếu không có, trả lời dựa trên kiến thức chuyên môn du lịch, và nêu rõ khi nào bạn suy đoán.
- Không tự chế thông tin nội bộ (như giá nội bộ, mã số..). Nếu thiếu, khuyến nghị cầu người chịu trách nhiệm cung cấp.

Dữ liệu nội bộ liên quan (tối đa 5 đoạn):
{related_text}
"""

        # 3) Call OpenAI
        try:
            reply_text = call_openai_chat(system_prompt, user_message)
        except Exception as e:
            # nếu OpenAI gặp lỗi, fallback trả lời qua search nội bộ
            print("⚠️ OpenAI call error:", e)
            if related:
                fallback = "Mình tạm thời không gọi được AI bên ngoài. Đây là thông tin liên quan từ cơ sở dữ liệu nội bộ:\n\n" + related_text
                return jsonify({"reply": fallback})
            return jsonify({"reply": f"Không thể xử lý yêu cầu do lỗi hệ thống: {e}"}), 500

        return jsonify({"reply": reply_text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"Lỗi server: {e}"}), 500

# ============================
# 5) Run (dev)
# ============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
