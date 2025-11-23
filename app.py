# app.py
import os
import json
import time
import threading
import traceback
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import faiss
import numpy as np

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")  # dims 1536
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")  # change as needed
TOP_K = int(os.environ.get("TOP_K", "5"))

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is missing. Set environment variable OPENAI_API_KEY.")
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_BASE_URL

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOWLEDGE = {}
FLATTENED_TEXTS: List[str] = []
# mapping: index -> {"source":..., "path":..., "text":...}
MAPPING: List[dict] = []
INDEX_LOCK = threading.Lock()
FAISS_INDEX = None  # will be faiss.IndexFlatIP or IndexIDMap


# ---------- Utilities ----------
def load_knowledge(path=KNOWLEDGE_PATH):
    global KNOWLEDGE, FLATTENED_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOWLEDGE = json.load(f)
        # flatten: collect small text chunks (strings) with some path label
        FLATTENED_TEXTS = []
        MAPPING = []

        def scan(obj, prefix="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    scan(v, prefix + "." + k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    scan(item, f"{prefix}[{i}]")
            elif isinstance(obj, str):
                text = obj.strip()
                if len(text) > 20:  # keep reasonably sized passages
                    FLATTENED_TEXTS.append(text)
                    MAPPING.append({"path": prefix, "text": text})
        scan(KNOWLEDGE, "root")
        print(f"✅ knowledge loaded: {len(FLATTENED_TEXTS)} passages")
    except Exception as e:
        print("⚠️ Could not load knowledge.json:", e)
        KNOWLEDGE = {}
        FLATTENED_TEXTS = []
        MAPPING = []


def save_mapping(path=FAISS_MAPPING_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        print("✅ Saved mapping.")
    except Exception as e:
        print("⚠️ Save mapping failed:", e)


def load_mapping(path=FAISS_MAPPING_PATH):
    global MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING = json.load(f)
        print("✅ Loaded mapping.")
    except Exception as e:
        print("⚠️ Could not load mapping:", e)
        MAPPING = []


# ---------- Embedding helpers ----------
@lru_cache(maxsize=4096)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Returns embedding vector (list) and dimension.
    Cached via lru_cache for repeated usage in same process.
    """
    if not text:
        return [], 0
    txt = text if len(text) < 2000 else text[:2000]  # guard length
    # call OpenAI embeddings
    try:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=txt)
        emb = resp["data"][0]["embedding"]
        return emb, len(emb)
    except Exception as e:
        print("Embedding error:", e)
        # fallback: simple hash-based vector (stable)
        h = abs(hash(txt)) % (10 ** 8)
        dim = 1536
        vec = [float((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dim)]
        return vec, dim


def build_faiss_index(force_rebuild=False):
    """
    Build FAISS index for FLATTENED_TEXTS and persist to disk.
    Safe to call multiple times; locked to avoid races.
    """
    global FAISS_INDEX, MAPPING, FLATTENED_TEXTS
    with INDEX_LOCK:
        # if index exists on disk and not forcing, try load
        if not force_rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
            try:
                idx = faiss.read_index(FAISS_INDEX_PATH)
                load_mapping(FAISS_MAPPING_PATH)
                FAISS_INDEX = idx
                print("✅ FAISS index loaded from disk.")
                return True
            except Exception as e:
                print("⚠️ Could not load FAISS index from disk:", e)
        # need to build
        if not FLATTENED_TEXTS:
            print("⚠️ No texts to index.")
            return False
        print("🔧 Building FAISS index ...")
        vectors = []
        dims = None
        for t in FLATTENED_TEXTS:
            emb, d = embed_text(t)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors:
            print("⚠️ No vectors created.")
            return False
        mat = np.vstack(vectors)
        # normalize for cos similarity
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(dims)  # inner product on normalized vectors => cosine
        index.add(mat)
        FAISS_INDEX = index
        # persist
        try:
            faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
            save_mapping()
            print("✅ FAISS index built and saved.")
        except Exception as e:
            print("⚠️ Could not save FAISS index:", e)
        return True


def query_faiss(query: str, top_k=TOP_K) -> List[Tuple[float, dict]]:
    """
    Return list of (score, mapping) for top_k similar passages.
    """
    if FAISS_INDEX is None:
        build_faiss_index(force_rebuild=False)
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    D, I = FAISS_INDEX.search(vec, top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(score), MAPPING[idx]))
    return results


# ---------- Prompt composition ----------
def compose_system_prompt(top_passages: List[Tuple[float, dict]]) -> str:
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn du lịch trải nghiệm, retreat, "
        "thiền, khí công và các hành trình chữa lành. Hãy trả lời ngắn gọn, thân thiện và chính xác.\n"
        "Ưu tiên trích dẫn thông tin từ nguồn nội bộ dưới đây. Nếu không đủ, hãy trả kiến thức chung chính xác.\n\n"
    )
    if not top_passages:
        return header
    content = header + "Dữ liệu nội bộ liên quan (sắp xếp theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nHướng dẫn: Dùng dữ liệu nội bộ trên nếu trả lời liên quan; trích dẫn nguồn nếu cần."
    return content


# ---------- Endpoints ----------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Ruby Wings FAISS backend running.",
        "knowledge_count": len(FLATTENED_TEXTS),
        "faiss_exists": FAISS_INDEX is not None
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    """
    Safe endpoint to rebuild index. Use when knowledge.json updated.
    """
    try:
        secret = request.headers.get("X-RBW-ADMIN", "")
        # very simple guard: if admin header not provided, disallow in public
        if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
            return jsonify({"error": "reindex not allowed without admin header or RBW_ALLOW_REINDEX=1"}), 403
        load_knowledge(KNOWLEDGE_PATH)
        ok = build_faiss_index(force_rebuild=True)
        return jsonify({"ok": ok, "count": len(FLATTENED_TEXTS)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Non-streaming chat: returns assistant reply (single response).
    Input JSON: { "message": "...", "max_tokens": 700, "top_k": 5 }
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập câu hỏi."})
        top_k = int(data.get("top_k", TOP_K))
        top = query_faiss(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        # call OpenAI ChatCompletion
        resp = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=int(data.get("max_tokens", 700)),
            top_p=0.95
        )
        content = ""
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            content = str(resp)
        return jsonify({
            "reply": content,
            "sources": [m for _, m in top]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stream", methods=["POST"])
def stream():
    """
    Streaming chat using Server-Sent Events (SSE).
    Client should call with Accept: text/event-stream or normal POST. 
    Input JSON: { "message": "...", "top_k": 5 }
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "empty message"}), 400
        top_k = int(data.get("top_k", TOP_K))
        top = query_faiss(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # create a generator to stream tokens
        def gen():
            try:
                # Use OpenAI streaming
                # Note: openai.ChatCompletion.create(..., stream=True) yields events
                for chunk in openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, stream=True):
                    # each chunk is a dict (depends on openai lib version)
                    try:
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            # SSE-style data
                            yield f"data: {json.dumps({'delta': content})}\n\n"
                    except Exception:
                        # some events might be 'finish_reason'
                        pass
                # at end, send sources
                yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"
            except Exception as ex:
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"
        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------- Startup ----------
if __name__ == "__main__":
    # initial load
    load_knowledge(KNOWLEDGE_PATH)
    # try to load index; if not available, try to build lazily
    try:
        # If mapping exists load it
        if os.path.exists(FAISS_MAPPING_PATH):
            load_mapping(FAISS_MAPPING_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
                print("✅ FAISS index loaded at startup.")
            except Exception as e:
                print("⚠️ Failed loading FAISS index at startup:", e)
                # attempt to build
                built = build_faiss_index(force_rebuild=True)
                print("Built index:", built)
        else:
            # build in background thread to reduce startup latency
            t = threading.Thread(target=build_faiss_index, kwargs={"force_rebuild": False}, daemon=True)
            t.start()
    except Exception as ex:
        print("Startup index error:", ex)

    port = int(os.environ.get("PORT", 10000))
    print(f"Server starting on port {port} ...")
    app.run(host="0.0.0.0", port=port)
