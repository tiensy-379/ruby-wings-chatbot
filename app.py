# app.py (final, ready)
import os
import json
import threading
import logging
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import numpy as np

# Try to import faiss
HAS_FAISS = False
FAISS_IMPORT_ERROR = None
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception as e:
    FAISS_IMPORT_ERROR = str(e)
    HAS_FAISS = False

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# default knowledge path; if you uploaded a cleaned file for testing, this file will be used automatically
DEFAULT_KNOWLEDGE = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
UPLOADED_TEST_PATH = "/mnt/data/knowledge_fixed.json"
if not os.path.exists(DEFAULT_KNOWLEDGE) and os.path.exists(UPLOADED_TEST_PATH):
    KNOWLEDGE_PATH = UPLOADED_TEST_PATH
else:
    KNOWLEDGE_PATH = DEFAULT_KNOWLEDGE

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED_ENV = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    try:
        openai.api_base = OPENAI_BASE_URL
    except Exception:
        pass
else:
    logger.warning("OPENAI_API_KEY is missing. Running in limited mode (local fallback embeddings).")

if not HAS_FAISS and FAISS_ENABLED_ENV:
    logger.warning("FAISS not available (%s). Falling back to numpy index. To silence set FAISS_ENABLED=0", FAISS_IMPORT_ERROR)

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOWLEDGE = {}
FLATTENED_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX_LOCK = threading.Lock()
INDEX = None
INDEX_IS_FAISS = False

# ---------- Numpy fallback index ----------
class NumpyFallbackIndex:
    def __init__(self, mat: np.ndarray = None):
        self.mat = mat.astype("float32") if (mat is not None and mat.size>0) else np.empty((0,0), dtype="float32")
        self.dim = None if self.mat.size==0 else self.mat.shape[1]
    def add(self, mat: np.ndarray):
        if mat is None or mat.size==0:
            return
        mat = mat.astype("float32")
        if self.mat.size==0:
            self.mat = mat.copy()
            self.dim = mat.shape[1]
        else:
            if mat.shape[1] != self.dim:
                raise ValueError("Dimension mismatch")
            self.mat = np.vstack([self.mat, mat])
    def search(self, qvec: np.ndarray, k:int):
        if self.mat is None or self.mat.size==0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec.astype("float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        mat = self.mat
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, mat.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")
    @property
    def ntotal(self):
        return 0 if self.mat is None or self.mat.size==0 else self.mat.shape[0]
    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
            logger.info("✅ Saved fallback vectors to %s", path)
        except Exception:
            logger.exception("Failed saving fallback vectors")
    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            return cls(mat=arr["mat"])
        except Exception:
            logger.exception("Failed loading fallback vectors")
            return cls(None)

# ---------- Utilities ----------
def load_knowledge(path=KNOWLEDGE_PATH):
    global KNOWLEDGE, FLATTENED_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOWLEDGE = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge.json; continuing with empty knowledge.")
        KNOWLEDGE = {}

    FLATTENED_TEXTS = []
    MAPPING = []

    def scan(obj, prefix="root"):
        # Recursively extract strings; also stringify numbers/booleans but be conservative
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        else:
            # leaf node: stringify
            try:
                text = obj if isinstance(obj, str) else str(obj)
            except Exception:
                text = ""
            text = text.strip()
            if not text:
                return
            # include if long enough OR contains digits (prices, years, group codes)
            if len(text) >= 20 or (len(text) >= 3 and any(ch.isdigit() for ch in text)):
                FLATTENED_TEXTS.append(text)
                MAPPING.append({"path": prefix, "text": text})

    scan(KNOWLEDGE, "root")
    logger.info("✅ knowledge loaded: %d passages", len(FLATTENED_TEXTS))
    return len(FLATTENED_TEXTS)

def save_mapping(path=FAISS_MAPPING_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        logger.info("✅ Saved mapping to %s", path)
    except Exception:
        logger.exception("Could not save mapping")

def load_mapping(path=FAISS_MAPPING_PATH):
    global MAPPING, FLATTENED_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING = json.load(f)
        FLATTENED_TEXTS = [m.get("text","") for m in MAPPING]
        logger.info("✅ Loaded mapping (%d entries).", len(MAPPING))
    except Exception:
        logger.exception("Could not load mapping; resetting")
        MAPPING = []
        FLATTENED_TEXTS = []

# ---------- Embedding helpers ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    # choose fallback dim by model
    if "large" in EMBEDDING_MODEL:
        fallback_dim = 3072
    else:
        fallback_dim = 1536
    if OPENAI_API_KEY:
        try:
            try:
                resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=short)
            except Exception:
                resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
            emb = None
            if isinstance(resp, dict) and "data" in resp and len(resp["data"])>0:
                emb = resp["data"][0].get("embedding") or resp["data"][0].get("vector")
            elif hasattr(resp, "data") and len(resp.data)>0:
                emb = getattr(resp.data[0], "embedding", None)
            if emb:
                return emb, len(emb)
            logger.warning("Embedding API returned no embedding field.")
        except Exception:
            logger.exception("OpenAI embedding call failed; falling back to synthetic embedding.")
    # deterministic fallback
    try:
        h = abs(hash(short)) % (10 ** 12)
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# ---------- Index management ----------
def build_index(force_rebuild=False):
    global INDEX, MAPPING, FLATTENED_TEXTS, INDEX_IS_FAISS
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED_ENV and HAS_FAISS
        # try load persisted
        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    INDEX_IS_FAISS = True
                    logger.info("✅ FAISS index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load existing FAISS index; will rebuild.")
            elif (not use_faiss) and os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    INDEX_IS_FAISS = False
                    logger.info("✅ Fallback index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load fallback index; will rebuild.")
        if not FLATTENED_TEXTS:
            logger.warning("No flattened texts to index (build aborted).")
            INDEX = None
            return False

        logger.info("🔧 Building index for %d passages... (faiss=%s)", len(FLATTENED_TEXTS), use_faiss)
        vectors = []
        dims = None
        for text in FLATTENED_TEXTS:
            emb, d = embed_text(text)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors or dims is None:
            logger.warning("No vectors produced; index build aborted.")
            INDEX = None
            return False

        try:
            mat = np.vstack(vectors).astype("float32")
            # normalize rows
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (row_norms + 1e-12)
            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                INDEX_IS_FAISS = True
                try:
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    save_mapping()
                except Exception:
                    logger.exception("Failed to persist FAISS index")
                logger.info("✅ FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
                return True
            else:
                idx = NumpyFallbackIndex(mat)
                INDEX = idx
                INDEX_IS_FAISS = False
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    save_mapping()
                except Exception:
                    logger.exception("Failed to persist fallback vectors")
                logger.info("✅ Numpy fallback index built (dims=%d, n=%d).", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

def query_index(query: str, top_k=TOP_K) -> List[Tuple[float, dict]]:
    global INDEX, MAPPING
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built or INDEX is None:
            logger.warning("Index not available; returning empty search results")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    try:
        vec = np.array(emb, dtype="float32").reshape(1, -1)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        if INDEX_IS_FAISS and HAS_FAISS:
            D, I = INDEX.search(vec, top_k)
        else:
            D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Error querying index")
        return []
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
        "thiền, khí công và các hành trình chữa lành. Trả lời lịch sự, tự nhiên và chuyên nghiệp.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp. Nếu cần, trả lời ngắn và đề nghị người dùng cung cấp thêm chi tiết."
    content = header + "Dữ liệu nội bộ (ưu tiên sử dụng):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        text = m.get("text", "")
        path = m.get("path", "?")
        # keep passage short per line
        content += f"[{i}] (score={score:.3f}) path={path}\n\"{text}\"\n"
    content += (
        "\n---\nHƯỚNG DẪN: Khi trả lời, hãy ưu tiên hoàn toàn thông tin trong 'Dữ liệu nội bộ' ở trên. "
        "Dùng OpenAI để diễn đạt tự nhiên, lịch sự và đầy đủ (2–4 câu). "
        "Khi trích dẫn trực tiếp, đặt trong ngoặc kép và ghi 'Từ cơ sở tri thức'. "
        "Nếu không có thông tin phù hợp, trả: 'Xin lỗi — không có thông tin trong cơ sở tri thức về yêu cầu này.'"
    )
    return content

# ---------- Endpoints ----------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Ruby Wings index backend running.",
        "knowledge_count": len(FLATTENED_TEXTS),
        "index_exists": INDEX is not None,
        "faiss_available": HAS_FAISS,
        "faiss_enabled_env": FAISS_ENABLED_ENV
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        secret = request.headers.get("X-RBW-ADMIN", "")
        if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
            return jsonify({"error": "reindex not allowed"}), 403
        load_knowledge(KNOWLEDGE_PATH)
        ok = build_index(force_rebuild=True)
        return jsonify({"ok": ok, "count": len(FLATTENED_TEXTS)})
    except Exception as e:
        logger.exception("Unhandled error in /reindex")
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập câu hỏi."})
        top_k = int(data.get("top_k", TOP_K))
        top = query_index(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        logger.info("CHAT: model=%s top_k=%d hits=%d", CHAT_MODEL, top_k, len(top))

        resp = None
        if OPENAI_API_KEY:
            try:
                resp = openai.ChatCompletion.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=int(data.get("max_tokens", 700)),
                    top_p=0.95
                )
            except Exception as e1:
                logger.warning("ChatCompletion.create failed: %s", e1)
                try:
                    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                        resp = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=False)
                    else:
                        resp = None
                except Exception as e2:
                    logger.exception("All OpenAI chat attempts failed")
                    return jsonify({"error": "OpenAI chat request failed", "detail": str(e2)}), 500

        if not resp:
            # fallback rule-based reply using top passages
            if top:
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}\n\nNếu bạn cần trích dẫn hoặc chi tiết, hãy hỏi cụ thể phần nào."
            else:
                reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan và API OpenAI chưa sẵn sàng."
            return jsonify({"reply": reply, "sources": [m for _, m in top]})

        # parse response
        content = ""
        try:
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    first = choices[0]
                    if isinstance(first.get("message"), dict):
                        content = first["message"].get("content", "") or ""
                    elif "text" in first:
                        content = first.get("text", "")
                    else:
                        content = str(first)
                else:
                    content = str(resp)
            else:
                choices = getattr(resp, "choices", None)
                if choices and len(choices)>0:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    if msg and isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = str(first)
                else:
                    content = str(resp)
        except Exception:
            logger.exception("Parsing OpenAI response failed")
            content = str(resp)

        return jsonify({"reply": content, "sources": [m for _, m in top]})
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        return jsonify({"error": str(e)}), 500

@app.route("/stream", methods=["POST"])
def stream():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "empty message"}), 400
        top_k = int(data.get("top_k", TOP_K))
        top = query_index(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        def gen():
            try:
                if not OPENAI_API_KEY:
                    yield f"data: {json.dumps({'error':'openai_key_missing'})}\n\n"
                    return
                try:
                    stream_iter = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, stream=True)
                except Exception as e1:
                    logger.warning("stream create failed: %s", e1)
                    try:
                        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                            stream_iter = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=True)
                        else:
                            stream_iter = None
                    except Exception as e2:
                        logger.exception("OpenAI streaming create failed")
                        yield f"data: {json.dumps({'error':'openai_stream_create_failed','detail':str(e2)})}\n\n"
                        return

                if stream_iter is None:
                    yield f"data: {json.dumps({'error':'openai_stream_iter_none'})}\n\n"
                    return

                for chunk in stream_iter:
                    try:
                        if not chunk:
                            continue
                        if isinstance(chunk, dict):
                            choices = chunk.get("choices", [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'delta': content})}\n\n"
                        else:
                            choices = getattr(chunk, "choices", None)
                            if choices and len(choices) > 0:
                                delta = getattr(choices[0], "delta", None)
                                if delta:
                                    content = delta.get("content", "") if isinstance(delta, dict) else ""
                                    if content:
                                        yield f"data: {json.dumps({'delta': content})}\n\n"
                    except Exception:
                        logger.exception("stream chunk processing error")
                        continue
                yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"
            except Exception as ex:
                logger.exception("stream generator error")
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"

        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    except Exception as e:
        logger.exception("Unhandled error in /stream")
        return jsonify({"error": str(e)}), 500

# ---------- Initialization (run on import so Gunicorn workers have index) ----------
try:
    count = load_knowledge(KNOWLEDGE_PATH)
    if os.path.exists(FAISS_MAPPING_PATH):
        load_mapping(FAISS_MAPPING_PATH)
    if FAISS_ENABLED_ENV and HAS_FAISS and os.path.exists(FAISS_INDEX_PATH):
        try:
            INDEX = faiss.read_index(FAISS_INDEX_PATH)
            INDEX_IS_FAISS = True
            logger.info("✅ FAISS index loaded at import time.")
        except Exception:
            logger.exception("Failed to load FAISS index at import; will rebuild in background.")
            t = threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True)
            t.start()
    elif (not FAISS_ENABLED_ENV or not HAS_FAISS) and os.path.exists(FALLBACK_VECTORS_PATH):
        try:
            INDEX = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
            INDEX_IS_FAISS = False
            logger.info("✅ Fallback numpy index loaded at import time.")
        except Exception:
            logger.exception("Failed to load fallback index; will rebuild in background.")
            t = threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True)
            t.start()
    else:
        t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
        t.start()
except Exception:
    logger.exception("Initialization error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("Server starting on port %d ...", port)
    app.run(host="0.0.0.0", port=port)
