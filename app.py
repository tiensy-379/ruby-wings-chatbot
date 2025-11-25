# app.py (patched)
import os
import json
import time
import threading
import traceback
import logging
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import numpy as np

# Try to import faiss; if unavailable, we'll use a numpy fallback
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
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")  # 1536-dim
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
# Allow toggling FAISS via env var (useful on platforms lacking faiss wheels)
FAISS_ENABLED_ENV = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    try:
        openai.api_base = OPENAI_BASE_URL
    except Exception:
        # older sdk might ignore this
        pass
else:
    logger.warning("OPENAI_API_KEY is missing. Running in limited mode (local fallback embeddings).")

if not HAS_FAISS and FAISS_ENABLED_ENV:
    logger.warning("FAISS not available (%s). Falling back to numpy index. Set FAISS_ENABLED=0 to silence.", FAISS_IMPORT_ERROR)

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOWLEDGE = {}
FLATTENED_TEXTS: List[str] = []
# mapping: list of {"path":..., "text":...}
MAPPING: List[dict] = []
INDEX_LOCK = threading.Lock()
# Index handle: may be faiss index or our NumpyFallbackIndex instance
INDEX = None

# ---------- Simple numpy fallback index ----------
class NumpyFallbackIndex:
    """
    Simple cosine-similarity index stored in memory.
    Keeps matrix (n,d) float32 and supports add and search.
    Persists to .npz (vectors + ids).
    """
    def __init__(self, mat: np.ndarray = None):
        self.mat = mat.astype("float32") if (mat is not None and mat.size>0) else np.empty((0,0), dtype="float32")
        if self.mat.size == 0:
            self.dim = None
        else:
            self.dim = self.mat.shape[1]
        self._ntotal = 0 if self.mat.size==0 else self.mat.shape[0]

    def add(self, mat: np.ndarray):
        if mat is None or mat.size == 0:
            return
        mat = mat.astype("float32")
        if self.mat.size == 0:
            self.mat = mat.copy()
            self.dim = mat.shape[1]
        else:
            if mat.shape[1] != self.dim:
                raise ValueError("Dimension mismatch in fallback index")
            self.mat = np.vstack([self.mat, mat])
        self._ntotal = self.mat.shape[0]

    def search(self, qvec: np.ndarray, k: int):
        # qvec: (1,d)
        if self.mat is None or self.mat.size == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        # normalize
        q = qvec.astype("float32")
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / (q_norm + 1e-12)
        mat_norm = np.linalg.norm(self.mat, axis=1, keepdims=True)
        mat_normed = self.mat / (mat_norm + 1e-12)
        sims = np.dot(q, mat_normed.T)  # (1, n)
        # argsort descending
        idx = np.argsort(-sims, axis=1)[:, :k]
        # gather scores and indices, pad if needed
        k_found = idx.shape[1]
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
            logger.exception("Failed saving fallback vectors to %s", path)

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            mat = arr["mat"]
            return cls(mat=mat)
        except Exception:
            logger.exception("Failed loading fallback vectors from %s", path)
            return cls(None)

# ---------- Utilities ----------
def load_knowledge(path=KNOWLEDGE_PATH):
    """
    Load knowledge.json and flatten textual passages into FLATTENED_TEXTS and MAPPING.
    """
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
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            text = obj.strip()
            if len(text) >= 1:
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
        FLATTENED_TEXTS = [m.get("text", "") for m in MAPPING]
        logger.info("✅ Loaded mapping (%d entries).", len(MAPPING))
    except Exception:
        logger.exception("Could not load mapping; resetting mapping/flattened_texts")
        MAPPING = []
        FLATTENED_TEXTS = []

# ---------- Embedding helpers ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Return embedding list and dimension.
    Robustly tries modern and legacy OpenAI clients; falls back to deterministic synthetic vector.
    """
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    # If OPENAI_API_KEY present, call actual API; else use deterministic fallback
    if OPENAI_API_KEY:
        try:
            # Preferred modern call: openai.Embeddings.create
            try:
                resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=short)
            except Exception:
                resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
            emb = None
            if isinstance(resp, dict) and "data" in resp and len(resp["data"]) > 0:
                emb = resp["data"][0].get("embedding") or resp["data"][0].get("vector")
            elif hasattr(resp, "data") and len(resp.data) > 0:
                emb = getattr(resp.data[0], "embedding", None)
            if emb:
                return emb, len(emb)
            logger.warning("Embedding API returned no embedding field; resp type=%s", type(resp))
        except Exception:
            logger.exception("OpenAI embedding call failed; falling back to synthetic embedding.")
    # deterministic synthetic fallback (stable across runs on same machine)
    try:
        h = abs(hash(short)) % (10 ** 12)
        dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
        return vec, dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# ---------- Index management ----------
def build_index(force_rebuild=False):
    """
    Build or load index. If FAISS enabled and available, use faiss IndexFlatIP,
    otherwise use NumpyFallbackIndex.
    Returns True when index available.
    """
    global INDEX, MAPPING, FLATTENED_TEXTS
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED_ENV and HAS_FAISS
        # Try loading persisted structures first
        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    logger.info("✅ FAISS index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load existing FAISS index; will rebuild.")
            elif (not use_faiss) and os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    logger.info("✅ Fallback numpy index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load existing fallback index; will rebuild.")

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
                try:
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    save_mapping()
                except Exception:
                    logger.exception("Failed to persist FAISS index or mapping")
                logger.info("✅ FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
                return True
            else:
                idx = NumpyFallbackIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    save_mapping()
                except Exception:
                    logger.exception("Failed to persist fallback vectors or mapping")
                logger.info("✅ Numpy fallback index built (dims=%d, n=%d).", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

def query_index(query: str, top_k=TOP_K) -> List[Tuple[float, dict]]:
    """
    Query the current index and return list of (score, mapping_entry).
    If index not ready, attempt lazy build once.
    """
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
        # normalize
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        if HAS_FAISS and FAISS_ENABLED_ENV and isinstance(INDEX, type(faiss.IndexFlatIP(1))):
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
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn nghành du lịch trải nghiệm, retreat, "
        "thiền, khí công, hành trình chữa lành - Hành trình tham quan linh hoạt theo nhhu cầu. Trả lời ngắn gọn, chính xác, tử tế.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header + "Dữ liệu nội bộ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nLưu ý: Ưu tiên sử dụng trích dẫn thông tin từ dữ liệu nội bộ ở trên. Nếu phải bổ sung, chỉ dùng kiến thức chuẩn xác, không được tự ý bịa ra khi chưa rõ đúng sai; sử dụng ngôn ngữ lịch sự, thân thiện, thông minh; khi khách gõ lời tạm biệt hoặc lời chúc thì chân thành cám ơn khách, chúc khách sức khoẻ tốt, may mắn, thành công..."
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
            return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
        load_knowledge(KNOWLEDGE_PATH)
        ok = build_index(force_rebuild=True)
        return jsonify({"ok": ok, "count": len(FLATTENED_TEXTS)})
    except Exception as e:
        logger.exception("Unhandled error in /reindex")
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Non-streaming chat endpoint.
    Input: JSON { "message": "...", "max_tokens": 700, "top_k": 5 }
    """
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

        # Try to call OpenAI chat; handle SDK variations
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
                logger.warning("ChatCompletion.create failed, trying alternate patterns: %s", e1)
                try:
                    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                        resp = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=False)
                    else:
                        resp = None
                except Exception as e2:
                    logger.exception("All OpenAI chat attempts failed")
                    return jsonify({"error": "OpenAI chat request failed", "detail": str(e2)}), 500

        # If no OpenAI response (no key or API failed), fall back to rule-based
        if not resp:
            if top:
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}\n\nNếu bạn cần trích dẫn hoặc chi tiết, hãy hỏi cụ thể phần nào."
            else:
                reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan và API OpenAI chưa sẵn sàng. Vui lòng thử lại sau."
            return jsonify({"reply": reply, "sources": [m for _, m in top]})

        # parse response robustly
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
                if choices and len(choices) > 0:
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
    """
    Streaming SSE endpoint. Input JSON: { "message": "...", "top_k": 5 }
    """
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
                    logger.warning("stream create failed, trying alternate: %s", e1)
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
                # final payload
                yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"
            except Exception as ex:
                logger.exception("stream generator error")
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"

        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    except Exception as e:
        logger.exception("Unhandled error in /stream")
        return jsonify({"error": str(e)}), 500

# ---------- Startup ----------
# ---------- Initialization (run on import so Gunicorn workers have index) ----------
try:
    # load knowledge from configured path
    count = load_knowledge(KNOWLEDGE_PATH)
    # try loading existing mapping/index
    if os.path.exists(FAISS_MAPPING_PATH):
        load_mapping(FAISS_MAPPING_PATH)
    # if FAISS index file exists and faiss enabled, try to load
    if FAISS_ENABLED_ENV and HAS_FAISS and os.path.exists(FAISS_INDEX_PATH):
        try:
            INDEX = faiss.read_index(FAISS_INDEX_PATH)
            logger.info("✅ FAISS index loaded at import time.")
        except Exception:
            logger.exception("Failed to load FAISS index at import; will rebuild in background.")
            t = threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True)
            t.start()
    elif (not FAISS_ENABLED_ENV or not HAS_FAISS) and os.path.exists(FALLBACK_VECTORS_PATH):
        try:
            INDEX = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
            logger.info("✅ Fallback numpy index loaded at import time.")
        except Exception:
            logger.exception("Failed to load fallback index at import; will rebuild in background.")
            t = threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True)
            t.start()
    else:
        # build in background if knowledge exists
        t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
        t.start()
except Exception:
    logger.exception("Initialization error")
    port = int(os.environ.get("PORT", 10000))
    logger.info("Server starting on port %d ...", port)
    app.run(host="0.0.0.0", port=port)
