# app.py — Optimized for openai==0.28.0 with auto-detect embedding dim
import os, json, threading, logging
from functools import lru_cache
from typing import List, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import numpy as np

# FAISS import
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# config (defaults)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

# default embedding model (will be overridden if index exists)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true")

app = Flask(__name__)
CORS(app)

# global state
KNOW = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()

# numpy fallback index
class NumpyIndex:
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if (isinstance(mat, np.ndarray) and mat.size>0) else np.empty((0,0), dtype="float32")
        self.dim = None if self.mat.size==0 else self.mat.shape[1]

    def search(self, qvec, k):
        if self.mat.size == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            mat = arr["mat"]
            return cls(mat=mat)
        except Exception:
            return cls(None)

    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except Exception:
            logger.exception("Failed to save fallback vectors")

# load knowledge (flatten)
def load_knowledge(path=KNOWLEDGE_PATH):
    global KNOW, FLAT_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge.json; continuing with empty knowledge.")
        KNOW = {}
    FLAT_TEXTS = []
    MAPPING = []
    def scan(obj, prefix="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                FLAT_TEXTS.append(t)
                MAPPING.append({"path": prefix, "text": t})
        else:
            try:
                s = str(obj).strip()
                if s:
                    FLAT_TEXTS.append(s)
                    MAPPING.append({"path": prefix, "text": s})
            except:
                pass
    scan(KNOW)
    logger.info("Knowledge loaded: %d passages", len(FLAT_TEXTS))

# helper - determine index dim
def _index_dim(idx):
    try:
        d = getattr(idx, "d", None)
        if isinstance(d, int) and d>0:
            return d
    except:
        pass
    try:
        d = getattr(idx, "dim", None)
        if isinstance(d, int) and d>0:
            return d
    except:
        pass
    # faiss IndexFlat* may have .ntotal and .d accessible via index.d
    try:
        if HAS_FAISS and isinstance(idx, faiss.Index):
            return int(idx.d)
    except:
        pass
    return None

# automatic embedding model selection from index dim
def choose_embedding_model_for_dim(dim):
    # common mapping: 1536 -> text-embedding-3-small, 3072 -> text-embedding-3-large
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    # fallback: keep current env or small
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

# embedding function (uses current EMBEDDING_MODEL variable)
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    try:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
        emb = None
        if isinstance(resp, dict) and "data" in resp and len(resp["data"])>0:
            emb = resp["data"][0].get("embedding") or resp["data"][0].get("vector")
        if emb:
            return emb, len(emb)
    except Exception:
        logger.exception("OpenAI embedding failed; using deterministic fallback")
    # fallback deterministic
    try:
        h = abs(hash(short)) % (10**12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        return [], 0

# build or load index - plus auto-detect embedding model if index exists
def build_index(force_rebuild=False):
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

        # try load existing index & mapping
        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text","") for m in MAPPING]
                    # detect dim and set embedding model accordingly
                    idx_dim = _index_dim(idx)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                        logger.info("Detected index dim=%s -> using embedding model=%s", idx_dim, EMBEDDING_MODEL)
                    INDEX = idx
                    logger.info("FAISS index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed loading FAISS index; will rebuild.")

            # fallback numpy vectors
            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    arr = np.load(FALLBACK_VECTORS_PATH)
                    mat = arr["mat"]
                    idx = NumpyIndex(mat)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text","") for m in MAPPING]
                    idx_dim = getattr(idx, "dim", None)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(int(idx_dim))
                        logger.info("Detected fallback vectors dim=%s -> using embedding model=%s", idx_dim, EMBEDDING_MODEL)
                    INDEX = idx
                    logger.info("Numpy fallback index loaded.")
                    return True
                except Exception:
                    logger.exception("Failed loading fallback vectors; will rebuild.")

        # if nothing loaded, build from FLAT_TEXTS (must be present)
        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index (build aborted).")
            return False

        logger.info("Building embeddings for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
        vecs = []
        dims = None
        for text in FLAT_TEXTS:
            emb, d = embed_text(text)
            if not emb:
                continue
            if dims is None:
                dims = d
            vecs.append(np.array(emb, dtype="float32"))
        if not vecs or dims is None:
            logger.warning("No vectors produced; abort build.")
            INDEX = None
            return False
        try:
            mat = np.vstack(vecs).astype("float32")
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                try:
                    faiss.write_index(index, FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Failed to persist FAISS index/mapping")
                logger.info("FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
                return True
            else:
                idx = NumpyIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Failed to persist fallback vectors/mapping")
                logger.info("Numpy index built (dims=%d, n=%d).", dims, idx.mat.shape[0])
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

# query index (with dimension safety)
def query_index(query: str, top_k=TOP_K):
    global INDEX
    if not query:
        return []
    if INDEX is None:
        ok = build_index(force_rebuild=False)
        if not ok or INDEX is None:
            logger.warning("Index not available")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    # check dims
    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.error("Query dim %s != index dim %s. Will attempt rebuild with matching model.", vec.shape[1], idx_dim)
        # attempt: pick embedding model matching index and rebuild embeddings (if we have API key)
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            logger.info("Setting EMBEDDING_MODEL=%s and rebuilding index...", EMBEDDING_MODEL)
            rebuilt = build_index(force_rebuild=True)
            if not rebuilt:
                logger.error("Rebuild failed; abort search.")
                return []
            # recompute query embedding with new model
            emb2, d2 = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1,-1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY; cannot rebuild model-matched index.")
            return []

    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Index search error")
        return []

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(score), MAPPING[idx]))
    return results

# compose system prompt
def compose_system_prompt(top_passages):
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn ngành du lịch trải nghiệm, retreat, thiền, khí công, hành trình chữa lành.\n"
        "Trả lời ngắn gọn, chính xác, tử tế.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header + "Dữ liệu nội bộ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nƯu tiên trích dẫn dữ liệu nội bộ; không bịa; văn phong lịch sự."
    return content

# endpoints
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "knowledge_count": len(FLAT_TEXTS),
        "index_exists": INDEX is not None,
        "index_dim": _index_dim(INDEX),
        "embedding_model": EMBEDDING_MODEL,
        "faiss_available": HAS_FAISS,
        "faiss_enabled": FAISS_ENABLED
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    # optionally require header or RBW_ALLOW_REINDEX
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX","") != "1":
        return jsonify({"error":"reindex not allowed"}), 403
    ok = build_index(force_rebuild=True)
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message","").strip()
    if not user_message:
        return jsonify({"reply":"Bạn chưa nhập câu hỏi."})
    top_k = int(data.get("top_k", TOP_K))
    top = query_index(user_message, top_k)
    system_prompt = compose_system_prompt(top)
    messages = [{"role":"system","content":system_prompt}, {"role":"user","content":user_message}]
    # call OpenAI ChatCompletion (SDK v0.28)
    reply = ""
    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=int(data.get("max_tokens",700)))
            reply = resp["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI chat failed; fallback to internal snippet")
    if not reply:
        if top:
            snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top[:5]])
            reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."
    return jsonify({"reply": reply, "sources":[m for _, m in top]})

# init on import
load_knowledge()
# if index files exist, this will auto-detect model and load index
build_index(force_rebuild=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",10000)))
