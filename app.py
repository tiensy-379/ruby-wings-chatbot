# app.py — Optimized for openai==0.28.0
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

# Try FAISS — fallback to numpy if missing
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------------- Config ----------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))

FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true")

# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Global State ----------------
KNOW = {}
FLAT_TEXTS = []
MAPPING = []
INDEX = None
INDEX_LOCK = threading.Lock()

# ---------------- Fallback Index (numpy) ----------------
class NumpyIndex:
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if mat is not None else np.zeros((0, 0), dtype="float32")
        self.dim = None if self.mat.size == 0 else self.mat.shape[1]

    def search(self, qvec, k):
        if self.mat.size == 0:
            return np.array([[0]]), np.array([[0]])
        q = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = q @ m.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            return cls(arr["mat"])
        except:
            return cls()

    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except:
            pass

# ---------------- Load Knowledge ----------------
def load_knowledge():
    global KNOW, FLAT_TEXTS, MAPPING
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    except:
        KNOW = {}

    FLAT_TEXTS = []
    MAPPING = []

    def scan(obj, p="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{p}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{p}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                FLAT_TEXTS.append(t)
                MAPPING.append({"path": p, "text": t})

    scan(KNOW)
    logger.info("Knowledge loaded: %d passages", len(FLAT_TEXTS))

# ---------------- Embedding (cached) ----------------
@lru_cache(maxsize=4096)
def embed_text(text: str):
    if not text:
        return [], 0
    text = text[:2000]

    try:
        r = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
        v = r["data"][0]["embedding"]
        return v, len(v)
    except Exception:
        # fallback deterministic vector
        h = abs(hash(text)) % (10**12)
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(1536)]
        return vec, 1536

# ---------------- Build or Load Index ----------------
def build_index(force=False):
    global INDEX, MAPPING, FLAT_TEXTS
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

        # Load existing
        if not force:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    INDEX = faiss.read_index(FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    FLAT_TEXTS[:] = [m["text"] for m in MAPPING]
                    logger.info("FAISS index loaded.")
                    return True
                except:
                    logger.warning("Failed to load FAISS index, rebuilding.")

            if (not use_faiss) and os.path.exists(FALLBACK_VECTORS_PATH):
                try:
                    INDEX = NumpyIndex.load(FALLBACK_VECTORS_PATH)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    logger.info("Numpy fallback index loaded.")
                    return True
                except:
                    logger.warning("Failed to load fallback index, rebuilding.")

        if not FLAT_TEXTS:
            return False

        logger.info("Building embeddings...")
        vecs = []
        dim = None
        for t in FLAT_TEXTS:
            v, d = embed_text(t)
            if v:
                if dim is None:
                    dim = d
                vecs.append(np.array(v, dtype="float32"))
        if not vecs:
            return False

        mat = np.vstack(vecs)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

        if use_faiss:
            index = faiss.IndexFlatIP(dim)
            index.add(mat)
            INDEX = index
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
            logger.info("FAISS index built.")
        else:
            idx = NumpyIndex(mat)
            INDEX = idx
            idx.save(FALLBACK_VECTORS_PATH)
            with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
            logger.info("Fallback index built.")

        return True

# ---------------- Query Index ----------------
def query_index(q: str, k=TOP_K):
    if not q:
        return []
    if INDEX is None:
        build_index()

    v, _ = embed_text(q)
    vec = np.array(v, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    try:
        if HAS_FAISS and isinstance(INDEX, faiss.Index):
            D, I = INDEX.search(vec, k)
        else:
            D, I = INDEX.search(vec, k)
    except Exception:
        return []

    out = []
    for s, i in zip(D[0], I[0]):
        if i < len(MAPPING):
            out.append((float(s), MAPPING[i]))
    return out

# ---------------- Prompt Composer ----------------
def compose_prompt(passages):
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên du lịch trải nghiệm, retreat, thiền, khí công.\n"
        "Trả lời ngắn gọn, chính xác, tử tế. Ưu tiên dùng dữ liệu nội bộ.\n\n"
    )
    if not passages:
        return header + "Không tìm thấy dữ liệu liên quan."

    s = header + "Dữ liệu nội bộ:\n"
    for i, (score, m) in enumerate(passages, 1):
        s += f"\n[{i}] (score={score:.3f}) {m['path']}:\n{m['text']}\n"
    s += "\n---\nTrả lời chính xác, không bịa."
    return s

# ---------------- Routes ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "knowledge": len(FLAT_TEXTS)})

@app.route("/reindex", methods=["POST"])
def reindex():
    build_index(force=True)
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Bạn chưa nhập nội dung."})

    top = query_index(user_msg, int(data.get("top_k", TOP_K)))
    prompt = compose_prompt(top)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg}
    ]

    try:
        r = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
            top_p=0.95
        )
        reply = r["choices"][0]["message"]["content"]
    except Exception as e:
        reply = "Lỗi OpenAI. Vui lòng thử lại."

    return jsonify({"reply": reply, "sources": [m for _, m in top]})

# ---------------- Startup ----------------
load_knowledge()
build_index(force=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
