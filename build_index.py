# build_index.py (compatible with app.py)
# Tạo FAISS index + mapping từ knowledge.json bằng OpenAI Embeddings
# Yêu cầu: python -m pip install faiss-cpu openai numpy

import os
import sys
import json
import time
import datetime
from typing import List, Tuple, Any
import numpy as np

# imports with helpful error messages
try:
    import faiss
except Exception as e:
    print("ERROR: Không import được faiss. Cài bằng: python -m pip install faiss-cpu", file=sys.stderr)
    raise

try:
    import openai
    from openai.error import OpenAIError
except Exception:
    print("ERROR: Không import được openai. Cài bằng: python -m pip install openai", file=sys.stderr)
    raise

# Config default values aligned with app.py
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY chưa đặt trong biến môi trường.", file=sys.stderr)
    sys.exit(1)
openai.api_key = OPENAI_KEY

# file paths (match app.py defaults)
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")

# embedding model - align with app.py default unless env overrides
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "16"))
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

def load_and_flatten_knowledge(path: str) -> List[dict]:
    """
    Load knowledge JSON and flatten strings into list of mapping entries:
    [{ "path": "...", "text": "..." }, ...]
    Accepts either object (dict), list, or other JSON that app.py's scan can handle.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} không tồn tại")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = []

    def scan(obj: Any, prefix: str = "root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            text = obj.strip()
            if len(text) >= 1:
                mapping.append({"path": prefix, "text": text})
        else:
            # non-string scalar -> stringify
            try:
                s = str(obj).strip()
                if s:
                    mapping.append({"path": prefix, "text": s})
            except Exception:
                pass

    scan(data, "root")
    return mapping

def call_embeddings_batch(inputs: List[str], model: str):
    """
    Robust embedding call with retry/backoff. Returns the resp object.
    Uses modern openai.Embeddings.create if available, else falls back.
    """
    attempt = 0
    while True:
        try:
            # Try modern method first
            try:
                resp = openai.Embeddings.create(model=model, input=inputs)
            except Exception:
                resp = openai.Embedding.create(model=model, input=inputs)
            return resp
        except OpenAIError as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"ERROR: Embedding failed after {RETRY_LIMIT} retries: {e}", file=sys.stderr)
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"Warning: Embedding request failed (attempt {attempt}/{RETRY_LIMIT}). Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"ERROR: Unexpected error while fetching embeddings: {e}", file=sys.stderr)
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"Warning: Unexpected error (attempt {attempt}/{RETRY_LIMIT}). Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)

def get_embeddings_for_texts(texts: List[str], model: str, batch_size: int = 16) -> np.ndarray:
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        resp = call_embeddings_batch(inputs, model)
        # normalize extraction for different SDK response shapes
        emb_batch = []
        if isinstance(resp, dict) and "data" in resp:
            for item in resp["data"]:
                if isinstance(item, dict):
                    emb = item.get("embedding") or item.get("vector")
                    emb_batch.append(emb)
                else:
                    # fallback
                    emb_batch.append(getattr(item, "embedding", None))
        else:
            # object-like
            data = getattr(resp, "data", None)
            if data:
                for item in data:
                    emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
                    emb_batch.append(emb)
        if len(emb_batch) != len(batch):
            raise RuntimeError("Embedding result length mismatch")
        all_emb.extend(emb_batch)
        # polite pause
        time.sleep(0.1)
    arr = np.array(all_emb, dtype="float32")
    return arr

def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / (norms + 1e-12)

def build_faiss_index(embs: np.ndarray) -> faiss.Index:
    if embs.size == 0:
        raise ValueError("Không có embedding để build index")
    dim = embs.shape[1]
    # Use inner-product index (FAISS IndexFlatIP) with normalized vectors -> cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

def write_mapping_file(mapping_list: List[dict], path: str):
    """
    Writes mapping in format expected by app.py: list of {"path":..., "text":...}
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping_list, f, ensure_ascii=False, indent=2)

def save_fallback_vectors(mat: np.ndarray, path: str):
    try:
        np.savez_compressed(path, mat=mat)
        print(f"Saved fallback vectors to {path}")
    except Exception:
        print(f"Warning: failed to save fallback vectors to {path}", file=sys.stderr)

def main():
    print("1) Load and flatten knowledge...", flush=True)
    mapping_list = load_and_flatten_knowledge(KNOW_PATH)
    if not mapping_list:
        print("ERROR: Không tìm thấy passages trong knowledge.json", file=sys.stderr)
        sys.exit(1)
    texts = [m["text"] for m in mapping_list]
    count_nonempty = sum(1 for t in texts if t and str(t).strip())
    print(f"  Tìm thấy {len(mapping_list)} passages, trong đó {count_nonempty} có text.", flush=True)
    if count_nonempty == 0:
        print("ERROR: Không có văn bản để lấy embedding.", file=sys.stderr)
        sys.exit(1)

    print(f"2) Tạo embeddings bằng model={EMBEDDING_MODEL} ...", flush=True)
    embeddings = get_embeddings_for_texts(texts, model=EMBEDDING_MODEL, batch_size=BATCH_SIZE)
    if embeddings.size == 0:
        print("ERROR: Embeddings rỗng", file=sys.stderr)
        sys.exit(1)

    print("3) Chuẩn hoá embeddings (unit norm) ...", flush=True)
    embeddings = l2_normalize_rows(embeddings).astype("float32")

    print("4) Build FAISS index (IndexFlatIP) ...", flush=True)
    index = build_faiss_index(embeddings)

    print(f"5) Ghi index ra: {FAISS_INDEX_PATH}", flush=True)
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
    except Exception as e:
        print(f"ERROR: Không ghi được index: {e}", file=sys.stderr)
        raise

    print(f"6) Ghi mapping ra: {FAISS_MAPPING_PATH}", flush=True)
    write_mapping_file(mapping_list, FAISS_MAPPING_PATH)

    print(f"7) Ghi fallback vectors (npz): {FALLBACK_VECTORS_PATH}", flush=True)
    save_fallback_vectors(embeddings, FALLBACK_VECTORS_PATH)

    # write some metadata file (optional) to make debugging easier
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_documents": len(mapping_list),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": int(embeddings.shape[1]),
        "faiss_version": getattr(faiss, "__version__", None) or getattr(faiss, "faiss_version", "")
    }
    META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("DONE: index created")
    print(f"- index: {FAISS_INDEX_PATH}")
    print(f"- mapping: {FAISS_MAPPING_PATH}")
    print(f"- fallback vectors: {FALLBACK_VECTORS_PATH}")
    print(f"- meta: {META_PATH}")

if __name__ == "__main__":
    main()
