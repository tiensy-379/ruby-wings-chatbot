#!/usr/bin/env python3
# build_index.py — optimized for low RAM, compatible with app.py (openai==0.28.0)
# Outputs: faiss_index.bin, faiss_mapping.json, vectors.npz, faiss_index_meta.json

import os
import sys
import json
import time
import datetime
from typing import Any, List
import numpy as np

# try imports with helpful error messages
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import openai
    from openai.error import OpenAIError
except Exception:
    openai = None
    OpenAIError = Exception

# ---------- Config (match app.py defaults) ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if OPENAI_KEY and openai is not None:
    openai.api_key = OPENAI_KEY

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))  # small for low RAM
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

TMP_EMB_FILE = "emb_tmp.bin"  # binary store for concatenated float32 rows

# ---------- Utilities ----------
def flatten_json(path: str) -> List[dict]:
    """Flatten knowledge.json into list of {'path':..., 'text':...} (order preserved)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
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
            t = obj.strip()
            if t:
                mapping.append({"path": prefix, "text": t})
        else:
            # convert scalars to str
            try:
                s = str(obj).strip()
                if s:
                    mapping.append({"path": prefix, "text": s})
            except Exception:
                pass

    scan(data, "root")
    return mapping

def call_embeddings_with_retry(inputs: List[str], model: str):
    """Call OpenAI embeddings with retries; returns list of vectors (lists)."""
    attempt = 0
    while True:
        try:
            # prefer modern name if available; for openai==0.28.0 use Embedding.create or Embeddings.create
            try:
                # many older SDKs expose Embedding.create
                resp = openai.Embedding.create(model=model, input=inputs)
            except Exception:
                # try plural
                resp = openai.Embeddings.create(model=model, input=inputs)
            # normalize extraction
            out = []
            if isinstance(resp, dict) and "data" in resp:
                for item in resp["data"]:
                    if isinstance(item, dict):
                        emb = item.get("embedding") or item.get("vector")
                        out.append(emb)
                    else:
                        out.append(getattr(item, "embedding", None))
            else:
                data_attr = getattr(resp, "data", None)
                if data_attr:
                    for item in data_attr:
                        emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
                        out.append(emb)
            return out
        except OpenAIError as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                raise
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"Warning: embedding API error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                raise
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"Warning: unexpected error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)

def synthetic_embedding(text: str, dim: int = 1536):
    """Deterministic synthetic embedding (fallback when no API)."""
    h = abs(hash(text)) % (10 ** 12)
    vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
    return vec

# ---------- Main build flow ----------
def build_index():
    print("Flatten knowledge.json...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m["text"] for m in mapping]
    n = len(texts)
    print(f"Found {n} passages.")
    if n == 0:
        print("No passages to index -> exit", file=sys.stderr)
        sys.exit(1)

    # prepare temporary binary file to append float32 rows
    if os.path.exists(TMP_EMB_FILE):
        os.remove(TMP_EMB_FILE)

    dim = None
    total_rows = 0

    # compute embeddings batch by batch, write to binary file
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        # ensure non-empty input
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        print(f"Embedding batch {i//BATCH_SIZE + 1} ...", flush=True)

        if openai is None or not OPENAI_KEY:
            # fallback to synthetic
            vecs = [synthetic_embedding(t) for t in inputs]
            # decide dim from first
            if dim is None:
                dim = len(vecs[0])
        else:
            vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL)
            if not vecs or any(v is None for v in vecs):
                # if API returned some None, fall back elementwise
                for j, v in enumerate(vecs):
                    if v is None:
                        vecs[j] = synthetic_embedding(inputs[j])
            if dim is None:
                dim = len(vecs[0])

        arr = np.array(vecs, dtype="float32")
        # normalize rows (unit norm) for cosine via inner product in FAISS
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / (norms + 1e-12)

        # append to binary file
        with open(TMP_EMB_FILE, "ab") as f:
            f.write(arr.tobytes())

        total_rows += arr.shape[0]

    if total_rows == 0:
        print("No embeddings created -> exit", file=sys.stderr)
        sys.exit(1)

    # load via memmap to avoid big RAM usage
    print("Loading embeddings with memmap...")
    # compute shape: (total_rows, dim)
    emb = np.memmap(TMP_EMB_FILE, dtype="float32", mode="r", shape=(total_rows, dim))

    # build FAISS index (IndexFlatIP with normalized vectors => cosine similarity)
    print("Building FAISS index...")
    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        # faiss expects contiguous array; memmap is okay if read-only contiguous; convert if needed
        try:
            index.add(np.asarray(emb))
        except Exception:
            # convert to a small chunk to add (safe but uses some RAM)
            index.add(np.array(emb, dtype="float32"))
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    else:
        # fallback: save vectors to npz and rely on app's NumpyIndex
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        print(f"FAISS not available: saved fallback vectors to {FALLBACK_VECTORS_PATH}")

    # write mapping as list of {"path","text"} to match app.py expectation
    print(f"Saving mapping to {FAISS_MAPPING_PATH} ...")
    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # also save fallback vectors (npz) for app when FAISS available (cheap)
    if HAS_FAISS:
        try:
            np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        except Exception:
            pass

    # write meta
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_passages": total_rows,
        "embedding_model": EMBEDDING_MODEL,
        "dimension": dim,
        "faiss_available": HAS_FAISS,
        "notes": "Built with small-batch memmap flow for low RAM"
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saving done. Clean up temporary file.")
    # keep tmp if you want; remove to save disk
    try:
        os.remove(TMP_EMB_FILE)
    except Exception:
        pass

    print("DONE. Index ready.")
    print(f"- index: {FAISS_INDEX_PATH if HAS_FAISS else '(none, fallback used)'}")
    print(f"- mapping: {FAISS_MAPPING_PATH}")
    print(f"- vectors (npz): {FALLBACK_VECTORS_PATH}")
    print(f"- meta: {META_PATH}")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR building index:", e, file=sys.stderr)
        raise
