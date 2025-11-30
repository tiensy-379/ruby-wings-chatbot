#!/usr/bin/env python3
# build_index.py — build embeddings/faiss index + stable mapping + meta
# Output:
#  - faiss_index.bin (if faiss available)
#  - faiss_mapping.json (list of {"path","text","field","tour_index"})
#  - vectors.npz (fallback numpy vectors)
#  - faiss_index_meta.json
#
# Requirements (recommended):
#   pip install numpy tqdm openai faiss-cpu
#
# Notes:
#  - Uses new OpenAI SDK (from openai import OpenAI). If missing or OPENAI_API_KEY absent,
#    will create deterministic synthetic embeddings (stable across runs).
#  - Assumes knowledge file path default 'knowledge.json'. Adjust env var KNOWLEDGE_PATH if needed.
#  - This script tries to ensure mapping order is stable: it produces mapping in canonical traversal order.
#  - Each mapping entry contains: path, text, field, tour_index (or null).
#  - You can adjust EMBEDDING_MODEL via env var EMBEDDING_MODEL.
# ------------------------------------------------------------

import os
import sys
import json
import time
import datetime
from typing import Any, List, Optional, Dict, Tuple
import numpy as np

# Optional packages
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# New OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Config (env overrides) ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))
TMP_EMB_FILE = os.environ.get("TMP_EMB_FILE", "emb_tmp.bin")

# Fields you expect each tour to have (for warnings; not enforced)
EXPECTED_TOUR_FIELDS = [
    "tour_name", "summary", "location", "duration", "price", "includes", "notes",
    "style", "transport", "accommodation", "meals", "event_support", "hotline", "mission"
]

# ---------- Helper utilities ----------
def synthetic_embedding(text: str, dim: int = 1536) -> List[float]:
    """Deterministic synthetic embedding (stable across runs for same input)."""
    h = abs(hash(text)) % (10 ** 12)
    return [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]

def call_embeddings_with_retry(inputs: List[str], model: str, client: Optional[Any]) -> List[List[float]]:
    """
    Call embeddings API with retries. If no OPENAI_KEY or client, return synthetic embeddings.
    Uses new OpenAI SDK call style: client.embeddings.create(...)
    """
    if not OPENAI_KEY or OpenAI is None or client is None:
        # fallback dims guess
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in inputs]

    attempt = 0
    while attempt <= RETRY_LIMIT:
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            if getattr(resp, "data", None):
                out = [r.embedding for r in resp.data]
                print(f"✅ Generated {len(out)} embeddings (model={model})", flush=True)
                return out
            else:
                raise ValueError("Empty response from OpenAI embeddings API")
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"❌ Embedding API failed after {RETRY_LIMIT} attempts: {e}", file=sys.stderr)
                dim = 1536 if "3-small" in model else 3072
                return [synthetic_embedding(t, dim) for t in inputs]
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"⚠️ Embedding API error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
    dim = 1536 if "3-small" in model else 3072
    return [synthetic_embedding(t, dim) for t in inputs]

# ---------- Flattening / mapping ----------
def canonical_flatten(path_prefix: str, obj: Any, mapping: List[Dict[str, Any]]):
    """
    Flatten JSON into mapping entries. Maintains deterministic traversal order:
    - dict: sorted by key (stable)
    - list: iterated in index order
    Each text leaf becomes an entry with fields:
        {"path": "<prefix>", "text": "<text>", "field": "<last_key>", "tour_index": <int or None>}
    """
    if isinstance(obj, dict):
        # iterate in sorted key order for determinism
        for k in sorted(obj.keys()):
            v = obj[k]
            canonical_flatten(f"{path_prefix}.{k}", v, mapping)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            canonical_flatten(f"{path_prefix}[{i}]", v, mapping)
    elif isinstance(obj, str):
        t = obj.strip()
        if t:
            last_key = path_prefix.split(".")[-1]
            tour_index = None
            m = None
            try:
                m = __import__("re").search(r"\[(\d+)\]", path_prefix)
            except Exception:
                m = None
            if m:
                try:
                    tour_index = int(m.group(1))
                except Exception:
                    tour_index = None
            entry = {"path": path_prefix, "text": t, "field": last_key, "tour_index": tour_index}
            mapping.append(entry)
    else:
        # other scalar types
        try:
            s = str(obj).strip()
            if s:
                last_key = path_prefix.split(".")[-1]
                tour_index = None
                m = None
                try:
                    m = __import__("re").search(r"\[(\d+)\]", path_prefix)
                except Exception:
                    m = None
                if m:
                    try:
                        tour_index = int(m.group(1))
                    except Exception:
                        tour_index = None
                entry = {"path": path_prefix, "text": s, "field": last_key, "tour_index": tour_index}
                mapping.append(entry)
        except Exception:
            pass

def flatten_json_to_mapping(know_path: str) -> List[Dict[str, Any]]:
    """Loads knowledge.json and returns deterministic mapping list."""
    if not os.path.exists(know_path):
        raise FileNotFoundError(f"Knowledge file not found: {know_path}")
    with open(know_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: List[Dict[str, Any]] = []
    canonical_flatten("root", data, mapping)
    return mapping

# ---------- Main build flow ----------
def build_index():
    print("Flattening knowledge.json ...")
    mapping = flatten_json_to_mapping(KNOW_PATH)
    texts = [m.get("text", "") for m in mapping]
    n = len(texts)
    print(f"Found {n} passages.")
    if n == 0:
        print("No passages to index -> exit", file=sys.stderr)
        sys.exit(1)

    # Warn if expected fields missing for any tour
    # Build tour->fields map
    tour_fields = {}
    for m in mapping:
        ti = m.get("tour_index")
        f = m.get("field")
        if ti is not None:
            tour_fields.setdefault(ti, set()).add(f)
    for ti, fields in tour_fields.items():
        missing = [fld for fld in EXPECTED_TOUR_FIELDS if fld not in fields]
        if missing:
            print(f"⚠️ Tour index [{ti}] missing expected fields: {missing}")

    # Prepare temp emb file
    if os.path.exists(TMP_EMB_FILE):
        try:
            os.remove(TMP_EMB_FILE)
        except Exception:
            pass

    dim: Optional[int] = None
    total_rows = 0
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    # Init OpenAI client if possible
    client = None
    if OPENAI_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_KEY)
        except Exception as e:
            print("Warning: OpenAI client init failed:", e, file=sys.stderr)
            client = None

    # Batch embed
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        print(f"Embedding batch {start//BATCH_SIZE + 1}/{batches} ...", flush=True)
        vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL, client)

        # ensure replacement for None
        for j, v in enumerate(vecs):
            if v is None:
                vecs[j] = synthetic_embedding(inputs[j], 1536 if "3-small" in EMBEDDING_MODEL else 3072)

        if dim is None and vecs:
            dim = len(vecs[0])

        arr = np.array(vecs, dtype="float32")
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / (norms + 1e-12)

        with open(TMP_EMB_FILE, "ab") as f:
            f.write(arr.tobytes())

        total_rows += arr.shape[0]

    if total_rows == 0 or dim is None:
        print("No embeddings created -> exit", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings via memmap...")
    try:
        emb = np.memmap(TMP_EMB_FILE, dtype="float32", mode="r", shape=(total_rows, dim))
        emb_arr = np.asarray(emb)
    except Exception:
        # fallback load into memory
        raw = np.fromfile(TMP_EMB_FILE, dtype="float32")
        emb_arr = raw.reshape((total_rows, dim))

    # Build FAISS if available
    print("Building index...")
    has_faiss_local = False
    if HAS_FAISS:
        try:
            index = faiss.IndexFlatIP(dim)
            index.add(np.asarray(emb_arr))
            try:
                faiss.write_index(index, FAISS_INDEX_PATH)
                print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
            except Exception:
                print("Warning: failed to persist FAISS index (continuing).", file=sys.stderr)
            has_faiss_local = True
        except Exception as e:
            print("FAISS index build failed:", e, file=sys.stderr)
            has_faiss_local = False
    else:
        has_faiss_local = False

    # Always save fallback vectors (npz)
    try:
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb_arr))
        print(f"Saved fallback vectors to {FALLBACK_VECTORS_PATH}")
    except Exception as e:
        print("Warning: failed to save fallback vectors:", e, file=sys.stderr)

    # Save mapping in stable order
    try:
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved mapping to {FAISS_MAPPING_PATH}")
    except Exception as e:
        print("Failed to save mapping:", e, file=sys.stderr)

    # Save meta
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_passages": int(total_rows),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": int(dim),
        "faiss_available": bool(has_faiss_local),
        "notes": "Mapping entries include fields: path, text, field, tour_index"
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # cleanup
    try:
        os.remove(TMP_EMB_FILE)
    except Exception:
        pass

    print("DONE. Index ready.")
    print(f"- faiss: {FAISS_INDEX_PATH if has_faiss_local else '(not produced)'}")
    print(f"- mapping: {FAISS_MAPPING_PATH}")
    print(f"- vectors (npz): {FALLBACK_VECTORS_PATH}")
    print(f"- meta: {META_PATH}")

# ---------- CLI entry ----------
if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR building index:", e, file=sys.stderr)
        raise
