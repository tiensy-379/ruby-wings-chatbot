#!/usr/bin/env python3
# build_index.py — build embeddings/faiss index + mapping + tour_entities.json (compatible with app.py & entities.py)
# Usage:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY="sk-..."
#   python build_index.py

import os
import sys
import json
import time
import datetime
from typing import Any, List, Optional
import numpy as np

# try imports with helpful fallbacks
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

# We use our own flatten_json that groups each tour into a single passage
# The function is defined below
def flatten_json(path: str) -> List[dict]:
    # fallback simple flattener - mỗi tour là 1 passage duy nhất
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = []
    
    # Xử lý about_company (giữ nguyên)
    about = data.get("about_company", {})
    for key, value in about.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.about_company.{key}",
                "text": value
            })
    
    # Xử lý tours - MỖI TOUR LÀ 1 PASSAGE DUY NHẤT
    tours = data.get("tours", [])
    for i, tour in enumerate(tours):
        tour_text_parts = []
        
        # Thêm các trường quan trọng
        fields_to_include = [
            ("tour_name", "Tên tour"),
            ("summary", "Tóm tắt"),
            ("location", "Địa điểm"),
            ("duration", "Thời lượng"),
            ("price", "Giá"),
            ("notes", "Lưu ý"),
            ("style", "Phong cách"),
            ("transport", "Phương tiện"),
            ("accommodation", "Chỗ ở"),
            ("meals", "Bữa ăn"),
            ("event_support", "Hỗ trợ sự kiện")
        ]
        
        for field_key, field_label in fields_to_include:
            if field_key in tour:
                value = tour[field_key]
                if isinstance(value, list):
                    tour_text_parts.append(f"{field_label}: {', '.join(str(v) for v in value)}")
                elif value and str(value).strip():
                    tour_text_parts.append(f"{field_label}: {value}")
        
        # Xử lý includes
        if "includes" in tour and tour["includes"]:
            includes_text = "Dịch vụ bao gồm: " + "; ".join(str(item) for item in tour["includes"])
            tour_text_parts.append(includes_text)
        
        # Gộp thành 1 passage
        full_tour_text = "\n".join(tour_text_parts)
        
        mapping.append({
            "path": f"root.tours[{i}]",
            "text": full_tour_text
        })
    
    # Xử lý FAQ và contact (giữ nguyên)
    faq = data.get("faq", {})
    for key, value in faq.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.faq.{key}",
                "text": value
            })
    
    contact = data.get("contact", {})
    for key, value in contact.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.contact.{key}",
                "text": value
            })
    
    return mapping

# Try to import entities.build_entity_index to persist tour_entities.json
try:
    from entities import build_entity_index, ENTITY_PATH_DEFAULT  # type: ignore
except Exception:
    build_entity_index = None
    ENTITY_PATH_DEFAULT = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")

# ---------- Config ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")
TOUR_ENTITIES_PATH = os.environ.get("TOUR_ENTITIES_PATH", ENTITY_PATH_DEFAULT)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

TMP_EMB_FILE = "emb_tmp.bin"

# ---------- Utilities ----------
def synthetic_embedding(text: str, dim: int = 1536) -> List[float]:
    h = abs(hash(text)) % (10 ** 12)
    return [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]

def call_embeddings_with_retry(inputs: List[str], model: str) -> List[List[float]]:
    if not OPENAI_KEY or OpenAI is None:
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in inputs]

    client = OpenAI(api_key=OPENAI_KEY)
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

# ---------- Main build flow ----------
def build_index():
    print("Flattening knowledge.json via common_utils...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m.get("text", "") for m in mapping]
    n = len(texts)
    print(f"Found {n} passages.")
    if n == 0:
        print("No passages to index -> exit", file=sys.stderr)
        sys.exit(1)

    # remove tmp if exists
    if os.path.exists(TMP_EMB_FILE):
        try:
            os.remove(TMP_EMB_FILE)
        except Exception:
            pass

    dim: Optional[int] = None
    total_rows = 0
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, n, BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        print(f"Embedding batch {start//BATCH_SIZE + 1}/{batches} ...", flush=True)
        vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL)

        # ensure no None entries
        for j, v in enumerate(vecs):
            if v is None:
                vecs[j] = synthetic_embedding(inputs[j], 1536 if "3-small" in EMBEDDING_MODEL else 3072)

        if dim is None and vecs:
            dim = len(vecs[0])

        arr = np.array(vecs, dtype="float32")
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
    except Exception:
        # fallback: load entire array into memory
        raw = np.fromfile(TMP_EMB_FILE, dtype="float32")
        emb = raw.reshape((total_rows, dim))

    # Build FAISS index if available
    print("Building index...")
    if HAS_FAISS:
        try:
            index = faiss.IndexFlatIP(dim)
            index.add(np.asarray(emb))
            try:
                faiss.write_index(index, FAISS_INDEX_PATH)
                print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
            except Exception:
                print("Warning: failed to persist FAISS index (continuing).", file=sys.stderr)
        except Exception as e:
            print("FAISS index build failed:", e, file=sys.stderr)
            HAS_FAISS_local = False
        else:
            HAS_FAISS_local = True
    else:
        HAS_FAISS_local = False

    # Always save fallback vectors (npz) for numpy fallback
    try:
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        print(f"Saved fallback vectors to {FALLBACK_VECTORS_PATH}")
    except Exception as e:
        print("Warning: failed to save fallback vectors:", e, file=sys.stderr)

    # Save mapping (list of {"path","text"}) expected by app.py
    print(f"Saving mapping to {FAISS_MAPPING_PATH} ...")
    try:
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save mapping:", e, file=sys.stderr)

    # Build tour_entities.json using entities.build_entity_index if available
    try:
        if build_entity_index is not None:
            print(f"Building entity index -> {TOUR_ENTITIES_PATH} ...")
            idx_map = build_entity_index(mapping, out_path=TOUR_ENTITIES_PATH)
            try:
                count_keys = len(idx_map) if isinstance(idx_map, dict) else 0
            except Exception:
                count_keys = 0
            print(f"Saved entity index keys: {count_keys}")
        else:
            print("entities.build_entity_index not available; skipping entity index build.")
    except Exception as e:
        print("Failed to build entity index:", e, file=sys.stderr)

    # write meta
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_passages": int(total_rows),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": int(dim),
        "faiss_available": bool(HAS_FAISS_local),
        "notes": "Built with batch memmap flow; tour_entities.json produced if entities module present"
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # cleanup temp
    try:
        os.remove(TMP_EMB_FILE)
    except Exception:
        pass

    print("DONE. Index ready.")
    print(f"- faiss: {FAISS_INDEX_PATH if HAS_FAISS_local else '(not produced)'}")
    print(f"- mapping: {FAISS_MAPPING_PATH}")
    print(f"- vectors (npz): {FALLBACK_VECTORS_PATH}")
    print(f"- entities: {TOUR_ENTITIES_PATH} (if produced)")
    print(f"- meta: {META_PATH}")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR building index:", e, file=sys.stderr)
        raise