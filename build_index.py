#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULL build_index.py – Đồng bộ hoàn chỉnh với app.py (MODE C – Auto-detect platform)

Tính năng:
- Tự nhận biết Windows / Render → chọn đúng đường dẫn.
- Chuẩn hoá knowledge.json → knowledge_fixed.json.
- Tạo passages → faiss_mapping.json.
- Sinh embeddings (OpenAI nếu có key, fallback deterministic nếu không).
- Tạo FAISS index (nếu có faiss) hoặc vectors.npz fallback.
- Lưu mọi file đúng cấu trúc app.py yêu cầu.
- Không sinh file rác, không sai định dạng.
"""

import os
import json
import hashlib
import logging
import numpy as np

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build")

# ---------------- Detect Environment ----------------
def is_windows():
    return os.name == "nt"

def base_path():
    """
    MODE C – Auto detect
    - Windows dùng local folder.
    - Render dùng /mnt/data/.
    """
    if is_windows():
        return os.getcwd()  # ví dụ: C:\project\ruby-wings-chatbot
    return "/mnt/data"

BASE = base_path()

# ---------------- Paths ----------------
KNOWLEDGE_IN  = os.path.join(BASE, "knowledge.json")
KNOWLEDGE_OUT = os.path.join(BASE, "knowledge_fixed.json")
MAPPING_PATH  = os.path.join(BASE, "faiss_mapping.json")
VECTORS_PATH  = os.path.join(BASE, "vectors.npz")
FAISS_INDEX_PATH = os.path.join(BASE, "faiss_index.bin")

# ---------------- Try import FAISS ----------------
try:
    import faiss
    HAS_FAISS = True
except Exception as e:
    faiss = None
    HAS_FAISS = False
    log.warning("FAISS unavailable: %s", e)

# ---------------- Try import OpenAI ----------------
try:
    import openai
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    else:
        log.warning("No OPENAI_API_KEY → fallback embeddings only")
except Exception:
    openai = None
    OPENAI_API_KEY = ""
    log.warning("OpenAI SDK not available → fallback embeddings only")

EMBEDDING_MODEL = "text-embedding-3-large"

# =========================
#  STEP 1 — LOAD & NORMALIZE
# =========================

def normalize_item(item):
    """
    Chuẩn hoá từng object tour — đảm bảo trường nào thiếu thì tạo.
    Không thay đổi nội dung của bạn, chỉ bổ sung cấu trúc chuẩn.
    """
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "short_description": item.get("short_description", ""),
        "vision": item.get("vision", ""),
        "mission": item.get("mission", ""),
        "core_values": item.get("core_values", []),
        "highlights": item.get("highlights", []),
        "itinerary": item.get("itinerary", {}),
        "price_info": item.get("price_info", {}),
        "target_audience": item.get("target_audience", ""),
        "contact": item.get("contact", ""),
        "tags": item.get("tags", []),
        "faqs": item.get("faqs", [])
    }

def normalize_knowledge():
    if not os.path.exists(KNOWLEDGE_IN):
        raise FileNotFoundError(f"Missing knowledge.json at: {KNOWLEDGE_IN}")

    with open(KNOWLEDGE_IN, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        normalized = {"items": [normalize_item(x) for x in data]}
    elif isinstance(data, dict) and "items" in data:
        normalized = {"items": [normalize_item(x) for x in data["items"]]}
    else:
        raise ValueError("knowledge.json must be list or dict with 'items'")

    with open(KNOWLEDGE_OUT, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    log.info("Wrote normalized file: %s", KNOWLEDGE_OUT)
    return normalized

# =========================
#  STEP 2 — EXTRACT PASSAGES
# =========================

def extract_passages(obj, prefix="root"):
    passages = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            passages += extract_passages(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            passages += extract_passages(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        text = obj.strip()
        if text:
            passages.append({"path": prefix, "text": text})

    return passages

# =========================
#  STEP 3 — EMBEDDING (OpenAI or fallback)
# =========================

def fallback_embedding(text: str) -> np.ndarray:
    """
    Fallback deterministic: SHA256 → vector 1536 dims normalized
    Đồng bộ 100% với app.py
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    needed = 1536 * 4
    rep = (h * ((needed // len(h)) + 1))[:needed]
    arr = np.frombuffer(rep, dtype=np.uint8).astype(np.float32)

    arr = arr.reshape(-1, 4)
    ints = (arr[:,0]*256**3 + arr[:,1]*256**2 + arr[:,2]*256 + arr[:,3]).astype(np.float64)
    floats = (ints % 1_000_000) / 1_000_000.0
    vec = np.resize(floats, 1536).astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm:
        vec = vec / norm
    return vec

def embed_batch(texts):
    """
    Embed batch.
    Nếu có OPENAI → dùng OpenAI.
    Nếu lỗi → fallback.
    """
    if OPENAI_API_KEY and openai is not None:
        try:
            res = openai.Embeddings.create(model=EMBEDDING_MODEL, input=texts)
            data = res.get("data", [])
            if data:
                out = []
                for d in data:
                    v = d.get("embedding")
                    if v:
                        out.append(np.array(v, dtype=np.float32))
                    else:
                        out.append(fallback_embedding(d))
                return out
        except Exception as e:
            log.warning("OpenAI embedding failed → fallback. Error: %s", e)

    return [fallback_embedding(t) for t in texts]

# =========================
#  STEP 4 — BUILD INDEX
# =========================

def build_index(vectors):
    mat = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-12)

    if HAS_FAISS:
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        faiss.write_index(index, FAISS_INDEX_PATH)
        log.info("FAISS index written → %s", FAISS_INDEX_PATH)
        return "faiss"
    else:
        np.savez_compressed(VECTORS_PATH, vectors=mat)
        log.info("Fallback vectors saved → %s", VECTORS_PATH)
        return "vectors"

# =========================
#  MAIN PIPELINE
# =========================

def run_pipeline():
    log.info("=== STEP 1: Normalize knowledge.json ===")
    knowledge = normalize_knowledge()

    log.info("=== STEP 2: Extract passages ===")
    passages = extract_passages(knowledge)
    log.info("Extracted %d passages", len(passages))

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    log.info("Wrote mapping → %s", MAPPING_PATH)

    log.info("=== STEP 3: Embedding passages ===")
    texts = [p["text"] for p in passages]
    vectors = embed_batch(texts)
    log.info("Generated embeddings: %s", len(vectors))

    log.info("=== STEP 4: Build index ===")
    mode = build_index(vectors)

    log.info("=== DONE ===")
    return {
        "passages": len(passages),
        "mode": mode,
        "knowledge_out": KNOWLEDGE_OUT,
        "mapping": MAPPING_PATH,
        "index": FAISS_INDEX_PATH if mode=="faiss" else VECTORS_PATH
    }

# =========================
#  RUN
# =========================

if __name__ == "__main__":
    out = run_pipeline()
    print(json.dumps(out, ensure_ascii=False, indent=2))
