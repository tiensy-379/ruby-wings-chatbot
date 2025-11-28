# build_index.py (super optimized for Render)
# - RAM thấp
# - CPU nhẹ
# - Tương thích 100% app.py và knowledge.json chuẩn RAG
# - Đảm bảo: dữ liệu tách riêng → embedding riêng

import os, sys, json, time, datetime
import numpy as np

try:
    import faiss
except:
    print("ERROR: cần cài faiss-cpu", file=sys.stderr)
    raise

try:
    import openai
    from openai.error import OpenAIError
except:
    print("ERROR: cần cài openai", file=sys.stderr)
    raise

# ==== CONFIG ===========================================================
openai.api_key = os.environ["OPENAI_API_KEY"]

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
BATCH_SIZE = 8   # rất nhỏ → cực nhẹ RAM
RETRY_LIMIT = 5
# =======================================================================


# ==== STEP 1: FLATTEN knowledge.json ===================================
def flatten_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = []

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
                mapping.append({"path": prefix, "text": t})
        else:
            try:
                t = str(obj).strip()
                if t:
                    mapping.append({"path": prefix, "text": t})
            except:
                pass

    scan(data, "root")
    return mapping


# ==== STEP 2: EMBEDDING WITH ULTRA-LOW RAM =============================
def embed_batch(batch):
    attempt = 0
    while True:
        try:
            try:
                resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=batch)
            except:
                resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            return [item["embedding"] for item in resp["data"]]
        except Exception as e:
            attempt += 1
            if attempt >= RETRY_LIMIT:
                raise
            time.sleep(1.2 * attempt)


# ==== STEP 3: BUILD INDEX STREAMING (KHÔNG GIỮ EMBEDDINGS TRONG RAM) ===
def build_index():
    print("Flatten knowledge.json...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m["text"] for m in mapping]
    print(f"Found {len(texts)} passages.")

    # Dùng file tạm để ghi embedding từng batch
    tmp_path = "emb_tmp.npy"
    dim = None
    total = 0

    with open(tmp_path, "wb") as tmp:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch = [t if t.strip() else " " for t in batch]

            print(f"Embedding batch {i//BATCH_SIZE + 1} ...")
            vecs = embed_batch(batch)
            mat = np.array(vecs, dtype="float32")

            # normalize unit vector (cosine)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (norms + 1e-12)

            if dim is None:
                dim = mat.shape[1]

            # ghi nhị phân để tránh chiếm RAM
            mat.tofile(tmp)
            total += mat.shape[0]

    # ==== Load lại file embedding bằng memmap (RAM siêu thấp) ==========
    print("Loading embeddings with memmap...")
    emb = np.memmap(tmp_path, dtype="float32", mode="r", shape=(total, dim))

    # ==== Build FAISS ===================================================
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving mapping.json ...")
    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("Saving fallback vectors (npz)...")
    np.savez_compressed(FALLBACK_VECTORS_PATH, mat=emb)

    print("DONE. Index ready.")


# ==== MAIN =============================================================
if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR building index:", e)
        sys.exit(1)
