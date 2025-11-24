# build_index.py
# Tạo FAISS index + metadata từ knowledge.json bằng OpenAI Embeddings
# Yêu cầu: openai, faiss, numpy (có trong requirements.txt thường)
import os, json, sys, time
import numpy as np

try:
    import faiss
except Exception as e:
    print("ERROR: Không import được faiss. Cài bằng pip nếu cần: pip install faiss-cpu", file=sys.stderr)
    raise

try:
    import openai
except Exception as e:
    print("ERROR: Không import được openai. Cài bằng pip install openai", file=sys.stderr)
    raise

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY chưa đặt trong biến môi trường.", file=sys.stderr)
    sys.exit(1)
openai.api_key = OPENAI_KEY

KNOW = "knowledge.json"
INDEX_PATH = "index.faiss"
META_PATH = "index_metadata.json"
MODEL = "text-embedding-3-small"  # ổn cho hầu hết use-case; đổi nếu bạn muốn

def load_docs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} không tồn tại")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("knowledge.json phải là mảng (list) các document")
    return data

def texts_from_docs(docs):
    texts = []
    ids = []
    metas = []
    for i, d in enumerate(docs):
        # Thích nghi: dùng 'text' hoặc 'description' hoặc 'content'
        text = d.get("text") or d.get("description") or d.get("name") or ""
        texts.append(text)
        ids.append(str(d.get("id", i)))
        metas.append(d)
    return texts, ids, metas

def get_embeddings(texts, model=MODEL, batch_size=16):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # tránh gửi empty strings
        inputs = [t if (t and t.strip()) else " " for t in batch]
        resp = openai.Embedding.create(model=model, input=inputs)
        emb_batch = [r["embedding"] for r in resp["data"]]
        all_emb.extend(emb_batch)
        time.sleep(0.1)
    return np.array(all_emb, dtype="float32")

def build_faiss(embs):
    if embs.size == 0:
        raise ValueError("Không có embedding để build index")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index

def main():
    print("1) Load documents...", flush=True)
    docs = load_docs(KNOW)
    texts, ids, metas = texts_from_docs(docs)
    count_nonempty = sum(1 for t in texts if t and t.strip())
    print(f"  Tìm thấy {len(docs)} documents, trong đó {count_nonempty} có text.", flush=True)
    if count_nonempty == 0:
        print("ERROR: Không có văn bản để lấy embedding.", file=sys.stderr)
        sys.exit(1)

    print("2) Tạo embeddings (gửi tới OpenAI)...", flush=True)
    embeddings = get_embeddings(texts)

    print("3) Build FAISS index...", flush=True)
    index = build_faiss(embeddings)

    print(f"4) Ghi index ra: {INDEX_PATH}", flush=True)
    faiss.write_index(index, INDEX_PATH)

    meta = {"ids": ids, "docs_meta": metas, "count": len(ids)}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("DONE: index created")
    print(f"- index: {INDEX_PATH}")
    print(f"- metadata: {META_PATH}")

if __name__ == "__main__":
    main()
