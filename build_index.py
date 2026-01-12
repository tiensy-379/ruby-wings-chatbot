import json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE = os.path.dirname(__file__)

KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "knowledge.json")
INDEX_FAISS = "index.faiss"
FAISS_BIN = "faiss_index.bin"
VECTORS_NPZ = "vectors.npz"
MAPPING_JSON = "faiss_mapping.json"
META_JSON = "index_metadata.json"
ENTITIES_JSON = "tour_entities.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tours = data.get("tours", [])
texts = []
entities = []

for i, t in enumerate(tours):
    blob = " ".join([
        t.get("tour_name",""), t.get("summary",""), t.get("location",""),
        t.get("duration",""), t.get("price",""), " ".join(t.get("includes",[])),
        t.get("notes",""), t.get("style",""), str(t.get("transport","")),
        str(t.get("accommodation","")), t.get("meals",""), t.get("event_support","")
    ])
    texts.append(blob)
    entities.append(t)

embeddings = model.encode(texts, show_progress_bar=True)
np.savez(VECTORS_NPZ, vectors=embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_FAISS)
faiss.write_index(index, FAISS_BIN)

mapping = {str(i): t["tour_name"] for i,t in enumerate(tours)}
with open(MAPPING_JSON,"w",encoding="utf-8") as f:
    json.dump(mapping,f,ensure_ascii=False,indent=2)

meta = {
    "count": len(tours),
    "dim": dim,
    "model": "all-MiniLM-L6-v2"
}
with open(META_JSON,"w") as f:
    json.dump(meta,f,indent=2)

with open(ENTITIES_JSON,"w",encoding="utf-8") as f:
    json.dump(entities,f,ensure_ascii=False,indent=2)

print("Build complete")