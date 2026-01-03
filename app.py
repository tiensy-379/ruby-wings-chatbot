# app.py — "HOÀN HẢO NHẤT" phiên bản tối ưu cho openai>=1.0.0, FAISS fallback, ưu tiên lấy FIELD trong cùng TOUR
# Mục tiêu: luôn trả lời bằng trường (field) đúng của tour khi user nhắc đến tên tour hoặc hỏi keyword liên quan.

from meta_capi import send_meta_pageview
import os
import json
import threading
import logging
import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
# ... các import hiện có ...





# Try FAISS; fallback to numpy-only index if missing
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ✅ OPENAI API MỚI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# ✅ KHỞI TẠO CLIENT OPENAI MỚI
client = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not set — embeddings/chat will fallback to deterministic behavior when possible.")

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Meta CAPI ----------
@app.before_request
def track_meta_pageview():
    send_meta_pageview(request)

# ---------- Global state ----------
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []  # list of {"path": "...", "text": "..."}
INDEX = None
INDEX_LOCK = threading.Lock()

# Mapping normalized tour name -> index (populated after load/build)
TOUR_NAME_TO_INDEX: Dict[str, int] = {}

# ---------- Keyword -> field mapping (priority) ----------
KEYWORD_FIELD_MAP: Dict[str, Dict] = {
    "tour_list": {
        "keywords": [
            "tên tour", "tour gì", "danh sách tour", "có những tour nào", "liệt kê tour",
            "show tour", "tour hiện có", "tour available", "liệt kê các tour đang có",
            "list tour", "tour đang bán", "tour hiện hành", "tour nào", "tours", "liệt kê các tour",
            "liệt kê các hành trình", "list tours", "show tours", "các tour hiện tại"
        ],
        "field": "tour_name"
    },
    "mission": {"keywords": ["tầm nhìn", "sứ mệnh", "giá trị cốt lõi", "triết lý", "vision", "mission"], "field": "mission"},
    "summary": {"keywords": ["tóm tắt chương trình tour", "tóm tắt", "overview", "brief", "mô tả ngắn"], "field": "summary"},
    "style": {"keywords": ["phong cách hành trình", "tính chất hành trình", "concept tour", "vibe tour", "style"], "field": "style"},
    "transport": {"keywords": ["vận chuyển", "phương tiện", "di chuyển", "xe gì", "transportation"], "field": "transport"},
    "includes": {"keywords": ["lịch trình chi tiết", "chương trình chi tiết", "chi tiết hành trình", "itinerary", "schedule", "includes"], "field": "includes"},
    "location": {"keywords": ["ở đâu", "đi đâu", "địa phương nào", "nơi nào", "điểm đến", "destination", "location"], "field": "location"},
    "duration": {"keywords": ["thời gian tour", "kéo dài", "mấy ngày", "bao lâu", "ngày đêm", "duration", "tour dài bao lâu", "tour bao nhiêu ngày", "2 ngày 1 đêm", "3 ngày 2 đêm"], "field": "duration"},
    "price": {"keywords": ["giá tour", "chi phí", "bao nhiêu tiền", "price", "cost"], "field": "price"},
    "notes": {"keywords": ["lưu ý", "ghi chú", "notes", "cần chú ý"], "field": "notes"},
    "accommodation": {"keywords": ["chỗ ở", "nơi lưu trú", "khách sạn", "homestay", "accommodation"], "field": "accommodation"},
    "meals": {"keywords": ["ăn uống", "ẩm thực", "meals", "thực đơn", "bữa"], "field": "meals"},
    "event_support": {"keywords": ["hỗ trợ", "dịch vụ hỗ trợ", "event support", "dịch vụ tăng cường"], "field": "event_support"},
    "cancellation_policy": {"keywords": ["phí huỷ", "chính sách huỷ", "cancellation", "refund policy"], "field": "cancellation_policy"},
    "booking_method": {"keywords": ["đặt chỗ", "đặt tour", "booking", "cách đặt"], "field": "booking_method"},
    "who_can_join": {"keywords": ["phù hợp đối tượng", "ai tham gia", "who should join"], "field": "who_can_join"},
    "hotline": {"keywords": ["hotline", "số điện thoại", "liên hệ", "contact number"], "field": "hotline"},
}

# ---------- Utilities ----------
def normalize_text_simple(s: str) -> str:
    """Lowercase, remove diacritics, strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove diacritics
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Index-tour-name helpers ----------
def index_tour_names():
    """Populate TOUR_NAME_TO_INDEX from MAPPING entries that end with .tour_name."""
    global TOUR_NAME_TO_INDEX
    TOUR_NAME_TO_INDEX = {}
    for m in MAPPING:
        path = m.get("path", "")
        if path.endswith(".tour_name"):
            txt = m.get("text", "") or ""
            norm = normalize_text_simple(txt)
            if not norm:
                continue
            # extract index within brackets: root.tours[2].tour_name
            match = re.search(r"\[(\d+)\]", path)
            if match:
                idx = int(match.group(1))
                # if duplicate normalized name, keep the first/longest or override heuristics
                prev = TOUR_NAME_TO_INDEX.get(norm)
                if prev is None:
                    TOUR_NAME_TO_INDEX[norm] = idx
                else:
                    # prefer longer original name (less likely to be ambiguous)
                    if len(txt) > len(MAPPING[next(i for i,m2 in enumerate(MAPPING) if re.search(rf"\[{prev}\]", m2.get('path','')) )].get("text","")):
                        TOUR_NAME_TO_INDEX[norm] = idx

def find_tour_indices_from_message(message: str) -> List[int]:
    """Improved tour detection with fuzzy matching"""
    if not message:
        return []
    
    msg_n = normalize_text_simple(message)
    if not msg_n:
        return []
    
    # Thêm fuzzy matching đơn giản
    matches = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        # Kiểm tra từng từ trong tên tour
        tour_words = set(norm_name.split())
        msg_words = set(msg_n.split())
        
        # Match nếu có từ khóa trùng
        common_words = tour_words & msg_words
        if len(common_words) >= 1:  # Giảm ngưỡng match
            matches.append((len(common_words), norm_name))
    
    if matches:
        matches.sort(reverse=True)
        best_score = matches[0][0]
        selected = [TOUR_NAME_TO_INDEX[nm] for sc, nm in matches if sc == best_score]
        return sorted(set(selected))
    
    return []

# ---------- MAPPING helpers ----------
def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    """
    Return passages whose path ends with field_name.
    If tour_indices provided, RESTRICT and PRIORITIZE entries matching those tour index brackets.
    Returned score is 2.0 for exact tour match, 1.0 for global match.
    """
    exact_matches: List[Tuple[float, dict]] = []
    global_matches: List[Tuple[float, dict]] = []
    
    for m in MAPPING:
        path = m.get("path", "")
        # match exact field location (ending with .field) or field somewhere in path
        if path.endswith(f".{field_name}") or f".{field_name}" in path:
            
            # Check if this passage belongs to any of the mentioned tours
            is_exact_match = False
            if tour_indices:
                for ti in tour_indices:
                    if f"[{ti}]" in path:
                        is_exact_match = True
                        break
            
            if is_exact_match:
                # ✅ ƯU TIÊN CAO: exact tour match
                exact_matches.append((2.0, m))
            elif not tour_indices:
                # ✅ Global match (no specific tour mentioned)
                global_matches.append((1.0, m))
    
    # ✅ COMBINE: Exact matches first, then global matches
    all_results = exact_matches + global_matches
    
    # ✅ SORT by score (exact matches will come first)
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    return all_results[:limit]

# ---------- Embeddings (robust) ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Return (embedding list, dim)
    Tries openai.Embedding.create (SDK 0.28.0). If API key missing or call fails, return deterministic fallback 1536-dim.
    """
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    
    # ✅ SỬ DỤNG OPENAI API MỚI
    if client is not None:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL, 
                input=short
            )
            # ✅ TRÍCH XUẤT DỮ LIỆU MỚI
            if resp.data and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding call failed — falling back to deterministic embedding.")
    
    # Deterministic fallback (stable across runs)
    try:
        h = abs(hash(short)) % (10 ** 12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# ---------- Index management ----------
def _index_dim(idx) -> Optional[int]:
    # Try common attributes then faiss-specific
    try:
        d = getattr(idx, "d", None)
        if isinstance(d, int) and d > 0:
            return d
    except Exception:
        pass
    try:
        d = getattr(idx, "dim", None)
        if isinstance(d, int) and d > 0:
            return d
    except Exception:
        pass
    try:
        if HAS_FAISS and isinstance(idx, faiss.Index):
            return int(idx.d)
    except Exception:
        pass
    return None

def choose_embedding_model_for_dim(dim: int) -> str:
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

class NumpyIndex:
    """Simple in-memory numpy index with cosine-similarity (via normalized dot product)."""
    def __init__(self, mat: Optional[np.ndarray] = None):
        if mat is None or getattr(mat, "size", 0) == 0:
            self.mat = np.empty((0, 0), dtype="float32")
            self.dim = None
        else:
            self.mat = mat.astype("float32")
            self.dim = self.mat.shape[1]

    def add(self, mat: np.ndarray):
        if getattr(mat, "size", 0) == 0:
            return
        mat = mat.astype("float32")
        if getattr(self.mat, "size", 0) == 0:
            self.mat = mat.copy()
            self.dim = mat.shape[1]
        else:
            if mat.shape[1] != self.dim:
                raise ValueError("Dimension mismatch")
            self.mat = np.vstack([self.mat, mat])

    def search(self, qvec: np.ndarray, k: int):
        if self.mat is None or getattr(self.mat, "size", 0) == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec.astype("float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")

    @property
    def ntotal(self):
        return 0 if getattr(self.mat, "size", 0) == 0 else self.mat.shape[0]

    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except Exception:
            logger.exception("Failed to save fallback vectors")

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            mat = arr["mat"]
            return cls(mat=mat)
        except Exception:
            logger.exception("Failed to load fallback vectors")
            return cls(None)

def load_mapping_from_disk(path=FAISS_MAPPING_PATH):
    global MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING[:] = json.load(f)
        FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
        logger.info("Loaded mapping from %s (%d entries)", path, len(MAPPING))
        return True
    except Exception:
        logger.exception("Failed to load mapping from disk")
        return False

def save_mapping_to_disk(path=FAISS_MAPPING_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        logger.info("Saved mapping to %s", path)
    except Exception:
        logger.exception("Failed to save mapping")

def build_index(force_rebuild: bool = False) -> bool:
    """
    Build or load index. If FAISS enabled and available, use it; otherwise NumpyIndex.
    Will auto-detect saved index+mapping and choose embedding model if dims known.
    """
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

        # try loading persisted structures first
        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    idx_dim = _index_dim(idx)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                        logger.info("Detected FAISS index dim=%s -> embedding_model=%s", idx_dim, EMBEDDING_MODEL)
                    INDEX = idx
                    index_tour_names()
                    logger.info("✅ FAISS index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load FAISS index; will rebuild.")
            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = NumpyIndex.load(FALLBACK_VECTORS_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    INDEX = idx
                    idx_dim = getattr(idx, "dim", None)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(int(idx_dim))
                        logger.info("Detected fallback vectors dim=%s -> embedding_model=%s", idx_dim, EMBEDDING_MODEL)
                    index_tour_names()
                    logger.info("✅ Fallback index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load fallback vectors; will rebuild.")

        # need to build from FLAT_TEXTS
        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index (build aborted).")
            INDEX = None
            return False

        logger.info("🔧 Building embeddings for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
        vectors = []
        dims = None
        for text in FLAT_TEXTS:
            emb, d = embed_text(text)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors or dims is None:
            logger.warning("No vectors produced; index build aborted.")
            INDEX = None
            return False

        try:
            mat = np.vstack(vectors).astype("float32")
            # normalize rows for cosine similarity
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (row_norms + 1e-12)

            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                try:
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    save_mapping_to_disk()
                except Exception:
                    logger.exception("Failed to persist FAISS index/mapping")
                index_tour_names()
                logger.info("✅ FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
                return True
            else:
                idx = NumpyIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    save_mapping_to_disk()
                except Exception:
                    logger.exception("Failed to persist fallback vectors/mapping")
                index_tour_names()
                logger.info("✅ Numpy fallback index built (dims=%d, n=%d).", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

# ---------- Query index ----------
def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    global INDEX
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built or INDEX is None:
            logger.warning("Index not available; semantic search skipped.")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.error("Query dim %s != index dim %s; will attempt rebuild with matching model.", vec.shape[1], idx_dim)
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            logger.info("Setting EMBEDDING_MODEL=%s and rebuilding index...", EMBEDDING_MODEL)
            rebuilt = build_index(force_rebuild=True)
            if not rebuilt:
                logger.error("Rebuild failed; cannot perform search.")
                return []
            emb2, d2 = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1, -1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY; cannot rebuild model-matched index.")
            return []
    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Error executing index.search")
        return []

    results: List[Tuple[float, dict]] = []
    try:
        scores = D[0].tolist() if getattr(D, "shape", None) else []
        idxs = I[0].tolist() if getattr(I, "shape", None) else []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(MAPPING):
                continue
            results.append((float(score), MAPPING[idx]))
    except Exception:
        logger.exception("Failed to parse search results")
    return results

# ---------- Prompt composition ----------
def compose_system_prompt(top_passages: List[Tuple[float, dict]]) -> str:
    header = (

        "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.\n"
        "TRẢ LỜI THEO CÁC NGUYÊN TẮC:\n"
        "1. ƯU TIÊN CAO: Thông tin từ dữ liệu được cung cấp\n"
        "2. Nếu thiếu thông tin CHI TIẾT, hãy trả lời dựa trên THÔNG TIN CHUNG có sẵn\n"
        "3. Đối với tour cụ thể: tìm thông tin đúng tour trước, sau đó mới dùng thông tin chung, nếu không có thông tin thì trả lời hiện nay Ruby Wings đang nâng cấp, cập nhật\n"
        "4. Luôn giữ thái độ nhiệt tình, hữu ích\n\n"
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn nghành du lịch trải nghiệm, retreat, "
        "thiền, khí công, hành trình chữa lành - Hành trình tham quan linh hoạt theo nhhu cầu. Trả lời ngắn gọn, chính xác, tử tế.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    
    content = header + "DỮ LIỆU NỘI BỘ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"

    
    content += "\n---\nTUÂN THỦ: Chỉ dùng dữ liệu trên; không bịa đặt nội dung không có thực; văn phong lịch sự."
    content += "\n---\nLưu ý: Ưu tiên sử dụng trích dẫn thông tin từ dữ liệu nội bộ ở trên. Nếu phải bổ sung, chỉ dùng kiến thức chuẩn xác, không được tự ý bịa ra khi chưa rõ đúng sai; sử dụng ngôn ngữ lịch sự, thân thiện, thông minh; khi khách gõ lời tạm biệt hoặc lời chúc thì chân thành cám ơn khách, chúc khách sức khoẻ tốt, may mắn, thành công..."
    return content

# ---------- Routes ----------
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
    # require header or env allow
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
    load_knowledge()  # reload raw knowledge before building
    ok = build_index(force_rebuild=True)
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint behavior:
      - If user message contains keywords mapping to a field, prioritize returning that field.
      - If a tour name is mentioned, restrict to that tour's field values.
      - If user asked for tour listing (tour_name), list all tour_name entries.
      - Else fallback to semantic search and LLM reply.
    """
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})

    text_l = user_message.lower()
    requested_field: Optional[str] = None
    # keyword detection (maintain insertion order of KEYWORD_FIELD_MAP)
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in text_l:
                requested_field = v["field"]
                break
        if requested_field:
            break

    # detect tour mentions
    tour_indices = find_tour_indices_from_message(user_message)

    top_results: List[Tuple[float, dict]] = []

    # If user explicitly asked for tour_name listing -> list all tour names (not restricted)
    if requested_field == "tour_name":
        top_results = get_passages_by_field("tour_name", tour_indices=None, limit=1000)
    elif requested_field and tour_indices:
        # user asked for a specific field AND mentioned a tour -> return that field restricted to the tour(s)
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=tour_indices)
        # If none found for the specific tour(s), fallback to global field
        if not top_results:
            top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
    elif requested_field:
        # user asked for a field but didn't name a tour -> return global matches for that field
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
        if not top_results:
            # fallback to semantic search
            top_results = query_index(user_message, TOP_K)
    else:
        # No keyword matched -> semantic search
        top_k = int(data.get("top_k", TOP_K))
        top_results = query_index(user_message, top_k)

    # Compose system prompt from the top_results and call LLM for nicer phrasing
    system_prompt = compose_system_prompt(top_results)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    reply = ""
    # ✅ SỬ DỤNG OPENAI CHAT API MỚI
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=int(data.get("max_tokens", 700)),
                top_p=0.95
            )
            # ✅ TRÍCH XUẤT DỮ LIỆU MỚI
            if resp.choices and len(resp.choices) > 0:
                reply = resp.choices[0].message.content or ""
        except Exception:
            logger.exception("OpenAI chat failed; will fallback to deterministic reply.")

    # If LLM returned nothing (or not allowed), build deterministic reply favoring requested_field and tour restriction
    if not reply:
        if top_results:
            # If requested_field was tour_name, return a clean deduped list of tour names
            if requested_field == "tour_name":
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                names_u = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            elif requested_field and tour_indices:
                # Provide the requested field values grouped by tour
                parts = []
                for ti in tour_indices:
                    # find tour_name by index
                    tour_name = None
                    for m in MAPPING:
                        p = m.get("path", "")
                        if p.endswith(f"tours[{ti}].tour_name"):
                            tour_name = m.get("text", "")
                            break
                    # collect requested field passages for this tour
                    field_passages = [m.get("text", "") for score, m in top_results if f"[{ti}]" in m.get("path", "")]
                    if not field_passages:
                        # explicit fetch per tour to ensure correctness if top_results were global
                        field_passages = [m.get("text", "") for _, m in get_passages_by_field(requested_field, limit=TOP_K, tour_indices=[ti])]
                    if field_passages:
                        label = f'Tour "{tour_name}"' if tour_name else f"Tour #{ti}"
                        parts.append(label + ":\n" + "\n".join(f"- {t}" for t in field_passages))
                if parts:
                    reply = "\n\n".join(parts)
                else:
                    # fallback to snippet list
                    snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                    reply = f"Tôi tìm thấy:\n\n{snippets}"
            else:
                # No tour restriction or not field-request -> provide top snippets
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."

    return jsonify({"reply": reply, "sources": [m for _, m in top_results]})

# ---------- Knowledge loader ----------
def load_knowledge(path: str = KNOWLEDGE_PATH):
    """Load knowledge.json and flatten into FLAT_TEXTS + MAPPING; then index tour names."""
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
            except Exception:
                pass

    scan(KNOW)
    index_tour_names()
    logger.info("✅ Knowledge loaded: %d passages", len(FLAT_TEXTS))

# ---------- Initialization ----------
# Load knowledge and try to load or build index at import time (safe for Gunicorn workers)
try:
    load_knowledge()
    # try loading existing mapping file into MAPPING if present (ensures mapping order stable)
    if os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            # only update if file_map length matches current flattened passages OR if MAPPING empty
            if file_map and (len(file_map) == len(MAPPING) or len(MAPPING) == 0):
                MAPPING[:] = file_map
                FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                index_tour_names()
                logger.info("Mapping overwritten from disk mapping.json")
        except Exception:
            logger.exception("Could not load FAISS_MAPPING_PATH at startup; proceeding with runtime-scan mapping.")

    # If index exists on disk, build_index will try to load it; otherwise it will build in background
    t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
    t.start()
except Exception:
    logger.exception("Initialization error")





# ==================================================
# LEAD SAVING ROUTE - TEST MODE (NO GOOGLE SHEETS)
# ==================================================
from datetime import datetime
from flask import request, jsonify

@app.route('/api/save-lead', methods=['POST'])
def save_lead():
    """
    Test mode: chỉ log lead, chưa ghi Google Sheets
    KHÔNG ảnh hưởng chatbot / Meta CAPI
    """
    try:
        # 1. Validate Content-Type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400

        # 2. Parse JSON safely
        data = request.get_json(silent=True) or {}

        phone = str(data.get('phone', '')).strip()
        page_url = data.get('page_url', '')
        action_type = data.get('action_type', '')
        source_channel = data.get('source_channel', 'Website')

        # 3. Validate phone (lead hợp lệ)
        if not phone:
            return jsonify({
                'status': 'error',
                'message': 'Phone number is required'
            }), 400

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 4. Log lead (test mode)
        print(
            f"[LEAD TEST] "
            f"time={now} | "
            f"phone={phone} | "
            f"action={action_type} | "
            f"source={source_channel} | "
            f"page={page_url}"
        )

        # 5. Trả JSON RÕ RÀNG cho frontend
        return jsonify({
            'status': 'ok',
            'mode': 'test',
            'phone': phone,
            'timestamp': now
        }), 200

    except Exception as e:
        # 6. Bắt lỗi an toàn – không làm crash app
        app.logger.error(f"[SAVE_LEAD_ERROR] {type(e).__name__}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500


# ==================================================
# TEST LEAD ROUTE - SIÊU ĐƠN GIẢN
# ==================================================

def save_lead():
    print("✅ /api/save-lead được gọi!")
    return jsonify({"success": True, "message": "Test OK"}), 200
# ==================================================
# Flask App Run (KHÔNG SỬA)
# ==================================================
    # ... code hiện tại ...
# When run directly, run flask dev server (note: for production use Gunicorn)
if __name__ == "__main__":
    # ensure mapping saved for reproducibility
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        save_mapping_to_disk()

        
    # block startup building index to ensure readiness
    built = build_index(force_rebuild=False)
    if not built:
        logger.warning("Index not ready at startup; endpoint will attempt on-demand build.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

