# app.py — "HOÀN HẢO NHẤT" (comprehensive optimized)
# Features:
# - session current_tour (in-memory or Redis)
# - fuzzy NER for tour_name (normalize + token_jaccard + levenshtein + substring)
# - two-stage retrieval: tour-restricted deterministic selection first, then semantic search
# - two-tier reranking: semantic top-K -> rerank by field/tour token overlap & exact match
# - deterministic fallback replies with clear messages
# - persistent mapping expectation: mapping entries include 'path','text','field','tour_index'
# - logging / trace for debug & audit (session, detected_tour, requested_field, chosen_sources, confidence)
# - FAISS if available, fallback numpy index
# - robust embedding with OpenAI SDK (new) and deterministic fallback
# - reindex endpoint with simple admin protection
# - Designed to work with build_index.py that produces stable mapping order
#
# To run:
#   pip install flask flask-cors numpy faiss-cpu openai  (faiss optional)
#   export OPENAI_API_KEY="sk-..."
#   python app.py
#
# Environment variables:
#   KNOWLEDGE_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH, FALLBACK_VECTORS_PATH
#   EMBEDDING_MODEL, CHAT_MODEL, TOP_K, FAISS_ENABLED (true/false)
#   SESSION_TIMEOUT (seconds), RBW_ALLOW_REINDEX, X-RBW-ADMIN header to allow reindex
#   REDIS_URL (optional) - if provided will switch to Redis session store (not implemented here but placeholder)
# ------------------------------------------------------------

import os
import json
import threading
import logging
import re
import unicodedata
import uuid
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np

# Optional FAISS
HAS_FAISS = False
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Optional new OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw_app")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
        logger.exception("Failed to init OpenAI client")

# Paths & models
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", str(60 * 5)))  # default 5 minutes
RBW_ALLOW_REINDEX = os.environ.get("RBW_ALLOW_REINDEX", "") == "1"

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []  # expected list of {"path": "...", "text": "...", "field": "...", "tour_index": <int>}
INDEX = None
INDEX_LOCK = threading.Lock()
TOUR_NAME_TO_INDEX: Dict[str, int] = {}  # normalized tour_name -> index

# Session store
USER_SESSIONS: Dict[str, dict] = {}  # simple in-memory; replace with Redis in prod

# Keyword -> field mapping (expandable)
KEYWORD_FIELD_MAP: Dict[str, Dict] = {
    "tour_list": {"keywords": ["tên tour", "tour gì", "danh sách tour", "liệt kê tour", "list tour", "tours"], "field": "tour_name"},
    "mission": {"keywords": ["tầm nhìn", "sứ mệnh", "mission"], "field": "mission"},
    "summary": {"keywords": ["tóm tắt chương trình", "tóm tắt", "overview", "mô tả", "summary"], "field": "summary"},
    "style": {"keywords": ["phong cách hành trình", "style"], "field": "style"},
    "transport": {"keywords": ["vận chuyển", "phương tiện", "di chuyển", "xe", "transport"], "field": "transport"},
    "includes": {"keywords": ["lịch trình chi tiết", "chương trình chi tiết", "includes", "itinerary"], "field": "includes"},
    "location": {"keywords": ["ở đâu", "đi đâu", "địa điểm", "location", "điểm đến"], "field": "location"},
    "duration": {"keywords": ["thời gian tour", "bao lâu", "mấy ngày", "duration"], "field": "duration"},
    "price": {"keywords": ["giá", "giá tour", "chi phí", "bao nhiêu tiền", "giá vé"], "field": "price"},
    "notes": {"keywords": ["lưu ý", "ghi chú", "notes"], "field": "notes"},
    "accommodation": {"keywords": ["chỗ ở", "khách sạn", "lưu trú", "accommodation"], "field": "accommodation"},
    "meals": {"keywords": ["ăn", "ăn uống", "bữa", "thực đơn", "meals"], "field": "meals"},
    "hotline": {"keywords": ["hotline", "sđt", "số điện thoại", "liên hệ", "contact"], "field": "hotline"},
    "event_support": {"keywords": ["hỗ trợ sự kiện", "hỗ trợ đoàn", "support", "event_support"], "field": "event_support"},
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

def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (iterative DP)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            insert = cur[j-1] + 1
            delete = prev[j] + 1
            replace = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(insert, delete, replace)
        prev = cur
    return prev[-1]

def token_jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    uni = sa | sb
    return len(inter) / len(uni)

# ---------- Index-tour-name helpers ----------
def index_tour_names():
    """Populate TOUR_NAME_TO_INDEX from MAPPING entries that end with .tour_name or field == 'tour_name'."""
    global TOUR_NAME_TO_INDEX
    TOUR_NAME_TO_INDEX = {}
    for m in MAPPING:
        # support both explicit 'field' and path.endswith
        field = m.get("field")
        path = m.get("path", "")
        if field == "tour_name" or path.endswith(".tour_name"):
            txt = (m.get("text") or "").strip()
            norm = normalize_text_simple(txt)
            if not norm:
                continue
            # extract index from path if present
            match = re.search(r"\[(\d+)\]", path)
            if match:
                idx = int(match.group(1))
                # if duplicate normalized name, keep first (mapping stable)
                if norm not in TOUR_NAME_TO_INDEX:
                    TOUR_NAME_TO_INDEX[norm] = idx

def fuzzy_find_tours(message: str, top_n: int = 3) -> List[Tuple[int, float]]:
    """Return list of (tour_index, score) sorted desc. Combine token overlap + levenshtein + substring."""
    if not message:
        return []
    msg_n = normalize_text_simple(message)
    results: List[Tuple[int, float]] = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        score = 0.0
        # token overlap (strong)
        score += 2.0 * token_jaccard(msg_n, norm_name)
        # substring
        if norm_name in msg_n or msg_n in norm_name:
            score += 1.5
        # normalized edit distance ratio
        ld = levenshtein(msg_n, norm_name)
        maxlen = max(1, len(norm_name))
        score += max(0.0, 1.0 - (ld / maxlen)) * 1.0
        # boost if any exact token present
        for w in norm_name.split():
            if w in msg_n.split():
                score += 0.3
        if score > 0:
            results.append((idx, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# ---------- Mapping helpers ----------
def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    """
    Return passages whose 'field' == field_name OR path contains .field_name
    If tour_indices provided, restrict and prioritize entries matching those tour index brackets.
    Score: 2.0 exact tour match, 1.0 global match
    """
    exact_matches: List[Tuple[float, dict]] = []
    global_matches: List[Tuple[float, dict]] = []
    for m in MAPPING:
        path = m.get("path", "")
        field = m.get("field") or ""
        # match by explicit field first or path
        if field == field_name or path.endswith(f".{field_name}") or f".{field_name}" in path:
            is_exact = False
            if tour_indices:
                for ti in tour_indices:
                    if f"[{ti}]" in path or m.get("tour_index") == ti:
                        is_exact = True
                        break
            if is_exact:
                exact_matches.append((2.0, m))
            elif not tour_indices:
                global_matches.append((1.0, m))
    all_results = exact_matches + global_matches
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:limit]

# ---------- Embeddings (robust) ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Return (embedding list, dim)
    Use OpenAI if available; deterministic fallback otherwise.
    """
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    if client is not None:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=short)
            if getattr(resp, "data", None) and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding call failed — falling back to deterministic embedding.")
    # deterministic fallback (stable)
    try:
        h = abs(hash(short)) % (10 ** 12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# ---------- Simple Numpy Index ----------
class NumpyIndex:
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

# ---------- Index management ----------
def _index_dim(idx) -> Optional[int]:
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
    # heuristics: 1536 -> small, 3072 -> large
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

def load_mapping_from_disk(path=FAISS_MAPPING_PATH):
    global MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            file_map = json.load(f)
        # validate mapping format: ensure list of dicts with path/text
        if isinstance(file_map, list):
            MAPPING[:] = file_map
            FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
            logger.info("Loaded mapping from %s (%d entries)", path, len(MAPPING))
            return True
        else:
            logger.error("Mapping file invalid format")
            return False
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
    Build or load index. Prefer FAISS if available+enabled, else NumpyIndex.
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

# ---------- Query index (semantic) ----------
def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    """
    Semantic search: returns list of (score, mapping_entry)
    """
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
        # Attempt to rebuild with matching model/dim if we have OpenAI key
        logger.warning("Query dim %s != index dim %s; attempt model-aligned rebuild.", vec.shape[1], idx_dim)
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY and desired_model != EMBEDDING_MODEL and client is not None:
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
            logger.error("No OPENAI_API_KEY or unable to rebuild index to match query dimension.")
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

# ---------- Two-tier reranking (semantic -> lexical/tour-field) ----------
def rerank_with_field_priority(semantic_hits: List[Tuple[float, dict]], requested_field: Optional[str], tour_indices: Optional[List[int]], top_n: int = 5) -> List[Tuple[float, dict]]:
    """
    Rerank semantic hits by boosting:
      - exact tour match (if tour_indices provided)
      - exact field match (mapping.field == requested_field)
      - token overlap between query and passage.text
    Returns sorted list (score, mapping) limited to top_n
    """
    enhanced: List[Tuple[float, dict]] = []
    for sem_score, m in semantic_hits:
        score = sem_score
        # boost for field match
        field = m.get("field", "")
        if requested_field and field == requested_field:
            score += 1.0
        # boost for tour match
        if tour_indices:
            ti = m.get("tour_index")
            if ti is not None and ti in tour_indices:
                score += 1.2
        # lexical overlap (token_jaccard)
        q = request.json.get("message", "") if request and request.json else ""
        qnorm = normalize_text_simple(q)
        txtnorm = normalize_text_simple(m.get("text", "") or "")
        score += 2.0 * token_jaccard(qnorm, txtnorm)
        enhanced.append((score, m))
    enhanced.sort(key=lambda x: x[0], reverse=True)
    return enhanced[:top_n]

# ---------- Session Management ----------
def get_or_create_session():
    """Obtain or create a session for the incoming request."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in USER_SESSIONS:
        session_id = str(uuid.uuid4())
        USER_SESSIONS[session_id] = {
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "last_tour_index": None,
            "last_tour_name": None,
            "last_field": None,
            "conversation_count": 0
        }
    # Update last activity
    USER_SESSIONS[session_id]["last_activity"] = datetime.utcnow()
    # Cleanup expired sessions opportunistically
    cleanup_expired_sessions()
    return session_id, USER_SESSIONS[session_id]

def cleanup_expired_sessions():
    """Remove expired sessions based on SESSION_TIMEOUT."""
    expired = []
    now = datetime.utcnow()
    for sid, sdata in list(USER_SESSIONS.items()):
        if now - sdata.get("last_activity", now) > timedelta(seconds=SESSION_TIMEOUT):
            expired.append(sid)
    for sid in expired:
        USER_SESSIONS.pop(sid, None)
        logger.debug("Expired session removed: %s", sid)

def update_session_context(session_data: dict, tour_indices: List[int], requested_field: Optional[str], user_message: str):
    """
    Update session context:
     - If new tour_indices detected -> update last_tour_index/name
     - If requested_field detected -> update last_field
     - Track conversation_count; reset if general/unrelated long conversation
    """
    if tour_indices:
        session_data["last_tour_index"] = tour_indices[0]
        # try to find tour name from mapping
        tour_name = None
        for m in MAPPING:
            if m.get("tour_index") == session_data["last_tour_index"] and (m.get("field") == "tour_name" or m.get("path", "").endswith(".tour_name")):
                tour_name = m.get("text")
                break
        session_data["last_tour_name"] = tour_name
        session_data["conversation_count"] = 1
    elif session_data.get("last_tour_index") is not None:
        session_data["conversation_count"] = session_data.get("conversation_count", 0) + 1

    if requested_field:
        session_data["last_field"] = requested_field

    # If user sends very generic questions repeatedly, clear context after threshold
    if session_data.get("conversation_count", 0) > 10 and is_general_question(user_message):
        session_data["last_tour_index"] = None
        session_data["last_tour_name"] = None
        session_data["last_field"] = None
        session_data["conversation_count"] = 0
        logger.debug("Session context reset due general questions (session cleared).")

def is_general_question(message: str) -> bool:
    """Heuristic to detect generic unrelated questions."""
    general_keywords = ["ai", "là gì", "cái gì", "ở đâu", "công ty", "ruby wings", "bạn là ai", "giới thiệu", "thông tin chung"]
    ml = message.lower()
    return any(k in ml for k in general_keywords)

# ---------- Prompt composition (for LLM when available) ----------
def compose_system_prompt(top_passages: List[Tuple[float, dict]], context_tour: Optional[str] = None) -> str:
    header = "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.\n\n"
    if context_tour:
        header += f"NGỮ CẢNH HIỆN TẠI: User đang hỏi về tour '{context_tour}'.\nƯU TIÊN TRẢ LỜI THEO TOUR NÀY.\n\n"
    header += (
        "TRẢ LỜI THEO NGUYÊN TẮC:\n"
        "1) ƯU TIÊN: Thông tin từ dữ liệu nội bộ được liệt kê bên dưới (theo thứ tự liên quan).\n"
        "2) Nếu thiếu chi tiết, trả lời ngắn gọn & yêu cầu bổ sung (nếu cần) — nhưng theo yêu cầu bạn không hỏi thêm.\n"
        "3) KHÔNG bịa thông tin.\n\n"
    )
    if not top_passages:
        header += "Không tìm thấy dữ liệu nội bộ phù hợp.\n"
        return header

    content = header + "DỮ LIỆU NỘI BỘ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')} (field={m.get('field','?')})\n{m.get('text','')}\n"
    content += "\n---\nHÃY TRẢ LỜI NGẮN GỌN, CHÍNH XÁC, VÀ QUY CHẾ: chỉ dùng dữ liệu trên."
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
        "faiss_enabled": FAISS_ENABLED,
        "active_sessions": len(USER_SESSIONS)
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    # simple protection: header or env allow
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and not RBW_ALLOW_REINDEX:
        return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN header)"}), 403
    # reload knowledge and rebuild index
    load_knowledge()  # reload raw knowledge before building
    ok = build_index(force_rebuild=True)
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint:
    - Detect requested_field by keyword mapping (keyword detection)
    - Detect tour mentions with fuzzy NER
    - Update per-session context
    - Two-stage retrieval: if tour known/or mentioned -> deterministic retrieval from that tour's fields
      else -> semantic search
    - Two-tier reranking for semantic results
    - Deterministic fallback replies if no reliable data
    """
    session_id, session_data = get_or_create_session()

    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})

    text_lower = user_message.lower()

    # 1) Keyword detection (field routing) - preserve map order priority
    requested_field: Optional[str] = None
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in text_lower:
                requested_field = v["field"]
                break
        if requested_field:
            break

    # 2) Tour detection via fuzzy NER
    detected_tours = []
    # quick exact normalized name check first
    msg_norm = normalize_text_simple(user_message)
    if msg_norm in TOUR_NAME_TO_INDEX:
        detected_tours = [TOUR_NAME_TO_INDEX[msg_norm]]
    else:
        fuzzy = fuzzy_find_tours(user_message, top_n=3)
        if fuzzy:
            # determine confidence threshold heuristics
            # if top score >> others or above threshold, pick them
            best_score = fuzzy[0][1] if fuzzy else 0
            # pick all with score >= 0.5 * best_score and > 0.2
            detected_tours = [idx for idx, sc in fuzzy if sc >= max(0.25, 0.5 * best_score)]

    # 3) Update session context
    update_session_context(session_data, detected_tours, requested_field, user_message)

    # 4) If no detected tours but session has last_tour_index -> use it
    tour_indices = detected_tours or ([session_data["last_tour_index"]] if session_data.get("last_tour_index") is not None else [])

    # 5) Two-stage deterministic retrieval:
    top_results: List[Tuple[float, dict]] = []
    used_deterministic = False

    if requested_field == "tour_name":
        # Always return list of tour names (global)
        top_results = get_passages_by_field("tour_name", limit=1000, tour_indices=None)
        used_deterministic = True
    elif requested_field and tour_indices:
        # Field requested and tour referenced -> prioritize deterministic tour-specific field
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=tour_indices)
        if top_results:
            used_deterministic = True
        else:
            # fallback to global field entries (deterministic)
            top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
            if top_results:
                used_deterministic = True
    elif requested_field and not tour_indices:
        # user asked field but no tour -> return global matches for that field deterministically
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
        if top_results:
            used_deterministic = True
        else:
            # fallback to semantic search
            top_results = query_index(user_message, TOP_K)
    else:
        # No field detected -> if tour_indices present, attempt to return key fields from that tour deterministically
        if tour_indices:
            # attempt to return the most likely fields: summary, price, duration, location
            candidates = []
            for f in ["summary", "price", "duration", "location", "includes", "notes"]:
                candidates.extend(get_passages_by_field(f, limit=1, tour_indices=tour_indices))
            if candidates:
                # dedupe by path and keep ordering
                seen_paths = set()
                dedup = []
                for score, m in candidates:
                    p = m.get("path")
                    if p not in seen_paths:
                        dedup.append((score, m))
                        seen_paths.add(p)
                top_results = dedup[:TOP_K]
                used_deterministic = True
            else:
                # fallback to semantic search
                top_results = query_index(user_message, TOP_K)
        else:
            # pure semantic search
            top_results = query_index(user_message, TOP_K)

    # 6) If semantic results returned (or in addition), perform two-tier reranking if needed
    if top_results and not used_deterministic:
        # prefer reranking to enforce field/tour boundaries
        top_results = rerank_with_field_priority(top_results, requested_field, tour_indices if tour_indices else None, top_n=TOP_K)

    # 7) Compose system prompt for LLM (if available) using top_results & session context
    system_prompt = compose_system_prompt(top_results, session_data.get("last_tour_name"))
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    reply = ""
    llm_used = False
    if client is not None:
        try:
            # Using new OpenAI SDK chat completions interface
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=int(data.get("max_tokens", 700)) if data.get("max_tokens") else 400,
                top_p=0.95
            )
            if getattr(resp, "choices", None) and len(resp.choices) > 0:
                choice = resp.choices[0]
                # new SDK structure: choice.message.content
                reply = getattr(choice, "message", {}).get("content", "") or getattr(choice, "text", "") or ""
                llm_used = bool(reply)
        except Exception:
            logger.exception("OpenAI chat failed; will fallback to deterministic reply.")

    # 8) If LLM not available or returned nothing -> deterministic reply building
    if not reply:
        if top_results:
            # If requested_field == tour_name: return clean list of tour names from mapping (dedup)
            if requested_field == "tour_name":
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                names_u = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            elif requested_field and tour_indices:
                # Provide requested field values grouped by tour
                parts = []
                for ti in tour_indices:
                    # find tour_name by index
                    tour_name = None
                    for m in MAPPING:
                        if m.get("tour_index") == ti and (m.get("field") == "tour_name" or m.get("path","").endswith(".tour_name")):
                            tour_name = m.get("text")
                            break
                    # collect requested field passages for this tour
                    field_passages = [m.get("text", "") for score, m in top_results if m.get("tour_index") == ti]
                    if not field_passages:
                        # explicit fetch per tour to ensure correctness if top_results were global
                        field_passages = [m.get("text", "") for _, m in get_passages_by_field(requested_field, limit=TOP_K, tour_indices=[ti])]
                    if field_passages:
                        label = f'Tour "{tour_name}"' if tour_name else f"Tour #{ti}"
                        parts.append(label + ":\n" + "\n".join(f"- {t}" for t in field_passages))
                if parts:
                    reply = "\n\n".join(parts)
                else:
                    snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                    reply = f"Tôi tìm thấy:\n\n{snippets}"
            else:
                # No tour restriction or not field-request -> provide top snippets
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            # No relevant data found: deterministic fallback (do not ask follow-up per user's prior instruction)
            if session_data.get("last_tour_name"):
                reply = "Hiện chưa tìm thấy thông tin cho tour hiện hành. Vui lòng nêu rõ tên tour nếu bạn muốn tôi kiểm tra cụ thể."
            else:
                reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."

    # 9) Logging trace
    trace = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "user_message": user_message,
        "requested_field": requested_field,
        "detected_tours": tour_indices,
        "used_deterministic": used_deterministic,
        "llm_used": llm_used,
        "results_count": len(top_results),
    }
    logger.info("QUERY TRACE: %s", json.dumps(trace, ensure_ascii=False))

    # 10) Build response
    resp_body = {
        "reply": reply,
        "sources": [m for _, m in top_results],
        "context_tour": session_data.get("last_tour_name"),
        "session_active": session_data.get("last_tour_name") is not None,
        "trace": trace  # include for debugging (could be removed in production)
    }
    response = make_response(jsonify(resp_body))
    response.set_cookie("session_id", session_id, max_age=SESSION_TIMEOUT, httponly=True)
    return response

# ---------- Knowledge loader ----------
def load_knowledge(path: str = KNOWLEDGE_PATH):
    """
    Load knowledge.json and flatten into FLAT_TEXTS + MAPPING; expects knowledge structured as:
      root.{about_company,...}.tours -> list of tours with fields
    Each mapping entry produced should have keys: path, text, field, tour_index
    """
    global KNOW, FLAT_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge.json; continuing with empty knowledge.")
        KNOW = {}
    FLAT_TEXTS = []
    MAPPING = []

    # Prefer structured flatten that maps known tour fields
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
                # infer field from prefix last token
                field = prefix.split(".")[-1]
                tour_index = None
                m = re.search(r"\[(\d+)\]", prefix)
                if m:
                    try:
                        tour_index = int(m.group(1))
                    except Exception:
                        tour_index = None
                entry = {"path": prefix, "text": t, "field": field, "tour_index": tour_index}
                FLAT_TEXTS.append(t)
                MAPPING.append(entry)
        else:
            try:
                s = str(obj).strip()
                if s:
                    field = prefix.split(".")[-1]
                    tour_index = None
                    m = re.search(r"\[(\d+)\]", prefix)
                    if m:
                        try:
                            tour_index = int(m.group(1))
                        except Exception:
                            tour_index = None
                    entry = {"path": prefix, "text": s, "field": field, "tour_index": tour_index}
                    FLAT_TEXTS.append(s)
                    MAPPING.append(entry)
            except Exception:
                pass

    scan(KNOW)
    # If there exists a mapping file produced by build_index.py, prefer its ordering (stable)
    if os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            # Replace only if lengths match or MAPPING empty
            if file_map and (len(file_map) == len(MAPPING) or len(MAPPING) == 0):
                MAPPING[:] = file_map
                FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                logger.info("Mapping overwritten from disk mapping.json")
        except Exception:
            logger.exception("Could not load FAISS_MAPPING_PATH at startup; proceeding with runtime-scan mapping.")
    index_tour_names()
    logger.info("✅ Knowledge loaded: %d passages", len(FLAT_TEXTS))

# ---------- Initialization ----------
try:
    load_knowledge()
    # build index in background thread (non-blocking for startup)
    t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
    t.start()
except Exception:
    logger.exception("Initialization error")

# ---------- Run server (dev) ----------
if __name__ == "__main__":
    # Ensure mapping persisted for reproducibility
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        save_mapping_to_disk()
    built = build_index(force_rebuild=False)
    if not built:
        logger.warning("Index not ready at startup; endpoint will attempt on-demand build.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
