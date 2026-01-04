# app.py — RUBY WINGS CHATBOT v2.1
# Enhanced with robust error handling for Google Sheets and file permissions

# === SAFE MODE FOR DEBUG ===
FLAT_TEXTS = []
INDEX = None
HAS_FAISS = False
FAISS_ENABLED = False

def _index_dim(idx):
    return None

# === IMPORTS ===
import os
import json
import threading
import logging
import re
import unicodedata
import traceback
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from google.auth.exceptions import GoogleAuthError
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound

# Meta CAPI
from meta_capi import send_meta_pageview

# Try FAISS
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# OpenAI API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =========== CONFIGURATION ===========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rbw")

# Environment variables with defaults
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

# Embedding and model config
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# Google Sheets config
GOOGLE_SHEET_ID = "1SdVbwkuxb8l1meEW--ddyfh4WmUvSXXMOPQ5bCyPkdk"
GOOGLE_SHEET_NAME = "RBW_Lead_Raw_Inbox"

# Feature flags
ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "true").lower() in ("1", "true", "yes")
ENABLE_FALLBACK_STORAGE = os.environ.get("ENABLE_FALLBACK_STORAGE", "true").lower() in ("1", "true", "yes")
FALLBACK_STORAGE_PATH = os.environ.get("FALLBACK_STORAGE_PATH", "leads_fallback.json")

# =========== GLOBAL STATE ===========
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.warning("OPENAI_API_KEY not set — embeddings/chat will use fallback behavior")

# Knowledge base state
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()
TOUR_NAME_TO_INDEX: Dict[str, int] = {}

# Google Sheets client cache
_gsheet_client = None
_gsheet_client_lock = threading.Lock()

# Fallback storage for leads
_fallback_storage_lock = threading.Lock()

# =========== KEYWORD MAPPING ===========
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

# =========== UTILITY FUNCTIONS ===========
def normalize_text_simple(s: str) -> str:
    """Lowercase, remove diacritics, strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_gspread_client(force_refresh: bool = False):
    """
    Get or create Google Sheets client with thread safety and error handling.
    Returns None if authentication fails.
    """
    global _gsheet_client
    
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        logger.error("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")
        return None
    
    with _gsheet_client_lock:
        if _gsheet_client is not None and not force_refresh:
            return _gsheet_client
        
        try:
            # Parse service account JSON
            try:
                info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
                return None
            
            # Define scopes
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            
            # Create credentials
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            _gsheet_client = gspread.authorize(creds)
            logger.info("Google Sheets client initialized successfully")
            return _gsheet_client
            
        except GoogleAuthError as e:
            logger.error(f"Google authentication error: {e}")
            _gsheet_client = None
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets client: {e}")
            _gsheet_client = None
            return None

def save_lead_to_fallback_storage(lead_data: dict) -> bool:
    """
    Save lead data to local JSON file as fallback when Google Sheets fails.
    """
    if not ENABLE_FALLBACK_STORAGE:
        return False
    
    try:
        lead_data["timestamp"] = datetime.utcnow().isoformat()
        lead_data["synced"] = False
        
        with _fallback_storage_lock:
            # Read existing data
            leads = []
            if os.path.exists(FALLBACK_STORAGE_PATH):
                try:
                    with open(FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                        leads = json.load(f)
                        if not isinstance(leads, list):
                            leads = []
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to read fallback storage: {e}")
                    leads = []
            
            # Append new lead
            leads.append(lead_data)
            
            # Write back
            with open(FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                json.dump(leads, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Lead saved to fallback storage: {FALLBACK_STORAGE_PATH}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to save lead to fallback storage: {e}")
        return False

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
            match = re.search(r"\[(\d+)\]", path)
            if match:
                idx = int(match.group(1))
                prev = TOUR_NAME_TO_INDEX.get(norm)
                if prev is None:
                    TOUR_NAME_TO_INDEX[norm] = idx
                else:
                    if len(txt) > len(MAPPING[next(i for i,m2 in enumerate(MAPPING) if re.search(rf"\[{prev}\]", m2.get('path','')) )].get("text","")):
                        TOUR_NAME_TO_INDEX[norm] = idx

def find_tour_indices_from_message(message: str) -> List[int]:
    """Improved tour detection with fuzzy matching"""
    if not message:
        return []
    
    msg_n = normalize_text_simple(message)
    if not msg_n:
        return []
    
    matches = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        tour_words = set(norm_name.split())
        msg_words = set(msg_n.split())
        common_words = tour_words & msg_words
        if len(common_words) >= 1:
            matches.append((len(common_words), norm_name))
    
    if matches:
        matches.sort(reverse=True)
        best_score = matches[0][0]
        selected = [TOUR_NAME_TO_INDEX[nm] for sc, nm in matches if sc == best_score]
        return sorted(set(selected))
    
    return []

# =========== MAPPING HELPERS ===========
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
        if path.endswith(f".{field_name}") or f".{field_name}" in path:
            is_exact_match = False
            if tour_indices:
                for ti in tour_indices:
                    if f"[{ti}]" in path:
                        is_exact_match = True
                        break
            
            if is_exact_match:
                exact_matches.append((2.0, m))
            elif not tour_indices:
                global_matches.append((1.0, m))
    
    all_results = exact_matches + global_matches
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:limit]

# =========== EMBEDDINGS ===========
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Return (embedding list, dim)
    Tries openai.Embedding.create. If API key missing or call fails, return deterministic fallback 1536-dim.
    """
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    
    if client is not None:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL, 
                input=short
            )
            if resp.data and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding call failed — falling back to deterministic embedding.")
    
    # Deterministic fallback
    try:
        h = abs(hash(short)) % (10 ** 12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# =========== INDEX MANAGEMENT ===========
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
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

class NumpyIndex:
    """Simple in-memory numpy index with cosine-similarity."""
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
            logger.info(f"Saved numpy index to {path}")
        except Exception as e:
            logger.error(f"Failed to save numpy index: {e}")

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            mat = arr["mat"]
            logger.info(f"Loaded numpy index from {path}")
            return cls(mat=mat)
        except Exception as e:
            logger.error(f"Failed to load numpy index: {e}")
            return cls(None)

def load_mapping_from_disk(path=FAISS_MAPPING_PATH):
    global MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING[:] = json.load(f)
        FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
        logger.info("Loaded mapping from %s (%d entries)", path, len(MAPPING))
        return True
    except Exception as e:
        logger.error(f"Failed to load mapping from disk: {e}")
        return False

def save_mapping_to_disk(path=FAISS_MAPPING_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        logger.info("Saved mapping to %s", path)
    except Exception as e:
        logger.error(f"Failed to save mapping: {e}")

def build_index(force_rebuild: bool = False) -> bool:
    """
    Build or load index. If FAISS enabled and available, use it; otherwise NumpyIndex.
    """
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

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
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
            
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
                except Exception as e:
                    logger.error(f"Failed to load fallback vectors: {e}")

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
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (row_norms + 1e-12)

            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                try:
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    save_mapping_to_disk()
                except Exception as e:
                    logger.error(f"Failed to persist FAISS index: {e}")
                index_tour_names()
                logger.info("✅ FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
                return True
            else:
                idx = NumpyIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    save_mapping_to_disk()
                except Exception as e:
                    logger.error(f"Failed to persist fallback vectors: {e}")
                index_tour_names()
                logger.info("✅ Numpy fallback index built (dims=%d, n=%d).", dims, idx.ntotal)
                return True
        except Exception as e:
            logger.error(f"Error while building index: {e}")
            INDEX = None
            return False

# =========== QUERY INDEX ===========
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
    except Exception as e:
        logger.error(f"Error executing index.search: {e}")
        return []

    results: List[Tuple[float, dict]] = []
    try:
        scores = D[0].tolist() if getattr(D, "shape", None) else []
        idxs = I[0].tolist() if getattr(I, "shape", None) else []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(MAPPING):
                continue
            results.append((float(score), MAPPING[idx]))
    except Exception as e:
        logger.error(f"Failed to parse search results: {e}")
    return results

# =========== PROMPT COMPOSITION ===========
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

# =========== KNOWLEDGE LOADER ===========
def load_knowledge(path: str = KNOWLEDGE_PATH):
    """Load knowledge.json and flatten into FLAT_TEXTS + MAPPING; then index tour names."""
    global KNOW, FLAT_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
        logger.info(f"Successfully loaded knowledge from {path}")
    except Exception as e:
        logger.error(f"Could not open {path}: {e}")
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

# =========== META CAPI ===========
@app.before_request
def track_meta_pageview():
    try:
        send_meta_pageview(request)
    except Exception as e:
        logger.error(f"Meta CAPI tracking failed: {e}")

# =========== ROUTES ===========
@app.route("/")
def home():
    try:
        return jsonify({
            "status": "ok",
            "knowledge_count": len(FLAT_TEXTS) if FLAT_TEXTS is not None else 0,
            "index_exists": INDEX is not None,
            "index_dim": _index_dim(INDEX) if INDEX is not None else None,
            "embedding_model": EMBEDDING_MODEL,
            "faiss_available": HAS_FAISS,
            "faiss_enabled": FAISS_ENABLED,
            "google_sheets_enabled": ENABLE_GOOGLE_SHEETS,
            "fallback_storage_enabled": ENABLE_FALLBACK_STORAGE,
            "service_status": "operational"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
    load_knowledge()
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
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in text_l:
                requested_field = v["field"]
                break
        if requested_field:
            break

    tour_indices = find_tour_indices_from_message(user_message)
    top_results: List[Tuple[float, dict]] = []

    if requested_field == "tour_name":
        top_results = get_passages_by_field("tour_name", tour_indices=None, limit=1000)
    elif requested_field and tour_indices:
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=tour_indices)
        if not top_results:
            top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
    elif requested_field:
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
        if not top_results:
            top_results = query_index(user_message, TOP_K)
    else:
        top_k = int(data.get("top_k", TOP_K))
        top_results = query_index(user_message, top_k)

    system_prompt = compose_system_prompt(top_results)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    reply = ""
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=int(data.get("max_tokens", 700)),
                top_p=0.95
            )
            if resp.choices and len(resp.choices) > 0:
                reply = resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")

    if not reply:
        if top_results:
            if requested_field == "tour_name":
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                names_u = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            elif requested_field and tour_indices:
                parts = []
                for ti in tour_indices:
                    tour_name = None
                    for m in MAPPING:
                        p = m.get("path", "")
                        if p.endswith(f"tours[{ti}].tour_name"):
                            tour_name = m.get("text", "")
                            break
                    field_passages = [m.get("text", "") for score, m in top_results if f"[{ti}]" in m.get("path", "")]
                    if not field_passages:
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
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."

    return jsonify({"reply": reply, "sources": [m for _, m in top_results]})

# =========== LEAD SAVING ROUTE ===========
@app.route('/api/save-lead', methods=['POST'])
def save_lead_to_sheet():
    """
    Save lead to Google Sheets with robust error handling and fallback storage.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False
            }), 400

        data = request.get_json() or {}
        
        # Extract and validate required fields
        phone = (data.get("phone") or "").strip()
        if not phone:
            return jsonify({
                "error": "Phone number is required",
                "success": False
            }), 400

        # Prepare lead data
        lead_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "source_channel": data.get("source_channel", "Website"),
            "action_type": data.get("action_type", "Click Call"),
            "page_url": data.get("page_url", ""),
            "contact_name": data.get("contact_name", ""),
            "phone": phone,
            "service_interest": data.get("service_interest", ""),
            "note": data.get("note", ""),
            "status": "New",
            "sync_method": "unknown"
        }
        
        logger.info(f"Processing lead: {phone}, source: {lead_data['source_channel']}")

        # Try Google Sheets first (if enabled)
        sheets_success = False
        if ENABLE_GOOGLE_SHEETS:
            try:
                gc = get_gspread_client()
                if gc is None:
                    logger.warning("Google Sheets client not available, trying fallback")
                else:
                    logger.info(f"Attempting to save to Google Sheet: {GOOGLE_SHEET_ID}")
                    
                    # Open spreadsheet
                    sh = gc.open_by_key(GOOGLE_SHEET_ID)
                    logger.info(f"Opened spreadsheet: {GOOGLE_SHEET_ID}")
                    
                    # Get worksheet
                    ws = sh.worksheet(GOOGLE_SHEET_NAME)
                    logger.info(f"Accessed worksheet: {GOOGLE_SHEET_NAME}")
                    
                    # Prepare row data
                    row = [
                        lead_data["timestamp"],
                        lead_data["source_channel"],
                        lead_data["action_type"],
                        lead_data["page_url"],
                        lead_data["contact_name"],
                        lead_data["phone"],
                        lead_data["service_interest"],
                        lead_data["note"],
                        lead_data["status"]
                    ]
                    
                    # Append row
                    ws.append_row(row, value_input_option="USER_ENTERED")
                    lead_data["sync_method"] = "google_sheets"
                    sheets_success = True
                    
                    logger.info(f"✅ Lead successfully saved to Google Sheets: {phone}")
                    
            except SpreadsheetNotFound:
                logger.error(f"Google Sheet not found: {GOOGLE_SHEET_ID}")
                lead_data["error"] = "Google Sheet not found"
            except WorksheetNotFound:
                logger.error(f"Worksheet not found: {GOOGLE_SHEET_NAME}")
                lead_data["error"] = f"Worksheet '{GOOGLE_SHEET_NAME}' not found"
            except APIError as e:
                error_msg = str(e)
                logger.error(f"Google Sheets API error: {error_msg}")
                lead_data["error"] = f"Google Sheets API error: {error_msg}"
                
                # Check for permission errors
                if "PERMISSION_DENIED" in error_msg or "forbidden" in error_msg.lower():
                    logger.error("Permission denied to access Google Sheet. Check sharing settings.")
            except Exception as e:
                logger.error(f"Unexpected Google Sheets error: {type(e).__name__}: {str(e)}")
                lead_data["error"] = f"Google Sheets error: {type(e).__name__}"
        else:
            logger.info("Google Sheets integration is disabled")

        # Save to fallback storage if Google Sheets failed or for redundancy
        fallback_success = False
        if not sheets_success or ENABLE_FALLBACK_STORAGE:
            fallback_success = save_lead_to_fallback_storage(lead_data)
            if fallback_success:
                logger.info(f"Lead saved to fallback storage: {phone}")
                lead_data["sync_method"] = "fallback_storage"

        # Determine response
        if sheets_success:
            return jsonify({
                "success": True,
                "message": "Lead saved successfully to Google Sheets",
                "data": {
                    "phone": phone,
                    "timestamp": lead_data["timestamp"],
                    "sync_method": lead_data["sync_method"]
                }
            }), 200
        elif fallback_success:
            return jsonify({
                "success": True,
                "message": "Lead saved to fallback storage (Google Sheets unavailable)",
                "warning": "Google Sheets synchronization failed, data saved locally",
                "data": {
                    "phone": phone,
                    "timestamp": lead_data["timestamp"],
                    "sync_method": lead_data["sync_method"]
                }
            }), 200
        else:
            logger.error(f"Failed to save lead by any method: {phone}")
            return jsonify({
                "success": False,
                "error": "Failed to save lead. Both Google Sheets and fallback storage failed.",
                "details": lead_data.get("error", "Unknown error")
            }), 500

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"SAVE_LEAD_CRITICAL_ERROR >>> Type: {error_type}")
        logger.error(f"SAVE_LEAD_CRITICAL_ERROR >>> Details: {error_details}")
        logger.error(f"SAVE_LEAD_CRITICAL_ERROR >>> Traceback: {error_traceback}")
        
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "error_type": error_type,
            "details": "Please check server logs for details"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check Google Sheets connectivity
        sheets_status = "disabled"
        if ENABLE_GOOGLE_SHEETS:
            try:
                gc = get_gspread_client()
                if gc:
                    # Quick test - try to get spreadsheet metadata
                    gc.open_by_key(GOOGLE_SHEET_ID)
                    sheets_status = "connected"
                else:
                    sheets_status = "client_error"
            except Exception as e:
                sheets_status = f"error: {type(e).__name__}"
        
        # Check fallback storage
        fallback_status = "disabled"
        if ENABLE_FALLBACK_STORAGE:
            try:
                if os.path.exists(FALLBACK_STORAGE_PATH):
                    with open(FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                        json.load(f)
                    fallback_status = "available"
                else:
                    fallback_status = "not_created"
            except Exception as e:
                fallback_status = f"error: {type(e).__name__}"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "google_sheets": sheets_status,
                "fallback_storage": fallback_status,
                "openai": "available" if client else "unavailable",
                "faiss": "available" if HAS_FAISS else "unavailable",
                "index": "loaded" if INDEX else "not_loaded"
            },
            "counts": {
                "knowledge_passages": len(FLAT_TEXTS),
                "mapping_entries": len(MAPPING),
                "tour_names": len(TOUR_NAME_TO_INDEX)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# =========== INITIALIZATION ===========
def initialize_application():
    """Initialize the application with proper error handling"""
    try:
        logger.info("Starting Ruby Wings Chatbot initialization...")
        
        # Load knowledge base
        load_knowledge()
        
        # Load existing mapping if available
        if os.path.exists(FAISS_MAPPING_PATH):
            try:
                with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                    file_map = json.load(f)
                if file_map and (len(file_map) == len(MAPPING) or len(MAPPING) == 0):
                    MAPPING[:] = file_map
                    FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    index_tour_names()
                    logger.info("Mapping loaded from disk")
            except Exception as e:
                logger.warning(f"Could not load mapping from disk: {e}")
        
        # Initialize Google Sheets client in background
        if ENABLE_GOOGLE_SHEETS and GOOGLE_SERVICE_ACCOUNT_JSON:
            def init_gsheets():
                try:
                    client = get_gspread_client()
                    if client:
                        logger.info("Google Sheets client initialized successfully")
                    else:
                        logger.warning("Google Sheets client initialization failed")
                except Exception as e:
                    logger.error(f"Background Google Sheets init failed: {e}")
            
            gsheet_thread = threading.Thread(target=init_gsheets, daemon=True)
            gsheet_thread.start()
        
        # Build index in background
        def build_index_background():
            try:
                built = build_index(force_rebuild=False)
                if built:
                    logger.info("Index built successfully")
                else:
                    logger.warning("Index building failed or deferred")
            except Exception as e:
                logger.error(f"Background index build failed: {e}")
        
        index_thread = threading.Thread(target=build_index_background, daemon=True)
        index_thread.start()
        
        logger.info("✅ Application initialization completed")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise

# =========== APPLICATION STARTUP ===========
if __name__ == "__main__":
    # Run initialization
    initialize_application()
    
    # Ensure mapping is saved if not exists
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        try:
            save_mapping_to_disk()
        except Exception as e:
            logger.error(f"Failed to save initial mapping: {e}")
    
    # Start Flask server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
else:
    # For Gunicorn/WSGI
    initialize_application()