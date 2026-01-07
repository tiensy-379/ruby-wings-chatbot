# app.py — RUBY WINGS CHATBOT v2.1.1
# Fixed critical UnboundLocalError in chat endpoint
# Enhanced with robust error handling and context-aware tour detection

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
import hashlib
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
from meta_capi import send_meta_lead

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
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox")

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
TOURS_DB: Dict[int, Dict[str, Any]] = {}  # Structured database: {tour_index: {field: value}}
TOUR_TAGS: Dict[int, List[str]] = {}  # Auto-generated tags for each tour

# Google Sheets client cache
_gsheet_client = None
_gsheet_client_lock = threading.Lock()

# Fallback storage for leads
_fallback_storage_lock = threading.Lock()

# =========== ENHANCED CONTEXT MANAGEMENT ===========
class EnhancedContext:
    def __init__(self):
        self.tour_mentions = []  # [(tour_id, confidence, timestamp)]
        self.user_preferences = {
            "duration_pref": None,
            "price_range": None, 
            "interests": [],
            "location_pref": None
        }
        self.conversation_stack = []  # Lưu 10 lượt gần nhất
        self.last_action = None
        self.timestamp = datetime.utcnow()
        # For backward compatibility during transition
        self.last_tour_indices = []
        self.conversation_history = []
        self.last_tour_name = None

# Global session storage
ENHANCED_SESSION_CONTEXT = {}
CONTEXT_TIMEOUT = 1800  # 30 phút

def cleanup_old_contexts():
    """Dọn dẹp context cũ"""
    now = datetime.utcnow()
    to_delete = []
    for session_id, context in ENHANCED_SESSION_CONTEXT.items():
        if (now - context.timestamp).total_seconds() > CONTEXT_TIMEOUT:
            to_delete.append(session_id)
    for session_id in to_delete:
        del ENHANCED_SESSION_CONTEXT[session_id]

def get_session_context(session_id):
    """Lấy enhanced context cho session"""
    cleanup_old_contexts()
    if session_id not in ENHANCED_SESSION_CONTEXT:
        ENHANCED_SESSION_CONTEXT[session_id] = EnhancedContext()
    return ENHANCED_SESSION_CONTEXT[session_id]

def update_tour_context(session_id, tour_indices, tour_name=None):
    """Cập nhật context tour"""
    context = get_session_context(session_id)
    if tour_indices:
        context.last_tour_indices = tour_indices
        # Update tour_mentions with confidence
        for idx in tour_indices:
            context.tour_mentions.append((idx, 1.0, datetime.utcnow()))
        # Keep only last 5 mentions
        if len(context.tour_mentions) > 5:
            context.tour_mentions = context.tour_mentions[-5:]
    if tour_name:
        context.last_tour_name = tour_name
    context.timestamp = datetime.utcnow()
    return context

# Keep old function names for compatibility during transition
SESSION_CONTEXT = ENHANCED_SESSION_CONTEXT  # Alias for backward compatibility

def extract_session_id(request_data, remote_addr):
    """Trích xuất session_id từ request"""
    # Ưu tiên session_id từ frontend
    session_id = request_data.get("session_id")
    
    if not session_id:
        # Tạo session_id ổn định từ IP + timestamp (giữ trong 30 phút)
        ip = remote_addr or "0.0.0.0"
        current_hour = datetime.utcnow().strftime("%Y%m%d%H")
        
        # Tạo session_id ổn định trong 1 giờ
        unique_str = f"{ip}_{current_hour}"
        session_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    return f"session_{session_id}"

def get_complete_tour_info(tour_indices):
    """Lấy thông tin đầy đủ của tour từ các indices"""
    result = {}
    
    for idx in tour_indices:
        tour_info = {}
        
        # Lấy tên tour
        for m in MAPPING:
            if f"[{idx}]" in m.get("path", "") and ".tour_name" in m.get("path", ""):
                tour_info["name"] = m.get("text", "")
                break
        
        # Lấy các field khác
        for field in TOUR_FIELDS:
            if field == "tour_name":
                continue
                
            passages = get_passages_by_field(field, limit=5, tour_indices=[idx])
            if passages:
                field_texts = [m.get("text", "") for _, m in passages]
                tour_info[field] = "\n".join(field_texts)
        
        result[idx] = tour_info
    
    return result

def get_suggested_questions(tour_indices, current_field):
    """Gợi ý câu hỏi tiếp theo"""
    suggestions = []
    
    if not tour_indices:
        suggestions.extend([
            "Bạn muốn hỏi về tour nào?",
            "Có tour nào về Huế không?",
            "Tour nào phù hợp cho gia đình?"
        ])
    else:
        common_fields = ["price", "includes", "accommodation", "meals", "duration"]
        current = current_field or ""
        
        for field in common_fields:
            if field != current:
                field_names = {
                    "price": "giá cả",
                    "includes": "lịch trình",
                    "accommodation": "chỗ ở", 
                    "meals": "ăn uống",
                    "duration": "thời gian"
                }
                suggestions.append(f"Tour có {field_names.get(field, field)} như thế nào?")
    
    return suggestions[:3]  # Chỉ 3 gợi ý

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
# =========== FIELD INFERENCE RULES ===========
FIELD_INFERENCE_RULES = {
    "price": {
        "default_for_1day": "800.000 - 1.500.000 VNĐ",
        "default_for_2day": "1.500.000 - 2.500.000 VNĐ",
        "missing_response": "Liệt kê tất cả tour có giá trong database"
    },
    "meals": {
        "1day_tours": "Các tour 1 ngày đều bao gồm ít nhất 1 bữa chính",
        "2day_tours": "Tour 2 ngày thường bao gồm 3 bữa chính/ngày + 2 bữa sáng"
    },
    "duration": {
        "tour_reference": "Các tour thường có thời gian 1 ngày hoặc 2 ngày 1 đêm"
    },
    "accommodation": {
        "1day_tours": "Tour 1 ngày không bao gồm chỗ ở qua đêm",
        "2day_tours": "Tour 2 ngày 1 đêm bao gồm 1 đêm lưu trú"
    }
}

COMMON_SENSE_RULES = {
    "1day_tour_meals": "Tour 1 ngày luôn bao gồm ít nhất 1 bữa ăn (thường là bữa trưa).",
    "2day_tour_accommodation": "Tour 2 ngày 1 đêm luôn bao gồm chỗ ở qua đêm.",
    "tour_includes_transport": "Tất cả các tour đều bao gồm phương tiện di chuyển (xe du lịch).",
    "tour_has_price": "Mọi tour đều có giá cả cụ thể, có thể thay đổi theo số lượng người.",
    "tour_has_location": "Mỗi tour đều có địa điểm cụ thể để tham quan.",
    "basic_includes": "Các tour đều bao gồm hướng dẫn viên và bảo hiểm du lịch."
}
# =========== TOUR FIELDS FOR COMPLETE INFO ===========
TOUR_FIELDS = [
    "tour_name", "summary", "location", "duration", "price",
    "includes", "notes", "style", "transport", "accommodation",
    "meals", "event_support", "cancellation_policy", 
    "booking_method", "who_can_join", "hotline"
]

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
def build_tours_db():
    """
    Build structured tour database from MAPPING.
    - TOURS_DB: {tour_index: {field_name: field_value}}
    - TOUR_TAGS: {tour_index: [tag1, tag2, ...]}
    """
    global TOURS_DB, TOUR_TAGS
    TOURS_DB.clear()
    TOUR_TAGS.clear()
    
    # First pass: collect all fields for each tour
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        if not path or not text:
            continue
        
                # Extract tour index from path pattern: tours[index].field (có thể có "root." prefix)
        tour_match = re.search(r'(?:root\.)?tours\[(\d+)\]', path)
        
        # Extract field name (remove array indices if present)
        # Example: root.tours[0].includes[0] -> includes
        field_match = re.search(r'(?:root\.)?tours\[\d+\]\.(\w+)(?:\[\d+\])?', path)
        if not field_match:
            continue
            
        field_name = field_match.group(1)
        
        # Initialize tour entry if not exists
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = {}
        
        # Handle field value accumulation
        # For list fields (like includes, transport), collect as list
        # For string fields, use the text directly
        current_value = TOURS_DB[tour_idx].get(field_name)
        
        if current_value is None:
            TOURS_DB[tour_idx][field_name] = text
        elif isinstance(current_value, list):
            current_value.append(text)
        elif isinstance(current_value, str):
            # Convert to list when encountering multiple values
            TOURS_DB[tour_idx][field_name] = [current_value, text]
    
    # Second pass: generate tags for each tour
    for tour_idx, tour_data in TOURS_DB.items():
        tags = []
        
        # Extract location tags
        location = tour_data.get("location", "")
        if location:
            locations = [loc.strip() for loc in location.split(",") if loc.strip()]
            tags.extend([f"location:{loc}" for loc in locations[:2]])
        
        # Extract duration tags
        duration = tour_data.get("duration", "")
        if duration:
            if "1 ngày" in duration or "1ngày" in duration:
                tags.append("duration:1day")
            elif "2 ngày" in duration or "2ngày" in duration:
                tags.append("duration:2day")
            elif "3 ngày" in duration or "3ngày" in duration:
                tags.append("duration:3day")
            else:
                tags.append(f"duration:{duration}")
        
        # Extract price range tags
        price = tour_data.get("price", "")
        if price:
            # Extract numeric price range
            price_nums = re.findall(r'[\d,\.]+', price)
            if price_nums:
                try:
                    # Clean and convert to float
                    clean_nums = []
                    for p in price_nums[:2]:
                        p_clean = p.replace(',', '').replace('.', '')
                        if p_clean.isdigit():
                            clean_nums.append(int(p_clean))
                    
                    if clean_nums:
                        avg_price = sum(clean_nums) / len(clean_nums)
                        if avg_price < 1000000:
                            tags.append("price:budget")
                        elif avg_price < 2000000:
                            tags.append("price:midrange")
                        else:
                            tags.append("price:premium")
                except:
                    pass
        
        # Extract style tags
        style = tour_data.get("style", "")
        if style:
            style_lower = style.lower()
            if "thiền" in style_lower or "chánh niệm" in style_lower:
                tags.append("style:meditation")
            if "lịch sử" in style_lower or "tri ân" in style_lower:
                tags.append("style:history")
            if "thiên nhiên" in style_lower or "rừng" in style_lower:
                tags.append("style:nature")
            if "retreat" in style_lower or "chữa lành" in style_lower:
                tags.append("style:retreat")
            if "văn hóa" in style_lower or "cộng đồng" in style_lower:
                tags.append("style:culture")
        
        # Add tags based on tour name
        tour_name = tour_data.get("tour_name", "")
        if tour_name:
            name_lower = tour_name.lower()
            if "bạch mã" in name_lower:
                tags.append("destination:bachma")
            if "trường sơn" in name_lower:
                tags.append("destination:truongson")
            if "quảng trị" in name_lower:
                tags.append("destination:quangtri")
            if "huế" in name_lower:
                tags.append("destination:hue")
        
        TOUR_TAGS[tour_idx] = list(set(tags))  # Remove duplicates
    
    logger.info(f"✅ Built tours database: {len(TOURS_DB)} tours, tags generated")




def build_tours_db():
    """
    Build structured tour database from MAPPING.
    - TOURS_DB: {tour_index: {field_name: field_value}}
    - TOUR_TAGS: {tour_index: [tag1, tag2, ...]}
    """
    global TOURS_DB, TOUR_TAGS
    TOURS_DB.clear()
    TOUR_TAGS.clear()
    
    
    
    
    
    # First pass: collect all fields for each tour
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        if not path or not text:
            continue
        
        # Extract tour index from path pattern: tours[index].field
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            # DEBUG: Log paths that don't match
            if "tour" in path.lower():
                logger.debug(f"DEBUG: Path doesn't match tour pattern: {path}")
            continue
            
        tour_idx = int(tour_match.group(1))
        
        # Extract field name (remove array indices if present)
        # Example: tours[0].includes[0] -> includes
        field_match = re.search(r'tours\[\d+\]\.(\w+)(?:\[\d+\])?', path)
        if not field_match:
            logger.debug(f"DEBUG: Could not extract field from path: {path}")
            continue
            
        field_name = field_match.group(1)
        
        # Initialize tour entry if not exists
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = {}
            logger.debug(f"DEBUG: Created tour entry for index {tour_idx}")
        
        # Handle field value accumulation
        # For list fields (like includes, transport), collect as list
        # For string fields, use the text directly
        current_value = TOURS_DB[tour_idx].get(field_name)
        
        if current_value is None:
            TOURS_DB[tour_idx][field_name] = text
        elif isinstance(current_value, list):
            current_value.append(text)
        elif isinstance(current_value, str):
            # Convert to list when encountering multiple values
            TOURS_DB[tour_idx][field_name] = [current_value, text]
    
    
    
    # Second pass: generate tags for each tour
    for tour_idx, tour_data in TOURS_DB.items():
        tags = []
        
        # Extract location tags
        location = tour_data.get("location", "")
        if location:
            locations = [loc.strip() for loc in location.split(",") if loc.strip()]
            tags.extend([f"location:{loc}" for loc in locations[:2]])
        
        # Extract duration tags
        duration = tour_data.get("duration", "")
        if duration:
            if "1 ngày" in duration or "1ngày" in duration:
                tags.append("duration:1day")
            elif "2 ngày" in duration or "2ngày" in duration:
                tags.append("duration:2day")
            elif "3 ngày" in duration or "3ngày" in duration:
                tags.append("duration:3day")
            else:
                tags.append(f"duration:{duration}")
        
        # Extract price range tags
        price = tour_data.get("price", "")
        if price:
            # Extract numeric price range
            price_nums = re.findall(r'[\d,\.]+', price)
            if price_nums:
                try:
                    # Clean and convert to float
                    clean_nums = []
                    for p in price_nums[:2]:
                        p_clean = p.replace(',', '').replace('.', '')
                        if p_clean.isdigit():
                            clean_nums.append(int(p_clean))
                    
                    if clean_nums:
                        avg_price = sum(clean_nums) / len(clean_nums)
                        if avg_price < 1000000:
                            tags.append("price:budget")
                        elif avg_price < 2000000:
                            tags.append("price:midrange")
                        else:
                            tags.append("price:premium")
                except:
                    pass
        
        # Extract style tags
        style = tour_data.get("style", "")
        if style:
            style_lower = style.lower()
            if "thiền" in style_lower or "chánh niệm" in style_lower:
                tags.append("style:meditation")
            if "lịch sử" in style_lower or "tri ân" in style_lower:
                tags.append("style:history")
            if "thiên nhiên" in style_lower or "rừng" in style_lower:
                tags.append("style:nature")
            if "retreat" in style_lower or "chữa lành" in style_lower:
                tags.append("style:retreat")
            if "văn hóa" in style_lower or "cộng đồng" in style_lower:
                tags.append("style:culture")
        
        # Add tags based on tour name
        tour_name = tour_data.get("tour_name", "")
        if tour_name:
            name_lower = tour_name.lower()
            if "bạch mã" in name_lower:
                tags.append("destination:bachma")
            if "trường sơn" in name_lower:
                tags.append("destination:truongson")
            if "quảng trị" in name_lower:
                tags.append("destination:quangtri")
            if "huế" in name_lower:
                tags.append("destination:hue")
        
        TOUR_TAGS[tour_idx] = list(set(tags))  # Remove duplicates
    
    logger.info(f"✅ Built tours database: {len(TOURS_DB)} tours, tags generated")

def resolve_tour_reference(message: str, context: EnhancedContext) -> List[int]:
    """
    Xác định tour được đề cập bằng 4 phương pháp:
    1. Tên tour trực tiếp (Bạch Mã, Trường Sơn)
    2. Đặc điểm: "tour 1 ngày", "tour dưới 2 triệu"
    3. Ngữ cảnh: "tour đó", "tour này", "cái tour"
    4. Lịch sử: tour vừa được mention gần nhất
    """
    if not message:
        return []
    
    text_l = message.lower().strip()
    msg_norm = normalize_text_simple(message)
    
    # 1. Tên tour trực tiếp
    direct_matches = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        tour_words = set(norm_name.split())
        msg_words = set(msg_norm.split())
        common_words = tour_words & msg_words
        if len(common_words) >= 1:
            score = len(common_words) / max(len(tour_words), 1)
            direct_matches.append((score, idx))
    
    if direct_matches:
        direct_matches.sort(reverse=True)
        best_score = direct_matches[0][0]
        if best_score >= 0.5:
            selected = [idx for score, idx in direct_matches if score == best_score]
            return sorted(set(selected))
    
    # 2. Đặc điểm (duration, price, location)
    feature_matches = []
    
    # Duration features
    if "1 ngày" in text_l or "1ngày" in text_l:
        duration_filter = "1 ngày"
    elif "2 ngày" in text_l or "2ngày" in text_l:
        duration_filter = "2 ngày"
    else:
        duration_filter = None
    
    # Price features
    price_match = re.search(r'dưới\s*(\d+)\s*tr', text_l)
    price_filter = None
    if price_match:
        max_price = int(price_match.group(1)) * 1000000  # Convert triệu to VNĐ
    
    # Location features
    location_keywords = []
    for loc in ['quảng trị', 'huế', 'bạch mã', 'trường sơn', 'đông hà', 'khe sanh']:
        if loc in text_l:
            location_keywords.append(loc)
    
    # Tìm tour phù hợp với features
    if duration_filter or location_keywords:
        for tour_idx, tour_data in TOURS_DB.items():
            score = 0.0
            
            if duration_filter and 'duration' in tour_data:
                if duration_filter in tour_data['duration'].lower():
                    score += 1.0
            
            if location_keywords and 'location' in tour_data:
                location = tour_data['location'].lower()
                for lk in location_keywords:
                    if lk in location:
                        score += 1.0
                        break
            
            if score > 0:
                feature_matches.append((score, tour_idx))
        
        if feature_matches:
            feature_matches.sort(reverse=True)
            best_score = feature_matches[0][0]
            selected = [idx for score, idx in feature_matches if score == best_score]
            return selected
    
    # 3. Ngữ cảnh (context reference)
    context_refs = ['tour này', 'tour đó', 'tour đang nói', 'cái tour', 'này', 'đó', 'nó']
    if any(ref in text_l for ref in context_refs):
        if context.last_tour_indices:
            return context.last_tour_indices
        if context.conversation_history:
            for msg in reversed(context.conversation_history[-3:]):
                if msg.get("tour_indices"):
                    return msg.get("tour_indices")
    
    # 4. Lịch sử (recent mentions)
    if context.tour_mentions:
        recent = sorted(context.tour_mentions, key=lambda x: x[2], reverse=True)
        for tour_id, confidence, _ in recent[:2]:
            if confidence >= 0.5:
                return [tour_id]
    
    return []

# Giữ tương thích với code cũ
def find_tour_indices_from_message(message: str) -> List[int]:
    """Legacy function, uses resolve_tour_reference with minimal context"""
    class TempContext:
        def __init__(self):
            self.last_tour_indices = []
            self.conversation_history = []
            self.tour_mentions = []
    
    temp_context = TempContext()
    return resolve_tour_reference(message, temp_context)

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


def handle_field_query(field_name: str, tour_indices: Optional[List[int]] = None, context: Optional[EnhancedContext] = None) -> Tuple[str, List[dict]]:
    """
    Thông minh xử lý truy vấn field với inference rules và common sense.
    Xử lý cả trường hợp có tour cụ thể và không có tour cụ thể.
    Trả về: (answer_text, source_passages)
    """
    passages = []
    
    # Nếu không có tour cụ thể, tổng hợp thông tin từ tất cả các tour
    if not tour_indices:
        # Nếu field là tour_name, trả về danh sách tên tour
        if field_name == "tour_name":
            tour_names = []
            for idx, tour_data in TOURS_DB.items():
                name = tour_data.get("tour_name")
                if name and name not in tour_names:
                    tour_names.append(name)
            
            if tour_names:
                answer = "Danh sách các tour hiện có:\n"
                for i, name in enumerate(tour_names[:10], 1):
                    answer += f"{i}. {name}\n"
                if len(tour_names) > 10:
                    answer += f"... và {len(tour_names) - 10} tour khác."
                return answer, []
            else:
                return "Hiện chưa có thông tin tour. Vui lòng liên hệ hotline 0935 304 338.", []
        
        # Tổng hợp thông tin field từ tất cả tour
        all_values = []
        for idx, tour_data in TOURS_DB.items():
            if field_name in tour_data:
                value = tour_data[field_name]
                if isinstance(value, list):
                    all_values.extend(value)
                else:
                    all_values.append(value)
        
        if all_values:
            # Lấy 5 giá trị đầu tiên
            sample = all_values[:5]
            answer = f"Thông tin về {field_name} từ các tour:\n"
            for val in sample:
                answer += f"• {val}\n"
            if len(all_values) > 5:
                answer += f"... và {len(all_values) - 5} thông tin khác."
            return answer, []
        else:
            # Dùng inference rules
            if field_name in FIELD_INFERENCE_RULES:
                rules = FIELD_INFERENCE_RULES[field_name]
                answer = f"Thông tin chung về {field_name}:\n"
                for rule_key, rule_value in rules.items():
                    answer += f"• {rule_value}\n"
                return answer, []
            else:
                # Common sense fallback
                for rule_key, rule_value in COMMON_SENSE_RULES.items():
                    if field_name in rule_key:
                        return rule_value, []
                
                return f"Hiện không có thông tin về {field_name} trong dữ liệu. Vui lòng liên hệ hotline 0935 304 338.", []
    
    # Nếu có tour cụ thể
    answers = []
    inference_used = False
    
    for idx in tour_indices:
        if idx in TOURS_DB:
            tour_data = TOURS_DB[idx]
            
            # Field có trong database
            if field_name in tour_data:
                field_value = tour_data[field_name]
                if isinstance(field_value, list):
                    answer = "\n".join([f"• {item}" for item in field_value])
                else:
                    answer = field_value
                
                # Lấy tên tour
                tour_name = tour_data.get("tour_name", f"Tour #{idx}")
                answers.append(f"**{tour_name}**:\n{answer}")
                
                # Ghi nhận source
                for m in MAPPING:
                    if f"[{idx}]" in m.get("path", "") and f".{field_name}" in m.get("path", ""):
                        passages.append(m)
            else:
                # Field không có trong database, sử dụng inference rules
                tour_name = tour_data.get("tour_name", f"Tour #{idx}")
                
                # Áp dụng inference rules dựa trên duration
                duration = tour_data.get("duration", "")
                inference_answer = None
                
                if field_name == "price" and FIELD_INFERENCE_RULES.get("price"):
                    if "1 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["price"]["default_for_1day"]
                    elif "2 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["price"]["default_for_2day"]
                
                elif field_name == "meals" and FIELD_INFERENCE_RULES.get("meals"):
                    if "1 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["meals"]["1day_tours"]
                    elif "2 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["meals"]["2day_tours"]
                
                elif field_name == "accommodation" and FIELD_INFERENCE_RULES.get("accommodation"):
                    if "1 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["accommodation"]["1day_tours"]
                    elif "2 ngày" in duration:
                        inference_answer = FIELD_INFERENCE_RULES["accommodation"]["2day_tours"]
                
                if inference_answer:
                    answers.append(f"**{tour_name}**:\n{inference_answer} (thông tin ước tính dựa trên loại tour)")
                    inference_used = True
                else:
                    # Common sense fallback
                    for rule_key, rule_value in COMMON_SENSE_RULES.items():
                        if field_name in rule_key:
                            answers.append(f"**{tour_name}**:\n{rule_value}")
                            inference_used = True
                            break
    
    if answers:
        answer_text = "\n\n".join(answers)
        if inference_used:
            answer_text += "\n\n*Ghi chú: Thông tin dựa trên ước tính thông thường. Vui lòng liên hệ hotline 0935 304 338 để biết chính xác.*"
        return answer_text, passages
    
    # Hoàn toàn không có thông tin
    if field_name == "price":
        return "Thông tin giá cả đang được cập nhật. Vui lòng liên hệ hotline 0935 304 338 để được báo giá chính xác.", []
    elif field_name == "meals":
        return "Thông tin về bữa ăn đang được cập nhật. Các tour thường bao gồm ít nhất 1 bữa chính mỗi ngày.", []
    else:
        return f"Thông tin về {field_name} đang được cập nhật. Vui lòng liên hệ hotline 0935 304 338 để biết thêm chi tiết.", []
def compare_tours(tour_ids: List[int], aspect: str) -> str:
    """So sánh 2+ tour theo aspect: thời gian, giá, phù hợp..."""
    if len(tour_ids) < 2:
        return "Cần ít nhất 2 tour để so sánh."
    
    if not TOURS_DB:
        return "Không có dữ liệu tour để so sánh."
    
    results = []
    
    # Lấy thông tin các tour
    tours_data = []
    for tid in tour_ids:
        if tid in TOURS_DB:
            tours_data.append((tid, TOURS_DB[tid]))
    
    if len(tours_data) < 2:
        return "Không đủ dữ liệu tour để so sánh."
    
    # So sánh theo aspect
    aspect = aspect.lower()
    
    if "giá" in aspect or "price" in aspect:
        results.append("**SO SÁNH VỀ GIÁ CẢ:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            price = tour.get("price", "Chưa có thông tin")
            duration = tour.get("duration", "")
            results.append(f"• **{name}** ({duration}): {price}")
    
    elif "thời gian" in aspect or "duration" in aspect:
        results.append("**SO SÁNH VỀ THỜI GIAN:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            duration = tour.get("duration", "Chưa có thông tin")
            results.append(f"• **{name}**: {duration}")
    
    elif "địa điểm" in aspect or "location" in aspect:
        results.append("**SO SÁNH VỀ ĐỊA ĐIỂM:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            location = tour.get("location", "Chưa có thông tin")
            results.append(f"• **{name}**: {location}")
    
    elif "ăn uống" in aspect or "meals" in aspect:
        results.append("**SO SÁNH VỀ ĂN UỐNG:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            meals = tour.get("meals", "Chưa có thông tin chi tiết")
            if isinstance(meals, list):
                meals = ", ".join(meals[:3]) + ("..." if len(meals) > 3 else "")
            results.append(f"• **{name}**: {meals}")
    
    elif "chỗ ở" in aspect or "accommodation" in aspect:
        results.append("**SO SÁNH VỀ CHỖ Ở:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            accommodation = tour.get("accommodation", "Chưa có thông tin chi tiết")
            if isinstance(accommodation, list):
                accommodation = ", ".join(accommodation[:2]) + ("..." if len(accommodation) > 2 else "")
            results.append(f"• **{name}**: {accommodation}")
    
    else:
        # So sánh tổng quát
        results.append("**SO SÁNH TỔNG QUAN CÁC TOUR:**\n")
        for tid, tour in tours_data:
            name = tour.get("tour_name", f"Tour #{tid}")
            duration = tour.get("duration", "?")
            location = tour.get("location", "?")
            price = tour.get("price", "?")
            summary = tour.get("summary", "")
            
            results.append(f"**{name}**")
            results.append(f"  • Thời gian: {duration}")
            results.append(f"  • Địa điểm: {location}")
            results.append(f"  • Giá: {price}")
            if summary and len(summary) > 0:
                short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                results.append(f"  • Mô tả: {short_summary}")
            results.append("")
    
    # Thêm lời khuyên nếu có 2 tour
    if len(tours_data) == 2:
        results.append("\n**GỢI Ý LỰA CHỌN:**")
        tour1 = tours_data[0][1]
        tour2 = tours_data[1][1]
        
        name1 = tour1.get("tour_name", "Tour 1")
        name2 = tour2.get("tour_name", "Tour 2")
        
        # So sánh duration
        dur1 = tour1.get("duration", "")
        dur2 = tour2.get("duration", "")
        
        if "1 ngày" in dur1 and "2 ngày" in dur2:
            results.append(f"• Nếu bạn có ít thời gian: **{name1}** (1 ngày)")
            results.append(f"• Nếu muốn trải nghiệm sâu hơn: **{name2}** (2 ngày)")
        
        # So sánh location
        loc1 = tour1.get("location", "").lower()
        loc2 = tour2.get("location", "").lower()
        
        if "bạch mã" in loc1 and "trường sơn" in loc2:
            results.append(f"• Nếu thích thiên nhiên, retreat: **{name1}** (Bạch Mã)")
            results.append(f"• Nếu thích lịch sử, văn hóa: **{name2}** (Trường Sơn)")
    
    return "\n".join(results)

def recommend_tours_by_preferences(prefs: dict, available_tours: list) -> List[Tuple[int, float]]:
    """Đề xuất tour dựa trên user preferences với confidence score"""
    recommendations = []
    
    # Nếu không có preference rõ ràng, trả về tour phổ biến
    if not prefs or all(v is None or (isinstance(v, list) and len(v) == 0) for v in prefs.values()):
        # Trả về tất cả tour với confidence mặc định
        for tour_idx in available_tours[:5]:
            recommendations.append((tour_idx, 0.5))
        return recommendations
    
    # Tính điểm cho mỗi tour
    for tour_idx in available_tours:
        if tour_idx not in TOURS_DB:
            continue
            
        tour_data = TOURS_DB[tour_idx]
        score = 0.0
        max_possible = 0
        
        # Kiểm tra duration preference
        if prefs.get("duration_pref") and "duration" in tour_data:
            duration = tour_data["duration"].lower()
            if prefs["duration_pref"] == "1day" and ("1 ngày" in duration or "1ngày" in duration):
                score += 1.0
            elif prefs["duration_pref"] == "2day" and ("2 ngày" in duration or "2ngày" in duration):
                score += 1.0
            max_possible += 1
        
        # Kiểm tra location preference
        if prefs.get("location_pref") and "location" in tour_data:
            location = tour_data["location"].lower()
            pref_location = prefs["location_pref"].lower()
            if pref_location in location:
                score += 1.0
            max_possible += 1
        
        # Kiểm tra interests
        if prefs.get("interests") and isinstance(prefs["interests"], list):
            interests = [i.lower() for i in prefs["interests"]]
            tour_tags = TOUR_TAGS.get(tour_idx, [])
            
            # Chuyển tags thành keywords
            tag_keywords = []
            for tag in tour_tags:
                # style:meditation -> meditation
                if ":" in tag:
                    tag_keywords.append(tag.split(":")[1])
                else:
                    tag_keywords.append(tag)
            
            # Kiểm tra mỗi interest
            for interest in interests:
                if any(interest in keyword for keyword in tag_keywords):
                    score += 0.5
            max_possible += 0.5 * len(interests)
        
        # Kiểm tra price range
        if prefs.get("price_range") and "price" in tour_data:
            price = tour_data["price"]
            price_nums = re.findall(r'[\d,\.]+', price)
            if price_nums:
                try:
                    # Lấy giá đầu tiên
                    first_price = price_nums[0].replace(',', '').replace('.', '')
                    if first_price.isdigit():
                        price_val = int(first_price)
                        
                        if prefs["price_range"] == "budget" and price_val < 1500000:
                            score += 1.0
                        elif prefs["price_range"] == "midrange" and 1500000 <= price_val <= 3000000:
                            score += 1.0
                        elif prefs["price_range"] == "premium" and price_val > 3000000:
                            score += 1.0
                except:
                    pass
            max_possible += 1
        
        # Tính confidence score
        if max_possible > 0:
            confidence = score / max_possible
        else:
            confidence = 0.5  # Mặc định
            
        recommendations.append((tour_idx, confidence))
    
    # Sắp xếp theo confidence
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations
    
    # Fallback về phương pháp cũ nếu không có trong database
    passages = get_passages_by_field(field_name, limit=5, tour_indices=tour_indices)
    if passages:
        texts = []
        for score, m in passages:
            text = m.get("text", "").strip()
            if text:
                texts.append(text)
        
        if texts:
            answer_text = "\n".join([f"• {text}" for text in texts[:3]])
            return answer_text, [m for _, m in passages]
    
    # Hoàn toàn không có thông tin
    if field_name == "price":
        return "Thông tin giá cả đang được cập nhật. Vui lòng liên hệ hotline 0332510486 để được báo giá chính xác.", []
    elif field_name == "meals":
        return "Thông tin về bữa ăn đang được cập nhật. Các tour thường bao gồm ít nhất 1 bữa chính mỗi ngày.", []
    else:
        return f"Thông tin về {field_name} đang được cập nhật. Vui lòng liên hệ hotline 0332510486 để biết thêm chi tiết.", []
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
            # Tự động hỗ trợ cả file cũ và mới
            index_files_to_try = [FAISS_INDEX_PATH, "index.faiss"]
            loaded = False
            
            for index_file in index_files_to_try:
                if use_faiss and os.path.exists(index_file) and os.path.exists(FAISS_MAPPING_PATH):
                    try:
                        idx = faiss.read_index(index_file)
                        if load_mapping_from_disk(FAISS_MAPPING_PATH):
                            FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                        idx_dim = _index_dim(idx)
                        if idx_dim:
                            EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                            logger.info("Detected FAISS index dim=%s -> embedding_model=%s", idx_dim, EMBEDDING_MODEL)
                        INDEX = idx
                        index_tour_names()
                        logger.info(f"✅ FAISS index loaded from {index_file}.")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {index_file}: {e}")
            
            if loaded:
                return True
            
            # Fallback: try numpy index
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
                    # Xóa file index cũ để tránh conflict
                    old_files = ["index.faiss", "faiss_index_meta.json"]
                    for old_file in old_files:
                        if os.path.exists(old_file):
                            try:
                                os.remove(old_file)
                                logger.info(f"Removed old index file: {old_file}")
                            except Exception:
                                pass
                    
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
def compose_enhanced_prompt(top_passages: List[Tuple[float, dict]], context: EnhancedContext, tour_indices: List[int], user_message: str) -> str:
    """
    Prompt thông minh hơn với:
    - Ngữ cảnh hội thoại
    - User preferences
    - Tour đang được thảo luận
    - Inference rules cho missing info
    """
    # Base header
    header = (
        "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.\n"
        "TRẢ LỜI THEO CÁC NGUYÊN TẮC:\n"
        "1. ƯU TIÊN CAO NHẤT: Luôn sử dụng thông tin từ dữ liệu nội bộ được cung cấp thông qua hệ thống.\n"
        "2. Nếu thiếu thông tin CHI TIẾT, hãy tổng hợp và trả lời dựa trên THÔNG TIN CHUNG có sẵn trong dữ liệu nội bộ.\n"
        "3. Đối với tour cụ thể: nếu tìm thấy bất kỳ dữ liệu nội bộ liên quan nào (dù là tóm tắt, giá, lịch trình, ghi chú), PHẢI tổng hợp và trình bày rõ ràng; chỉ trả lời đang nâng cấp hoặc chưa có thông tin khi hoàn toàn không tìm thấy dữ liệu phù hợp.\n"
        "4. TUYỆT ĐỐI KHÔNG nói rằng bạn không đọc được file, không truy cập dữ liệu, hoặc từ chối trả lời khi đã có dữ liệu liên quan.\n"
        "5. Luôn giữ thái độ nhiệt tình, hữu ích, trả lời trực tiếp vào nội dung người dùng hỏi.\n\n"
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn ngành du lịch trải nghiệm, retreat, "
        "thiền, khí công, hành trình chữa lành và các hành trình tham quan linh hoạt theo nhu cầu. "
        "Trả lời ngắn gọn, chính xác, rõ ràng, tử tế và bám sát dữ liệu Ruby Wings.\n\n"
    )
    
    # Add context information
    context_info = ""
    if context.user_preferences:
        prefs = []
        if context.user_preferences.get("duration_pref"):
            prefs.append(f"Thích tour {context.user_preferences['duration_pref']}")
        if context.user_preferences.get("price_range"):
            prefs.append(f"Ngân sách {context.user_preferences['price_range']}")
        if context.user_preferences.get("interests"):
            prefs.append(f"Quan tâm: {', '.join(context.user_preferences['interests'])}")
        if prefs:
            context_info = "THÔNG TIN NGƯỜI DÙNG (từ hội thoại trước):\n" + "\n".join(f"- {p}" for p in prefs) + "\n\n"
    
    # Add tour information if available
    tour_info = ""
    if tour_indices and TOURS_DB:
        tour_info = "TOUR ĐANG ĐƯỢC THẢO LUẬN:\n"
        for idx in tour_indices[:2]:  # Limit to 2 tours
            if idx in TOURS_DB:
                tour = TOURS_DB[idx]
                name = tour.get("tour_name", f"Tour #{idx}")
                duration = tour.get("duration", "Không rõ")
                location = tour.get("location", "Không rõ")
                tour_info += f"- {name} (Thời gian: {duration}, Địa điểm: {location})\n"
        tour_info += "\n"
    
    # Add inference rules for common questions
    inference_info = ""
    # Check if the user is asking about a field that might have inference rules
    lower_msg = user_message.lower()
    for field, rules in FIELD_INFERENCE_RULES.items():
        # Tìm từ khóa cho field này trong KEYWORD_FIELD_MAP
        keywords_for_field = []
        for k, v in KEYWORD_FIELD_MAP.items():
            if v.get("field") == field:
                keywords_for_field.extend(v.get("keywords", []))
                    
        
        # Nếu câu hỏi có chứa từ khóa của field này
        if any(keyword in lower_msg for keyword in keywords_for_field):
            inference_info = "QUY TẮC SUY LUẬN CHO THÔNG TIN THIẾU:\n"
            for rule_key, rule_value in rules.items():
                inference_info += f"- {rule_value}\n"
            inference_info += "\n"
            break
    
    # Build the data section
    if not top_passages:
        data_section = "Không tìm thấy dữ liệu nội bộ phù hợp."
    else:
        data_section = "DỮ LIỆU NỘI BỘ (theo độ liên quan):\n"
        for i, (score, m) in enumerate(top_passages, start=1):
            data_section += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    
    # Combine all parts
    prompt = header + context_info + tour_info + inference_info + data_section
    
    # Add footer with instructions
    prompt += "\n---\nTUÂN THỦ: Chỉ dùng dữ liệu trên; không bịa đặt nội dung không có thực; văn phong lịch sự."
    prompt += "\n---\nLưu ý: Ưu tiên sử dụng trích dẫn thông tin từ dữ liệu nội bộ ở trên. Nếu phải bổ sung, chỉ dùng kiến thức chuẩn xác, không được tự ý bịa ra khi chưa rõ đúng sai; sử dụng ngôn ngữ lịch sự, thân thiện, thông minh; khi khách gõ lời tạm biệt hoặc lời chúc thì chân thành cám ơn khách, chúc khách sức khoẻ tốt, may mắn, thành công..."
    
    return prompt

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
    # CHỈ scan, không build tours ở đây nữa
    # Tours sẽ được build sau khi MAPPING đã load từ file
    logger.info("✅ Knowledge scanned: %d passages", len(FLAT_TEXTS))

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

       # =========== CONTEXT AWARE PROCESSING ===========
    # Get session context
    session_id = extract_session_id(data, request.remote_addr)
    context = get_session_context(session_id)
    last_tour_indices = context.last_tour_indices
    last_tour_name = context.last_tour_name
        # Update user preferences from current message
    # Extract interests
    interests_to_add = []
    if any(word in text_l for word in ["thiên nhiên", "rừng", "cây cối", "núi"]):
        interests_to_add.append("nature")
    if any(word in text_l for word in ["lịch sử", "tri ân", "chiến tranh", "di tích"]):
        interests_to_add.append("history") 
    if any(word in text_l for word in ["văn hóa", "cộng đồng", "dân tộc", "truyền thống"]):
        interests_to_add.append("culture")
    if any(word in text_l for word in ["thiền", "chánh niệm", "tĩnh tâm", "yoga", "khí công"]):
        interests_to_add.append("meditation")
    if any(word in text_l for word in ["retreat", "chữa lành", "thư giãn", "nghỉ dưỡng"]):
        interests_to_add.append("retreat")
    
    for interest in interests_to_add:
        if interest not in context.user_preferences["interests"]:
            context.user_preferences["interests"].append(interest)
    
    # Extract duration preference
    if "1 ngày" in text_l or "1ngày" in text_l:
        context.user_preferences["duration_pref"] = "1day"
    elif "2 ngày" in text_l or "2ngày" in text_l:
        context.user_preferences["duration_pref"] = "2day"
    
    # Extract price range preference  
    if "dưới 1 triệu" in text_l or "dưới 1tr" in text_l:
        context.user_preferences["price_range"] = "budget"
    elif "dưới 2 triệu" in text_l or "dưới 2tr" in text_l:
        context.user_preferences["price_range"] = "budget"
    elif "từ 2 đến 3 triệu" in text_l or "2-3 triệu" in text_l:
        context.user_preferences["price_range"] = "midrange"
    elif "trên 3 triệu" in text_l:
        context.user_preferences["price_range"] = "premium"
    # Detect requested field
    text_l = user_message.lower()
    requested_field: Optional[str] = None
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in text_l:
                requested_field = v["field"]
                break
        if requested_field:
            break
        
            # Detect recommendation request
    is_recommendation_request = False
    if any(word in text_l for word in ["phù hợp", "recommend", "gợi ý", "nên chọn", "tư vấn tour", "tour nào tốt"]):
        is_recommendation_request = True
        
                # Detect comparison request
    is_comparison_request = False
    compare_aspect = ""
    if any(word in text_l for word in ["so sánh", "sánh", "compare", "khác nhau", "giống nhau"]):
        is_comparison_request = True
        
        # Xác định aspect cần so sánh
        if "giá" in text_l or "price" in text_l:
            compare_aspect = "giá cả"
        elif "thời gian" in text_l or "duration" in text_l:
            compare_aspect = "thời gian"
        elif "địa điểm" in text_l or "location" in text_l:
            compare_aspect = "địa điểm"
        elif "ăn" in text_l or "meals" in text_l:
            compare_aspect = "ăn uống"
        elif "chỗ ở" in text_l or "accommodation" in text_l:
            compare_aspect = "chỗ ở"
        else:
            compare_aspect = "tổng quát"
    # Tour detection with context awareness
    tour_indices = resolve_tour_reference(user_message, context)
    
    # Nếu không tìm thấy tour, KIỂM TRA KỸ các reference
    if not tour_indices:
        # Danh sách từ tham chiếu MỞ RỘNG
        ref_keywords = [
            "tour này", "tour đó", "tour đang nói", 
            "cái tour", "này", "đó", "nó",
            "tour bach ma", "bạch mã", "bach ma"
        ]
        
        has_reference = any(keyword in text_l for keyword in ref_keywords)
        
        if has_reference and last_tour_indices:
            tour_indices = last_tour_indices
            logger.info(f"✅ Using CONTEXT tour indices: {tour_indices} for reference: '{user_message}'")
        elif has_reference and not last_tour_indices:
            # Người dùng nói "tour này" nhưng chưa có context
            # Thử tìm tour gần nhất trong lịch sử
            if context.conversation_history:
                # Tìm tour được mention gần nhất trong history
                for msg in reversed(context.conversation_history[-5:]):
                    if msg.get("type") == "tour_mentioned":
                        tour_indices = msg.get("tour_indices", [])
                        if tour_indices:
                            break
    
    # Update context if we have tour indices
    if tour_indices:
        # Find tour name for these indices
        tour_name = None
        for idx in tour_indices:
            for m in MAPPING:
                if f"[{idx}]" in m.get("path", "") and ".tour_name" in m.get("path", ""):
                    tour_name = m.get("text", "")
                    break
            if tour_name:
                break
        
        update_tour_context(session_id, tour_indices, tour_name)
        
        # Update user preferences based on tour selection
        if tour_indices and len(tour_indices) > 0:
            first_tour_idx = tour_indices[0]
            if first_tour_idx in TOURS_DB:
                tour_data = TOURS_DB[first_tour_idx]
                
                # Extract duration preference
                if "duration" in tour_data:
                    duration = tour_data["duration"]
                    if "1 ngày" in duration or "1ngày" in duration:
                        context.user_preferences["duration_pref"] = "1day"
                    elif "2 ngày" in duration or "2ngày" in duration:
                        context.user_preferences["duration_pref"] = "2day"
                
                # Extract location preference
                if "location" in tour_data:
                    context.user_preferences["location_pref"] = tour_data["location"].split(",")[0].strip()

    # Update conversation history
    context.conversation_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": user_message,
        "tour_indices": tour_indices,
        "requested_field": requested_field,
        "type": "tour_mentioned" if tour_indices else "general"
    })
    
    # Giữ history tối đa 10 messages
    if len(context.conversation_history) > 10:
        context.conversation_history = context.conversation_history[-10:]
    
    # Update conversation stack
    context.conversation_stack.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.utcnow()
    })
    if len(context.conversation_stack) > 10:
        context.conversation_stack = context.conversation_stack[-10:]
    


    
    
    # =========== CHECK FOR LIST REQUEST PATTERNS ===========
    # Initialize is_list_request to False (FIXED CRITICAL BUG)
    is_list_request = False
    
    list_patterns = [
        r"liệt kê.*tour",
        r"có những tour nào",
        r"danh sách tour", 
        r"tour.*nổi bật",
        r"show tour",
        r"tour available"
    ]
    
    is_list_request = any(re.search(pattern, text_l) for pattern in list_patterns)
    
    # =========== DEBUG LOGGING FOR TOUR CONTEXT ===========
    logger.info(f"🎯 TOUR DETECTION DEBUG:")
    logger.info(f"  User message: '{user_message}'")
    logger.info(f"  Found indices: {tour_indices}")
    logger.info(f"  Last tour indices from context: {last_tour_indices}")
    logger.info(f"  Session ID: {session_id}")
    logger.info(f"  Requested field: {requested_field}")
    logger.info(f"  Is list request: {is_list_request}")
    
    # Log tour names if indices exist
    if tour_indices:
        for idx in tour_indices:
            tour_name = None
            for m in MAPPING:
                if f"[{idx}]" in m.get("path", "") and ".tour_name" in m.get("path", ""):
                    tour_name = m.get("text", "")
                    logger.info(f"  Tour index {idx}: '{tour_name}'")
                    break
    
        # Special handling for tour listing requests
    top_results: List[Tuple[float, dict]] = []
    
    # Handle "liệt kê tour" requests
    if is_list_request:
        # Determine how many tours to list
        limit = 3  # Default
        num_match = re.search(r"(\d+)\s*tour", user_message)
        if num_match:
            limit = int(num_match.group(1))
        elif "tất cả" in text_l or "all" in text_l:
            limit = 50  # Large number for "all"
        
        top_results = get_passages_by_field("tour_name", tour_indices=None, limit=limit)
    elif requested_field == "tour_name":
        top_results = get_passages_by_field("tour_name", tour_indices=None, limit=1000)



    elif requested_field:
        # Sử dụng hàm xử lý field thông minh mới
        field_answer, field_sources = handle_field_query(requested_field, tour_indices, context)
        
        # Luôn dùng field_answer từ handle_field_query (đã có inference rules)
        reply = field_answer
        
        # Convert field_sources thành top_results format
        for source in field_sources:
            top_results.append((1.0, source))
        
        # Trả về ngay, không cần qua LLM
        return jsonify({
            "reply": reply, 
            "sources": [m for _, m in top_results],
            "context": {
                "tour_indices": tour_indices,
                "session_id": session_id,
                "last_tour_name": context.last_tour_name,
                "user_preferences": context.user_preferences,
                "suggested_next": get_suggested_questions(tour_indices, requested_field)
            }
        })


                    # Handle comparison request
    if is_comparison_request and tour_indices and len(tour_indices) >= 2:
        comparison_result = compare_tours(tour_indices, compare_aspect)
        
        return jsonify({
            "reply": comparison_result,
            "sources": [],
            "context": {
                "tour_indices": tour_indices,
                "session_id": session_id,
                "last_tour_name": context.last_tour_name,
                "user_preferences": context.user_preferences,
                "suggested_next": ["So sánh về điểm khác", "Tour nào phù hợp hơn với tôi?"]
            }
        })
    

        # Handle recommendation request
    if is_recommendation_request:
        # Get available tours (all tours in TOURS_DB)
        available_tours = list(TOURS_DB.keys())
        
        if not available_tours:
            return jsonify({
                "reply": "Hiện chưa có đủ dữ liệu tour để đề xuất. Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp.",
                "sources": [],
                "context": {
                    "tour_indices": tour_indices,
                    "session_id": session_id,
                    "last_tour_name": context.last_tour_name,
                    "user_preferences": context.user_preferences,
                    "suggested_next": ["Tour 1 ngày nào phổ biến?", "Tour nào về Quảng Trị?"]
                }
            })
        
        # Get recommendations
        recommendations = recommend_tours_by_preferences(context.user_preferences, available_tours)
        
        if not recommendations:
            # Fallback: recommend top 3 tours
            recommendations = [(tid, 0.5) for tid in available_tours[:3]]
        
        # Format recommendation response
        if recommendations:
            reply_lines = ["**DỰA TRÊN SỞ THÍCH CỦA BẠN, TÔI ĐỀ XUẤT:**\n"]
            
            for i, (tour_idx, confidence) in enumerate(recommendations[:3], 1):
                if tour_idx in TOURS_DB:
                    tour = TOURS_DB[tour_idx]
                    name = tour.get("tour_name", f"Tour #{tour_idx}")
                    duration = tour.get("duration", "")
                    location = tour.get("location", "")
                    summary = tour.get("summary", "")
                    
                    confidence_star = "★" * int(confidence * 5)
                    if confidence > 0.7:
                        match_text = "Rất phù hợp"
                    elif confidence > 0.4:
                        match_text = "Khá phù hợp"
                    else:
                        match_text = "Có thể phù hợp"
                    
                    reply_lines.append(f"**{i}. {name}**")
                    reply_lines.append(f"   ⭐ Độ phù hợp: {match_text} {confidence_star}")
                    reply_lines.append(f"   🕒 Thời gian: {duration}")
                    reply_lines.append(f"   📍 Địa điểm: {location}")
                    
                    if summary:
                        short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                        reply_lines.append(f"   📝 {short_summary}")
                    
                    reply_lines.append("")
            
            # Add explanation based on preferences
            if context.user_preferences["duration_pref"]:
                reply_lines.append(f"*Đã ưu tiên tour {context.user_preferences['duration_pref']} theo yêu cầu của bạn.*")
            if context.user_preferences["interests"]:
                interests_str = ", ".join(context.user_preferences["interests"])
                reply_lines.append(f"*Đã ưu tiên tour có chủ đề: {interests_str}.*")
            
            reply_lines.append("\n💡 **Gợi ý tiếp theo**: Bạn có thể hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour.")
            
            reply = "\n".join(reply_lines)
            
            return jsonify({
                "reply": reply,
                "sources": [],
                "context": {
                    "tour_indices": [tid for tid, _ in recommendations[:2]],
                    "session_id": session_id,
                    "last_tour_name": context.last_tour_name,
                    "user_preferences": context.user_preferences,
                    "suggested_next": ["Chi tiết về tour đầu tiên?", "So sánh 2 tour đầu tiên?"]
                }
            })
    else:
        top_k = int(data.get("top_k", TOP_K))
        top_results = query_index(user_message, top_k)

        system_prompt = compose_enhanced_prompt(top_results, context, tour_indices, user_message)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    reply = ""
    
    # =========== SPECIAL HANDLING FOR LIST REQUESTS ===========
    if is_list_request and top_results:
        # Format beautiful tour list response
        names = []
        for _, m in top_results:
            tour_name = m.get("text", "").strip()
            if tour_name and tour_name not in names:
                names.append(tour_name)
        
        if names:
            # Determine limit from message or use all found
            limit = len(names)
            num_match = re.search(r"(\d+)\s*tour", user_message)
            if num_match:
                limit = min(int(num_match.group(1)), len(names))
            
            reply = f"✨ **Ruby Wings hiện có {len(names)} tour trải nghiệm đặc sắc:** ✨\n\n"
            
            for i, name in enumerate(names[:limit], 1):
                # Find tour index for this name
                tour_idx = None
                for idx, m2 in enumerate(MAPPING):
                    if m2.get("text", "").strip() == name and ".tour_name" in m2.get("path", ""):
                        # Extract index from path like "tours[3].tour_name"
                        match = re.search(r'\[(\d+)\]', m2.get("path", ""))
                        if match:
                            tour_idx = int(match.group(1))
                        break
                
                # Get summary for this tour
                summary = ""
                duration = ""
                if tour_idx is not None:
                    for m2 in MAPPING:
                        if f"[{tour_idx}]" in m2.get("path", ""):
                            if ".summary" in m2.get("path", ""):
                                summary = m2.get("text", "").strip()
                            elif ".duration" in m2.get("path", ""):
                                duration = m2.get("text", "").strip()
                
                reply += f"**{i}. {name}**"
                if duration:
                    reply += f" ({duration})"
                reply += "\n"
                
                if summary:
                    reply += f"   📝 *{summary[:120]}"
                    if len(summary) > 120:
                        reply += "...*"
                    else:
                        reply += "*"
                
                reply += "\n"
            
            reply += "\n💡 **Gợi ý:** Bạn có thể hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour hoặc hỏi về: giá cả, lịch trình, chỗ ở, ẩm thực..."
        
        else:
            reply = "Hiện chưa có thông tin tour trong hệ thống. Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp."
    
    # =========== OPENAI CHAT ===========
    elif client is not None and not is_list_request:
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
    
    # =========== FALLBACK RESPONSE GENERATION ===========
    if not reply:
        if top_results:
            if is_list_request:
                # Should have been handled above, but as backup
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                names_u = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            
            elif requested_field == "tour_name":
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                names_u = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            
            elif requested_field == "accommodation" and tour_indices:
                # Special formatting for accommodation
                parts = []
                for ti in tour_indices:
                    # Get tour name
                    tour_name = None
                    for m in MAPPING:
                        p = m.get("path", "")
                        if f"[{ti}]" in p and ".tour_name" in p:
                            tour_name = m.get("text", "")
                            break
                    
                    # Get accommodation text for this tour
                    acc_texts = []
                    for score, m in top_results:
                        if f"[{ti}]" in m.get("path", ""):
                            acc_texts.append(m.get("text", ""))
                    
                    # Also check directly from mapping
                    if not acc_texts:
                        for m2 in MAPPING:
                            if f"[{ti}]" in m2.get("path", "") and ".accommodation" in m2.get("path", ""):
                                acc_texts.append(m2.get("text", ""))
                    
                    if acc_texts:
                        label = f'🏨 **Tour "{tour_name}"**' if tour_name else f"Tour"
                        parts.append(f"{label}:\n" + "\n".join(f"   • {txt}" for txt in acc_texts))
                
                if parts:
                    reply = "**Thông tin chỗ ở:**\n\n" + "\n\n".join(parts)
                    
                    # Add helpful note for 1-day tours
                    if any("1 ngày" in p.lower() for p in parts):
                        reply += "\n\n💡 *Lưu ý: Tour 1 ngày thường không bao gồm chỗ ở qua đêm. Nếu bạn cần lưu trú, vui lòng liên hệ để được tư vấn thêm.*"
                else:
                    reply = "Thông tin chỗ ở đang được cập nhật. Vui lòng liên hệ hotline 0332510486 để biết thêm chi tiết về chỗ nghỉ."
            
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
                        
                        # Special formatting for different fields
                        if requested_field == "includes":
                            parts.append(f"**{label} - Lịch trình chi tiết:**\n" + "\n".join(f"   • {t}" for t in field_passages))
                        elif requested_field == "price":
                            parts.append(f"**{label} - Giá tour:**\n" + "\n".join(f"   💰 {t}" for t in field_passages))
                        elif requested_field == "duration":
                            parts.append(f"**{label} - Thời gian:**\n" + "\n".join(f"   ⏱️ {t}" for t in field_passages))
                        else:
                            parts.append(f"**{label}:**\n" + "\n".join(f"   • {t}" for t in field_passages))
                
                if parts:
                    reply = "\n\n".join(parts)
                else:
                    snippets = "\n\n".join([f"• {m.get('text')}" for _, m in top_results[:5]])
                    reply = f"**Thông tin liên quan:**\n\n{snippets}"
            
            else:
                snippets = "\n\n".join([f"• {m.get('text')}" for _, m in top_results[:5]])
                reply = f"**Thông tin nội bộ liên quan:**\n\n{snippets}"
        
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan. Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp."
    
    # =========== VALIDATE DURATION TO AVOID INCORRECT INFO ===========
    # Check if reply contains unrealistic duration (like "5 ngày 4 đêm")
    if reply and ("ngày" in reply or "đêm" in reply):
        # re đã import ở global scope

        
        # Tìm tất cả các pattern duration trong reply
        duration_patterns = [
            r'(\d+)\s*ngày\s*(\d+)\s*đêm',
            r'(\d+)\s*ngày',
            r'(\d+)\s*đêm'
        ]
        
        for pattern in duration_patterns:
            matches = list(re.finditer(pattern, reply))
            for match in matches:
                try:
                    if match.lastindex == 2:  # "X ngày Y đêm"
                        days = int(match.group(1))
                        nights = int(match.group(2))
                        
                        # Kiểm tra tính hợp lý: tour du lịch thường days = nights hoặc days = nights + 1
                        # Và không quá 7 ngày cho tour thông thường
                        if days > 7 or nights > 7 or abs(days - nights) > 1:
                            logger.warning(f"⚠️ Unrealistic duration detected: {days} ngày {nights} đêm")
                            # Thay thế chỉ phần duration không hợp lý
                            old_duration = match.group(0)
                            new_duration = "thời gian phù hợp"
                            reply = reply.replace(old_duration, new_duration)
                            
                    elif match.lastindex == 1:  # "X ngày" hoặc "Y đêm"
                        num = int(match.group(1))
                        if num > 7:  # Quá dài cho tour thông thường
                            logger.warning(f"⚠️ Unrealistic duration detected: {num}")
                            old_duration = match.group(0)
                            new_duration = "thời gian phù hợp"
                            reply = reply.replace(old_duration, new_duration)
                            
                except (ValueError, IndexError):
                    continue
    
    # Nếu sau validation mà reply bị thay đổi nhiều, kiểm tra lại
    if "thời gian phù hợp" in reply and "tour" in user_message.lower():
        # Đảm bảo reply vẫn có ý nghĩa
        if "Thông tin thời gian tour" not in reply:
            reply = "Thông tin thời gian tour đang được cập nhật. Vui lòng liên hệ hotline 0332510486 để biết lịch trình cụ thể."
    
        return jsonify({
        "reply": reply, 
        "sources": [m for _, m in top_results],
        "context": {
            "tour_indices": tour_indices,
            "session_id": session_id,
            "last_tour_name": context.last_tour_name,
            "user_preferences": context.user_preferences,
            "suggested_next": get_suggested_questions(tour_indices, requested_field)
        }
    })

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
                    
                # --- ADD-ONLY: Meta CAPI Lead (SAFE HOOK) ---
                try:
                    send_meta_lead(
                        request=request,
                        event_name="Lead",
                        phone=lead_data.get("phone"),
                        value=200000,
                        currency="VND",
                        content_name=lead_data.get("action_type", "Call / Consult")
                    )
                except Exception as e:
                    logger.warning(f"Meta CAPI lead tracking failed: {e}")

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
        fallback_backup = False
        
        if ENABLE_FALLBACK_STORAGE:
            if not sheets_success:
                # Google Sheets failed, use fallback as primary
                fallback_success = save_lead_to_fallback_storage(lead_data)
                if fallback_success:
                    logger.info(f"Lead saved to fallback storage: {phone}")
                    lead_data["sync_method"] = "fallback_storage"
            else:
                # Google Sheets succeeded, also save to fallback for backup
                # BUT DO NOT CHANGE sync_method - keep it as google_sheets
                fallback_backup = save_lead_to_fallback_storage(lead_data)
                if fallback_backup:
                    logger.info(f"Lead also backed up to fallback storage: {phone}")

        # Determine response - FIXED: sync_method always accurate
        if sheets_success:
            return jsonify({
                "success": True,
                "message": "Lead saved successfully to Google Sheets",
                "data": {
                    "phone": phone,
                    "timestamp": lead_data["timestamp"],
                    "sync_method": "google_sheets"  # Always google_sheets when successful
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
                    "sync_method": "fallback_storage"  # Always fallback_storage when primary
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

# =========== TRACK CALL BUTTON CLICKS - ENHANCED FOR META CAPI ===========
@app.route('/api/track-call', methods=['POST', 'OPTIONS'])
def track_call_event():
    """
    Enhanced endpoint for tracking call button clicks with proper Meta CAPI integration
    Tương thích với tracking script hiện tại từ frontend
    """
    try:
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', 'https://www.rubywings.vn')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
        
        # Process POST request
        data = request.get_json() or {}
        logger.info(f"Call button clicked: {data.get('phone', 'unknown')} - {data.get('call_type')}")
        
        # Gọi Meta CAPI với đầy đủ tham số mới (FIXED VERSION)
        try:
            from meta_capi import send_meta_call_button
            
            # Lấy user_agent từ frontend hoặc request
            user_agent = data.get('user_agent')
            if not user_agent:
                user_agent = request.headers.get('User-Agent')
            
            # Gọi hàm đã fix
            send_meta_call_button(
                request=request,
                page_url=data.get('page_url'),
                user_agent=user_agent,
                phone=data.get('phone'),
                call_type=data.get('call_type', 'regular'),
                fbp=data.get('fbp'),
                fbc=data.get('fbc'),
                event_id=data.get('event_id'),
                pixel_id=data.get('pixel_id'),
                event_name=data.get('event_name', 'CallButtonClick'),
                value=data.get('value', 150000)
            )
        except Exception as e:
            logger.warning(f"Meta CAPI call tracking failed: {e}")
        
        # Log vào file riêng (giữ nguyên chức năng cũ)
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "call_button_click",
                "data": {
                    "phone": data.get('phone'),
                    "call_type": data.get('call_type'),
                    "page_url": data.get('page_url')
                }
            }
            
            # Lưu vào file log
            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            log_file = os.path.join(logs_dir, f"call_clicks_{datetime.utcnow().strftime('%Y-%m-%d')}.json")
            
            logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save call log: {e}")
        
        # Thêm CORS headers cho POST response
        response = jsonify({
            "success": True, 
            "message": "Call event tracked successfully",
            "meta_capi_sent": True
        })
        response.headers.add('Access-Control-Allow-Origin', 'https://www.rubywings.vn')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    except Exception as e:
        logger.error(f"Track call error: {e}")
        response = jsonify({
            "success": False,
            "error": str(e)
        })
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', 'https://www.rubywings.vn')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

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
                "index": "loaded" if (INDEX is not None or os.path.exists(FAISS_INDEX_PATH)) else "not_loaded"
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
                # Load existing mapping if available
        if os.path.exists(FAISS_MAPPING_PATH):
            try:
                with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                    file_map = json.load(f)
                if file_map:
                    MAPPING[:] = file_map
                    FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    index_tour_names()
                    build_tours_db()  # <--- THÊM DÒNG NÀY
                    logger.info("Mapping loaded from disk")
                    logger.info(f"✅ Built tours database from disk: {len(TOURS_DB)} tours")
            except Exception as e:
                logger.warning(f"Could not load mapping from disk: {e}")
                # Fallback: build from knowledge
                index_tour_names()
                build_tours_db()
        else:
            # No mapping file, build from knowledge
            index_tour_names()
            build_tours_db()
        
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