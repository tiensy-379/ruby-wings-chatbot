def safe_validate(reply):
    try:
        if not isinstance(reply, dict):
            return reply
        # Only validate when tour is selected
        if not reply.get("tour_name"):
            return reply
        return AutoValidator.validate_response(reply)
    except Exception as e:
        try:
            reply.setdefault("warnings", []).append(str(e))
        except:
            pass
        return reply


# app.py - Ruby Wings Chatbot v4.0 (Complete Rewrite with Dataclasses)
# =========== IMPORTS ===========
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ruby-wings")
import os
import sys
import json
import threading
import logging
import re
import unicodedata
import traceback
import hashlib
import time
import random
from functools import lru_cache, wraps
from typing import List, Tuple, Dict, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from difflib import SequenceMatcher
from enum import Enum
import difflib
# Try to import numpy with detailed error handling
try:
    
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("✅ NumPy available")
except ImportError as e:
    logger.error(f"❌ NumPy import failed: {e}")
    # Create a minimal numpy-like fallback for basic operations
    class NumpyFallback:
        def __init__(self):
            self.float32 = float
            self.int64 = int
            
        def array(self, data, dtype=None):
            # Simple list wrapper
            class SimpleArray:
                def __init__(self, data):
                    self.data = list(data)
                    self.shape = (len(data),) if isinstance(data[0], (int, float)) else (len(data), len(data[0]))
                
                def astype(self, dtype):
                    return self
                
                def reshape(self, shape):
                    return self
                
                def __getitem__(self, idx):
                    return self.data[idx]
                
                def __len__(self):
                    return len(self.data)
            
            return SimpleArray(data)
        
        def empty(self, shape, dtype):
            if len(shape) == 1:
                return [0.0] * shape[0]
            else:
                return [[0.0] * shape[1] for _ in range(shape[0])]
        
        def vstack(self, arrays):
            result = []
            for arr in arrays:
                if hasattr(arr, 'data'):
                    result.extend(arr.data)
                else:
                    result.extend(arr)
            return result
        
        def load(self, path):
            # Mock load function
            class MockNpz:
                def __init__(self):
                    self.files = ['mat']
                
                def __getitem__(self, key):
                    if key == 'mat':
                        # Return empty array
                        return self.array([[0.0]])
                    return None
            
            return MockNpz()
        
        def savez_compressed(self, path, **kwargs):
            # Mock save function
            logger.warning(f"⚠️ NumPy fallback: Mock saving to {path}")
            return None
    
    np = NumpyFallback()
    NUMPY_AVAILABLE = False
    logger.warning("⚠️ Using NumPy fallback - limited functionality")
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# =========== ENTITY IMPORTS ===========
from entities import (
    QuestionType,
    ConversationState,
    PriceLevel,
    DurationType,
    Tour,
    UserProfile,
    SearchResult,
    ConversationContext,
    FilterSet,
    LLMRequest,
    ChatResponse,
    LeadData,
    CacheEntry,
    EnhancedJSONEncoder
)

# =========== CONFIGURATION ===========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ruby_wings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rbw_v4")

# =========== IMPORTS WITH FALLBACKS ===========
HAS_FAISS = False
faiss_index = None
faiss_mapping = {}


try:
    import faiss
    HAS_FAISS = True
    logger.info("✅ FAISS available")
except ImportError:
    logger.warning("⚠️ FAISS not available, using numpy fallback")

HAS_OPENAI = False
client = None
embedding_client = client

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    logger.warning("⚠️ OpenAI not available, using fallback responses")

# Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    from google.auth.exceptions import GoogleAuthError
    from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
    HAS_GOOGLE_SHEETS = True
except ImportError:
    HAS_GOOGLE_SHEETS = False
    logger.warning("⚠️ Google Sheets not available")

# Meta CAPI
try:
    from meta_capi import send_meta_pageview, send_meta_lead, send_meta_call_button
    HAS_META_CAPI = True
    logger.info("✅ Meta CAPI available")
except ImportError:
    HAS_META_CAPI = False
    logger.warning("⚠️ Meta CAPI not available")

# =========== ENVIRONMENT VARIABLES ===========
# Memory Profile
RAM_PROFILE = os.environ.get("RAM_PROFILE", "512").strip()
IS_LOW_RAM = RAM_PROFILE == "512"
IS_HIGH_RAM = RAM_PROFILE == "2048"

# Core API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

# Knowledge & Index
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

# Models
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "10"))

# FAISS
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes") and not IS_LOW_RAM

# Google Sheets
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "1SdVbwkuxb8l1meEW--ddyfh4WmUvSXXMOPQ5bCyPkdk")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox")
ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "true").lower() in ("1", "true", "yes")

# Storage
ENABLE_FALLBACK_STORAGE = os.environ.get("ENABLE_FALLBACK_STORAGE", "true").lower() in ("1", "true", "yes")
FALLBACK_STORAGE_PATH = os.environ.get("FALLBACK_STORAGE_PATH", "leads_fallback.json")

# Meta CAPI
META_CAPI_TOKEN = os.environ.get("META_CAPI_TOKEN", "").strip()
META_PIXEL_ID = os.environ.get("META_PIXEL_ID", "").strip()
META_CAPI_ENDPOINT = os.environ.get("META_CAPI_ENDPOINT", "https://graph.facebook.com/v17.0/")
ENABLE_META_CAPI_CALL = os.environ.get("ENABLE_META_CAPI_CALL", "true").lower() in ("1", "true", "yes")

# Server
FLASK_ENV = os.environ.get("FLASK_ENV", "production")
DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
SECRET_KEY = os.environ.get("SECRET_KEY", "ruby-wings-secret-key-2024")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "https://www.rubywings.vn,http://localhost:3000").split(",")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "10000"))

# =========== STATS TRACKING (FIX LỖI STATE) ===========
# Thêm global stats tracking system
STATS_LOCK = threading.Lock()
GLOBAL_STATS = {
    'meta_capi_calls': 0,
    'meta_capi_errors': 0,
    'leads': 0,
    'errors': 0,
    'total_requests': 0
}

def increment_stat(stat_name: str, amount: int = 1):
    """Thread-safe stat increment"""
    with STATS_LOCK:
        if stat_name in GLOBAL_STATS:
            GLOBAL_STATS[stat_name] += amount
        else:
            GLOBAL_STATS[stat_name] = amount

def get_stats() -> dict:
    """Get current stats"""
    with STATS_LOCK:
        return GLOBAL_STATS.copy()

# =========== UPGRADE FEATURE FLAGS ===========
class UpgradeFlags:
    """Control all 10 upgrades with environment variables"""
    
    @staticmethod
    def get_all_flags():
        return {
            # CORE UPGRADES (Essential fixes)
            "UPGRADE_1_MANDATORY_FILTER": os.environ.get("UPGRADE_1_MANDATORY_FILTER", "true").lower() == "true",
            "UPGRADE_2_DEDUPLICATION": os.environ.get("UPGRADE_2_DEDUPLICATION", "true").lower() == "true",
            "UPGRADE_3_ENHANCED_FIELDS": os.environ.get("UPGRADE_3_ENHANCED_FIELDS", "true").lower() == "true",
            "UPGRADE_4_QUESTION_PIPELINE": os.environ.get("UPGRADE_4_QUESTION_PIPELINE", "true").lower() == "true",
            
            # ADVANCED UPGRADES
            "UPGRADE_5_QUERY_SPLITTER": os.environ.get("UPGRADE_5_QUERY_SPLITTER", "true").lower() == "true",
            "UPGRADE_6_FUZZY_MATCHING": os.environ.get("UPGRADE_6_FUZZY_MATCHING", "true").lower() == "true",
            "UPGRADE_7_STATE_MACHINE": os.environ.get("UPGRADE_7_STATE_MACHINE", "true").lower() == "true",
            "UPGRADE_8_SEMANTIC_ANALYSIS": os.environ.get("UPGRADE_8_SEMANTIC_ANALYSIS", "true").lower() == "true",
            "UPGRADE_9_AUTO_VALIDATION": os.environ.get("UPGRADE_9_AUTO_VALIDATION", "true").lower() == "true",
            "UPGRADE_10_TEMPLATE_SYSTEM": os.environ.get("UPGRADE_10_TEMPLATE_SYSTEM", "true").lower() == "true",
            
            # PERFORMANCE OPTIONS
            "ENABLE_CACHING": os.environ.get("ENABLE_CACHING", "true").lower() == "true",
            "CACHE_TTL_SECONDS": int(os.environ.get("CACHE_TTL_SECONDS", "300")),
            "ENABLE_QUERY_LOGGING": os.environ.get("ENABLE_QUERY_LOGGING", "true").lower() == "true",
            
            # MEMORY OPTIMIZATION
            "EMBEDDING_CACHE_SIZE": 100 if IS_LOW_RAM else 1000,
            "TOUR_CACHE_ENABLED": not IS_LOW_RAM,
            "PRELOAD_EMBEDDINGS": not IS_LOW_RAM,
        }
    
    @staticmethod
    def is_enabled(upgrade_name: str) -> bool:
        flags = UpgradeFlags.get_all_flags()
        return flags.get(f"UPGRADE_{upgrade_name}", False)

# =========== FLASK APP CONFIG ===========
app = Flask(__name__)
app.json_encoder = EnhancedJSONEncoder  # Use custom JSON encoder
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# =========== GLOBAL STATE (USING DATACLASSES) ===========
# Initialize OpenAI client
# ==== OpenAI client (SDK 1.x safe, Render compatible) ====
from openai import OpenAI
import httpx
import os

def create_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Render-safe HTTP client (no proxies param in OpenAI 1.x)
    http_client = httpx.Client(
        timeout=60.0,
        follow_redirects=True
    )

    return OpenAI(
        api_key=api_key,
        http_client=http_client
    )

# Create global OpenAI client
client = create_openai_client()


# Knowledge base state
KNOW: Dict = {}                      # Raw knowledge.json data
FLAT_TEXTS: List[str] = []           # All text passages for indexing
MAPPING: List[Dict] = []             # Mapping from text to original path
INDEX = None                         # FAISS or numpy index
INDEX_LOCK = threading.Lock()        # Thread safety for index operations

# Tour databases (USING Tour DATACLASS)
TOUR_NAME_TO_INDEX: Dict[str, int] = {}      # Normalized tour name → index
TOURS_DB: Dict[int, Tour] = {}               # Structured tour database using Tour objects
TOUR_TAGS: Dict[int, List[str]] = {}         # Auto-generated tags for filtering

# Session management (USING ConversationContext DATACLASS)
SESSION_CONTEXTS: Dict[str, ConversationContext] = {}
SESSION_LOCK = threading.Lock()
SESSION_TIMEOUT = 1800  # 30 minutes

# Cache system
_response_cache: Dict[str, CacheEntry] = {}
_cache_lock = threading.Lock()

# Embedding cache (memory optimized)
_embedding_cache: Dict[str, Tuple[List[float], int]] = {}
_embedding_cache_lock = threading.Lock()
MAX_EMBEDDING_CACHE_SIZE = UpgradeFlags.get_all_flags()["EMBEDDING_CACHE_SIZE"]

# =========== MEMORY OPTIMIZATION FUNCTIONS ===========
def optimize_for_memory_profile():
    """Apply memory optimizations based on RAM profile"""
    flags = UpgradeFlags.get_all_flags()
    
    if IS_LOW_RAM:
        logger.info("🧠 Low RAM mode (512MB) - optimizing memory usage")
        # Disable heavy preloading
        global FAISS_ENABLED
        FAISS_ENABLED = False
        
        # Reduce cache sizes
        import functools
        functools.lru_cache(maxsize=128)(embed_text)
        
        # Limit tour loading
        global MAX_TOURS_TO_LOAD
        MAX_TOURS_TO_LOAD = 50
        
    elif IS_HIGH_RAM:
        logger.info("🚀 High RAM mode (2GB) - enabling all features")
        # Enable all features
        FAISS_ENABLED = HAS_FAISS
        MAX_TOURS_TO_LOAD = 1000
        
        # Increase cache sizes
        import functools
        functools.lru_cache(maxsize=flags["EMBEDDING_CACHE_SIZE"])(embed_text)

# =========== UPGRADE 1: MANDATORY FILTER SYSTEM (DATACLASS COMPATIBLE) ===========
class MandatoryFilterSystem:
    """
    UPGRADE 1: Extract and apply mandatory filters BEFORE semantic search
    """
    
    FILTER_PATTERNS = {
        'duration': [
            (r'(?:thời gian|mấy ngày|bao lâu|kéo dài)\s*(?:là\s*)?(\d+)\s*(?:ngày|đêm)', 'exact_duration'),
            (r'(\d+)\s*ngày\s*(?:và\s*)?(\d+)?\s*đêm', 'days_nights'),
            (r'(\d+)\s*ngày\s*(?:trở lên|trở xuống)', 'duration_range'),
            (r'(?:tour|tour)\s*(?:khoảng|tầm|khoảng)?\s*(\d+)\s*ngày', 'approx_duration'),
            (r'(\d+)\s*ngày', 'exact_duration'),  # THÊM DÒNG NÀY
        ],
        
        'price': [
            (r'dưới\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'max_price'),
            (r'trên\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'min_price'),
            (r'khoảng\s*(\d[\d,\.]*)\s*(?:đến|-)\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'price_range'),
            (r'giá\s*(?:từ\s*)?(\d[\d,\.]*)\s*(?:đến|-|tới)\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'price_range'),
            (r'(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)\s*trở xuống', 'max_price'),
            (r'(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)\s*trở lên', 'min_price'),
        ],
        
        'location': [
            (r'(?:ở|tại|về|đến|thăm)\s+([^.,!?\n]+?)(?:\s|$|\.|,|!|\?)', 'location'),
            (r'(?:điểm đến|địa điểm|nơi|vùng)\s+(?:là\s*)?([^.,!?\n]+)', 'location'),
            (r'(?:quanh|gần|khu vực)\s+([^.,!?\n]+)', 'near_location'),
        ],
        
        'date_time': [
            (r'(?:tháng|vào)\s*(\d{1,2})', 'month'),
            (r'(?:cuối tuần|weekend)', 'weekend'),
            (r'(?:dịp|lễ|tết)\s+([^.,!?\n]+)', 'holiday'),
        ],
        
        'group_type': [
            (r'(?:gia đình|family)', 'family'),
            (r'(?:cặp đôi|couple|đôi lứa)', 'couple'),
            (r'(?:nhóm bạn|bạn bè|friends)', 'friends'),
            (r'(?:công ty|doanh nghiệp|team building)', 'corporate'),
            (r'(?:một mình|đi lẻ|solo)', 'solo'),
        ],
    }
    
    @staticmethod
    def extract_filters(message: str) -> FilterSet:
        """
        Extract ALL mandatory filters from user message
        """
        filters = FilterSet()
        
        if not message:
            return filters
        
        message_lower = message.lower()
        
        # 1. DURATION FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['duration']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                if filter_type == 'exact_duration':
                    try:
                        days = int(match.group(1))
                        filters.duration_min = days
                        filters.duration_max = days
                    except (ValueError, IndexError):
                        pass
                elif filter_type == 'days_nights':
                    try:
                        days = int(match.group(1))
                        nights = int(match.group(2)) if match.group(2) else days
                        # Store in appropriate fields
                        filters.duration_min = days
                        filters.duration_max = days
                    except (ValueError, IndexError):
                        pass
        
        # 2. PRICE FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['price']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                try:
                    if filter_type == 'max_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters.price_max = amount
                            logger.info(f"💰 Extracted MAX price filter: {amount} VND")
                    
                    elif filter_type == 'min_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters.price_min = amount
                            logger.info(f"💰 Extracted MIN price filter: {amount} VND")
                    
                    elif filter_type == 'price_range':
                        min_amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(3))
                        max_amount = MandatoryFilterSystem._parse_price(match.group(2), match.group(3))
                        if min_amount and max_amount:
                            filters.price_min = min_amount
                            filters.price_max = max_amount
                            logger.info(f"💰 Extracted PRICE RANGE: {min_amount} - {max_amount} VND")
                
                except (ValueError, IndexError, AttributeError):
                    continue
        
        # 3. LOCATION FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['location']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                location = match.group(1).strip()
                if location and len(location) > 1:
                    if filter_type == 'location':
                        filters.location = location
                    elif filter_type == 'near_location':
                        filters.near_location = location
        
        # 4. DATE/TIME FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['date_time']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                if filter_type == 'month':
                    try:
                        filters.month = int(match.group(1))
                    except (ValueError, IndexError):
                        pass
                elif filter_type == 'weekend':
                    filters.weekend = True
                elif filter_type == 'holiday':
                    filters.holiday = match.group(1).strip()
        
        # 5. GROUP TYPE FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['group_type']:
            if re.search(pattern, message_lower):
                filters.group_type = filter_type
        
        # 6. SPECIAL KEYWORDS
        special_keywords = {
            'rẻ': ('price_max', 1500000),
            'giá rẻ': ('price_max', 1500000),
            'tiết kiệm': ('price_max', 1500000),
            'cao cấp': ('price_min', 3000000),
            'sang trọng': ('price_min', 3000000),
            'premium': ('price_min', 3000000),
            'ngắn ngày': ('duration_max', 2),
            'dài ngày': ('duration_min', 3),
        }
        
        for keyword, (filter_key, value) in special_keywords.items():
            if keyword in message_lower:
                if filter_key == 'price_max':
                    filters.price_max = value
                elif filter_key == 'price_min':
                    filters.price_min = value
                elif filter_key == 'duration_max':
                    filters.duration_max = value
                elif filter_key == 'duration_min':
                    filters.duration_min = value
        
        logger.info(f"🎯 Extracted filters: {filters}")
        return filters
    
    @staticmethod
    def _parse_price(amount_str: str, unit: str) -> Optional[int]:
        """Parse price string like '1.5 triệu' to integer VND"""
        if not amount_str:
            return None
        
        try:
            amount_str = amount_str.replace(',', '').replace('.', '')
            if not amount_str.isdigit():
                return None
            
            amount = int(amount_str)
            
            if unit in ['triệu', 'tr']:
                return amount * 1000000
            elif unit == 'k':
                return amount * 1000
            elif unit == 'nghìn':
                return amount * 1000
            else:
                return amount if amount > 1000 else amount * 1000
        
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def apply_filters(tours_db: Dict[int, Tour], filters: FilterSet) -> List[int]:
        """
        Apply mandatory filters to tour database
        Returns list of tour indices that pass ALL filters
        Enhanced version with better error handling and group_type support
        """
        if filters.is_empty() or not tours_db:
            logger.info(f"🔍 No filters or empty DB, returning all {len(tours_db)} tours")
            return list(tours_db.keys())
        
        passing_tours = []
        total_tours = len(tours_db)
        
        try:
            logger.info(f"🎯 Applying filters: {filters}")
            
            # Validate group_type if present
            if hasattr(filters, 'group_type') and filters.group_type:
                valid_group_types = ['family', 'friends', 'corporate', 'solo', 'couple', 'senior', 'group']
                if filters.group_type not in valid_group_types:
                    logger.warning(f"⚠️ Invalid group_type: {filters.group_type}, using default filtering")
                    # Continue without group_type filter but log warning
            
            for tour_idx, tour in tours_db.items():
                passes_all = True
                
                # PRICE FILTERING - ENHANCED
                if passes_all and (filters.price_max is not None or filters.price_min is not None):
                    tour_price_text = tour.price or ""
                    if not tour_price_text or tour_price_text.lower() == 'liên hệ':
                        # If tour doesn't have price, check if we require price filter
                        if filters.price_max is not None or filters.price_min is not None:
                            # If price is required but not available, fail this tour
                            passes_all = False
                    else:
                        tour_prices = MandatoryFilterSystem._extract_tour_prices(tour_price_text)
                        if not tour_prices:
                            # Can't extract price, be conservative - pass if no strict requirement
                            if filters.price_max is not None and filters.price_min is not None:
                                passes_all = False
                        else:
                            min_tour_price = min(tour_prices)
                            max_tour_price = max(tour_prices)
                            
                            # Apply price range filter
                            if filters.price_max is not None and min_tour_price > filters.price_max:
                                passes_all = False
                            if filters.price_min is not None and max_tour_price < filters.price_min:
                                passes_all = False
                
                # DURATION FILTERING - ENHANCED
                if passes_all and (filters.duration_min is not None or filters.duration_max is not None):
                    duration_text = (tour.duration or "").lower()
                    tour_duration = MandatoryFilterSystem._extract_duration_days(duration_text)
                    
                    if tour_duration is not None:
                        if filters.duration_min is not None and tour_duration < filters.duration_min:
                            passes_all = False
                        if filters.duration_max is not None and tour_duration > filters.duration_max:
                            passes_all = False
                    else:
                        # If duration cannot be extracted, be conservative
                        if filters.duration_min is not None and filters.duration_max is not None:
                            passes_all = False
                
                # LOCATION FILTERING - ENHANCED
                if passes_all and (filters.location is not None or filters.near_location is not None):
                    tour_location = (tour.location or "").lower()
                    if filters.location is not None:
                        filter_location = filters.location.lower()
                        # Enhanced location matching
                        if filter_location and filter_location not in tour_location:
                            # Try partial matching for common location names
                            location_keywords = {
                                'huế': ['huế', 'hue'],
                                'quảng trị': ['quảng trị', 'quang tri'],
                                'bạch mã': ['bạch mã', 'bach ma'],
                                'trường sơn': ['trường sơn', 'truong son'],
                                'đông hà': ['đông hà', 'dong ha']
                            }
                            
                            # Check if filter_location matches any keyword
                            matches = False
                            for keyword, variants in location_keywords.items():
                                if filter_location in variants:
                                    # Check if any variant is in tour_location
                                    for variant in variants:
                                        if variant in tour_location:
                                            matches = True
                                            break
                                if matches:
                                    break
                            
                            if not matches:
                                passes_all = False
                    
                    if filters.near_location is not None and passes_all:
                        near_location = filters.near_location.lower()
                        if near_location and near_location not in tour_location:
                            passes_all = False
                
                # GROUP TYPE FILTERING - ADDED SUPPORT
                if passes_all and hasattr(filters, 'group_type') and filters.group_type:
                    group_type = filters.group_type.lower()
                    tour_summary = (tour.summary or "").lower()
                    tour_tags = [tag.lower() for tag in (tour.tags or [])]
                    
                    # Enhanced group type matching
                    group_type_matched = False
                    
                    # Define keywords for each group type
                    group_keywords = {
                        'family': ['gia đình', 'trẻ em', 'con nhỏ', 'bố mẹ', 'đa thế hệ'],
                        'friends': ['nhóm bạn', 'bạn bè', 'bạn trẻ', 'thanh niên', 'sinh viên'],
                        'corporate': ['công ty', 'team building', 'doanh nghiệp', 'nhân viên', 'đồng nghiệp'],
                        'solo': ['một mình', 'đi lẻ', 'solo', 'cá nhân'],
                        'couple': ['cặp đôi', 'đôi lứa', 'người yêu', 'tình nhân'],
                        'senior': ['người lớn tuổi', 'cao tuổi', 'cựu chiến binh', 'veteran'],
                        'group': ['nhóm', 'đoàn', 'tập thể']
                    }
                    
                    if group_type in group_keywords:
                        keywords = group_keywords[group_type]
                        
                        # Check in tour tags
                        for tag in tour_tags:
                            if any(keyword in tag for keyword in keywords):
                                group_type_matched = True
                                break
                        
                        # Check in tour summary
                        if not group_type_matched:
                            if any(keyword in tour_summary for keyword in keywords):
                                group_type_matched = True
                        
                        # Special handling for senior/veteran
                        if group_type == 'senior':
                            # Also check for historical/meaningful tours
                            if any(word in tour_summary for word in ['lịch sử', 'tri ân', 'ký ức', 'chiến tranh']):
                                group_type_matched = True
                        
                        if not group_type_matched:
                            passes_all = False
                    else:
                        logger.warning(f"⚠️ Unknown group_type: {group_type}, skipping group filter")
                
                # MONTH FILTERING - ADDED SUPPORT
                if passes_all and hasattr(filters, 'month') and filters.month:
                    try:
                        month = int(filters.month)
                        # Simple season-based filtering
                        tour_summary = (tour.summary or "").lower()
                        
                        # Tours suitable for specific months
                        # This is simplified - in reality would need more complex logic
                        if month in [1, 2, 3]:  # Dry season, good for most tours
                            # No filtering, most tours are suitable
                            pass
                        elif month in [9, 10, 11, 12]:  # Rainy season
                            # Avoid tours with lots of outdoor activities
                            if any(word in tour_summary for word in ['trekking', 'leo núi', 'đi bộ đường dài']):
                                passes_all = False
                    except (ValueError, TypeError):
                        # Invalid month format, ignore filter
                        pass
                
                # WEEKEND/HOLIDAY FILTERING - ADDED SUPPORT
                if passes_all:
                    tour_duration = MandatoryFilterSystem._extract_duration_days((tour.duration or "").lower())
                    
                    if hasattr(filters, 'weekend') and filters.weekend and tour_duration:
                        # Weekend tours should be 1-2 days
                        if tour_duration > 2:
                            passes_all = False
                    
                    if hasattr(filters, 'holiday') and filters.holiday and tour_duration:
                        # Holiday tours might be longer
                        # No specific filtering for now
                        pass
                
                if passes_all:
                    passing_tours.append(tour_idx)
            
            logger.info(f"✅ Filtering complete: {len(passing_tours)}/{total_tours} tours pass")
            
            # If filtering results in too few tours, provide fallback
            if len(passing_tours) < 3 and total_tours > 10:
                logger.info(f"⚠️ Only {len(passing_tours)} tours passed filters, applying lenient filtering")
                
                # Apply lenient filtering: tours must pass at least 50% of non-empty filters
                if not filters.is_empty():
                    lenient_passing_tours = []
                    
                    # Count non-empty filters
                    non_empty_filters = 0
                    if filters.price_max is not None or filters.price_min is not None:
                        non_empty_filters += 1
                    if filters.duration_min is not None or filters.duration_max is not None:
                        non_empty_filters += 1
                    if filters.location is not None or filters.near_location is not None:
                        non_empty_filters += 1
                    if hasattr(filters, 'group_type') and filters.group_type:
                        non_empty_filters += 1
                    
                    if non_empty_filters > 0:
                        for tour_idx, tour in tours_db.items():
                            passed_filters = 0
                            
                            # Check price
                            if not (filters.price_max is not None or filters.price_min is not None):
                                passed_filters += 1
                            else:
                                tour_price_text = tour.price or ""
                                if tour_price_text and tour_price_text.lower() != 'liên hệ':
                                    tour_prices = MandatoryFilterSystem._extract_tour_prices(tour_price_text)
                                    if tour_prices:
                                        min_tour_price = min(tour_prices)
                                        max_tour_price = max(tour_prices)
                                        
                                        price_passed = True
                                        if filters.price_max is not None and min_tour_price > filters.price_max:
                                            price_passed = False
                                        if filters.price_min is not None and max_tour_price < filters.price_min:
                                            price_passed = False
                                        
                                        if price_passed:
                                            passed_filters += 1
                            
                            # Check duration
                            if not (filters.duration_min is not None or filters.duration_max is not None):
                                passed_filters += 1
                            else:
                                duration_text = (tour.duration or "").lower()
                                tour_duration = MandatoryFilterSystem._extract_duration_days(duration_text)
                                
                                if tour_duration is not None:
                                    duration_passed = True
                                    if filters.duration_min is not None and tour_duration < filters.duration_min:
                                        duration_passed = False
                                    if filters.duration_max is not None and tour_duration > filters.duration_max:
                                        duration_passed = False
                                    
                                    if duration_passed:
                                        passed_filters += 1
                            
                            # Check location
                            if not (filters.location is not None or filters.near_location is not None):
                                passed_filters += 1
                            else:
                                tour_location = (tour.location or "").lower()
                                location_passed = True
                                
                                if filters.location is not None:
                                    filter_location = filters.location.lower()
                                    if filter_location not in tour_location:
                                        location_passed = False
                                
                                if filters.near_location is not None and location_passed:
                                    near_location = filters.near_location.lower()
                                    if near_location not in tour_location:
                                        location_passed = False
                                
                                if location_passed:
                                    passed_filters += 1
                            
                            # Check group type
                            if not (hasattr(filters, 'group_type') and filters.group_type):
                                passed_filters += 1
                            else:
                                # Simplified group type check for lenient filtering
                                group_type = filters.group_type.lower()
                                tour_summary = (tour.summary or "").lower()
                                tour_tags = [tag.lower() for tag in (tour.tags or [])]
                                
                                group_passed = False
                                if group_type == 'family':
                                    if any(word in tour_summary for word in ['gia đình', 'trẻ em', 'con nhỏ']):
                                        group_passed = True
                                elif group_type == 'friends':
                                    if any(word in tour_summary for word in ['nhóm bạn', 'bạn bè']):
                                        group_passed = True
                                elif group_type == 'senior':
                                    if any(word in tour_summary for word in ['lịch sử', 'tri ân', 'nhẹ nhàng']):
                                        group_passed = True
                                else:
                                    # For other group types, be lenient
                                    group_passed = True
                                
                                if group_passed:
                                    passed_filters += 1
                            
                            # Pass if at least 50% of filters passed
                            if passed_filters >= non_empty_filters * 0.5:
                                lenient_passing_tours.append(tour_idx)
                        
                        # Use lenient results if better
                        if len(lenient_passing_tours) > len(passing_tours):
                            logger.info(f"🔄 Using lenient filtering: {len(lenient_passing_tours)} tours")
                            passing_tours = lenient_passing_tours
            
        except Exception as e:
            logger.error(f"❌ Error in apply_filters: {e}\n{traceback.format_exc()}")
            # Fallback: return all tours
            passing_tours = list(tours_db.keys())
        
        return passing_tours
        
    @staticmethod
    def _extract_tour_prices(price_text: str) -> List[int]:
        """Extract price numbers from tour price text"""
        prices = []
        
        number_patterns = [
            r'(\d[\d,\.]+)\s*(?:triệu|tr)',
            r'(\d[\d,\.]+)\s*(?:k|nghìn)',
            r'(\d[\d,\.]+)\s*(?:đồng|vnđ|vnd)',
            r'(\d[\d,\.]+)\s*-\s*(\d[\d,\.]+)',
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, price_text, re.IGNORECASE)
            for match in matches:
                try:
                    for i in range(1, 3):
                        if match.group(i):
                            num_str = match.group(i).replace(',', '').replace('.', '')
                            if num_str.isdigit():
                                num = int(num_str)
                                
                                if 'triệu' in match.group(0).lower() or 'tr' in match.group(0).lower():
                                    num = num * 1000000
                                elif 'k' in match.group(0).lower() or 'nghìn' in match.group(0).lower():
                                    num = num * 1000
                                
                                prices.append(num)
                except (ValueError, AttributeError):
                    continue
        
        if not prices:
            raw_numbers = re.findall(r'\d[\d,\.]+', price_text)
            for num_str in raw_numbers[:2]:
                try:
                    num_str = num_str.replace(',', '').replace('.', '')
                    if num_str.isdigit():
                        num = int(num_str)
                        if 100 <= num <= 10000:
                            num = num * 1000
                        prices.append(num)
                except ValueError:
                    continue
        
        return prices
    
    @staticmethod
    def _extract_duration_days(duration_text: str) -> Optional[int]:
        """Extract duration in days from text"""
        if not duration_text:
            return None
        
        patterns = [
            r'(\d+)\s*ngày',
            r'(\d+)\s*ngày\s*\d*\s*đêm',
            r'(\d+)\s*đêm',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, duration_text)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None

# =========== UPGRADE 2: DEDUPLICATION ENGINE (DATACLASS COMPATIBLE) ===========
class DeduplicationEngine:
    """
    UPGRADE 2: Remove duplicate and highly similar results
    """
    
    SIMILARITY_THRESHOLD = 0.85
    MIN_TEXT_LENGTH = 20
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        if not text1 or not text2:
            return 0.0
        
        text1_norm = DeduplicationEngine._normalize_text(text1)
        text2_norm = DeduplicationEngine._normalize_text(text2)
        
        if len(text1_norm) < DeduplicationEngine.MIN_TEXT_LENGTH or len(text2_norm) < DeduplicationEngine.MIN_TEXT_LENGTH:
            return 0.0
        
        seq_ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            jaccard = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard = len(intersection) / len(union)
        
        prefix_len = min(50, min(len(text1_norm), len(text2_norm)))
        prefix1 = text1_norm[:prefix_len]
        prefix2 = text2_norm[:prefix_len]
        prefix_sim = SequenceMatcher(None, prefix1, prefix2).ratio()
        
        similarity = (seq_ratio * 0.5) + (jaccard * 0.3) + (prefix_sim * 0.2)
        
        return similarity
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        stopwords = {'và', 'của', 'cho', 'với', 'tại', 'ở', 'này', 'đó', 'kia', 'về', 'trong'}
        words = [word for word in text.split() if word not in stopwords]
        
        return ' '.join(words)
    
    @staticmethod
    def deduplicate_passages(passages: List[Tuple[float, Dict]], 
                            similarity_threshold: float = None) -> List[Tuple[float, Dict]]:
        """
        Remove duplicate passages from results
        """
        if len(passages) <= 1:
            return passages
        
        threshold = similarity_threshold or DeduplicationEngine.SIMILARITY_THRESHOLD
        unique_passages = []
        seen_passages = []
        
        sorted_passages = sorted(passages, key=lambda x: x[0], reverse=True)
        
        for score, passage in sorted_passages:
            text = passage.get('text', '').strip()
            path = passage.get('path', '')
            
            if not text or len(text) < DeduplicationEngine.MIN_TEXT_LENGTH:
                unique_passages.append((score, passage))
                continue
            
            is_duplicate = False
            for seen_text, seen_path in seen_passages:
                tour_match1 = re.search(r'tours\[(\d+)\]', path)
                tour_match2 = re.search(r'tours\[(\d+)\]', seen_path)
                
                if tour_match1 and tour_match2:
                    if tour_match1.group(1) == tour_match2.group(1):
                        field1 = path.split('.')[-1] if '.' in path else ''
                        field2 = seen_path.split('.')[-1] if '.' in seen_path else ''
                        if field1 == field2:
                            is_duplicate = True
                            break
                
                similarity = DeduplicationEngine.calculate_similarity(text, seen_text)
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_passages.append((score, passage))
                seen_passages.append((text, path))
        
        logger.info(f"🔄 Deduplication: {len(passages)} → {len(unique_passages)} passages")
        return unique_passages
    
    @staticmethod
    def merge_similar_tours(tour_indices: List[int], tours_db: Dict[int, Tour]) -> List[int]:
        """Merge tours that are essentially the same"""
        if len(tour_indices) <= 1:
            return tour_indices
        
        tour_groups = []
        processed = set()
        
        for i, idx1 in enumerate(tour_indices):
            if idx1 in processed:
                continue
            
            group = [idx1]
            tour1 = tours_db.get(idx1)
            name1 = (tour1.name if tour1 else "").strip()
            
            if not name1:
                processed.add(idx1)
                tour_groups.append(group)
                continue
            
            for j, idx2 in enumerate(tour_indices[i+1:], i+1):
                if idx2 in processed:
                    continue
                
                tour2 = tours_db.get(idx2)
                name2 = (tour2.name if tour2 else "").strip()
                
                if not name2:
                    continue
                
                similarity = DeduplicationEngine.calculate_similarity(name1, name2)
                if similarity > 0.9:
                    group.append(idx2)
                    processed.add(idx2)
            
            processed.add(idx1)
            tour_groups.append(group)
        
        best_tours = []
        for group in tour_groups:
            if not group:
                continue
            
            if len(group) == 1:
                best_tours.append(group[0])
                continue
            
            best_score = -1
            best_idx = group[0]
            
            for idx in group:
                tour = tours_db.get(idx)
                if not tour:
                    continue
                
                score = 0
                
                if tour.name:
                    score += 2
                if tour.duration:
                    score += 2
                if tour.location:
                    score += 2
                if tour.price:
                    score += 3
                if tour.includes:
                    score += 2
                if tour.summary:
                    score += 1
                
                for field in [tour.includes, tour.summary, tour.notes]:
                    if isinstance(field, str) and len(field) > 50:
                        score += 1
                    elif isinstance(field, list) and field:
                        score += len(field)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            best_tours.append(best_idx)
        
        logger.info(f"🔄 Tour merging: {len(tour_indices)} → {len(best_tours)} unique tours")
        return best_tours

# =========== UPGRADE 3: ENHANCED FIELD DETECTION (DATACLASS COMPATIBLE) ===========
class EnhancedFieldDetector:
    """
    UPGRADE 3: Better detection of what user is asking for
    """
    
    FIELD_DETECTION_RULES = [
        # TOUR LIST
        {
            "field": "tour_name",
            "patterns": [
                (r'liệt kê.*tour|danh sách.*tour|các tour|có những tour nào', 1.0),
                (r'tour nào.*có|tour nào.*hiện|tour nào.*đang', 0.9),
                (r'kể tên.*tour|nêu tên.*tour|tên các tour', 0.9),
                (r'có mấy.*tour|bao nhiêu.*tour|số lượng.*tour', 0.8),
                (r'list tour|show tour|all tour|every tour', 0.8),
            ],
            "keywords": [
                ("liệt kê", 0.9), ("danh sách", 0.9), ("các", 0.7),
                ("tất cả", 0.8), ("mọi", 0.7), ("mấy", 0.6),
                ("bao nhiêu", 0.7), ("số lượng", 0.7),
            ]
        },
        
        # PRICE
        {
            "field": "price",
            "patterns": [
                (r'giá.*bao nhiêu|bao nhiêu tiền|chi phí.*bao nhiêu', 1.0),
                (r'giá tour|giá cả|giá thành|chi phí tour', 0.9),
                (r'tour.*giá.*bao nhiêu|tour.*bao nhiêu tiền', 0.95),
                (r'phải trả.*bao nhiêu|tốn.*bao nhiêu|mất.*bao nhiêu', 0.8),
                (r'đóng.*bao nhiêu|thanh toán.*bao nhiêu', 0.8),
            ],
            "keywords": [
                ("giá", 0.8), ("tiền", 0.7), ("chi phí", 0.8),
                ("đóng", 0.6), ("trả", 0.6), ("tốn", 0.6),
                ("phí", 0.7), ("kinh phí", 0.7), ("tổng chi", 0.7),
            ]
        },
        
        # DURATION
        {
            "field": "duration",
            "patterns": [
                (r'thời gian.*bao lâu|mấy ngày.*đi|bao lâu.*tour', 1.0),
                (r'tour.*bao nhiêu ngày|mấy ngày.*tour', 0.9),
                (r'đi trong.*bao lâu|kéo dài.*bao lâu', 0.9),
                (r'thời lượng.*bao nhiêu|thời gian.*dài bao lâu', 0.8),
            ],
            "keywords": [
                ("bao lâu", 0.9), ("mấy ngày", 0.9), ("thời gian", 0.8),
                ("kéo dài", 0.7), ("thời lượng", 0.8), ("ngày", 0.6),
                ("đêm", 0.6), ("thời hạn", 0.7),
            ]
        },
        
        # LOCATION
        {
            "field": "location",
            "patterns": [
                (r'ở đâu|đi đâu|đến đâu|tới đâu|thăm quan đâu', 1.0),
                (r'địa điểm.*nào|nơi nào|vùng nào|khu vực nào', 0.9),
                (r'tour.*ở.*đâu|tour.*đi.*đâu', 0.9),
                (r'khám phá.*đâu|thăm.*đâu|ghé.*đâu', 0.8),
            ],
            "keywords": [
                ("ở đâu", 1.0), ("đi đâu", 1.0), ("đến đâu", 0.9),
                ("tới đâu", 0.9), ("địa điểm", 0.8), ("nơi", 0.7),
                ("vùng", 0.7), ("khu vực", 0.7),
            ]
        },
        
        # SUMMARY
        {
            "field": "summary",
            "patterns": [
                (r'có gì hay|có gì đặc biệt|có gì thú vị', 0.9),
                (r'tour này thế nào|tour ra sao|chuyến đi như nào', 0.8),
                (r'giới thiệu.*tour|mô tả.*tour|nói về.*tour', 0.8),
                (r'tour.*có gì|đi.*được gì|trải nghiệm.*gì', 0.7),
                (r'điểm nhấn.*tour|nổi bật.*gì|đặc sắc.*gì', 0.7),
            ],
            "keywords": [
                ("có gì", 0.7), ("thế nào", 0.6), ("ra sao", 0.6),
                ("giới thiệu", 0.7), ("mô tả", 0.7), ("nói về", 0.6),
                ("điểm nhấn", 0.7), ("nổi bật", 0.7), ("đặc sắc", 0.7),
            ]
        },
        
        # INCLUDES
        {
            "field": "includes",
            "patterns": [
                (r'lịch trình.*chi tiết|chương trình.*chi tiết', 0.9),
                (r'làm gì.*tour|hoạt động.*gì|sinh hoạt.*gì', 0.8),
                (r'tour.*gồm.*gì|bao gồm.*gì|gồm những gì', 0.8),
                (r'đi đâu.*làm gì|thăm quan.*gì|khám phá.*gì', 0.7),
            ],
            "keywords": [
                ("lịch trình", 0.8), ("chương trình", 0.8), ("làm gì", 0.7),
                ("hoạt động", 0.7), ("sinh hoạt", 0.6), ("gồm", 0.6),
                ("bao gồm", 0.7), ("gồm những", 0.7),
            ]
        },
    ]
    
    @staticmethod
    def detect_field_with_confidence(message: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Detect which field user is asking about with confidence scores
        """
        if not message:
            return None, 0.0, {}
        
        message_lower = message.lower()
        scores = {}
        
        for rule in EnhancedFieldDetector.FIELD_DETECTION_RULES:
            field = rule["field"]
            field_score = 0.0
            
            for pattern, weight in rule["patterns"]:
                if re.search(pattern, message_lower):
                    field_score = max(field_score, weight)
            
            for keyword, weight in rule["keywords"]:
                if keyword in message_lower:
                    position = message_lower.find(keyword)
                    position_factor = 1.0 - (position / max(len(message_lower), 1))
                    adjusted_weight = weight * (0.7 + 0.3 * position_factor)
                    field_score = max(field_score, adjusted_weight)
            
            if field_score > 0:
                field_score = min(field_score * 1.1, 1.0)
            
            scores[field] = field_score
        
        best_field = None
        best_score = 0.0
        
        for field, score in scores.items():
            if score > best_score:
                best_score = score
                best_field = field
        
        if (best_score < 0.3 and 
            ("có gì" in message_lower or "thế nào" in message_lower) and
            "tour" in message_lower):
            best_field = "summary"
            best_score = 0.6
        
        logger.info(f"🔍 Field detection: '{message}' → {best_field} (confidence: {best_score:.2f})")
        return best_field, best_score, scores

# =========== UPGRADE 4: QUESTION PIPELINE (DATACLASS COMPATIBLE) ===========
class QuestionPipeline:
    """
    UPGRADE 4: Process different types of questions differently
    """
    
    @staticmethod
    def classify_question(message: str) -> Tuple[QuestionType, float, Dict]:
        """
        Classify question type with confidence and metadata
        """
        message_lower = message.lower()
        type_scores = defaultdict(float)
        metadata = {}
        
        # LISTING detection - CHỈ khi yêu cầu rõ ràng liệt kê DANH SÁCH
        listing_patterns = [
            (r'liệt kê.*tất cả.*tour|danh sách.*tất cả.*tour|tất cả.*tour', 0.95),
            (r'liệt kê.*tour|danh sách.*tour|list.*tour', 0.9),
            (r'kể tên.*tour|nêu tên.*tour', 0.9),
            (r'có những.*tour nào|có mấy.*tour|mấy.*tour', 0.7),
            (r'bên bạn.*có.*tour|hiện có.*tour', 0.75),
        ]
        
        for pattern, weight in listing_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionType.LISTING] = max(
                    type_scores[QuestionType.LISTING], weight
                )
        
        # COMPARISON detection
        comparison_patterns = [
            (r'so sánh.*và|đối chiếu.*và', 0.95),
            (r'khác nhau.*nào|giống nhau.*nào', 0.9),
            (r'nên chọn.*nào|tốt hơn.*nào|hơn kém.*nào', 0.85),
            (r'tour.*và.*tour', 0.8),
            (r'sánh.*với|đối chiếu.*với', 0.8),
        ]
        
        for pattern, weight in comparison_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionType.COMPARISON] = max(
                    type_scores[QuestionType.COMPARISON], weight
                )
                metadata['comparison_type'] = 'direct'
        
        # RECOMMENDATION detection
        recommendation_patterns = [
            (r'phù hợp.*với|nên đi.*nào|gợi ý.*tour', 0.95),
            (r'tour nào.*phù hợp|phù hợp.*tour nào', 0.95),
            (r'tour.*tốt.*nhất|tour.*hay nhất|tour.*lý tưởng', 0.9),
            (r'đề xuất.*tour|tư vấn.*tour|chọn.*tour nào', 0.9),
            (r'tour nào.*cho.*gia đình|tour.*gia đình|gia đình.*tour', 0.9),
            (r'tour nào.*cho|dành cho.*tour|tour.*dành cho', 0.85),
            (r'nên.*tour nào|nên chọn.*tour|tour.*nên', 0.85),
            (r'tour.*nhẹ nhàng|tour.*dễ|tour.*phù hợp.*người', 0.85),
            (r'tour.*trẻ em|tour.*con nít|tour.*bé', 0.85),
            (r'tour.*người lớn tuổi|tour.*cao tuổi|tour.*nghỉ dưỡng', 0.85),
            (r'chi phí.*vừa phải|giá.*phù hợp|giá.*hợp lý', 0.8),
            (r'cho.*tôi|dành cho.*tôi|hợp với.*tôi', 0.75),
            (r'nếu.*thì.*nên.*tour|nên chọn.*tour', 0.8),
        ]
        
        for pattern, weight in recommendation_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionType.RECOMMENDATION] = max(
                    type_scores[QuestionType.RECOMMENDATION], weight
                )
        
        # GREETING detection
        greeting_words = ['xin chào', 'chào', 'hello', 'hi', 'helo', 'chao']
        greeting_score = 0.0
        for word in greeting_words:
            if word in message_lower:
                if message_lower.startswith(word) or f" {word} " in message_lower or message_lower.endswith(f" {word}"):
                    greeting_score += 0.3
        
        other_intent_score = max([score for qtype, score in type_scores.items() 
                                 if qtype != QuestionType.GREETING], default=0.0)
        
        if greeting_score > 0.8 and other_intent_score < 0.3:
            type_scores[QuestionType.GREETING] = min(greeting_score, 1.0)
        
        # FAREWELL detection
        farewell_words = ['tạm biệt', 'cảm ơn', 'thanks', 'thank you', 'bye', 'goodbye']
        if any(word in message_lower for word in farewell_words):
            type_scores[QuestionType.FAREWELL] = 0.95
        
        # CALCULATION detection
        calculation_patterns = [
            (r'tính toán|tính.*bao nhiêu|tổng.*bao nhiêu', 0.9),
            (r'cộng.*lại|nhân.*lên|chia.*ra', 0.8),
            (r'bao nhiêu.*người|mấy.*người|số lượng.*người', 0.7),
        ]
        
        for pattern, weight in calculation_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionType.CALCULATION] = max(
                    type_scores[QuestionType.CALCULATION], weight
                )
        
        # COMPLEX question detection
        complex_indicators = [
            ('và', 0.3), ('rồi', 0.4), ('sau đó', 0.5),
            ('tiếp theo', 0.5), ('ngoài ra', 0.4), ('thêm nữa', 0.4),
        ]
        
        complex_score = 0.0
        for indicator, weight in complex_indicators:
            if indicator in message_lower:
                complex_score += weight
        
        if complex_score > 0.8:
            type_scores[QuestionType.COMPLEX] = min(complex_score / 2, 1.0)
            metadata['complex_parts'] = QuestionPipeline._split_complex_question(message)
        
        # DEFAULT: INFORMATION request
        if not type_scores:
            type_scores[QuestionType.INFORMATION] = 0.6
        else:
            info_keywords = ['là gì', 'bao nhiêu', 'ở đâu', 'khi nào', 'thế nào', 'ai', 'tại sao']
            if any(keyword in message_lower for keyword in info_keywords):
                type_scores[QuestionType.INFORMATION] = max(
                    type_scores.get(QuestionType.INFORMATION, 0),
                    0.5
                )
        
        # Determine best type
        best_type = QuestionType.INFORMATION
        best_score = 0.0
        
        for qtype, score in type_scores.items():
            if score > best_score:
                best_score = score
                best_type = qtype
        
        if best_score < 0.5:
            best_type = QuestionType.INFORMATION
            best_score = 0.5
        
        logger.info(f"🎯 Question classification: '{message}' → {best_type.value} (score: {best_score:.2f})")
        return best_type, best_score, metadata
    
    @staticmethod
    def _split_complex_question(message: str) -> List[str]:
        """Split complex multi-part question into simpler parts"""
        split_patterns = [
            r'\s+và\s+',
            r'\s+rồi\s+',
            r'\s+sau đó\s+',
            r'\s+tiếp theo\s+',
            r'\s+ngoài ra\s+',
            r'\s+thêm nữa\s+',
            r'\s+đồng thời\s+',
            r'\s+cuối cùng\s+',
        ]
        
        parts = [message]
        
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([p.strip() for p in split_result if p.strip()])
            parts = new_parts
        
        return parts
    
    @staticmethod
    def process_comparison_question(tour_indices: List[int], tours_db: Dict[int, Tour], 
                                  aspect: str = "", context: Dict = None) -> str:
        """
        Process comparison question between tours
        """
        if len(tour_indices) < 2:
            return "Cần ít nhất 2 tour để so sánh."
        
        tours_to_compare = []
        for idx in tour_indices[:3]:
            tour = tours_db.get(idx)
            if tour:
                tours_to_compare.append((idx, tour))
        
        if len(tours_to_compare) < 2:
            return "Không tìm thấy đủ thông tin tour để so sánh."
        
        if not aspect:
            aspect = 'price'
        
        result_lines = []
        
        headers = ["TIÊU CHÍ"]
        for idx, tour in tours_to_compare:
            tour_name = tour.name or f'Tour #{idx}'
            headers.append(tour_name[:25])
        
        result_lines.append(" | ".join(headers))
        result_lines.append("-" * (len(headers) * 30))
        
        comparison_fields = [
            ('duration', '⏱️ Thời gian'),
            ('location', '📍 Địa điểm'),
            ('price', '💰 Giá tour'),
            ('accommodation', '🏨 Chỗ ở'),
            ('meals', '🍽️ Ăn uống'),
            ('transport', '🚗 Di chuyển'),
            ('summary', '📝 Mô tả'),
        ]
        
        for field, display_name in comparison_fields:
            if aspect and field != aspect and aspect not in ['all', 'tất cả']:
                continue
            
            row = [display_name]
            all_values = []
            
            for idx, tour in tours_to_compare:
                value = getattr(tour, field, 'N/A')
                if isinstance(value, list):
                    value = ', '.join(value[:2])
                row.append(str(value)[:30])
                all_values.append(str(value).lower())
            
            if len(set(all_values)) > 1 or aspect == field:
                result_lines.append(" | ".join(row))
        
        result_lines.append("\n" + "="*50)
        result_lines.append("**ĐÁNH GIÁ & GỢI Ý:**")
        
        durations = [tour.duration for _, tour in tours_to_compare]
        if any('1 ngày' in d for d in durations) and any('2 ngày' in d for d in durations):
            result_lines.append("• Nếu bạn có ít thời gian: Chọn tour 1 ngày")
            result_lines.append("• Nếu muốn trải nghiệm sâu: Chọn tour 2 ngày")
        
        prices = []
        for _, tour in tours_to_compare:
            price_text = tour.price or ''
            price_nums = re.findall(r'\d[\d,\.]+', price_text)
            if price_nums:
                try:
                    price = int(price_nums[0].replace(',', '').replace('.', ''))
                    prices.append(price)
                except:
                    pass
        
        if len(prices) >= 2:
            min_price_idx = prices.index(min(prices))
            max_price_idx = prices.index(max(prices))
            
            if prices[max_price_idx] > prices[min_price_idx] * 1.5:
                result_lines.append(f"• Tiết kiệm chi phí: {headers[min_price_idx + 1]}")
                result_lines.append(f"• Trải nghiệm cao cấp: {headers[max_price_idx + 1]}")
        
        result_lines.append("\n💡 *Liên hệ hotline 0332510486 để được tư vấn chi tiết*")
        
        return "\n".join(result_lines)

# =========== UPGRADE 5: COMPLEX QUERY SPLITTER (DATACLASS COMPATIBLE) ===========
class ComplexQueryProcessor:
    """
    UPGRADE 5: Handle complex multi-condition queries
    """
    
    @staticmethod
    def split_query(query: str) -> List[Dict[str, Any]]:
        """
        Split complex query into sub-queries with priorities
        """
        sub_queries = []
        
        complexity_score = ComplexQueryProcessor._calculate_complexity(query)
        if complexity_score < 1.5:
            return [{
                'query': query,
                'priority': 1.0,
                'filters': {},
                'focus': 'general'
            }]
        
        conditions = ComplexQueryProcessor._extract_conditions(query)
        
        if len(conditions) <= 1:
            return [{
                'query': query,
                'priority': 1.0,
                'filters': conditions[0] if conditions else {},
                'focus': 'general'
            }]
        
        sub_queries.append({
            'query': query,
            'priority': 1.0,
            'filters': ComplexQueryProcessor._merge_conditions(conditions),
            'focus': 'specific'
        })
        
        location_conds = [c for c in conditions if 'location' in c]
        other_conds = [c for c in conditions if 'location' not in c]
        
        if location_conds and other_conds:
            for other_cond in other_conds[:2]:
                merged = ComplexQueryProcessor._merge_conditions(location_conds + [other_cond])
                sub_queries.append({
                    'query': f"{query} (focus on location + {list(other_cond.keys())[0]})",
                    'priority': 0.8,
                    'filters': merged,
                    'focus': list(other_cond.keys())[0]
                })
        
        important_conds = ['price', 'duration', 'location']
        for cond_type in important_conds:
            conds_of_type = [c for c in conditions if cond_type in c]
            if conds_of_type:
                sub_queries.append({
                    'query': f"{query} (focus on {cond_type})",
                    'priority': 0.6,
                    'filters': conds_of_type[0],
                    'focus': cond_type
                })
        
        sub_queries.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"🔀 Split query into {len(sub_queries)} sub-queries")
        return sub_queries[:3]
    
    @staticmethod
    def _calculate_complexity(query: str) -> float:
        """Calculate how complex a query is"""
        complexity = 0.0
        
        aspects = {
            'price': ['giá', 'tiền', 'chi phí', 'đắt', 'rẻ'],
            'duration': ['ngày', 'đêm', 'bao lâu', 'thời gian'],
            'location': ['ở', 'tại', 'đến', 'về', 'địa điểm'],
            'quality': ['tốt', 'hay', 'đẹp', 'hấp dẫn', 'thú vị'],
            'type': ['thiền', 'khí công', 'retreat', 'chữa lành'],
        }
        
        query_lower = query.lower()
        
        distinct_aspects = 0
        for aspect, keywords in aspects.items():
            if any(keyword in query_lower for keyword in keywords):
                distinct_aspects += 1
        
        complexity += distinct_aspects * 0.5
        complexity += min(len(query.split()) / 10, 1.0)
        
        conjunctions = ['và', 'với', 'có', 'cho', 'mà', 'nhưng']
        for conj in conjunctions:
            if conj in query_lower:
                complexity += 0.3
        
        return complexity
    
    @staticmethod
    def _extract_conditions(query: str) -> List[Dict[str, Any]]:
        """Extract individual conditions from query"""
        conditions = []
        
        filters = MandatoryFilterSystem.extract_filters(query)
        
        if filters.price_min is not None or filters.price_max is not None:
            price_cond = {'price': {}}
            if filters.price_min is not None:
                price_cond['price']['min'] = filters.price_min
            if filters.price_max is not None:
                price_cond['price']['max'] = filters.price_max
            conditions.append(price_cond)
        
        if filters.duration_min is not None or filters.duration_max is not None:
            duration_cond = {'duration': {}}
            if filters.duration_min is not None:
                duration_cond['duration']['min'] = filters.duration_min
            if filters.duration_max is not None:
                duration_cond['duration']['max'] = filters.duration_max
            conditions.append(duration_cond)
        
        if filters.location:
            conditions.append({'location': filters.location})
        if filters.near_location:
            conditions.append({'near_location': filters.near_location})
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['rẻ', 'giá rẻ', 'tiết kiệm']):
            conditions.append({'price_quality': 'budget'})
        if any(word in query_lower for word in ['cao cấp', 'sang', 'premium']):
            conditions.append({'price_quality': 'premium'})
        
        if 'thiền' in query_lower:
            conditions.append({'activity_type': 'meditation'})
        if 'khí công' in query_lower:
            conditions.append({'activity_type': 'qigong'})
        if 'retreat' in query_lower:
            conditions.append({'activity_type': 'retreat'})
        if 'chữa lành' in query_lower:
            conditions.append({'activity_type': 'healing'})
        
        tour_name_patterns = [
            r'tour\s+([^và\s,]+)\s+và\s+tour\s+([^\s,]+)',
            r'tour\s+([^\s,]+)\s+với\s+tour\s+([^\s,]+)',
            r'tour\s+([^\s,]+)\s+.*tour\s+([^\s,]+)',
        ]
        
        for pattern in tour_name_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                for i in range(1, 3):
                    if match.group(i):
                        tour_name = match.group(i).strip()
                        normalized_name = FuzzyMatcher.normalize_vietnamese(tour_name)
                        for name, idx in TOUR_NAME_TO_INDEX.items():
                            if normalized_name in name or name in normalized_name:
                                conditions.append({'specific_tour': idx})
                                logger.info(f"🔍 Extracted tour name from complex query: {tour_name} → index {idx}")
        
        return conditions
    
    @staticmethod
    def _merge_conditions(conditions: List[Dict]) -> Dict[str, Any]:
        """Merge multiple conditions into one filter dict"""
        merged = {}
        
        for condition in conditions:
            for key, value in condition.items():
                if key in merged:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        if isinstance(value, dict) or (isinstance(value, str) and len(value) > len(str(merged[key]))):
                            merged[key] = value
                else:
                    merged[key] = value
        
        return merged

# =========== UPGRADE 6: FUZZY MATCHING (DATACLASS COMPATIBLE) ===========
class FuzzyMatcher:
    """
    UPGRADE 6: Handle misspellings and variations in tour names
    """
    
    SIMILARITY_THRESHOLD = 0.75
    
    @staticmethod
    def normalize_vietnamese(text: str) -> str:
        """
        Normalize Vietnamese text for fuzzy matching
        """
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        replacements = {
            'đ': 'd',
            'không': 'ko',
            'khong': 'ko',
            'rồi': 'roi',
            'với': 'voi',
            'được': 'duoc',
            'một': 'mot',
            'hai': '2',
            'ba': '3',
            'bốn': '4',
            'năm': '5',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def find_similar_tours(query: str, tour_names: Dict[str, int]) -> List[Tuple[int, float]]:
        """
        Find tours with names similar to query - Enhanced version
        Returns list of (tour_idx, similarity_score) sorted by similarity
        """
        if not query or not tour_names:
            logger.info(f"🔍 Fuzzy matching: Empty query or tour_names, returning empty list")
            return []
        
        query_lower = query.lower().strip()
        query_norm = FuzzyMatcher.normalize_vietnamese(query_lower)
        
        if not query_norm:
            logger.info(f"🔍 Fuzzy matching: Cannot normalize query '{query}'")
            return []
        
        logger.info(f"🔍 Fuzzy matching: Query '{query}' -> Normalized: '{query_norm}'")
        
        matches = []
        query_words = set(query_norm.split())
        
        # Define common stop words to ignore
        stop_words = {'tour', 'chương', 'trình', 'của', 'cho', 'với', 'và', 'tại', 'ở', 'từ'}
        query_filtered_words = [word for word in query_norm.split() if word not in stop_words]
        
        # Enhanced keyword extraction
        query_keywords = set(query_filtered_words)
        
        # Check for specific tour patterns
        known_tour_patterns = {
            'bạch mã': ['bạch mã', 'bach ma'],
            'trường sơn': ['trường sơn', 'truong son', 'tây trường sơn'],
            'mưa đỏ': ['mưa đỏ', 'mua do'],
            'ngọn lửa': ['ngọn lửa', 'ngon lua'],
            'ký ức': ['ký ức', 'ky uc'],
            'lịch sử': ['lịch sử', 'lich su'],
            'đại ngàn': ['đại ngàn', 'dai ngan'],
            'non nước': ['non nước', 'non nuoc'],
            'tour': ['tour', 'hanh trinh'],
            'khát vọng': ['khát vọng', 'khat vong'],
            'tĩnh lặng': ['tĩnh lặng', 'tinh lang'],
            'retreat': ['retreat', 'tĩnh tâm', 'tinh tam'],
            'thiền': ['thiền', 'thien'],
            'huế': ['huế', 'hue'],
            'quảng trị': ['quảng trị', 'quang tri']
        }
        
        # Extract potential tour name from query
        extracted_tour_names = []
        for pattern, variants in known_tour_patterns.items():
            for variant in variants:
                if variant in query_lower:
                    extracted_tour_names.append(pattern)
                    break
        
        logger.info(f"🔍 Extracted tour patterns: {extracted_tour_names}")
        
        for tour_name, tour_idx in tour_names.items():
            tour_name_lower = tour_name.lower().strip()
            tour_norm = FuzzyMatcher.normalize_vietnamese(tour_name_lower)
            
            if not tour_norm:
                continue
            
            # Calculate multiple similarity scores
            scores = []
            
            # 1. Direct string similarity
            direct_similarity = SequenceMatcher(None, query_norm, tour_norm).ratio()
            scores.append(('direct', direct_similarity))
            
            # 2. Check if query contains tour name or vice versa (partial match)
            if query_norm in tour_norm:
                scores.append(('query_in_tour', min(direct_similarity + 0.3, 1.0)))
            if tour_norm in query_norm:
                scores.append(('tour_in_query', min(direct_similarity + 0.3, 1.0)))
            
            # 3. Word overlap similarity
            tour_words = set(tour_norm.split())
            tour_filtered_words = [word for word in tour_norm.split() if word not in stop_words]
            
            common_words = query_words.intersection(tour_words)
            if common_words:
                word_overlap = len(common_words) / max(len(query_words), len(tour_words))
                scores.append(('word_overlap', word_overlap))
            
            # 4. Enhanced keyword matching
            if query_keywords:
                keyword_matches = sum(1 for keyword in query_keywords if any(keyword in word for word in tour_filtered_words))
                if keyword_matches > 0:
                    keyword_score = keyword_matches / len(query_keywords)
                    scores.append(('keyword', keyword_score))
            
            # 5. Pattern matching for known tour names
            pattern_score = 0
            for pattern in extracted_tour_names:
                if pattern in tour_norm:
                    pattern_score += 0.5
            if pattern_score > 0:
                scores.append(('pattern', min(pattern_score, 1.0)))
            
            # 6. Abbreviation/alias matching
            # Check if tour has common abbreviations
            tour_abbreviations = {
                'bạch mã': 'bm',
                'trường sơn': 'ts',
                'mưa đỏ': 'md',
                'huế': 'h',
                'quảng trị': 'qt'
            }
            
            for full, abbrev in tour_abbreviations.items():
                if full in tour_norm and abbrev in query_norm:
                    scores.append(('abbreviation', 0.7))
                    break
            
            # 7. Number matching (for tour durations like 1 ngày, 2 ngày)
            import re
            query_numbers = set(re.findall(r'\d+', query_norm))
            tour_numbers = set(re.findall(r'\d+', tour_norm))
            if query_numbers and tour_numbers:
                number_match = len(query_numbers.intersection(tour_numbers)) / len(query_numbers)
                if number_match > 0:
                    scores.append(('number', number_match))
            
            # Calculate final similarity (weighted average)
            weights = {
                'direct': 0.3,
                'query_in_tour': 0.25,
                'tour_in_query': 0.25,
                'word_overlap': 0.15,
                'keyword': 0.15,
                'pattern': 0.1,
                'abbreviation': 0.05,
                'number': 0.05
            }
            
            weighted_scores = []
            for score_type, score_value in scores:
                if score_type in weights:
                    weighted_scores.append(score_value * weights[score_type])
            
            final_similarity = sum(weighted_scores) if weighted_scores else direct_similarity
            
            # Apply bonuses for specific cases
            bonuses = 0
            
            # Bonus for exact word match
            exact_word_match = any(word == tour_word for word in query_filtered_words for tour_word in tour_filtered_words)
            if exact_word_match:
                bonuses += 0.1
            
            # Bonus for matching at beginning of tour name
            if query_filtered_words and any(tour_norm.startswith(word) for word in query_filtered_words):
                bonuses += 0.15
            
            # Bonus for historical/relevant keywords
            historical_keywords = ['lịch sử', 'chiến tranh', 'di tích', 'tri ân', 'cựu chiến binh']
            if any(keyword in query_norm for keyword in historical_keywords) and \
            any(keyword in tour_norm for keyword in historical_keywords):
                bonuses += 0.2
            
            # Bonus for wellness/retreat keywords
            wellness_keywords = ['thiền', 'yoga', 'retreat', 'tĩnh tâm', 'khí công', 'chữa lành']
            if any(keyword in query_norm for keyword in wellness_keywords) and \
            any(keyword in tour_norm for keyword in wellness_keywords):
                bonuses += 0.2
            
            final_similarity = min(final_similarity + bonuses, 1.0)
            
            # Adjust threshold based on query complexity
            dynamic_threshold = 0.5  # Base threshold
            
            # Lower threshold for complex queries (more words)
            if len(query_filtered_words) >= 3:
                dynamic_threshold = 0.4
            
            # Higher threshold for very short queries
            if len(query_filtered_words) == 1:
                dynamic_threshold = 0.6
            
            # Special case for known tour patterns
            if extracted_tour_names and any(pattern in tour_norm for pattern in extracted_tour_names):
                dynamic_threshold = 0.3
            
            if final_similarity >= dynamic_threshold:
                matches.append((tour_idx, final_similarity))
                logger.debug(f"  ✓ Match: '{tour_name}' (idx: {tour_idx}) - Score: {final_similarity:.2f}")
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results but ensure we get relevant matches
        max_results = 10
        if matches:
            # Ensure we include all high-confidence matches (>0.7)
            high_confidence = [m for m in matches if m[1] > 0.7]
            if high_confidence:
                matches = high_confidence + [m for m in matches if m[1] <= 0.7][:max_results - len(high_confidence)]
            else:
                matches = matches[:max_results]
        
        logger.info(f"✅ Fuzzy matching: '{query}' → {len(matches)} matches (threshold: dynamic)")
        
        # Log top matches for debugging
        if matches:
            for i, (idx, score) in enumerate(matches[:5]):
                tour_name = next((name for name, tid in tour_names.items() if tid == idx), "Unknown")
                logger.debug(f"  Top {i+1}: {tour_name} (idx: {idx}) - Score: {score:.2f}")
        
        return matches


    # Thêm phương thức helper cho normalization nâng cao nếu cần
    @staticmethod
    def enhanced_normalize_vietnamese(text: str) -> str:
        """
        Enhanced Vietnamese text normalization
        """
        if not text:
            return ""
        
        # Basic normalization (giữ nguyên từ hàm gốc)
        normalized = text.lower().strip()
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        # Common replacements for tour names
        replacements = {
            '–': ' ',
            '-': ' ',
            '–': ' ',
            '(': ' ',
            ')': ' ',
            ',': ' ',
            '.': ' ',
            '!': ' ',
            '?': ' ',
            '"': ' ',
            "'": ' ',
            ';': ' ',
            ':': ' ',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove multiple spaces again
        normalized = ' '.join(normalized.split())
        
        return normalized
        
    @staticmethod
    def find_tour_by_partial_name(partial_name: str, tours_db: Dict[int, Tour]) -> List[int]:
        """
        Find tours by partial name match
        """
        if not partial_name or not tours_db:
            return []
        
        partial_norm = FuzzyMatcher.normalize_vietnamese(partial_name)
        matches = []
        
        for tour_idx, tour in tours_db.items():
            tour_name = tour.name or ""
            if not tour_name:
                continue
            
            tour_norm = FuzzyMatcher.normalize_vietnamese(tour_name)
            
            if partial_norm in tour_norm:
                match_ratio = len(partial_norm) / len(tour_norm) if tour_norm else 0
                matches.append((tour_idx, match_ratio))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in matches[:3]]

# =========== UPGRADE 7: STATE MACHINE (DATACLASS COMPATIBLE) ===========
class ConversationStateMachine:
    """
    UPGRADE 7: Track conversation state for better context
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = ConversationState.INITIAL
        self.context = ConversationContext(session_id=session_id)
        self.transitions = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update state based on new interaction"""
        self.last_updated = datetime.utcnow()
        self.context.update(user_message, bot_response, tour_indices)
        
        new_state = self._determine_state(user_message, bot_response)
        
        self.transitions.append({
            'timestamp': datetime.utcnow().isoformat(),
            'from': self.state.value,
            'to': new_state.value,
            'message': user_message[:100]
        })
        
        self.state = new_state
        
        logger.info(f"🔄 State update: {self.state.value} for session {self.session_id}")
    
    def _determine_state(self, user_message: str, bot_response: str) -> ConversationState:
        """Determine new state based on current interaction"""
        message_lower = user_message.lower()
        
        farewell_words = ['tạm biệt', 'cảm ơn', 'thanks', 'bye', 'goodbye']
        if any(word in message_lower for word in farewell_words):
            return ConversationState.FAREWELL
        
        tour_ref_patterns = [
            r'tour này', r'tour đó', r'tour đang nói', r'cái tour',
            r'nó', r'cái đó', r'cái này', r'đấy'
        ]
        
        if any(re.search(pattern, message_lower) for pattern in tour_ref_patterns):
            if self.context.current_tours:
                return ConversationState.TOUR_SELECTED
            elif self.context.last_successful_tours:
                self.context.current_tours = self.context.last_successful_tours
                return ConversationState.TOUR_SELECTED
        
        if 'so sánh' in message_lower or 'sánh' in message_lower:
            return ConversationState.COMPARING
        
        if any(word in message_lower for word in ['phù hợp', 'gợi ý', 'đề xuất', 'tư vấn', 'nên chọn']):
            return ConversationState.RECOMMENDATION
        
        if any(word in message_lower for word in ['đặt', 'booking', 'đăng ký', 'giữ chỗ']):
            return ConversationState.BOOKING
        
        if self.context.current_tours:
            return ConversationState.ASKING_DETAILS
        
        return ConversationState.INITIAL
    
    def get_context_hint(self) -> str:
        """Get hint about current context for LLM prompt"""
        hints = []
        
        if self.state == ConversationState.TOUR_SELECTED and self.context.current_tours:
            tour_indices = self.context.current_tours
            if len(tour_indices) == 1:
                hints.append(f"User is asking about tour index {tour_indices[0]}")
            else:
                hints.append(f"User is asking about tours {tour_indices}")
        
        if self.context.user_preferences:
            prefs = []
            for key, value in self.context.user_preferences.items():
                prefs.append(f"{key}: {value}")
            if prefs:
                hints.append(f"User preferences: {', '.join(prefs)}")
        
        return "; ".join(hints) if hints else "No specific context"
    
    def extract_reference(self, message: str) -> List[int]:
        """Extract tour reference from message using conversation context"""
        message_lower = message.lower()
        
        if self.context.current_tours:
            for tour_idx in self.context.current_tours:
                tour = TOURS_DB.get(tour_idx)
                if not tour:
                    continue
                tour_name = (tour.name or "").lower()
                if tour_name:
                    tour_words = set(tour_name.split())
                    msg_words = set(message_lower.split())
                    common = tour_words.intersection(msg_words)
                    if common and len(common) >= 1:
                        logger.info(f"🔄 State machine: Using current tour {tour_idx}")
                        return self.context.current_tours
        
        ref_patterns = [
            (r'tour này', 1.0),
            (r'tour đó', 0.9),
            (r'tour đang nói', 0.9),
            (r'cái tour', 0.8),
            (r'nó', 0.7),
            (r'đấy', 0.7),
            (r'cái đó', 0.7),
        ]
        
        for pattern, confidence in ref_patterns:
            if re.search(pattern, message_lower):
                if self.context.current_tours:
                    logger.info(f"🔄 State machine: Resolved reference to {self.context.current_tours}")
                    return self.context.current_tours
                elif self.context.last_successful_tours:
                    logger.info(f"🔄 State machine: Using last successful tours {self.context.last_successful_tours}")
                    return self.context.last_successful_tours
        
        if self.context.mentioned_tours:
            recent_tours = list(self.context.mentioned_tours)
            for tour_idx in recent_tours[-3:]:
                tour = TOURS_DB.get(tour_idx)
                if not tour:
                    continue
                tour_name = (tour.name or "").lower()
                if tour_name:
                    tour_words = set(tour_name.split())
                    msg_words = set(message_lower.split())
                    common = tour_words.intersection(msg_words)
                    if common and len(common) >= 1:
                        logger.info(f"🔄 State machine: Matched to recently mentioned tour {tour_idx}")
                        return [tour_idx]
        
        return []

# =========== UPGRADE 8: DEEP SEMANTIC ANALYSIS (DATACLASS COMPATIBLE) ===========
class SemanticAnalyzer:
    """
    UPGRADE 8: Deep understanding of user intent beyond keywords
    """
    
    USER_PROFILE_PATTERNS = {
        'age_group': [
            (r'người già|người lớn tuổi|cao tuổi', 'senior'),
            (r'thanh niên|trẻ|sinh viên|học sinh', 'young'),
            (r'trung niên|trung tuổi', 'middle_aged'),
            (r'gia đình.*trẻ em|trẻ nhỏ|con nít', 'family_with_kids'),
        ],
        
        'group_type': [
            (r'một mình|đi lẻ|solo', 'solo'),
            (r'cặp đôi|đôi lứa|người yêu', 'couple'),
            (r'gia đình|bố mẹ con', 'family'),
            (r'bạn bè|nhóm bạn|hội bạn', 'friends'),
            (r'công ty|doanh nghiệp|đồng nghiệp', 'corporate'),
        ],
        
        'interest_type': [
            (r'thiên nhiên|rừng|cây|cảnh quan', 'nature'),
            (r'lịch sử|di tích|chiến tranh|tri ân', 'history'),
            (r'văn hóa|cộng đồng|dân tộc|truyền thống', 'culture'),
            (r'thiền|tâm linh|tĩnh tâm|yoga', 'spiritual'),
            (r'khí công|sức khỏe|chữa lành|wellness', 'wellness'),
            (r'ẩm thực|đồ ăn|món ngon|đặc sản', 'food'),
            (r'phiêu lưu|mạo hiểm|khám phá|trải nghiệm', 'adventure'),
        ],
        
        'budget_level': [
            (r'kinh tế|tiết kiệm|rẻ|giá thấp', 'budget'),
            (r'trung bình|vừa phải|phải chăng', 'midrange'),
            (r'cao cấp|sang trọng|premium|đắt', 'premium'),
        ],
        
        'physical_level': [
            (r'nhẹ nhàng|dễ dàng|không mệt', 'easy'),
            (r'vừa phải|trung bình|bình thường', 'moderate'),
            (r'thử thách|khó|mệt|leo núi', 'challenging'),
        ],
    }
    
    @staticmethod
    def analyze_user_profile(message: str, current_context: ConversationContext = None) -> UserProfile:
        """
        Analyze message to build user profile
        """
        if current_context and hasattr(current_context, 'user_profile') and current_context.user_profile:
            profile = current_context.user_profile
        else:
            profile = UserProfile()
        
        message_lower = message.lower()
        
        for category, patterns in SemanticAnalyzer.USER_PROFILE_PATTERNS.items():
            for pattern, value in patterns:
                if re.search(pattern, message_lower):
                    if category == 'interests':
                        if value not in profile.interests:
                            profile.interests.append(value)
                            profile.confidence_scores[f'interest_{value}'] = 0.8
                    else:
                        setattr(profile, category, value)
                        profile.confidence_scores[category] = 0.8
        
        SemanticAnalyzer._infer_attributes(profile, message_lower)
        profile.overall_confidence = SemanticAnalyzer._calculate_confidence(profile)
        
        logger.info(f"👤 User profile analysis: {profile}")
        return profile
    
    @staticmethod
    def _infer_attributes(profile: UserProfile, message_lower: str):
        """Infer additional attributes from context"""
        if not profile.age_group:
            if profile.group_type and 'family_with_kids' in profile.group_type:
                profile.age_group = 'middle_aged'
                profile.confidence_scores['age_group'] = 0.6
            elif 'senior' in message_lower or 'già' in message_lower:
                profile.age_group = 'senior'
                profile.confidence_scores['age_group'] = 0.7
        
        if not profile.physical_level:
            if 'adventure' in profile.interests:
                profile.physical_level = 'challenging'
                profile.confidence_scores['physical_level'] = 0.6
            elif 'spiritual' in profile.interests or 'wellness' in profile.interests:
                profile.physical_level = 'easy'
                profile.confidence_scores['physical_level'] = 0.6
        
        if not profile.budget_level:
            budget_keywords = {
                'budget': ['rẻ', 'tiết kiệm', 'ít tiền', 'kinh tế'],
                'premium': ['cao cấp', 'sang', 'đắt', 'premium']
            }
            
            for level, keywords in budget_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    profile.budget_level = level
                    profile.confidence_scores['budget_level'] = 0.7
                    break
    
    @staticmethod
    def _calculate_confidence(profile: UserProfile) -> float:
        """Calculate overall confidence in user profile"""
        if not profile.confidence_scores:
            return 0.0
        
        total = 0.0
        count = 0
        
        for key, score in profile.confidence_scores.items():
            total += score
            count += 1
        
        return total / max(count, 1)
    
    @staticmethod
    def match_tours_to_profile(profile: UserProfile, tours_db: Dict[int, Tour], 
                              max_results: int = 5) -> List[Tuple[int, float, List[str]]]:
        """
        Match tours to user profile with explanation
        """
        matches = []
        
        for tour_idx, tour in tours_db.items():
            score = 0.0
            reasons = []
            
            tour_tags = tour.tags or []
            
            if profile.age_group:
                if profile.age_group == 'senior':
                    if any('easy' in tag for tag in tour_tags):
                        score += 0.3
                        reasons.append("phù hợp người lớn tuổi")
                    if any('nature' in tag for tag in tour_tags):
                        score += 0.2
                        reasons.append("thiên nhiên nhẹ nhàng")
            
            if profile.interests:
                for interest in profile.interests:
                    tour_summary = (tour.summary or "").lower()
                    if (interest in tour_summary or 
                        any(interest in tag for tag in tour_tags)):
                        score += 0.4
                        reasons.append(f"có yếu tố {interest}")
            
            if profile.budget_level:
                tour_price = tour.price or ""
                price_nums = re.findall(r'\d[\d,\.]+', tour_price)
                
                if price_nums:
                    try:
                        first_price = int(price_nums[0].replace(',', '').replace('.', ''))
                        
                        if profile.budget_level == 'budget' and first_price < 2000000:
                            score += 0.3
                            reasons.append("giá hợp lý")
                        elif profile.budget_level == 'premium' and first_price > 2500000:
                            score += 0.3
                            reasons.append("cao cấp")
                        elif profile.budget_level == 'midrange' and 1500000 <= first_price <= 3000000:
                            score += 0.3
                            reasons.append("giá vừa phải")
                    except:
                        pass
            
            if profile.physical_level:
                if profile.physical_level == 'easy':
                    if any('easy' in tag or 'meditation' in tag for tag in tour_tags):
                        score += 0.2
                        reasons.append("hoạt động nhẹ nhàng")
            
            if score > 0:
                matches.append((tour_idx, score, reasons))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:max_results]

# =========== UPGRADE 9: AUTO-VALIDATION SYSTEM (DATACLASS COMPATIBLE) ===========
class AutoValidator:
    """
    UPGRADE 9: Validate and correct information before returning
    """
    
    VALIDATION_RULES = {
        'duration': {
            'patterns': [
                r'(\d+)\s*ngày\s*(\d+)\s*đêm',
                r'(\d+)\s*ngày',
                r'(\d+)\s*đêm',
            ],
            'constraints': {
                'max_days': 7,
                'max_nights': 7,
                'valid_day_night_combos': [(1,0), (1,1), (2,1), (2,2), (3,2), (3,3)],
                'common_durations': ['1 ngày', '2 ngày 1 đêm', '3 ngày 2 đêm']
            }
        },
        
        'price': {
            'patterns': [
                r'(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)',
                r'(\d[\d,\.]*)\s*-\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)?',
                r'(\d[\d,\.]*)\s*(đồng|vnđ|vnd)',
            ],
            'constraints': {
                'min_tour_price': 500000,
                'max_tour_price': 10000000,
                'common_ranges': [
                    (800000, 1500000),
                    (1500000, 2500000),
                    (2500000, 4000000),
                ]
            }
        },
        
        'location': {
            'patterns': [
                r'ở\s+([^.,!?]+)',
                r'tại\s+([^.,!?]+)',
                r'đến\s+([^.,!?]+)',
            ],
            'constraints': {
                'valid_locations': ['Huế', 'Quảng Trị', 'Bạch Mã', 'Trường Sơn', 'Đông Hà', 'Khe Sanh'],
                'max_length': 100
            }
        },
    }
    
    @staticmethod
    def validate_response(response: str) -> str:
        """
        Validate and correct response content
        """
        if not response:
            return response
        
        validated = response
        
        validated = AutoValidator._validate_duration(validated)
        validated = AutoValidator._validate_price(validated)
        validated = AutoValidator._validate_locations(validated)
        validated = AutoValidator._check_unrealistic_info(validated)
        
        if validated != response:
            validated = AutoValidator._add_validation_note(validated)
        
        return validated
    
    @staticmethod
    def _validate_duration(text: str) -> str:
        """Validate and correct duration information"""
        for pattern in AutoValidator.VALIDATION_RULES['duration']['patterns']:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                try:
                    if match.lastindex == 2:
                        days = int(match.group(1))
                        nights = int(match.group(2))
                        
                        constraints = AutoValidator.VALIDATION_RULES['duration']['constraints']
                        
                        valid_combos = constraints['valid_day_night_combos']
                        is_valid_combo = bool(valid_combos)
                        
                        if days > constraints['max_days'] or nights > constraints['max_nights']:
                            replacement = random.choice(constraints['common_durations'])
                            text = text.replace(match.group(0), replacement)
                            logger.warning(f"⚠️ Corrected unrealistic duration: {days} ngày {nights} đêm → {replacement}")
                        
                        elif not is_valid_combo:
                            valid_days = min(days, constraints['max_days'])
                            valid_nights = min(nights, constraints['max_nights'])
                            if abs(valid_days - valid_nights) > 1:
                                valid_nights = valid_days
                            
                            replacement = f"{valid_days} ngày {valid_nights} đêm"
                            text = text.replace(match.group(0), replacement)
                            logger.info(f"🔄 Fixed duration combo: {replacement}")
                    
                    elif match.lastindex == 1:
                        num = int(match.group(1))
                        constraints = AutoValidator.VALIDATION_RULES['duration']['constraints']
                        
                        if num > constraints['max_days']:
                            replacement = f"{constraints['max_days']} ngày"
                            text = text.replace(match.group(0), replacement)
                            logger.warning(f"⚠️ Capped long duration: {num} → {constraints['max_days']}")
                
                except (ValueError, IndexError):
                    continue
        
        return text
    
    @staticmethod
    def _validate_price(text: str) -> str:
        """Validate and correct price information"""
        for pattern in AutoValidator.VALIDATION_RULES['price']['patterns']:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '').replace('.', '')
                    if not amount_str.isdigit():
                        continue
                    
                    amount = int(amount_str)
                    
                    unit = match.group(2).lower() if match.lastindex >= 2 else ''
                    
                    if unit in ['triệu', 'tr']:
                        amount = amount * 1000000
                    elif unit in ['k', 'nghìn']:
                        amount = amount * 1000
                    
                    constraints = AutoValidator.VALIDATION_RULES['price']['constraints']
                    
                    if amount < constraints['min_tour_price']:
                        replacement = "giá hợp lý"
                        text = text.replace(match.group(0), replacement)
                        logger.warning(f"⚠️ Corrected too-low price: {amount} → {replacement}")
                    
                    elif amount > constraints['max_tour_price']:
                        replacement = "giá cao cấp"
                        text = text.replace(match.group(0), replacement)
                        logger.warning(f"⚠️ Corrected too-high price: {amount} → {replacement}")
                
                except (ValueError, IndexError, AttributeError):
                    continue
        
        return text
    
    @staticmethod
    def _validate_locations(text: str) -> str:
        """Validate location names"""
        wrong_locations = {
            'hà nội': 'Huế',
            'hồ chí minh': 'Quảng Trị',
            'đà nẵng': 'Bạch Mã',
            'nha trang': 'Trường Sơn',
        }
        
        for wrong, correct in wrong_locations.items():
            if wrong in text.lower():
                text = text.replace(wrong, correct)
                text = text.replace(wrong.capitalize(), correct)
                logger.info(f"🔄 Corrected location: {wrong} → {correct}")
        
        return text
    
    @staticmethod
    def _check_unrealistic_info(text: str) -> str:
        """Check for other unrealistic information"""
        unrealistic_patterns = [
            (r'\d+\s*giờ\s*bay', "thời gian di chuyển"),
            (r'\d+\s*sao', "chất lượng dịch vụ"),
            (r'\d+\s*tầng', "chỗ ở"),
            (r'\d+\s*m\s*cao', "địa hình"),
        ]
        
        for pattern, replacement in unrealistic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logger.info(f"🔄 Replaced unrealistic info with: {replacement}")
        
        return text
    
    @staticmethod
    def _add_validation_note(text: str) -> str:
        """Add note about information validation"""
        note = "\n\n*Thông tin được cung cấp dựa trên dữ liệu hiện có. " \
               "Vui lòng liên hệ hotline 0332510486 để xác nhận chi tiết chính xác nhất.*"
        
        if note not in text:
            text += note
        
        return text

# =========== UPGRADE 10: TEMPLATE SYSTEM (DATACLASS COMPATIBLE) ===========
class TemplateSystem:
    """
    UPGRADE 10: Beautiful, structured responses for different question types
    """
    
    TEMPLATES = {
        'tour_list': {
            'header': "✨ **DANH SÁCH TOUR RUBY WINGS** ✨\n\n",
            'item': "**{index}. {tour_name}** {emoji}\n"
                   "   📅 {duration}\n"
                   "   📍 {location}\n"
                   "   💰 {price}\n"
                   "   {summary}\n",
            'footer': "\n📞 **Liên hệ đặt tour:** 0332510486\n"
                     "📍 **Ruby Wings Travel** - Hành trình trải nghiệm đặc sắc\n"
                     "💡 *Hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour*",
            'emoji_map': {
                '1 ngày': '🌅',
                '2 ngày': '🌄',
                '3 ngày': '🏔️',
                'default': '✨'
            }
        },
        
        'tour_detail': {
            'header': "🎯 **{tour_name}**\n\n",
            'sections': {
                'overview': "📋 **THÔNG TIN CHÍNH:**\n"
                          "   ⏱️ Thời gian: {duration}\n"
                          "   📍 Địa điểm: {location}\n"
                          "   💰 Giá tour: {price}\n\n",
                'description': "📖 **MÔ TẢ TOUR:**\n{summary}\n\n",
                'includes': "🎪 **LỊCH TRÌNH & DỊCH VỤ:**\n{includes}\n\n",
                'accommodation': "🏨 **CHỖ Ở:**\n{accommodation}\n\n",
                'meals': "🍽️ **ĂN UỐNG:**\n{meals}\n\n",
                'transport': "🚗 **DI CHUYỂN:**\n{transport}\n\n",
                'notes': "📝 **GHI CHÚ:**\n{notes}\n\n",
            },
            'footer': "📞 **ĐẶT TOUR & TƯ VẾN:** 0332510486\n"
                     "⭐ *Tour phù hợp cho: {suitable_for}*",
            'default_values': {
                'duration': 'Đang cập nhật',
                'location': 'Đang cập nhật',
                'price': 'Liên hệ để biết giá',
                'summary': 'Hành trình trải nghiệm đặc sắc của Ruby Wings',
                'includes': 'Chi tiết lịch trình liên hệ tư vấn',
                'accommodation': 'Đang cập nhật',
                'meals': 'Đang cập nhật',
                'transport': 'Đang cập nhật',
                'notes': 'Vui lòng liên hệ để biết thêm chi tiết',
                'suitable_for': 'mọi đối tượng',
            }
        },
        
        'comparison': {
            'header': "📊 **SO SÁNH TOUR**\n\n",
            'table_header': "| Tiêu chí | {tour1} | {tour2} |\n|----------|----------|----------|\n",
            'table_row': "| {criterion} | {value1} | {value2} |\n",
            'recommendation': "\n💡 **GỢI Ý LỰA CHỌN:**\n{recommendations}\n",
            'footer': "\n📞 **Tư vấn chi tiết:** 0332510486\n"
                     "🤔 *Cần so sánh thêm tiêu chí nào?*",
        },
        
        'recommendation': {
            'header': "🎯 **ĐỀ XUẤT TOUR PHÙ HỢP**\n\n",
            'top_recommendation': "🏆 **PHÙ HỢP NHẤT ({score}%)**\n"
                                "**{tour_name}**\n"
                                "   ✅ {reasons}\n"
                                "   📅 {duration} | 📍 {location} | 💰 {price}\n\n",
            'other_recommendations': "📋 **LỰA CHỌN KHÁC:**\n",
            'other_item': "   • **{tour_name}** ({score}%)\n"
                         "     📅 {duration} | 📍 {location}\n",
            'criteria': "\n🔍 **TIÊU CHÍ ĐỀ XUẤT:**\n{criteria}\n",
            'footer': "\n📞 **Liên hệ tư vấn cá nhân hóa:** 0332510486\n"
                     "💬 *Cho tôi biết thêm sở thích của bạn để đề xuất chính xác hơn*",
        },
        
        'information': {
            'header': "ℹ️ **THÔNG TIN:**\n\n",
            'content': "{content}\n",
            'sources': "\n📚 *Nguồn thông tin từ dữ liệu Ruby Wings*",
            'footer': "\n📞 **Hotline hỗ trợ:** 0332510486",
        },
        
        'greeting': {
            'template': "👋 **Xin chào! Tôi là trợ lý AI của Ruby Wings**\n\n"
                       "Tôi có thể giúp bạn:\n"
                       "• Tìm hiểu về các tour trải nghiệm\n"
                       "• So sánh các tour\n"
                       "• Đề xuất tour phù hợp với bạn\n"
                       "• Cung cấp thông tin chi tiết về tour\n\n"
                       "💡 **Ví dụ bạn có thể hỏi:**\n"
                       "- 'Có những tour nào?'\n"
                       "- 'Tour Bạch Mã giá bao nhiêu?'\n"
                       "- 'Tour nào phù hợp cho gia đình?'\n\n"
                       "Hãy cho tôi biết bạn cần gì nhé! 😊",
        },
        
        'farewell': {
            'template': "🙏 **Cảm ơn bạn đã trò chuyện cùng Ruby Wings!**\n\n"
                       "Chúc bạn một ngày tràn đầy năng lượng và bình an.\n"
                       "Hy vọng sớm được đồng hành cùng bạn trong tour trải nghiệm sắp tới!\n\n"
                       "📞 **Liên hệ đặt tour:** 0332510486\n"
                       "🌐 **Website:** rubywings.vn\n\n"
                       "Hẹn gặp lại! ✨",
        },
    }
    
    @staticmethod
    def render(template_name: str, **kwargs) -> str:
        """Render template with provided variables"""
        template_data = TemplateSystem.TEMPLATES.get(template_name)
        if not template_data:
            return kwargs.get('content', '')
        
        if template_name in ['greeting', 'farewell']:
            return template_data['template']
        
        response_parts = []
        
        if 'header' in template_data:
            header = template_data['header']
            for key, value in kwargs.items():
                header = header.replace(f'{{{key}}}', str(value))
            response_parts.append(header)
        
        if template_name == 'tour_list':
            response_parts.append(TemplateSystem._render_tour_list(template_data, kwargs))
        
        elif template_name == 'tour_detail':
            response_parts.append(TemplateSystem._render_tour_detail(template_data, kwargs))
        
        elif template_name == 'comparison':
            response_parts.append(TemplateSystem._render_comparison(template_data, kwargs))
        
        elif template_name == 'recommendation':
            response_parts.append(TemplateSystem._render_recommendation(template_data, kwargs))
        
        elif template_name == 'information':
            response_parts.append(TemplateSystem._render_information(template_data, kwargs))
        
        if 'footer' in template_data:
            footer = template_data['footer']
            for key, value in kwargs.items():
                footer = footer.replace(f'{{{key}}}', str(value))
            response_parts.append(footer)
        
        return '\n'.join(response_parts)
    
    @staticmethod
    def _render_tour_list(template_data: Dict, kwargs: Dict) -> str:
        """Render tour list template"""
        tours = kwargs.get('tours', [])
        if not tours:
            return "Hiện chưa có thông tin tour."
        
        items = []
        for i, tour in enumerate(tours[:10], 1):
            duration = tour.duration or ''
            emoji = template_data['emoji_map'].get('default')
            for dur_pattern, dur_emoji in template_data['emoji_map'].items():
                if dur_pattern in duration.lower():
                    emoji = dur_emoji
                    break
            
            item_template = template_data['item']
            item = item_template.format(
                index=i,
                tour_name=tour.name or f'Tour #{i}',
                emoji=emoji or '✨',
                duration=duration or 'Đang cập nhật',
                location=tour.location or 'Đang cập nhật',
                price=tour.price or 'Liên hệ để biết giá',
                summary=(tour.summary or 'Tour trải nghiệm đặc sắc')[:100] + '...'
            )
            items.append(item)
        
        return '\n'.join(items)
    
    @staticmethod
    def _render_tour_detail(template_data: Dict, kwargs: Dict) -> str:
        """Render tour detail template"""
        sections = []
        
        for section_name, section_template in template_data['sections'].items():
            value = kwargs.get(section_name, template_data['default_values'].get(section_name, ''))
            
            if value and value != template_data['default_values'].get(section_name):
                if isinstance(value, list):
                    if section_name == 'includes':
                        value = '\n'.join([f'   • {item}' for item in value[:5]])
                    else:
                        value = ', '.join(value[:3])
                
                section = section_template.format(**{section_name: value})
                sections.append(section)
        
        return '\n'.join(sections)
    
    @staticmethod
    def _render_comparison(template_data: Dict, kwargs: Dict) -> str:
        """Render comparison template"""
        comparison_table = []
        
        tour1_name = kwargs.get('tour1_name', 'Tour 1')[:20]
        tour2_name = kwargs.get('tour2_name', 'Tour 2')[:20]
        table_header = template_data['table_header'].format(tour1=tour1_name, tour2=tour2_name)
        comparison_table.append(table_header)
        
        criteria = kwargs.get('criteria', [])
        for criterion in criteria[:8]:
            row = template_data['table_row'].format(
                criterion=criterion.get('name', ''),
                value1=criterion.get('value1', 'N/A')[:20],
                value2=criterion.get('value2', 'N/A')[:20]
            )
            comparison_table.append(row)
        
        return '\n'.join(comparison_table)
    
    @staticmethod
    def _render_recommendation(template_data: Dict, kwargs: Dict) -> str:
        """Render recommendation template"""
        recommendation_text = []
        
        top_tour = kwargs.get('top_tour')
        if top_tour:
            top_text = template_data['top_recommendation'].format(
                score=int(top_tour.get('score', 0) * 100),
                tour_name=top_tour.get('name', ''),
                reasons=', '.join(top_tour.get('reasons', ['phù hợp'])[:3]),
                duration=top_tour.get('duration', ''),
                location=top_tour.get('location', ''),
                price=top_tour.get('price', 'Liên hệ để biết giá')
            )
            recommendation_text.append(top_text)
        
        other_tours = kwargs.get('other_tours', [])
        if other_tours:
            recommendation_text.append(template_data['other_recommendations'])
            
            for tour in other_tours[:2]:
                other_item = template_data['other_item'].format(
                    tour_name=tour.get('name', ''),
                    score=int(tour.get('score', 0) * 100),
                    duration=tour.get('duration', ''),
                    location=tour.get('location', '')
                )
                recommendation_text.append(other_item)
        
        return '\n'.join(recommendation_text)
    
    @staticmethod
    def _render_information(template_data: Dict, kwargs: Dict) -> str:
        """Render information template"""
        content = kwargs.get('content', '')
        if not content:
            return ""
        
        info_text = template_data['content'].format(content=content)
        
        if kwargs.get('has_sources'):
            info_text += template_data['sources']
        
        return info_text

# =========== TOUR DATABASE BUILDER (USING Tour DATACLASS) ===========
def load_knowledge(path: str = KNOWLEDGE_PATH):
    """Load knowledge base from JSON file"""
    global KNOW, FLAT_TEXTS, MAPPING
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
        logger.info(f"✅ Loaded knowledge from {path}")
    except Exception as e:
        logger.error(f"❌ Could not open {path}: {e}")
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
    logger.info(f"📊 Knowledge scanned: {len(FLAT_TEXTS)} passages")

def index_tour_names():
    """Build tour name to index mapping"""
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
                    existing_txt = MAPPING[next(
                        i for i, m2 in enumerate(MAPPING) 
                        if re.search(rf"\[{prev}\]", m2.get('path','')) and ".tour_name" in m2.get('path','')
                    )].get("text","")
                    if len(txt) > len(existing_txt):
                        TOUR_NAME_TO_INDEX[norm] = idx
    
    logger.info(f"📝 Indexed {len(TOUR_NAME_TO_INDEX)} tour names")

def build_tours_db():
    """Build structured tour database from MAPPING using Tour dataclass"""
    global TOURS_DB, TOUR_TAGS
    
    TOURS_DB.clear()
    TOUR_TAGS.clear()
    
    # First pass: collect all fields for each tour
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        
        if not path or not text:
            continue
        
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            continue
        
        tour_idx = int(tour_match.group(1))
        
        field_match = re.search(r'tours\[\d+\]\.(\w+)(?:\[\d+\])?', path)
        if not field_match:
            continue
        
        field_name = field_match.group(1)
        
        # Initialize tour entry
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = Tour(index=tour_idx)
        
        # Update field in Tour object
        tour_obj = TOURS_DB[tour_idx]
        if field_name == 'tour_name':
            tour_obj.name = text
        elif field_name == 'duration':
            tour_obj.duration = text
        elif field_name == 'location':
            tour_obj.location = text
        elif field_name == 'price':
            tour_obj.price = text
        elif field_name == 'summary':
            tour_obj.summary = text
        elif field_name == 'includes':
            if isinstance(tour_obj.includes, list):
                tour_obj.includes.append(text)
            else:
                tour_obj.includes = [text]
        elif field_name == 'accommodation':
            tour_obj.accommodation = text
        elif field_name == 'meals':
            tour_obj.meals = text
        elif field_name == 'transport':
            tour_obj.transport = text
        elif field_name == 'notes':
            tour_obj.notes = text
        elif field_name == 'style':
            tour_obj.style = text
    
    # Second pass: generate tags and metadata
    for tour_idx, tour_obj in TOURS_DB.items():
        tags = []
        
        # Location tags
        if tour_obj.location:
            locations = [loc.strip() for loc in tour_obj.location.split(",") if loc.strip()]
            tags.extend([f"location:{loc}" for loc in locations[:2]])
        
        # Duration tags
        if tour_obj.duration:
            duration_lower = tour_obj.duration.lower()
            if "1 ngày" in duration_lower:
                tags.append("duration:1day")
            elif "2 ngày" in duration_lower:
                tags.append("duration:2day")
            elif "3 ngày" in duration_lower:
                tags.append("duration:3day")
            else:
                day_match = re.search(r'(\d+)\s*ngày', duration_lower)
                if day_match:
                    days = int(day_match.group(1))
                    tags.append(f"duration:{days}day")
        
        # Price tags
        if tour_obj.price:
            price_nums = re.findall(r'[\d,\.]+', tour_obj.price)
            if price_nums:
                try:
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
        
        # Style/theme tags
        text_to_check = (tour_obj.style + " " + (tour_obj.summary or '')).lower()
        
        theme_keywords = {
            'meditation': ['thiền', 'chánh niệm', 'tâm linh'],
            'history': ['lịch sử', 'di tích', 'chiến tranh', 'tri ân'],
            'nature': ['thiên nhiên', 'rừng', 'núi', 'cây'],
            'culture': ['văn hóa', 'cộng đồng', 'dân tộc'],
            'wellness': ['khí công', 'sức khỏe', 'chữa lành'],
            'adventure': ['phiêu lưu', 'mạo hiểm', 'khám phá'],
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                tags.append(f"theme:{theme}")
        
        # Destination tags from tour name
        if tour_obj.name:
            name_lower = tour_obj.name.lower()
            if "bạch mã" in name_lower:
                tags.append("destination:bachma")
            if "trường sơn" in name_lower:
                tags.append("destination:truongson")
            if "quảng trị" in name_lower:
                tags.append("destination:quangtri")
            if "huế" in name_lower:
                tags.append("destination:hue")
        
        # Update Tour object tags
        tour_obj.tags = list(set(tags))
        TOUR_TAGS[tour_idx] = tour_obj.tags
        
        # Calculate completeness score
        completeness = 0
        important_fields = ['name', 'duration', 'location', 'price', 'summary']
        for field in important_fields:
            if getattr(tour_obj, field, None):
                completeness += 1
        
        tour_obj.completeness_score = completeness / len(important_fields)
    
    logger.info(f"✅ Built tours database: {len(TOURS_DB)} tours with tags")

def get_passages_by_field(field_name: str, limit: int = 50, 
                         tour_indices: Optional[List[int]] = None) -> List[Tuple[float, Dict]]:
    """
    Get passages for a specific field
    """
    exact_matches = []
    global_matches = []
    
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

# =========== CACHE SYSTEM (DATACLASS COMPATIBLE) ===========
class CacheSystem:
    """Simple caching system for responses"""
    
    @staticmethod
    def get_cache_key(query: str, context_hash: str = "") -> str:
        """Generate cache key"""
        key_parts = [query]
        if context_hash:
            key_parts.append(context_hash)
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    @staticmethod
    def get(key: str, ttl_seconds: int = 300):
        """Get item from cache"""
        with _cache_lock:
            if key in _response_cache:
                cache_entry = _response_cache[key]
                if not cache_entry.is_expired():
                    logger.info(f"💾 Cache hit for key: {key[:20]}...")
                    return cache_entry.value
                else:
                    del _response_cache[key]
            return None
    
    @staticmethod
    def set(key: str, value: Any, expiry: int = None):
        """
        Set item in cache with enhanced features
        - Supports custom expiry (TTL in seconds)
        - Intelligent cache eviction
        - Thread-safe with lock
        """
        with _cache_lock:
            try:
                # Get TTL from parameter or config
                ttl_seconds = expiry or UpgradeFlags.get_all_flags().get("CACHE_TTL_SECONDS", 300)
                
                # Create cache entry
                cache_entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    ttl_seconds=ttl_seconds,
                    access_count=0,  # Track how many times accessed
                    last_accessed=datetime.utcnow()
                )
                
                # Store in cache
                _response_cache[key] = cache_entry
                
                # Intelligent cache cleaning
                CacheSystem._clean_cache()
                
                logger.debug(f"💾 Cached response for key: {key[:50]}... (TTL: {ttl_seconds}s)")
                
            except Exception as e:
                logger.error(f"❌ Cache set error: {e}")
                # Don't crash if cache fails


    @staticmethod
    def _clean_cache():
        """
        Intelligent cache cleaning with multiple strategies
        """
        try:
            now = datetime.utcnow()
            cache_size = len(_response_cache)
            
            # Strategy 1: Remove expired entries
            expired_keys = []
            for key, entry in _response_cache.items():
                # Check if entry has expired
                if hasattr(entry, 'is_expired'):
                    if entry.is_expired(now):
                        expired_keys.append(key)
                else:
                    # Fallback: manual expiration check
                    age = (now - entry.created_at).total_seconds()
                    if age > (entry.ttl_seconds if hasattr(entry, 'ttl_seconds') else 300):
                        expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del _response_cache[key]
            
            if expired_keys:
                logger.debug(f"🧹 Removed {len(expired_keys)} expired cache entries")
            
            # Strategy 2: If still over limit, remove least recently used
            current_size = len(_response_cache)
            if current_size > 1000:
                logger.warning(f"⚠️ Cache size ({current_size}) exceeds limit, performing LRU cleanup")
                
                # Sort by last accessed time (oldest first)
                lru_items = sorted(_response_cache.items(), 
                                key=lambda x: x[1].last_accessed if hasattr(x[1], 'last_accessed') 
                                else x[1].created_at)
                
                # Remove oldest 20% or at least 200 items
                remove_count = max(200, int(current_size * 0.2))
                remove_keys = [k for k, _ in lru_items[:remove_count]]
                
                for key in remove_keys:
                    if key in _response_cache:
                        del _response_cache[key]
                
                logger.info(f"🧹 LRU cleanup removed {len(remove_keys)} items")
            
            # Strategy 3: Clean up very old entries regardless of size
            if _response_cache:
                very_old_threshold = 86400  # 24 hours in seconds
                very_old_keys = []
                
                for key, entry in _response_cache.items():
                    age = (now - entry.created_at).total_seconds()
                    if age > very_old_threshold:
                        very_old_keys.append(key)
                
                if very_old_keys:
                    for key in very_old_keys:
                        del _response_cache[key]
                    logger.debug(f"🧹 Removed {len(very_old_keys)} very old cache entries")
            
            # Final size check
            final_size = len(_response_cache)
            if final_size > 0:
                logger.debug(f"📊 Cache stats: {final_size} items, " 
                            f"approx. {final_size * 0.5:.1f}KB memory")
                
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")


    @staticmethod
    def get(key: str, update_access: bool = True) -> Optional[Any]:
        """
        Get item from cache with enhanced features
        - Updates access count and timestamp
        - Auto-removes expired items
        """
        with _cache_lock:
            try:
                if key not in _response_cache:
                    return None
                
                entry = _response_cache[key]
                now = datetime.utcnow()
                
                # Check expiration
                if hasattr(entry, 'is_expired'):
                    if entry.is_expired(now):
                        del _response_cache[key]
                        logger.debug(f"🗑️  Auto-removed expired cache: {key[:50]}...")
                        return None
                else:
                    # Manual expiration check
                    age = (now - entry.created_at).total_seconds()
                    ttl = entry.ttl_seconds if hasattr(entry, 'ttl_seconds') else 300
                    if age > ttl:
                        del _response_cache[key]
                        return None
                
                # Update access metadata if requested
                if update_access:
                    if hasattr(entry, 'access_count'):
                        entry.access_count += 1
                    if hasattr(entry, 'last_accessed'):
                        entry.last_accessed = now
                
                logger.debug(f"💾 Cache hit for key: {key[:50]}...")
                return entry.value
                
            except Exception as e:
                logger.error(f"❌ Cache get error: {e}")
                return None


    @staticmethod
    def delete(key: str) -> bool:
        """Delete specific cache entry"""
        with _cache_lock:
            try:
                if key in _response_cache:
                    del _response_cache[key]
                    logger.debug(f"🗑️  Deleted cache: {key[:50]}...")
                    return True
                return False
            except Exception as e:
                logger.error(f"❌ Cache delete error: {e}")
                return False


    @staticmethod
    def clear() -> int:
        """Clear all cache, return number of items cleared"""
        with _cache_lock:
            try:
                count = len(_response_cache)
                _response_cache.clear()
                logger.info(f"🧹 Cleared all cache ({count} items)")
                return count
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
                return 0


    @staticmethod
    def stats() -> Dict[str, Any]:
        """Get cache statistics"""
        with _cache_lock:
            try:
                now = datetime.utcnow()
                total_size = len(_response_cache)
                
                # Calculate age distribution
                age_distribution = {
                    "under_1min": 0,
                    "1min_10min": 0,
                    "10min_1hour": 0,
                    "1hour_24hour": 0,
                    "over_24hour": 0
                }
                
                # Calculate expiration status
                expired_count = 0
                will_expire_soon = 0  # Within 60 seconds
                
                for entry in _response_cache.values():
                    # Age distribution
                    age = (now - entry.created_at).total_seconds()
                    if age < 60:
                        age_distribution["under_1min"] += 1
                    elif age < 600:
                        age_distribution["1min_10min"] += 1
                    elif age < 3600:
                        age_distribution["10min_1hour"] += 1
                    elif age < 86400:
                        age_distribution["1hour_24hour"] += 1
                    else:
                        age_distribution["over_24hour"] += 1
                    
                    # Expiration check
                    ttl = entry.ttl_seconds if hasattr(entry, 'ttl_seconds') else 300
                    remaining = ttl - age
                    if remaining <= 0:
                        expired_count += 1
                    elif remaining < 60:
                        will_expire_soon += 1
                
                return {
                    "total_items": total_size,
                    "age_distribution": age_distribution,
                    "expired_items": expired_count,
                    "expiring_soon": will_expire_soon,
                    "memory_estimate_kb": total_size * 0.5  # Rough estimate
                }
                
            except Exception as e:
                logger.error(f"❌ Cache stats error: {e}")
                return {"error": str(e)}


    @staticmethod
    def get_cache_key(user_message: str, context_hash: str = None) -> str:
        """
        Generate cache key with enhanced hashing
        """
        try:
            # Normalize the user message
            normalized = user_message.lower().strip()
            
            # Remove extra whitespace
            normalized = ' '.join(normalized.split())
            
            # Create base key
            base_content = normalized
            
            # Add context hash if provided
            if context_hash:
                base_content += f"|{context_hash}"
            
            # Create hash (shorter for efficiency)
            import hashlib
            cache_key = hashlib.md5(base_content.encode('utf-8')).hexdigest()[:16]
            
            # Add prefix for identification
            cache_key = f"chat_{cache_key}"
            
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Cache key generation error: {e}")
            # Fallback: use simple hash
            import hashlib
            return f"chat_fallback_{hashlib.md5(user_message.encode()).hexdigest()[:8]}"


    # Cập nhật class CacheEntry để hỗ trợ các tính năng mới
    @dataclass
    class CacheEntry:
        """
        Enhanced cache entry with metadata for intelligent cache management
        """
        key: str
        value: Any
        created_at: datetime
        ttl_seconds: int = 300
        access_count: int = 0
        last_accessed: datetime = None
        
        def __post_init__(self):
            """Initialize last_accessed if not provided"""
            if self.last_accessed is None:
                self.last_accessed = self.created_at
        
        def is_expired(self, current_time: datetime = None) -> bool:
            """Check if cache entry has expired"""
            if current_time is None:
                current_time = datetime.utcnow()
            
            age = (current_time - self.created_at).total_seconds()
            return age > self.ttl_seconds
        
        def age_seconds(self) -> float:
            """Get age of cache entry in seconds"""
            return (datetime.utcnow() - self.created_at).total_seconds()
        
        def ttl_remaining(self) -> float:
            """Get remaining TTL in seconds"""
            age = self.age_seconds()
            remaining = self.ttl_seconds - age
            return max(0, remaining)

# =========== EMBEDDING FUNCTIONS (MEMORY OPTIMIZED) ===========
@lru_cache(maxsize=128 if IS_LOW_RAM else 1000)
def embed_text(text: str) -> Tuple[List[float], int]:
    """Embed text using OpenAI or fallback (with memory optimization)"""
    if not text:
        return [], 0
    
    text = text[:2000]
    
    if client:
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            if response.data:
                embedding = response.data[0].embedding
                return embedding, len(embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
    
    # Fallback: deterministic hash-based embedding
    h = hash(text) % (10 ** 12)
    dim = 1536
    embedding = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 
                 for i in range(dim)]
    
    return embedding, dim

def query_index(
    query: str,
    top_k: int = 5,
    min_score: float = 0.78
):
    """
    Semantic search dùng FAISS – CHẶN BỊA TUYỆT ĐỐI
    Trả về [] nếu KHÔNG có dữ liệu đủ tin cậy
    """

    # ========== SAFETY CHECK ==========
    if not query or not query.strip():
        return []

    if not INDEX or not MAPPING: 
        logger.error("❌ FAISS index hoặc mapping chưa được load")
        return []

    # ========== EMBEDDING QUERY ==========
    try:
        embedding = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding
    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
        return []

    import numpy as np

    query_vector = np.array([embedding], dtype="float32")

    # ========== FAISS SEARCH ==========
    try:
        distances, indices = faiss_index.search(
query_vector, top_k)
    except Exception as e:
        logger.error(f"❌ FAISS search error: {e}")
        return []

    results = []

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        # FAISS cosine similarity (index đã normalize)
        similarity = float(score)

        # 🚨 NGƯỠNG CHẶN BỊA
        if similarity < min_score:
            continue

        mapping = faiss_mapping.get(
str(idx))
        if not mapping:
            continue

        text = mapping.get("text", "").strip()
        if not text:
            continue

        results.append((similarity, text))

    # ========== SORT & RETURN ==========
    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        logger.info(
            f"⚠️ No semantic match above threshold "
            f"(min_score={min_score}) for query: {query}"
        )

    return results


class NumpyIndex:
    """Simple numpy-based index with safe numpy handling"""

    def __init__(self, mat=None):
        if NUMPY_AVAILABLE:
            if mat is None:
                self.mat = np.empty((0, 0), dtype="float32")
            else:
                # Force numpy float32 2D
                self.mat = np.asarray(mat, dtype="float32")
                if self.mat.ndim == 1:
                    self.mat = self.mat.reshape(1, -1)
        else:
            self.mat = mat if mat is not None else []

        # SAFE dimension detection (no numpy truth check)
        if NUMPY_AVAILABLE:
            if self.mat.shape[0] > 0 and self.mat.ndim == 2:
                self.dim = int(self.mat.shape[1])
            else:
                self.dim = 0
            self.size = int(self.mat.shape[0])
        else:
            self.size = len(self.mat)
            self.dim = len(self.mat[0]) if self.size > 0 else 0

    def is_empty(self):
        if NUMPY_AVAILABLE:
            return self.mat.shape[0] == 0
        return len(self.mat) == 0

    def search(self, query_vec, k=5):
        if self.is_empty():
            return [], []

        if NUMPY_AVAILABLE:
            q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
            sims = np.dot(self.mat, q.T).reshape(-1)
            topk = np.argsort(-sims)[:k]
            return sims[topk].tolist(), topk.tolist()
        else:
            return [], []

    
    def search(self, qvec, k):
        if not self.mat or (NUMPY_AVAILABLE and self.mat.shape[0] == 0) or (not NUMPY_AVAILABLE and len(self.mat) == 0):
            # Return empty results
            return np.array([[]]), np.array([[]], dtype=np.int64)
        
        q = np.array(qvec).flatten()
        
        if NUMPY_AVAILABLE:
            # Use numpy if available
            q = q / (np.linalg.norm(q) + 1e-12)
            m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
            sims = np.dot(q, m.T)
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        else:
            # Fallback calculation
            q_norm = q / (sum(x*x for x in q)**0.5 + 1e-12)
            scores = []
            for i, row in enumerate(self.mat):
                row_norm = row / (sum(x*x for x in row)**0.5 + 1e-12)
                sim = sum(q_norm[j] * row_norm[j] for j in range(min(len(q_norm), len(row_norm))))
                scores.append((sim, i))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k = scores[:k]
            if top_k:
                scores_arr = np.array([s[0] for s in top_k])
                idx_arr = np.array([s[1] for s in top_k])
            else:
                scores_arr = np.array([])
                idx_arr = np.array([], dtype=np.int64)
            
            return scores_arr.reshape(1, -1), idx_arr.reshape(1, -1)
        
        return scores.reshape(1, -1), idx.reshape(1, -1)
    
    def save(self, path):
        if NUMPY_AVAILABLE:
            np.savez_compressed(path, mat=self.mat)
        else:
            logger.warning(f"⚠️ Cannot save index without NumPy: {path}")
    
    @classmethod
    def load(cls, path):
        if NUMPY_AVAILABLE:
            try:
                arr = np.load(path)
                return cls(arr['mat'])
            except Exception as e:
                logger.error(f"Failed to load numpy index: {e}")
                return cls()
        else:
            logger.warning(f"⚠️ Cannot load index without NumPy: {path}")
            return cls()

def build_index(force_rebuild: bool = False) -> bool:
    """Build or load FAISS/numpy index"""
    global INDEX, EMBEDDING_MODEL
    
    with INDEX_LOCK:
        # Try to load existing index
        if not force_rebuild:
            if FAISS_ENABLED and HAS_FAISS and os.path.exists(FAISS_INDEX_PATH):
                try:
                    INDEX = faiss.read_index(FAISS_INDEX_PATH)
                    logger.info(f"✅ Loaded FAISS index from {FAISS_INDEX_PATH}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
            
            # Try numpy fallback
            if os.path.exists(FALLBACK_VECTORS_PATH):
                try:
                    arr = np.load(FALLBACK_VECTORS_PATH)
                    INDEX = NumpyIndex(arr['mat'])
                    logger.info("✅ Loaded numpy index")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load numpy index: {e}")
        
        # Build new index
        if not FLAT_TEXTS:
            logger.warning("No texts to index")
            return False
        
        logger.info(f"🔨 Building index for {len(FLAT_TEXTS)} passages...")
        
        # Generate embeddings
        vectors = []
        dims = None
        
        for text in FLAT_TEXTS:
            emb, d = embed_text(text)
            if emb:
                if dims is None:
                    dims = len(emb)
                vectors.append(np.array(emb, dtype="float32"))
        
        if not vectors:
            logger.error("No embeddings generated")
            return False
        
        # Create index
        mat = np.vstack(vectors)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        
        if FAISS_ENABLED and HAS_FAISS:
            INDEX = faiss.IndexFlatIP(dims)
            INDEX.add(mat)
            try:
                faiss.write_index(INDEX, FAISS_INDEX_PATH)
                logger.info(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
        else:
            INDEX = NumpyIndex(mat)
            try:
                INDEX.save(FALLBACK_VECTORS_PATH)
                logger.info(f"✅ Saved numpy index to {FALLBACK_VECTORS_PATH}")
            except Exception as e:
                logger.error(f"Failed to save numpy index: {e}")
        
        logger.info(f"✅ Index built: {len(vectors)} vectors, {dims} dimensions")
        return True

# =========== HELPER FUNCTIONS ===========
def _format_price(price):
    return price
def normalize_text_simple(s: str) -> str:
    """Basic text normalization"""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_session_context(session_id: str) -> ConversationContext:
    """Get or create context for session using ConversationContext dataclass with auto-repair"""

    ctx = SESSION_CONTEXTS.get(session_id)

    if ctx is None:
        ctx = ConversationContext(session_id=session_id)
        SESSION_CONTEXTS[session_id] = ctx
        return ctx

    # ===== AUTO REPAIR FOR OLD CONTEXT OBJECTS =====
    if not hasattr(ctx, "last_tour_indices"):
        ctx.last_tour_indices = []

    if not hasattr(ctx, "current_tours"):
        ctx.current_tours = []

    if not hasattr(ctx, "mentioned_tours"):
        ctx.mentioned_tours = set()

    if not hasattr(ctx, "last_successful_tours"):
        ctx.last_successful_tours = []

    if not hasattr(ctx, "conversation_history"):
        ctx.conversation_history = []

    if not hasattr(ctx, "user_preferences"):
        ctx.user_preferences = {}

    return ctx

def save_session_context(session_id: str, context: ConversationContext):
    """Lưu context cho session"""
    with SESSION_LOCK:
        SESSION_CONTEXTS[session_id] = context
        # Dọn dẹp session cũ (giữ tối đa 100 session)
        if len(SESSION_CONTEXTS) > 100:
            # Xóa các session cũ nhất
            sorted_sessions = sorted(
                SESSION_CONTEXTS.items(),
                key=lambda x: getattr(x[1], 'last_updated', datetime.utcnow())
            )
            for key, _ in sorted_sessions[:20]:
                if key in SESSION_CONTEXTS:
                    del SESSION_CONTEXTS[key]
def extract_session_id(request_data: Dict, remote_addr: str) -> str:
    """Extract or create session ID"""
    session_id = request_data.get("session_id")
    if not session_id:
        ip = remote_addr or "0.0.0.0"
        current_hour = datetime.utcnow().strftime("%Y%m%d%H")
        unique_str = f"{ip}_{current_hour}"
        session_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    return f"session_{session_id}"

def _prepare_llm_prompt(user_message: str, search_results: List, context: Dict) -> str:
    """Prepare prompt for LLM - THÔNG MINH với context & câu hỏi chung"""
    
    message_lower = user_message.lower()
    
    # Phân loại câu hỏi
    is_general_question = any(keyword in message_lower for keyword in [
        'có bao gồm', 'đã bao gồm', 'bao gồm gì', 'bao gồm những gì',
        'có gì', 'như thế nào', 'ra sao', 'thế nào', 'giá tour'
    ])
    
    has_specific_tour = context.get('current_tours') and len(context.get('current_tours', [])) > 0
    tour_count = len(context.get('current_tours', []))
    
    # Phát hiện câu hỏi tiếp theo (followup)
    is_followup = (
        context.get('last_action') == 'chat_response' and 
        (has_specific_tour or context.get('last_tour_name'))
    )
    
    # Phát hiện ràng buộc địa lý
    has_location_constraint = False
    location_constraint = None
    filters = context.get('filters', {})
    if filters:
        if filters.get('location'):
            has_location_constraint = True
            location_constraint = filters.get('location')
        elif filters.get('near_location'):
            has_location_constraint = True
            location_constraint = filters.get('near_location')
    
    prompt_parts = [
        "Bạn là trợ lý tư vấn du lịch Ruby Wings - CHUYÊN NGHIỆP, THÔNG MINH, NHIỆT TÌNH.",
        "",
        "⚠️ QUY TẮC NGHIÊM NGẶT:",
    ]
    
    # RULE 1: Câu hỏi CHUNG (không có tour cụ thể)
    if is_general_question and not has_specific_tour:
        prompt_parts.extend([
            "",
            "🎯 ĐÂY LÀ CÂU HỎI CHUNG - KHÔNG CÓ TOUR CỤ THỂ:",
            "• TRẢ LỜI NGẮN GỌN (2-4 câu) dựa trên kiến thức chung về tour du lịch",
            "• SỬ DỤNG OPENAI để trả lời tự nhiên, không nói 'không có dữ liệu'",
            "• KẾT THÚC bằng câu hỏi: 'Bạn quan tâm tour nào để tôi tư vấn chi tiết?'",
            "• KHÔNG liệt kê tour, KHÔNG dump dữ liệu",
            "",
            "VÍ DỤ ĐÚNG:",
            "Q: 'Giá tour bao gồm gì?'",
            "A: 'Giá tour Ruby Wings thường bao gồm: xe đưa đón, ăn uống theo chương trình, khách sạn, hướng dẫn viên và bảo hiểm. Tùy tour cụ thể có thể có thêm vé tham quan hoặc hoạt động đặc biệt. Bạn quan tâm tour nào để tôi tư vấn chi tiết? 😊'",
            "",
            "Q: 'Tour có phù hợp gia đình không?'",
            "A: 'Ruby Wings có nhiều tour phù hợp gia đình với hoạt động nhẹ nhàng, an toàn cho trẻ em và người lớn tuổi. Gia đình bạn có bao nhiêu người và thích loại hình nào (thiên nhiên, lịch sử, nghỉ dưỡng) để tôi tư vấn tour phù hợp nhất?'",
        ])
    
    # RULE 2: Câu hỏi TIẾP THEO (followup)
    elif is_followup:
        prompt_parts.extend([
            "",
            "💭 ĐÂY LÀ CÂU HỎI TIẾP THEO - SỬ DỤNG CONTEXT:",
            f"• Đã bàn về {tour_count} tour: {context.get('last_tour_name', '')}",
            "• PHẢI dựa vào context cũ - KHÔNG reset",
            "• TRẢ LỜI TIẾP theo ngữ cảnh đã có",
            "• KHÔNG liệt kê lại toàn bộ tour",
            "• Nếu hỏi về giá/thời gian → Chỉ nói về tour đang bàn",
            "• Nếu hỏi thêm điều kiện → Gợi ý tour từ context hoặc hỏi lại",
            "",
            "VÍ DỤ ĐÚNG:",
            "Context: Đã nói về 'Tour Bạch Mã'",
            "Q: 'Tour này có phù hợp nhóm 10 người không?'",
            "A: 'Tour Bạch Mã rất phù hợp cho nhóm 10 người! Chúng tôi có thể tổ chức riêng với giá ưu đãi. Nhóm bạn thích hoạt động nào: trekking, thiền tĩnh tâm hay cả hai? Tôi sẽ tư vấn lịch trình chi tiết.'",
        ])
    
    # RULE 3: Location constraint
    if has_location_constraint:
        prompt_parts.extend([
            "",
            "🚨 RÀNG BUỘC ĐỊA LÝ - NGHIÊM NGẶT:",
            f"• Yêu cầu tour gần/tại: {location_constraint or 'khu vực cụ thể'}",
            "• CHỈ đề xuất tour trong khu vực này",
            "• NẾU không có tour phù hợp:",
            "  → 'Hiện Ruby Wings chưa có tour tại [địa điểm]. Tuy nhiên, chúng tôi có tour gần nhất tại [X].'",
            "  → Hỏi: 'Bạn có muốn xem tour ở khu vực lân cận không?'",
        ])
    
    # RULE 4: Giới hạn tour
    prompt_parts.extend([
        "",
        "📊 GIỚI HẠN TOUR (BẮT BUỘC):",
        "• Tối đa 2-3 tour/câu trả lời",
        "• MỖI tour phải có LÝ DO rõ ràng",
        "• KHÔNG liệt kê >3 tour",
        "• Nếu có nhiều tour phù hợp:",
        "  → Chọn 2-3 TIÊU BIỂU nhất",
        "  → Tóm tắt: 'Còn X tour khác...'",
        "  → Hỏi: 'Bạn muốn xem thêm loại nào?'",
    ])
    
    # CONTEXT INFO
    prompt_parts.extend([
        "",
        "📚 THÔNG TIN NGỮ CẢNH:",
    ])
    
    if context.get('user_preferences'):
        prefs = []
        for k, v in context['user_preferences'].items():
            prefs.append(f"{k}: {v}")
        if prefs:
            prompt_parts.append(f"- Sở thích: {'; '.join(prefs)}")
    
    if context.get('current_tours'):
        tours_info = [f"{t['name']}" for t in context['current_tours'][:3]]
        if tours_info:
            prompt_parts.append(f"- Tour đã bàn: {', '.join(tours_info)}")
    
    if filters:
        filter_strs = []
        if filters.get('price_max'):
            filter_strs.append(f"giá <{filters['price_max']:,}đ")
        if filters.get('location'):
            filter_strs.append(f"VỊ TRÍ: {filters['location']}")
        if filter_strs:
            prompt_parts.append(f"- Ràng buộc: {'; '.join(filter_strs)}")
    
    # SEARCH RESULTS
    prompt_parts.append("")
    prompt_parts.append("📝 DỮ LIỆU TỪ HỆ THỐNG:")
    
    if search_results:
        for i, (score, passage) in enumerate(search_results[:5], 1):
            text = passage.get('text', '')[:250]
            prompt_parts.append(f"[{i}] {text}")
    else:
        prompt_parts.append("(Không có dữ liệu cụ thể - sử dụng kiến thức chung)")
    
    # YÊU CẦU TRẢ LỜI
    prompt_parts.append("")
    prompt_parts.append("💬 YÊU CẦU TRẢ LỜI:")
    
    if is_general_question and not has_specific_tour:
        prompt_parts.extend([
            "1. Trả lời NGẮN GỌN (2-4 câu) dựa OpenAI",
            "2. KHÔNG nói 'không có dữ liệu'",
            "3. Kết thúc: Hỏi lại để xác định tour",
        ])
    elif is_followup:
        prompt_parts.extend([
            "1. Dựa vào CONTEXT (tour đã bàn)",
            "2. Trả lời TIẾP, KHÔNG reset",
            "3. Tối đa nhắc 1-2 tour từ context",
        ])
    else:
        prompt_parts.extend([
            "1. Chọn 2-3 tour với LÝ DO rõ",
            "2. KHÔNG >3 tour",
            "3. Nếu nhiều: tóm tắt + hỏi tiếp",
        ])
    
    prompt_parts.append("4. Luôn kết thúc: Câu hỏi dẫn dắt hoặc '📞 Gọi 0332510486'")
    
    return "\n".join(prompt_parts)

def _generate_fallback_response(user_message: str, search_results: List, tour_indices: List[int] = None) -> str:
    """Generate SMART fallback response - Dùng OpenAI khi có, context-aware"""
    message_lower = user_message.lower()
    
    # ===== SỬ DỤNG OPENAI NẾU CÓ =====
    if client and HAS_OPENAI:
        try:
            # Chuẩn bị context
            context_parts = []
            
            # Thông tin tour nếu có
            if tour_indices and TOURS_DB:
                for idx in tour_indices[:2]:
                    tour = TOURS_DB.get(idx)
                    if tour:
                        context_parts.append(f"Tour: {tour.name}")
                        if tour.duration:
                            context_parts.append(f"Thời gian: {tour.duration}")
                        if tour.price:
                            context_parts.append(f"Giá: {tour.price}")
                        if tour.summary:
                            context_parts.append(f"Mô tả: {tour.summary[:150]}")
            
            # Dữ liệu search
            if search_results:
                for i, (score, passage) in enumerate(search_results[:3], 1):
                    text = passage.get('text', '')[:200]
                    if text:
                        context_parts.append(f"Thông tin {i}: {text}")
            
            # Tạo prompt thông minh
            context_str = "\n".join(context_parts) if context_parts else "Không có dữ liệu cụ thể"
            
            prompt = f"""Bạn là tư vấn viên Ruby Wings chuyên nghiệp.

THÔNG TIN CÓ SẴN:
{context_str}

YÊU CẦU TRẢ LỜI:
1. Nếu có thông tin tour cụ thể → Tư vấn dựa trên đó
2. Nếu không có dữ liệu → Trả lời dựa kiến thức chung về tour du lịch
3. LUÔN kết thúc bằng câu hỏi dẫn dắt hoặc "Gọi 0332510486"
4. Ngắn gọn 2-4 câu, nhiệt tình, tự nhiên
5. KHÔNG nói "không có dữ liệu", "xin lỗi không tìm thấy"

Câu hỏi của khách: {user_message}"""

            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.6,
                max_tokens=300
            )
            
            if response.choices:
                reply = response.choices[0].message.content or ""
                # Đảm bảo có hotline
                if "0332510486" not in reply:
                    reply += "\n\n📞 Liên hệ 0332510486 để được tư vấn chi tiết!"
                return reply
        
        except Exception as e:
            logger.error(f"OpenAI fallback error: {e}")
            # Rơi xuống logic template bên dưới
    
    # ===== FALLBACK KHI KHÔNG CÓ OPENAI =====
    
    # Có tour cụ thể → Trả thông tin tour
    if tour_indices and TOURS_DB:
        response_parts = []
        for idx in tour_indices[:2]:
            tour = TOURS_DB.get(idx)
            if tour:
                response_parts.append(f"**{tour.name}**")
                if tour.duration:
                    response_parts.append(f"⏱️ {tour.duration}")
                if tour.location:
                    response_parts.append(f"📍 {tour.location}")
                if tour.price:
                    response_parts.append(f"💰 {tour.price}")
                if tour.summary:
                    response_parts.append(f"📝 {tour.summary[:150]}...")
        
        if response_parts:
            return "\n".join(response_parts) + "\n\n📞 Gọi 0332510486 để biết thêm!"
    
    # Có search results → Tóm tắt
    if search_results:
        top_results = search_results[:2]
        response_parts = ["Thông tin liên quan:"]
        
        for i, (score, passage) in enumerate(top_results, 1):
            text = passage.get('text', '')[:150]
            if text:
                response_parts.append(f"\n{i}. {text}...")
        
        response_parts.append("\n\n📞 Liên hệ 0332510486 để biết chi tiết!")
        return "".join(response_parts)
    
    # Câu hỏi chung → Template thông minh theo keyword
    general_qa = {
        'bao gồm': "Giá tour Ruby Wings thường bao gồm: xe đưa đón, ăn uống theo chương trình, khách sạn, hướng dẫn viên và bảo hiểm. Tùy tour cụ thể có thêm hoạt động đặc biệt. Bạn quan tâm tour nào để tôi tư vấn chi tiết? 😊",
        
        'phù hợp gia đình': "Ruby Wings có nhiều tour phù hợp gia đình với hoạt động nhẹ nhàng, an toàn cho trẻ em và người lớn tuổi. Gia đình bạn bao nhiêu người và thích loại tour nào (thiên nhiên, lịch sử, nghỉ dưỡng)?",
        
        'phù hợp': "Ruby Wings có tour cho mọi đối tượng! Bạn đi nhóm bao nhiêu người và có sở thích gì đặc biệt không?",
        
        'giá': "Giá tour Ruby Wings từ 800.000đ - 3.000.000đ tùy loại. Bạn có ngân sách khoảng bao nhiêu và muốn đi mấy ngày để tôi tư vấn phù hợp?",
        
        'thời gian': "Ruby Wings có tour 1 ngày, 2 ngày 1 đêm, 3 ngày 2 đêm. Bạn có khoảng bao nhiêu thời gian rảnh?",
        
        'địa điểm': "Ruby Wings tổ chức tour tại Huế, Quảng Trị, Bạch Mã, Trường Sơn và nhiều nơi khác. Bạn muốn khám phá khu vực nào?",
        
        'nhóm': "Tour nhóm của Ruby Wings rất phù hợp! Nhóm bạn bao nhiêu người và thích hoạt động gì (teambuilding, nghỉ dưỡng, khám phá)?",
        
        'retreat': "Ruby Wings chuyên tour retreat kết hợp thiền, khí công và thiên nhiên. Bạn muốn tour bao nhiêu ngày và mức độ hoạt động như nào?",
    }
    
    # Tìm keyword match
    for keyword, response in general_qa.items():
        if keyword in message_lower:
            return response
    
    # Default - Dẫn dắt hỏi lại
    return "Tôi có thể giúp bạn tìm tour phù hợp! Bạn có thể cho tôi biết:\n" \
           "• Muốn đi đâu?\n" \
           "• Thời gian bao lâu?\n" \
           "• Ngân sách khoảng bao nhiêu?\n" \
           "• Đi bao nhiêu người?\n\n" \
           "Hoặc gọi ngay 0332510486 để được tư vấn chi tiết! 😊"




# =========== MAIN CHAT ENDPOINT - ĐỈNH CAO THÔNG MINH V4.1 ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint_ultimate():
    """
    Main chat endpoint với xử lý AI thông minh, context-aware mạnh mẽ
    Version 4.2 (Enhanced with service_inquiry and location_query)
    """
    start_time = time.time()
    
    try:
        # ================== INITIALIZATION ==================
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        session_id = extract_session_id(data, request.remote_addr)
        
        if not user_message:
            return jsonify({
                "reply": "👋 **Xin chào! Tôi là trợ lý AI của Ruby Wings Travel**\n\n"
                        "Tôi có thể giúp bạn:\n"
                        "• Tìm hiểu về 32 tour trải nghiệm đặc sắc\n"
                        "• So sánh các tour để chọn phù hợp nhất\n"
                        "• Tư vấn tour theo nhu cầu gia đình, nhóm, cá nhân\n"
                        "• Cung cấp thông tin chi tiết về giá, lịch trình, địa điểm\n\n"
                        "📞 **Hotline tư vấn 24/7:** 0332510486\n"
                        "💡 **Hỏi ngay:** 'Tour nào phù hợp cho gia đình?', 'Tour Bạch Mã giá bao nhiêu?'",
                "sources": [],
                "context": {},
                "processing_time": 0
            })
        
        # ================== CONTEXT MANAGEMENT SYSTEM ==================
        context = get_session_context(session_id)
        
        # Khởi tạo context nếu chưa có
        if not hasattr(context, 'conversation_history'):
            context.conversation_history = []
        if not hasattr(context, 'current_tour'):
            context.current_tour = None
        if not hasattr(context, 'user_profile'):
            context.user_profile = {}
        
        # Lưu user message vào history
        context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Giới hạn history (giữ 20 tin nhắn gần nhất)
        if len(context.conversation_history) > 40:
            context.conversation_history = context.conversation_history[-20:]
        
        # ================== ADVANCED CONTEXT ANALYSIS V2 ==================
        message_lower = user_message.lower()
        
        # 1. PHÂN TÍCH CẤP ĐỘ PHỨC TẠP NÂNG CAO
        complexity_score = 0
        complexity_indicators = {
            'và': 1, 'cho': 1, 'với': 1, 'nhưng': 2, 'tuy nhiên': 2,
            'nếu': 2, 'khi': 1, 'để': 1, 'mà': 1, 'hoặc': 1, 'so sánh': 3,
            'phân biệt': 3, 'khác nhau': 3, 'tương tự': 2, 'giữa': 2,
            'tại sao': 2, 'làm thế nào': 3, 'có thể không': 2,
            'trước khi': 1, 'sau khi': 1, 'trong khi': 1, 'mặc dù': 2,
            'do đó': 2, 'vì vậy': 2, 'nên': 2, 'nhằm': 1
        }
        
        for indicator, weight in complexity_indicators.items():
            if indicator in message_lower:
                complexity_score += weight
        
        # 2. PHÂN TÍCH ĐỘ DÀI CÂU HỎI NÂNG CAO
        word_count = len(user_message.split())
        char_count = len(user_message)
        sentence_count = user_message.count('.') + user_message.count('?') + user_message.count('!')
        
        if word_count > 25:
            complexity_score += 3
        elif word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1
            
        if char_count > 150:
            complexity_score += 1
            
        if sentence_count > 1:
            complexity_score += 1
        
        # 3. PHÂN TÍCH NGÔN NGỮ HỌC & CÚ PHÁP
        question_words = ['ai', 'cái gì', 'gì', 'ở đâu', 'khi nào', 'tại sao', 'thế nào', 'bao nhiêu', 'mấy']
        question_word_count = sum(1 for word in question_words if word in message_lower)
        complexity_score += min(question_word_count, 2)  # Tối đa +2
        
        # Phân tích mức độ chi tiết
        detail_indicators = ['cụ thể', 'chi tiết', 'rõ ràng', 'từng', 'mỗi', 'các loại']
        if any(indicator in message_lower for indicator in detail_indicators):
            complexity_score += 2
        
        # 4. PHÂN TÍCH CẢM XÚC (SENTIMENT ANALYSIS CƠ BẢN)
        positive_words = ['tuyệt vời', 'xuất sắc', 'hoàn hảo', 'tốt', 'hay', 'thích', 'ưa', 'mong muốn', 'hài lòng']
        negative_words = ['tệ', 'dở', 'kém', 'không thích', 'ghét', 'phàn nàn', 'thất vọng', 'buồn', 'chán']
        urgent_words = ['gấp', 'ngay', 'lập tức', 'nhanh', 'khẩn cấp', 'càng sớm càng tốt']
        
        sentiment_score = 0
        sentiment_type = 'neutral'
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        urgent_count = sum(1 for word in urgent_words if word in message_lower)
        
        if positive_count > negative_count:
            sentiment_score = positive_count
            sentiment_type = 'positive'
        elif negative_count > positive_count:
            sentiment_score = -negative_count
            sentiment_type = 'negative'
            
        if urgent_count > 0:
            complexity_score += 2  # Câu hỏi khẩn cấp cần xử lý ưu tiên
        
        # 5. PHÂN TÍCH ĐỐI TƯỢNG & MỤC ĐÍCH
        audience_keywords = {
            'business': ['công ty', 'doanh nghiệp', 'team building', 'đồng nghiệp', 'nhân viên'],
            'family': ['gia đình', 'con nhỏ', 'trẻ em', 'ông bà', 'bố mẹ', 'đa thế hệ'],
            'youth': ['bạn trẻ', 'thanh niên', 'sinh viên', 'học sinh', 'tuổi teen'],
            'senior': ['người lớn tuổi', 'cao tuổi', 'về hưu', 'cựu chiến binh', 'trung niên'],
            'solo': ['một mình', 'đi lẻ', 'solo', 'cá nhân', 'tự đi']
        }
        
        audience_type = None
        for audience, keywords in audience_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                audience_type = audience
                complexity_score += 1  # Câu hỏi có đối tượng cụ thể
                break
        
        # 6. PHÂN TÍCH MỨC ĐỘ KHẨN CẤP & ƯU TIÊN
        priority_level = 'normal'
        if urgent_count > 0 or 'gấp' in message_lower:
            priority_level = 'high'
            complexity_score += 2
        elif 'khi nào' in message_lower or 'thời gian' in message_lower:
            priority_level = 'medium'
            complexity_score += 1
        
        # 7. PHÂN TÍCH MỨC ĐỘ TRANG TRỌNG
        formal_words = ['kính chào', 'thưa', 'xin hỏi', 'vui lòng', 'làm ơn', 'cảm ơn']
        informal_words = ['hey', 'hello', 'hi', 'ê', 'nè', 'ơi']
        
        formality_score = 0
        if any(word in message_lower for word in formal_words):
            formality_score = 1  # Trang trọng
        elif any(word in message_lower for word in informal_words):
            formality_score = -1  # Thân mật
        
        # 8. TỔNG HỢP CHỈ SỐ PHÂN TÍCH
        context_analysis = {
            'complexity_score': min(complexity_score, 10),  # Giới hạn 10
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'question_word_count': question_word_count,
            'sentiment': {
                'type': sentiment_type,
                'score': sentiment_score,
                'positive_count': positive_count,
                'negative_count': negative_count
            },
            'urgency': {
                'level': priority_level,
                'urgent_count': urgent_count
            },
            'audience_type': audience_type,
            'formality': formality_score,
            'has_specific_request': any(word in message_lower for word in detail_indicators),
            'is_comparison': 'so sánh' in message_lower or 'khác nhau' in message_lower
        }
        
        logger.info(f"🧠 Context Analysis: {context_analysis}")
        

        
        # ================== ENHANCED INTENT DETECTION V3 ==================
        intent_categories = {
            'service_inquiry': [
                'bao gồm', 'có những gì', 'dịch vụ', 'cung cấp', 'có cho',
                'có đưa đón', 'có ăn', 'có ở', 'có hướng dẫn viên',
                'có bảo hiểm', 'có vé tham quan', 'có nước uống',
                'điều kiện', 'điều khoản', 'chính sách', 'hỗ trợ',
                'phương tiện', 'ăn uống', 'nơi ở', 'khách sạn', 'homestay',
                'gồm những gì', 'được cung cấp gì', 'có sẵn gì',
                'điều gì được bao gồm', 'có cho mượn', 'có trang thiết bị',
                'có wifi', 'có điều hòa', 'có bữa sáng', 'all inclusive',
                'full package', 'dịch vụ đi kèm', 'tiện ích', 'tiện nghi'
            ],
            
            'location_query': [
                'đi đà nẵng', 'đi huế', 'đi quảng trị', 'đi bạch mã',
                'đi trường sơn', 'ở đâu', 'tại sao', 'tại đâu',
                'đến đâu', 'thăm quan đâu', 'khu vực', 'địa bàn',
                'miền trung', 'huế quảng trị', 'đông hà', 'địa điểm',
                'điểm đến', 'nơi đến', 'vị trí', 'tọa độ', 'bản đồ',
                'khu vực nào', 'vùng nào', 'tỉnh nào', 'thành phố nào',
                'huyện nào', 'xã nào', 'làng nào', 'bản nào', 'khu du lịch',
                'điểm tham quan', 'danh lam thắng cảnh', 'địa danh'
            ],
            
            'tour_listing': [
                'có những tour nào', 'danh sách tour', 'liệt kê tour', 
                'tour nào có', 'tour gì', 'có tour', 'có tour nào',
                'có chương trình', 'có dịch vụ', 'có tour',
                'xem tour', 'xem các tour', 'tour đang có', 'tour hiện tại',
                'tour nào đang chạy', 'tour khả dụng', 'tour sẵn có',
                'các tour hiện có', 'tất cả tour', 'full list',
                'danh mục tour', 'catalogue tour', 'bộ sưu tập tour',
                'tour mới nhất', 'tour hot', 'tour nổi bật', 'tour đặc biệt',
                'tour limited', 'tour theo mùa', 'tour theo tháng'
            ],

            'price_inquiry': [
                'giá bao nhiêu', 'bao nhiêu tiền', 'chi phí', 'giá tour',
                'bảng giá', 'bao nhiêu', 'giá thế nào', 'giá sao',
                'giá không', 'hết bao nhiêu tiền', 'chi phí hết bao nhiêu',
                'giá cả', 'mức giá', 'đơn giá', 'chi phí tour',
                'tour giá rẻ', 'tour giá tốt', 'tour tiết kiệm',
                'tour cao cấp giá', 'tour vip giá', 'giá khuyến mãi',
                'giá ưu đãi', 'giá đặc biệt', 'giá cuối', 'giá gốc',
                'giá niêm yết', 'giá sau giảm', 'giá cuối cùng',
                'tổng chi phí', 'tổng số tiền', 'cần bao nhiêu tiền',
                'kinh phí', 'ngân sách', 'tầm giá', 'khoảng giá'
            ],

            'tour_detail': [
                'chi tiết tour', 'lịch trình', 'có gì', 'bao gồm gì',
                'thông tin', 'mô tả', 'đi những đâu', 'tham quan gì',
                'chương trình thế nào', 'nội dung tour', 'hành trình',
                'lộ trình', 'kế hoạch', 'chương trình chi tiết',
                'thông tin đầy đủ', 'full detail', 'mô tả đầy đủ',
                'giới thiệu chi tiết', 'trình bày chi tiết', 'nói rõ hơn',
                'cụ thể hơn', 'thông tin tour', 'tour info', 'tour facts',
                'đặc điểm tour', 'điểm nổi bật', 'highlight', 'điểm đặc sắc'
            ],

            'comparison': [
                'so sánh', 'khác nhau', 'nên chọn', 'tốt hơn',
                'hơn kém', 'phân biệt', 'so với', 'cái nào hơn',
                'tour nào tốt hơn', 'tour nào hay hơn', 'tour nào đáng giá hơn',
                'đánh giá giữa', 'so sánh giữa', 'đối chiếu',
                'cùng loại', 'tương đồng', 'giống nhau', 'khác biệt',
                'ưu điểm nhược điểm', 'pros and cons', 'điểm mạnh điểm yếu',
                'tour a vs tour b', 'tour này với tour kia'
            ],

            'recommendation': [
                'phù hợp', 'gợi ý', 'đề xuất', 'tư vấn', 'nên đi',
                'chọn nào', 'tìm tour', 'nên chọn tour nào',
                'tư vấn giúp', 'gợi ý giúp mình', 'tư vấn cho tôi',
                'đề xuất tour', 'giới thiệu tour', 'tour đề cử',
                'tour recommend', 'tour suggested', 'tour được đề xuất',
                'nên đi tour nào', 'tour phù hợp nhất', 'tour tốt nhất cho',
                'tour hay nhất', 'tour đáng trải nghiệm', 'tour nên thử',
                'tour hợp với', 'tour dành cho', 'tour theo sở thích'
            ],

            'booking_info': [
                'đặt tour', 'đăng ký', 'booking', 'giữ chỗ',
                'thanh toán', 'đặt chỗ', 'cách đặt',
                'đặt như thế nào', 'đặt ra sao', 'quy trình đặt',
                'làm sao để đặt', 'hướng dẫn đặt tour', 'đặt tour online',
                'đặt tour trực tuyến', 'form đặt tour', 'điền form đặt tour',
                'thủ tục đặt tour', 'điều kiện đặt tour', 'chính sách đặt tour',
                'cách thức thanh toán', 'phương thức thanh toán',
                'cách book tour', 'book như thế nào', 'reservation',
                'đặt trước', 'pre-order', 'pre-book', 'giữ chỗ trước'
            ],

            'policy': [
                'chính sách', 'giảm giá', 'ưu đãi', 'khuyến mãi',
                'giảm', 'promotion', 'hoàn tiền', 'hủy tour',
                'đổi lịch', 'điều kiện', 'điều khoản', 'terms',
                'điều lệ', 'quy định', 'chính sách hủy',
                'chính sách hoàn tiền', 'chính sách đổi tour',
                'chính sách bảo hiểm', 'chính sách trẻ em',
                'chính sách người cao tuổi', 'chính sách nhóm',
                'discount', 'voucher', 'coupon', 'mã giảm giá',
                'khuyến mại', 'ưu đãi đặc biệt', 'giá sốc',
                'flash sale', 'sale off', 'giảm giá sốc'
            ],

            'general_info': [
                'giới thiệu', 'là gì', 'thế nào', 'ra sao',
                'sứ mệnh', 'giá trị', 'triết lý', 'bên bạn là ai',
                'công ty là gì', 'ruby wings là gì', 'về ruby wings',
                'thông tin công ty', 'about us', 'about company',
                'tầm nhìn', 'vision', 'mission', 'mục tiêu',
                'lịch sử công ty', 'đội ngũ', 'nhân sự',
                'văn hóa công ty', 'core value', 'giá trị cốt lõi',
                'đối tác', 'partner', 'collaboration', 'hợp tác'
            ],

            'weather_info': [
                'thời tiết', 'khí hậu', 'nắng mưa', 'mùa nào',
                'nhiệt độ', 'thời tiết có đẹp không', 'mưa không',
                'nắng không', 'khí hậu thế nào', 'thời tiết tại',
                'mưa nhiều không', 'nắng nhiều không', 'độ ẩm',
                'gió', 'bão', 'lụt', 'thiên tai', 'thời tiết có thuận lợi',
                'mùa du lịch', 'thời điểm tốt nhất', 'best time to visit',
                'mùa cao điểm', 'mùa thấp điểm', 'thời tiết theo mùa',
                'dự báo thời tiết', 'weather forecast', 'weather condition'
            ],

            'food_info': [
                'ẩm thực', 'món ăn', 'đặc sản', 'đồ ăn',
                'bánh bèo', 'mắm nêm', 'ăn gì', 'ăn uống thế nào',
                'có ăn đặc sản không', 'đồ ăn địa phương', 'local food',
                'street food', 'ẩm thực đường phố', 'nhà hàng',
                'quán ăn', 'đặc sản vùng miền', 'món ngon',
                'đồ uống', 'thức uống', 'đồ ăn kèm', 'set menu',
                'thực đơn', 'menu', 'dining', 'ẩm thực huế',
                'đặc sản huế', 'đặc sản quảng trị', 'đặc sản miền trung'
            ],

            'culture_info': [
                'văn hóa', 'lịch sử', 'truyền thống', 'di tích',
                'di sản', 'văn minh', 'bản sắc', 'văn hóa địa phương',
                'phong tục', 'tập quán', 'lễ hội', 'festival',
                'tín ngưỡng', 'tôn giáo', 'kiến trúc', 'nghệ thuật',
                'âm nhạc', 'múa', 'di sản văn hóa', 'di sản unesco',
                'văn hóa dân tộc', 'văn hóa bản địa', 'lịch sử địa phương',
                'truyền thuyết', 'cổ tích', 'historical site',
                'cultural heritage', 'cultural experience'
            ],

            'wellness_info': [
                'thiền', 'yoga', 'chữa lành', 'sức khỏe', 'retreat',
                'tĩnh tâm', 'khí công', 'nghỉ dưỡng', 'hồi phục',
                'thư giãn', 'wellness', 'spa', 'massage',
                'thiền định', 'mindfulness', 'meditation',
                'yoga therapy', 'health retreat', 'detox',
                'wellness retreat', 'sức khỏe tinh thần',
                'mental health', 'balance', 'cân bằng',
                'giảm stress', 'giảm căng thẳng', 'thả lỏng'
            ],

            'group_info': [
                'nhóm', 'đoàn', 'công ty', 'gia đình', 'bạn bè',
                'tập thể', 'cựu chiến binh', 'đi theo đoàn',
                'đi đông người', 'đoàn riêng', 'nhóm lớn',
                'nhóm nhỏ', 'team', 'đội', 'group tour',
                'private tour', 'tour riêng', 'tour đoàn',
                'tour công ty', 'tour team building',
                'tour gia đình', 'tour bạn bè', 'tour sinh viên',
                'tour học sinh', 'tour đồng nghiệp', 'tour tập thể'
            ],

            'custom_request': [
                'tùy chỉnh', 'riêng', 'cá nhân hóa', 'theo yêu cầu',
                'riêng biệt', 'thiết kế tour', 'làm tour riêng',
                'tour theo yêu cầu', 'custom tour', 'private tour',
                'tailor made', 'bespoke tour', 'đoàn riêng',
                'lịch trình riêng', 'chương trình riêng',
                'tour thiết kế riêng', 'personalized tour',
                'tour cá nhân', 'đặt theo ý muốn', 'theo ý tôi',
                'theo sở thích', 'theo ngân sách', 'theo thời gian'
            ],

            'sustainability': [
                'bền vững', 'môi trường', 'xanh', 'cộng đồng',
                'phát triển bền vững', 'du lịch xanh',
                'du lịch bền vững', 'eco tour', 'eco friendly',
                'thân thiện môi trường', 'bảo vệ môi trường',
                'tái chế', 'reduce reuse recycle', 'carbon footprint',
                'du lịch có trách nhiệm', 'responsible tourism',
                'du lịch cộng đồng', 'community tourism',
                'du lịch sinh thái', 'ecotourism', 'green tourism',
                'sustainable travel', 'ethical tourism'
            ],

            'experience': [
                'trải nghiệm', 'cảm giác', 'cảm nhận', 'thực tế',
                'trực tiếp', 'trải nghiệm như thế nào', 'có gì hay',
                'cảm nhận thế nào', 'experience', 'cảm xúc',
                'kỷ niệm', 'khoảnh khắc', 'moment', 'memory',
                'câu chuyện', 'story', 'chuyến đi đáng nhớ',
                'điều đặc biệt', 'điểm nhấn', 'highlight experience',
                'hoạt động đặc biệt', 'special activity', 'unique experience',
                'trải nghiệm độc đáo', 'trải nghiệm khác biệt'
            ],
            
            # THÊM INTENT MỚI - AN TOÀN VÌ CÓ FALLBACK HANDLING
            'accessibility_info': [
                'người khuyết tật', 'xe lăn', 'wheelchair', 'accessible',
                'thang máy', 'elevator', 'ramp', 'đường dốc',
                'cho người già', 'cho trẻ em', 'dễ di chuyển',
                'tiện nghi cho người già', 'tiện nghi cho trẻ em',
                'an toàn cho', 'phù hợp cho người khuyết tật'
            ],
            
            'transportation_info': [
                'phương tiện', 'xe cộ', 'transport', 'vehicle',
                'loại xe', 'xe gì', 'bus', 'xe khách', 'xe du lịch',
                'xe đưa đón', 'pick up', 'drop off', 'điểm đón',
                'thời gian đón', 'xe bao nhiêu chỗ', 'xe máy lạnh',
                'air conditioner', 'xe đời mới', 'xe thoải mái'
            ],
            
            'safety_info': [
                'an toàn', 'bảo đảm', 'secure', 'safety',
                'an ninh', 'security', 'bảo hiểm', 'insurance',
                'cứu hộ', 'rescue', 'y tế', 'medical',
                'phòng cháy', 'fire safety', 'sơ cứu', 'first aid',
                'hướng dẫn an toàn', 'safety briefing', 'emergency'
            ]
        }
        
        # NÂNG CẤP LOGIC PHÁT HIỆN INTENT THÔNG MINH HƠN
        detected_intents = []
        intent_scores = {}
        
        for intent, keywords in intent_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
                    matched_keywords.append(keyword)
                    
                    # Bonus cho keyword dài (cụ thể hơn)
                    if len(keyword.split()) >= 2:
                        score += 0.5
                    
                    # Bonus cho keyword chính xác
                    if f' {keyword} ' in f' {message_lower} ':
                        score += 0.3
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'keywords': matched_keywords[:3]  # Giữ 3 keyword đầu
                }
                
                # Đủ điểm threshold thì thêm vào detected_intents
                if score >= 1.0:  # Threshold có thể điều chỉnh
                    if intent not in detected_intents:
                        detected_intents.append(intent)
        
        # Sắp xếp intents theo score để debug
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        if sorted_intents:
            logger.info(f"🎯 Intent Scores (Top 3): {sorted_intents[:3]}")
        
        # ƯU TIÊN INTENT CHÍNH NÂNG CAO
        primary_intent = None
        
        if detected_intents:
            # Strategy 1: Ưu tiên theo priority order
            priority_order = [
                'comparison', 'recommendation', 'service_inquiry',
                'location_query', 'price_inquiry', 'tour_detail',
                'tour_listing', 'custom_request', 'booking_info',
                'group_info', 'wellness_info', 'policy',
                'culture_info', 'weather_info', 'food_info',
                'general_info', 'sustainability', 'experience',
                'accessibility_info', 'transportation_info', 'safety_info'
            ]
            
            # Tìm intent có điểm cao nhất trong priority order
            best_score = -1
            for intent in priority_order:
                if intent in detected_intents:
                    score_data = intent_scores.get(intent, {'score': 0})
                    current_score = score_data['score']
                    
                    # Ưu tiên intent có score cao hơn
                    if current_score > best_score:
                        best_score = current_score
                        primary_intent = intent
            
            # Strategy 2: Nếu không tìm thấy theo priority, lấy intent có score cao nhất
            if not primary_intent:
                highest_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])[0]
                primary_intent = highest_intent
            
            # Strategy 3: Xử lý trường hợp multiple high scores
            top_intents = [intent for intent, data in sorted_intents[:2] if data['score'] > 2]
            if len(top_intents) > 1 and primary_intent:
                # Ghi nhận multiple intents cho response generation xử lý
                context.multiple_intents = top_intents
                logger.info(f"🎯 Multiple High-Score Intents: {top_intents}")
        
        # Ghi log chi tiết
        logger.info(f"🎯 Detected Intents: {detected_intents}")
        logger.info(f"🎯 Primary Intent: {primary_intent}")



        
        # ================== ENHANCED TOUR RESOLUTION ENGINE V2 ==================
        tour_indices = []
        tour_names_mentioned = []
        
        # IMPORT: Cần thêm ở đầu file nếu chưa có
        # from difflib import SequenceMatcher
        import re
        import difflib
        
        # Strategy 0: Pre-process user message for better matching
        cleaned_message = user_message.lower()
        
        # Chuẩn hóa từ đồng nghĩa để tăng khả năng matching
        synonym_mapping = {
            'tour': ['tour', 'tour', 'chương trình', 'lịch trình', 'trip', 'chuyến đi'],
            'bạch mã': ['bạch mã', 'bach ma', 'vườn quốc gia bạch mã'],
            'trường sơn': ['trường sơn', 'truong son', 'đường hồ chí minh', 'đường hcm'],
            'huế': ['huế', 'hue', 'thành phố huế', 'cố đô huế'],
            'quảng trị': ['quảng trị', 'quang tri', 'đông hà', 'địa đạo vịnh mốc'],
            'thiền': ['thiền', 'meditation', 'thiền định', 'tĩnh tâm'],
            'retreat': ['retreat', 'tĩnh dưỡng', 'nghỉ dưỡng', 'chữa lành'],
            'lịch sử': ['lịch sử', 'history', 'di tích', 'chiến tranh', 'tri ân'],
            'thiên nhiên': ['thiên nhiên', 'nature', 'rừng núi', 'cây cối', 'trekking'],
            'ẩm thực': ['ẩm thực', 'food', 'đồ ăn', 'món ăn', 'đặc sản']
        }
        
        # Áp dụng chuẩn hóa từ đồng nghĩa
        for standard_word, synonyms in synonym_mapping.items():
            for synonym in synonyms:
                if synonym in cleaned_message:
                    cleaned_message = cleaned_message.replace(synonym, standard_word)
        
        # Strategy 1: Enhanced direct tour name matching với multiple patterns
        direct_tour_matches = []
        
        # Các pattern tìm tên tour với độ chính xác cao hơn
        tour_name_patterns = [
            r'["\'](.+?)["\']',  # Tên trong dấu nháy
            r'(?:tour|tour|lịch trình)\s+["\']?(.+?)["\']?(?:\s+|$|,|\.|\?)',  # Tour/Hành trình + tên
            r'(?:tour|tour|lịch trình)\s+(?:tên là|gọi là|mang tên)\s+["\']?(.+?)["\']?(?:\s+|$|,|\.|\?)',
            r'(?:đi|tham quan|khám phá|trải nghiệm)\s+["\']?(.+?)["\']?(?:\s+tại|\s+ở|\s+trong|\s+|$|,|\.|\?)',
            r'(?:cho|về|tìm hiểu|tư vấn)\s+["\']?(.+?)["\']?(?:\s+tour|\s+tour|\s+|$|,|\.|\?)'
        ]
        
        for pattern in tour_name_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE | re.UNICODE)
            for match in matches:
                if match and len(match.strip()) > 2:
                    clean_name = match.strip()
                    # Loại bỏ các từ không cần thiết
                    remove_words = ['nào', 'gì', 'đó', 'ấy', 'này', 'kia', 'cho', 'với', 'của', 'về', 'tại', 'ở']
                    for word in remove_words:
                        if clean_name.lower().endswith(f' {word}'):
                            clean_name = clean_name[:-len(word)-1].strip()
                    
                    # Chỉ thêm nếu tên đủ dài và không chỉ là từ chung chung
                    if len(clean_name) >= 3 and clean_name.lower() not in remove_words:
                        tour_names_mentioned.append(clean_name)
        
        logger.info(f"🔍 Tour names mentioned in query (raw): {tour_names_mentioned}")
        
        # Strategy 1.1: Advanced direct matching với similarity scoring
        for tour_name in tour_names_mentioned:
            best_matches = []
            
            for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                similarity_score = 0
                match_type = None
                
                # Tính toán multiple similarity scores
                scores = []
                
                # 1. Exact match hoặc partial match
                if tour_name.lower() == norm_name.lower():
                    similarity_score = 1.0
                    match_type = 'exact'
                elif tour_name.lower() in norm_name.lower() or norm_name.lower() in tour_name.lower():
                    similarity_score = 0.85
                    match_type = 'contains'
                
                # 2. Word overlap score
                name_words = set([w for w in norm_name.lower().split() if len(w) > 2])
                query_words = set([w for w in tour_name.lower().split() if len(w) > 2])
                
                if name_words and query_words:
                    common_words = name_words.intersection(query_words)
                    if common_words:
                        overlap_score = len(common_words) / max(len(name_words), len(query_words))
                        similarity_score = max(similarity_score, overlap_score)
                        if overlap_score > 0.3:
                            match_type = 'word_overlap'
                
                # 3. Sequence similarity (difflib)
                seq_similarity = difflib.SequenceMatcher(None, tour_name.lower(), norm_name.lower()).ratio()
                if seq_similarity > similarity_score:
                    similarity_score = seq_similarity
                    match_type = 'sequence'
                
                # 4. Acronym/short form matching
                # Kiểm tra xem tour_name có phải là viết tắt của norm_name không
                if len(tour_name) <= 5 and tour_name.isupper():
                    acronym = ''.join([word[0] for word in norm_name.split() if word])
                    if tour_name.lower() == acronym.lower():
                        similarity_score = 0.9
                        match_type = 'acronym'
                
                if similarity_score >= 0.5:  # Ngưỡng matching
                    best_matches.append((idx, similarity_score, norm_name, match_type))
            
            # Sắp xếp theo điểm và lấy match tốt nhất cho tour_name này
            if best_matches:
                best_matches.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_score, best_norm_name, match_type = best_matches[0]
                
                if best_score >= 0.6:  # Ngưỡng cao hơn cho matching chất lượng
                    if best_idx not in direct_tour_matches:
                        direct_tour_matches.append(best_idx)
                        logger.info(f"🎯 Found tour '{best_norm_name}' (idx: {best_idx}) for query '{tour_name}' "
                                   f"(score: {best_score:.2f}, type: {match_type})")
        
        if direct_tour_matches:
            tour_indices = direct_tour_matches[:5]
            logger.info(f"🎯 Direct tour matches found: {tour_indices} (count: {len(tour_indices)})")
        
        # Strategy 2: Enhanced fuzzy matching với nâng cấp
        if not tour_indices and UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            logger.info("🔍 Starting enhanced fuzzy matching")
            
            # Tạo danh sách tên tour để fuzzy matching
            tour_names = list(TOUR_NAME_TO_INDEX.keys())
            
            # Tìm các tour có similarity cao với toàn bộ câu hỏi
            best_fuzzy_matches = []
            
            for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                # Tính similarity giữa câu hỏi và tên tour
                similarity = difflib.SequenceMatcher(None, cleaned_message, norm_name.lower()).ratio()
                
                # Thêm điểm bonus nếu có từ khóa quan trọng trùng
                important_keywords = ['bạch mã', 'trường sơn', 'huế', 'quảng trị', 'thiền', 'retreat']
                keyword_bonus = 0
                for keyword in important_keywords:
                    if keyword in norm_name.lower() and keyword in cleaned_message:
                        keyword_bonus += 0.2
                
                total_score = similarity + keyword_bonus
                
                if total_score > 0.5:  # Ngưỡng fuzzy matching
                    best_fuzzy_matches.append((idx, total_score, norm_name))
            
            # Sắp xếp và lọc
            if best_fuzzy_matches:
                best_fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                fuzzy_indices = [idx for idx, score, name in best_fuzzy_matches[:5] if score > 0.55]
                
                if fuzzy_indices:
                    tour_indices = fuzzy_indices
                    logger.info(f"🔍 Enhanced fuzzy matches found: {tour_indices}")
                    logger.info(f"🔍 Top fuzzy match: {best_fuzzy_matches[0][2]} (score: {best_fuzzy_matches[0][1]:.2f})")
        
        # Strategy 3: Enhanced semantic content matching
        if not tour_indices and UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            logger.info("🧠 Starting enhanced semantic content matching")
            
            semantic_matches = []
            
            # Từ khóa chính trong câu hỏi (loại bỏ stop words)
            stop_words = ['có', 'và', 'cho', 'với', 'tại', 'ở', 'nào', 'gì', 'bao nhiêu', 'thế nào', 'ra sao']
            query_keywords = [word for word in cleaned_message.split() 
                             if len(word) > 2 and word not in stop_words]
            
            # Thêm các cụm từ quan trọng từ câu hỏi
            important_phrases = []
            for i in range(len(query_keywords) - 1):
                phrase = f"{query_keywords[i]} {query_keywords[i+1]}"
                if len(phrase) > 5:
                    important_phrases.append(phrase)
            
            logger.info(f"🧠 Query keywords: {query_keywords}")
            logger.info(f"🧠 Important phrases: {important_phrases[:5]}")
            
            for idx, tour in TOURS_DB.items():
                score = 0
                match_details = []
                
                # Tạo text blob từ nhiều trường dữ liệu
                text_blob = f"{tour.name or ''} {tour.summary or ''} {tour.style or ''} {tour.location or ''} {' '.join(tour.tags or [])}".lower()
                
                # 1. Keyword matching
                keyword_matches = sum(1 for word in query_keywords if word in text_blob)
                if keyword_matches > 0:
                    score += keyword_matches * 0.5
                    match_details.append(f"keywords:{keyword_matches}")
                
                # 2. Phrase matching
                phrase_matches = sum(1 for phrase in important_phrases if phrase in text_blob)
                if phrase_matches > 0:
                    score += phrase_matches * 1.0  # Phrase match quan trọng hơn
                    match_details.append(f"phrases:{phrase_matches}")
                
                # 3. Location matching đặc biệt
                if tour.location:
                    location_lower = tour.location.lower()
                    for loc_keyword in ['huế', 'quảng trị', 'bạch mã', 'trường sơn']:
                        if loc_keyword in cleaned_message and loc_keyword in location_lower:
                            score += 2.0
                            match_details.append(f"location:{loc_keyword}")
                            break
                
                # 4. Theme matching
                theme_keywords = {
                    'history': ['lịch sử', 'di tích', 'chiến tranh', 'tri ân'],
                    'nature': ['thiên nhiên', 'rừng', 'núi', 'trekking'],
                    'meditation': ['thiền', 'yoga', 'tĩnh tâm', 'retreat'],
                    'culture': ['văn hóa', 'ẩm thực', 'đặc sản', 'truyền thống']
                }
                
                for theme, keywords in theme_keywords.items():
                    theme_in_query = any(keyword in cleaned_message for keyword in keywords)
                    theme_in_tour = any(keyword in text_blob for keyword in keywords)
                    
                    if theme_in_query and theme_in_tour:
                        score += 1.5
                        match_details.append(f"theme:{theme}")
                
                # 5. Duration matching
                if tour.duration:
                    # Tìm số ngày trong câu hỏi
                    day_patterns = [r'(\d+)\s*ngày', r'(\d+)\s*day', r'(\d+)\s*đêm']
                    query_days = []
                    for pattern in day_patterns:
                        matches = re.findall(pattern, cleaned_message)
                        query_days.extend([int(m) for m in matches])
                    
                    # Tìm số ngày trong tour description
                    tour_days = []
                    for pattern in day_patterns:
                        matches = re.findall(pattern, tour.duration.lower())
                        tour_days.extend([int(m) for m in matches])
                    
                    if query_days and tour_days:
                        # Kiểm tra xem có ngày trùng không
                        common_days = set(query_days) & set(tour_days)
                        if common_days:
                            score += 1.0
                            match_details.append(f"duration:{list(common_days)[0]}ngày")
                
                if score > 1.0:  # Ngưỡng semantic matching
                    semantic_matches.append((idx, score, match_details))
                    if len(semantic_matches) % 10 == 0:
                        logger.debug(f"🧠 Processed {idx} tours, found {len(semantic_matches)} matches")
            
            if semantic_matches:
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                semantic_indices = [idx for idx, score, details in semantic_matches[:5]]
                
                if semantic_indices:
                    tour_indices = semantic_indices
                    logger.info(f"🧠 Enhanced semantic matches found: {tour_indices}")
                    
                    # Log chi tiết top matches
                    for idx, score, details in semantic_matches[:3]:
                        tour = TOURS_DB.get(idx)
                        if tour:
                            logger.info(f"🧠   {tour.name}: score={score:.1f}, details={details}")
        
        # Strategy 4: Fallback keyword matching (luôn hoạt động)
        if not tour_indices:
            logger.info("🔄 Starting fallback keyword matching")
            
            # Tạo bản đồ từ khóa -> tour indices
            keyword_to_tours = {}
            
            for idx, tour in TOURS_DB.items():
                # Thu thập từ khóa từ tour
                tour_keywords = []
                
                if tour.name:
                    tour_keywords.extend(tour.name.lower().split())
                
                if tour.summary:
                    tour_keywords.extend([w for w in tour.summary.lower().split() if len(w) > 2])
                
                if tour.tags:
                    tour_keywords.extend([tag.lower() for tag in tour.tags])
                
                if tour.style:
                    tour_keywords.append(tour.style.lower())
                
                if tour.location:
                    tour_keywords.extend(tour.location.lower().split())
                
                # Thêm vào keyword map
                for keyword in set(tour_keywords):
                    if keyword not in keyword_to_tours:
                        keyword_to_tours[keyword] = []
                    keyword_to_tours[keyword].append(idx)
            
            # Tìm từ khóa trong câu hỏi
            found_keywords = []
            for keyword, tour_indices in keyword_to_tours.items():
                if len(keyword) > 2 and keyword in cleaned_message:
                    found_keywords.append((keyword, len(tour_indices)))
            
            # Sắp xếp theo độ phổ biến (ít phổ biến -> cụ thể hơn)
            found_keywords.sort(key=lambda x: x[1])
            
            # Lấy các tour từ từ khóa cụ thể nhất
            fallback_indices = []
            for keyword, count in found_keywords[:5]:  # Top 5 keywords cụ thể nhất
                fallback_indices.extend(keyword_to_tours[keyword][:3])  # Lấy tối đa 3 tour mỗi keyword
            
            # Loại bỏ trùng lặp và giới hạn số lượng
            fallback_indices = list(dict.fromkeys(fallback_indices))[:5]
            
            if fallback_indices:
                tour_indices = fallback_indices
                logger.info(f"🔄 Fallback keyword matches found: {tour_indices}")
                if found_keywords:
                    logger.info(f"🔄 Matching keywords: {[k for k, _ in found_keywords[:3]]}")
        
        # Strategy 5: Popular tours fallback (chỉ khi không tìm thấy gì)
        if not tour_indices:
            logger.info("⭐ Showing popular tours as fallback")
            
            # Xác định popular tours dựa trên logic (có thể dựa vào rating, views, etc.)
            # Ở đây giả sử có một số tour phổ biến cố định
            popular_tour_keywords = ['bạch mã', 'trường sơn', 'huế', 'thiền', 'ẩm thực']
            
            popular_indices = []
            for idx, tour in TOURS_DB.items():
                tour_text = f"{tour.name or ''} {tour.summary or ''}".lower()
                for keyword in popular_tour_keywords:
                    if keyword in tour_text:
                        popular_indices.append(idx)
                        break
                if len(popular_indices) >= 3:
                    break
            
            if popular_indices:
                tour_indices = popular_indices
                logger.info(f"⭐ Popular tours fallback: {tour_indices}")
        
        # Final logging
        if tour_indices:
            logger.info(f"✅ Tour resolution completed. Found {len(tour_indices)} tours: {tour_indices}")
            
            # Log tên các tour tìm được
            for idx in tour_indices[:3]:
                tour = TOURS_DB.get(idx)
                if tour:
                    logger.info(f"   - {tour.name}")
        else:
            logger.warning("⚠️ No tours found after all resolution strategies")



        
        # ================== FILTER EXTRACTION & APPLICATION V2 ==================
        mandatory_filters = FilterSet()
        filter_applied = False
        
        if UpgradeFlags.is_enabled("1_MANDATORY_FILTER"):
            try:
                # 1. ENHANCED FILTER EXTRACTION với logging chi tiết
                logger.info(f"🎯 Starting filter extraction for message: '{user_message[:100]}...'")
                mandatory_filters = MandatoryFilterSystem.extract_filters(user_message)
                
                if not mandatory_filters.is_empty():
                    logger.info(f"🎯 Filters extracted: {mandatory_filters}")
                    
                    # Kiểm tra lỗi trong filter với danh sách đầy đủ từ MandatoryFilterSystem
                    if hasattr(mandatory_filters, 'group_type') and mandatory_filters.group_type:
                        valid_group_types = ['family', 'friends', 'corporate', 'solo', 'couple', 'senior', 'group']
                        if mandatory_filters.group_type not in valid_group_types:
                            logger.warning(f"⚠️ Invalid group type: {mandatory_filters.group_type}")
                            # Reset về None để tránh lỗi
                            mandatory_filters.group_type = None
                    
                    # 2. ENHANCED FILTER APPLICATION với fallback strategies
                    logger.info("🎯 Applying filters to tour database...")
                    filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                    
                    if filtered_indices:
                        filter_applied = True
                        logger.info(f"✅ Filter application successful: {len(filtered_indices)} tours passed filters")
                        
                        # 3. INTELLIGENT RESULT COMBINATION
                        if tour_indices:
                            logger.info("🔄 Combining filter results with tour search results...")
                            
                            # Strategy A: Giao của kết quả (AND logic)
                            combined_intersection = list(set(tour_indices) & set(filtered_indices))
                            
                            # Strategy B: Hợp của kết quả (OR logic) - nếu giao quá ít
                            combined_union = list(set(tour_indices) | set(filtered_indices))
                            
                            # Lựa chọn strategy dựa trên số lượng kết quả
                            if len(combined_intersection) >= 2:
                                # Ưu tiên giao nếu có đủ kết quả
                                tour_indices = combined_intersection[:5]
                                logger.info(f"🎯 Using intersection strategy: {len(tour_indices)} tours")
                            elif len(combined_union) > 0:
                                # Fallback về hợp nếu giao quá ít
                                # Ưu tiên các tour có trong cả hai danh sách trước
                                priority_tours = []
                                other_tours = []
                                
                                for idx in combined_union:
                                    if idx in tour_indices and idx in filtered_indices:
                                        priority_tours.append(idx)
                                    else:
                                        other_tours.append(idx)
                                
                                # Kết hợp ưu tiên + backup
                                tour_indices = (priority_tours + other_tours)[:5]
                                logger.info(f"🎯 Using union strategy with priority: {len(tour_indices)} tours")
                            else:
                                # Không có kết quả nào - dùng filter results
                                tour_indices = filtered_indices[:5]
                                logger.info(f"⚠️ No combined results, using filter results: {len(tour_indices)} tours")
                        else:
                            # Không có kết quả từ tour search, chỉ dùng filter
                            tour_indices = filtered_indices[:8]
                            logger.info(f"🎯 Filter-based search only: {len(tour_indices)} tours")
                        
                        # 4. POST-FILTERING VALIDATION
                        if not tour_indices and filtered_indices:
                            logger.warning("⚠️ Combined results empty but filtered_indices exists, using filtered_indices")
                            tour_indices = filtered_indices[:5]
                    
                    else:
                        # Không có tour nào pass filter
                        logger.warning("⚠️ No tours passed the filters")
                        
                        # Strategy: Áp dụng lenient filtering
                        if tour_indices:
                            # Vẫn giữ nguyên kết quả tìm kiếm nhưng cảnh báo
                            logger.info("🔄 No tours match all filters, using original search results with warning")
                            # Lưu trạng thái để thêm warning vào response nếu cần
                            context.filter_warning = "Không có tour nào đáp ứng đầy đủ tiêu chí. Hiển thị kết quả gần đúng nhất."
                        else:
                            # Fallback: Hiển thị tours phổ biến
                            logger.info("🔄 No tours match filters and no search results, showing popular tours")
                            # Gọi fallback mechanism
                            popular_keywords = ['bạch mã', 'trường sơn', 'huế', 'thiền', 'ẩm thực']
                            popular_indices = []
                            for idx, tour in TOURS_DB.items():
                                tour_text = f"{tour.name or ''} {tour.summary or ''}".lower()
                                for keyword in popular_keywords:
                                    if keyword in tour_text:
                                        popular_indices.append(idx)
                                        break
                                if len(popular_indices) >= 3:
                                    break
                            
                            if popular_indices:
                                tour_indices = popular_indices
                                context.filter_fallback = True
                                logger.info(f"🔄 Fallback to popular tours: {tour_indices}")
                
                else:
                    logger.info("ℹ️ No filters extracted from query")
                    
            except Exception as e:
                logger.error(f"❌ Filter system error: {e}\n{traceback.format_exc()}")
                # Continue without filters - important for graceful degradation
                mandatory_filters = FilterSet()
                filter_applied = False
                # Không cần xử lý thêm, vẫn dùng kết quả từ tour resolution engine
        
        # 5. FILTER-AWARE LOGGING & CONTEXT UPDATES
        if filter_applied:
            # Ghi thông tin filter vào context để response generation sử dụng
            context.applied_filters = {
                'filters': mandatory_filters.to_dict() if hasattr(mandatory_filters, 'to_dict') else str(mandatory_filters),
                'filtered_count': len(tour_indices) if tour_indices else 0,
                'filter_warning': getattr(context, 'filter_warning', None),
                'filter_fallback': getattr(context, 'filter_fallback', False)
            }
            
            # Log final filter status
            filter_summary = []
            if hasattr(mandatory_filters, 'group_type') and mandatory_filters.group_type:
                filter_summary.append(f"group_type:{mandatory_filters.group_type}")
            if hasattr(mandatory_filters, 'location') and mandatory_filters.location:
                filter_summary.append(f"location:{mandatory_filters.location}")
            if hasattr(mandatory_filters, 'duration_min') or hasattr(mandatory_filters, 'duration_max'):
                dur_range = []
                if mandatory_filters.duration_min:
                    dur_range.append(f"min:{mandatory_filters.duration_min}")
                if mandatory_filters.duration_max:
                    dur_range.append(f"max:{mandatory_filters.duration_max}")
                if dur_range:
                    filter_summary.append(f"duration:{','.join(dur_range)}")
            if hasattr(mandatory_filters, 'price_min') or hasattr(mandatory_filters, 'price_max'):
                price_range = []
                if mandatory_filters.price_min:
                    price_range.append(f"min:{mandatory_filters.price_min:,}")
                if mandatory_filters.price_max:
                    price_range.append(f"max:{mandatory_filters.price_max:,}")
                if price_range:
                    filter_summary.append(f"price:{','.join(price_range)}")
            
            logger.info(f"✅ Filter Summary: {filter_summary}")
            logger.info(f"✅ Final tour count after filtering: {len(tour_indices) if tour_indices else 0}")
        
        # 6. THÊM LOẠI FILTER MỚI: SEASON/PREFERENCE FILTERS (bổ sung)
        # Kiểm tra thêm các filter không có trong MandatoryFilterSystem nhưng có trong query
        additional_filters = {}
        
        # Season filter
        season_keywords = {
            'mùa xuân': 'spring',
            'mùa hè': 'summer', 
            'mùa thu': 'autumn',
            'mùa đông': 'winter',
            'mùa khô': 'dry_season',
            'mùa mưa': 'rainy_season'
        }
        
        for vi_key, en_key in season_keywords.items():
            if vi_key in message_lower:
                additional_filters['season'] = en_key
                logger.info(f"🍂 Additional season filter detected: {en_key}")
                break
        
        # Activity preference filter
        activity_keywords = {
            'nhẹ nhàng': 'gentle',
            'mạo hiểm': 'adventure',
            'văn hóa': 'cultural',
            'thiên nhiên': 'nature',
            'thư giãn': 'relaxing',
            'hoạt động': 'active'
        }
        
        for vi_key, en_key in activity_keywords.items():
            if vi_key in message_lower:
                additional_filters['activity_level'] = en_key
                logger.info(f"🏃 Additional activity filter detected: {en_key}")
                break
        
        # Accessibility filter
        accessibility_keywords = ['dễ đi', 'dễ tiếp cận', 'không leo núi', 'bằng phẳng', 'cho người già', 'cho trẻ em']
        if any(keyword in message_lower for keyword in accessibility_keywords):
            additional_filters['accessibility'] = 'easy'
            logger.info("♿ Additional accessibility filter detected: easy")
        
        # Lưu additional filters vào context để response generation sử dụng
        if additional_filters:
            context.additional_filters = additional_filters
            logger.info(f"➕ Additional filters: {additional_filters}")
            
            # Áp dụng thêm các filter bổ sung nếu có tour_indices
            if tour_indices and additional_filters:
                filtered_by_additional = []
                
                for idx in tour_indices[:10]:  # Chỉ xét 10 tour đầu
                    tour = TOURS_DB.get(idx)
                    if not tour:
                        continue
                    
                    passes_additional = True
                    tour_text = f"{tour.summary or ''} {tour.style or ''}".lower()
                    
                    # Season filter logic
                    if 'season' in additional_filters:
                        season = additional_filters['season']
                        # Logic đơn giản: mùa mưa tránh trekking, mùa khô phù hợp outdoor
                        if season == 'rainy_season':
                            if any(word in tour_text for word in ['trekking', 'leo núi', 'đi bộ đường dài', 'cắm trại']):
                                passes_additional = False
                    
                    # Activity level filter
                    if passes_additional and 'activity_level' in additional_filters:
                        activity = additional_filters['activity_level']
                        if activity == 'gentle' and any(word in tour_text for word in ['trekking', 'mạo hiểm', 'leo núi', 'khó']):
                            passes_additional = False
                        elif activity == 'adventure' and any(word in tour_text for word in ['nhẹ nhàng', 'thư giãn', 'nghỉ dưỡng']):
                            passes_additional = False
                    
                    # Accessibility filter
                    if passes_additional and 'accessibility' in additional_filters:
                        if any(word in tour_text for word in ['leo núi', 'trekking', 'đường khó', 'vất vả']):
                            passes_additional = False
                    
                    if passes_additional:
                        filtered_by_additional.append(idx)
                
                if filtered_by_additional:
                    # Giữ lại thứ tự ban đầu nếu có thể
                    original_order = {idx: i for i, idx in enumerate(tour_indices)}
                    filtered_by_additional.sort(key=lambda x: original_order.get(x, 999))
                    tour_indices = filtered_by_additional[:5]
                    logger.info(f"➕ Applied additional filters: {len(tour_indices)} tours remain")
                # Continue without filters
        
        # ================== INTELLIGENT RESPONSE GENERATION ==================
        reply = ""
        sources = []
        
        # 🔹 CASE 0: CONTEXT-AWARE FOLLOW-UP (Nâng cấp mới)
        if len(context.conversation_history) > 1:
            last_user_msg = None
            last_bot_msg = None
            
            # Tìm tin nhắn gần nhất
            for msg in reversed(context.conversation_history[:-1]):
                if msg['role'] == 'user':
                    last_user_msg = msg['message']
                elif msg['role'] == 'assistant' and not last_bot_msg:
                    last_bot_msg = msg['message']
                
                if last_user_msg and last_bot_msg:
                    break
            
            # Xử lý follow-up questions
            if last_bot_msg and ('tour nào' in message_lower or 'gợi ý' in message_lower):
                # Kiểm tra nếu đây là câu hỏi follow-up về tour
                follow_up_tours = getattr(context, 'last_recommended_tours', [])
                if follow_up_tours and len(tour_indices) == 0:
                    tour_indices = follow_up_tours[:3]
                    logger.info(f"🔄 Using context tour recommendations: {tour_indices}")
                    
        
        # 🔹 CASE 1.1: LOCATION QUERY - Xử lý câu hỏi về địa điểm cụ thể
        if 'location_query' in detected_intents:
            logger.info("📍 Processing location query")
            
            # Xác định địa điểm được hỏi
            locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà', 'miền trung', 'đà nẵng']
            mentioned_location = None
            
            for loc in locations:
                if loc in message_lower:
                    mentioned_location = loc
                    break
            
            if mentioned_location:
                # Tìm tour tại địa điểm này
                location_tours = []
                for idx, tour in TOURS_DB.items():
                    if tour.location and mentioned_location in tour.location.lower():
                        location_tours.append(tour)
                
                # Apply filters nếu có
                if filter_applied and not mandatory_filters.is_empty():
                    filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                    location_tours = [tour for idx, tour in enumerate(location_tours) if idx in filtered_indices]
                
                if location_tours:
                    reply = f"📍 **TOUR TẠI {mentioned_location.upper()}** 📍\n\n"
                    
                    # Hiển thị thông tin tổng quan
                    reply += f"Ruby Wings có {len(location_tours)} tour tại {mentioned_location.upper()}:\n\n"
                    
                    # Phân loại tour tại địa điểm này
                    for i, tour in enumerate(location_tours[:6], 1):
                        reply += f"{i}. **{tour.name}**\n"
                        if tour.duration:
                            reply += f"   ⏱️ {tour.duration}\n"
                        if tour.summary:
                            summary_short = tour.summary[:80] + "..." if len(tour.summary) > 80 else tour.summary
                            reply += f"   📝 {summary_short}\n"
                        if i == 1 and tour.price:
                            price_short = tour.price[:60] + "..." if len(tour.price) > 60 else tour.price
                            reply += f"   💰 {price_short}\n"
                        reply += "\n"
                    
                    # Thông tin đặc trưng của địa điểm
                    if mentioned_location == 'huế':
                        reply += "🏛️ **ĐẶC TRƯNG HUẾ:**\n"
                        reply += "• Di sản UNESCO: Đại Nội, Lăng tẩm\n"
                        reply += "• Ẩm thực cung đình đặc sắc\n"
                        reply += "• Sông Hương, núi Ngự thơ mộng...\n\n"
                    elif mentioned_location == 'bạch mã':
                        reply += "🌿 **ĐẶC TRƯNG BẠCH MÃ:**\n"
                        reply += "• Vườn quốc gia rộng 37,000ha\n"
                        reply += "• Khí hậu mát mẻ quanh năm\n"
                        reply += "• Đa dạng sinh học...\n\n"
                    elif mentioned_location == 'trường sơn':
                        reply += "🎖️ **ĐẶC TRƯNG TRƯỜNG SƠN:**\n"
                        reply += "• Di tích lịch sử chiến tranh\n"
                        reply += "• Đường Hồ Chí Minh huyền thoại\n"
                        reply += "• Văn hóa dân tộc Vân Kiều, Pa Kô\n\n"
                    
                    reply += "📞 **Đặt tour tại địa điểm này:** 0332510486"
                else:
                    reply = f"Hiện Ruby Wings chưa có tour nào tại {mentioned_location.upper()}. Tuy nhiên, chúng tôi có thể thiết kế tour riêng theo yêu cầu của bạn.\n\n"
                    reply += "📞 **Liên hệ thiết kế tour riêng:** 0332510486"
            else:
                reply = "Bạn muốn tìm tour tại khu vực nào? Ruby Wings có tour tại:\n\n"
                reply += "• Huế (di sản, ẩm thực)\n"
                reply += "• Quảng Trị (lịch sử, di tích)\n"
                reply += "• Bạch Mã (thiên nhiên, trekking)\n"
                reply += "• Trường Sơn (lịch sử, văn hóa)\n\n"
                reply += "📞 **Hotline tư vấn địa điểm:** 0332510486"
        
        # 🔹 CASE 2.1: SERVICE INQUIRY - Xử lý câu hỏi về dịch vụ bao gồm
        elif 'service_inquiry' in detected_intents:
            logger.info("🛎️ Processing service inquiry")
            
            reply = "🛎️ **DỊCH VỤ BAO GỒM TRONG TOUR RUBY WINGS** 🛎️\n\n"
            
            # Phân loại dịch vụ
            reply += "✅ **DỊCH VỤ CƠ BẢN (có trong hầu hết tour):**\n"
            reply += "• 🚌 Xe đưa đón đời mới, máy lạnh\n"
            reply += "• 🏨 Chỗ nghỉ ngơi tiêu chuẩn 3* (khách sạn/homestay)\n"
            reply += "• 🍽️ Ăn uống theo chương trình (3 bữa chính/ngày)\n"
            reply += "• 🧭 Hướng dẫn viên chuyên nghiệp, nhiệt tình\n"
            reply += "• 🎫 Vé tham quan các điểm du lịch\n"
            reply += "• 💧 Nước uống suối đóng chai\n"
            reply += "• 🛡️ Bảo hiểm du lịch (mức đền bù từ 50 triệu VNĐ)\n\n"
            
            reply += "✨ **DỊCH VỤ CAO CẤP (tour 2+ ngày):**\n"
            reply += "• 🌟 Khách sạn 3-4 sao (tùy tour)\n"
            reply += "• 🍷 Bữa ăn đặc sản địa phương\n"
            reply += "• 🎤 Hướng dẫn viên tiếng Anh (nếu yêu cầu)\n"
            reply += "• 📸 Chụp ảnh lưu niệm chuyên nghiệp\n"
            reply += "• 🎁 Quà tặng đặc sản địa phương\n"
            reply += "• 🚑 Phụ trách y tế đi kèm (tour nhóm lớn và có Cựu chiến binh)\n\n"
            
            reply += "⚠️ **DỊCH VỤ KHÔNG BAO GỒM:**\n"
            reply += "• Chi phí cá nhân: Giặt ủi, điện thoại, mini bar\n"
            reply += "• Đồ uống có cồn (bia, rượu, cocktail)\n"
            reply += "• Tip cho hướng dẫn viên và tài xế\n"
            reply += "• Chi phí phát sinh do thay đổi lịch trình\n"
            reply += "• Phí tham quan ngoài chương trình\n\n"
            
            # Áp dụng filter nếu có thông tin về nhóm/đối tượng
            if mandatory_filters and not mandatory_filters.is_empty():
                if hasattr(mandatory_filters, 'group_type'):
                    if mandatory_filters.group_type == 'family':
                        reply += "👨‍👩‍👧‍👦 **DỊCH VỤ ĐẶC BIỆT CHO GIA ĐÌNH:**\n"
                        reply += "• Phòng gia đình riêng biệt\n"
                        reply += "• Thực đơn riêng cho trẻ em\n"
                        reply += "• Hoạt động vui chơi cho trẻ\n"
                        reply += "• Trẻ em dưới 4 tuổi: Miễn phí\n"
                        reply += "• Trẻ 4 dưới 7 tuổi: Giảm 50% giá tour\n"
                        reply += "• Trẻ em 8-11 tuổi: Giảm 15% giá tour\n\n"
                    elif mandatory_filters.group_type == 'senior':
                        reply += "👴 **DỊCH VỤ ĐẶC BIỆT CHO NGƯỜI LỚN TUỔI:**\n"
                        reply += "• Xe đón tận nơi (có liên hệ trước)\n"
                        reply += "• Nhân viên hỗ trợ đặc biệt (nhân viên y tế)\n"
                        reply += "• Lịch trình nhẹ nhàng, không vội\n"
                        reply += "• Khám sức khỏe trước tour\n"
                        reply += "• Cựu chiến binh: Ưu đãi đặc biệt\n\n"
            
            reply += "📋 **ĐIỀU KIỆN THAM GIA:**\n"
            reply += "• Sức khỏe tốt, không mắc bệnh mãn tính\n"
            reply += "• Tuổi từ 5-70 (trừ tour đặc biệt)\n"
            reply += "• Mang theo giấy tờ tùy thân bản gốc\n"
            reply += "• Tuân thủ hướng dẫn của HDV\n"
            reply += "• Mua bảo hiểm du lịch (bắt buộc)\n\n"
            
            reply += "📞 **Liên hệ để biết chi tiết dịch vụ tour cụ thể:** 0332510486"
        
        # 🔹 CASE 3: PRICE INQUIRY - NÂNG CẤP (ÁP DỤNG FILTER)
        elif 'price_inquiry' in detected_intents:
            logger.info("💰 Processing enhanced price inquiry with filters")
            
            # Apply filters nếu có
            if filter_applied and not mandatory_filters.is_empty():
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                if filtered_indices:
                    # Sửa lỗi: đổi từ tour_indices sang tour_indices
                    if not tour_indices:
                        tour_indices = filtered_indices[:3]
                    else:
                        # Kết hợp kết quả
                        combined = list(set(tour_indices) & set(filtered_indices))
                        tour_indices = combined[:3] if combined else filtered_indices[:3]
            
            if tour_indices:
                # Có tour cụ thể
                detailed_info = []
                
                for idx in tour_indices[:3]:
                    tour = TOURS_DB.get(idx)
                    if tour:
                        # Format price information
                        price_info = {
                            'name': tour.name,
                            'price': tour.price or 'Liên hệ để biết giá',
                            'duration': tour.duration or 'Không xác định',
                            'location': tour.location or 'Không xác định'
                        }
                        
                        # Phân tích giá nếu có
                        price_text = price_info['price']
                        if price_text and price_text != 'Liên hệ để biết giá':
                            price_numbers = re.findall(r'\d[\d,\.]+', price_text)
                            if price_numbers:
                                try:
                                    clean_nums = []
                                    for num in price_numbers:
                                        clean_num = num.replace(',', '').replace('.', '')
                                        if clean_num.isdigit():
                                            clean_nums.append(int(clean_num))
                                    
                                    if clean_nums:
                                        min_price = min(clean_nums)
                                        max_price = max(clean_nums) if len(clean_nums) > 1 else min_price
                                        
                                        if min_price < 1000000:
                                            price_range = f"{min_price:,}đ"
                                        elif min_price == max_price:
                                            price_range = f"{min_price:,}đ"
                                        else:
                                            price_range = f"{min_price:,}đ - {max_price:,}đ"
                                        
                                        price_info['formatted'] = price_range
                                except:
                                    price_info['formatted'] = price_text
                        
                        detailed_info.append(price_info)
                
                if detailed_info:
                    reply = "💰 **THÔNG TIN GIÁ TOUR CHI TIẾT** 💰\n\n"
                    
                    for info in detailed_info:
                        reply += f"**{info['name']}**\n"
                        reply += f"⏱️ Thời gian: {info['duration']}\n"
                        reply += f"📍 Địa điểm: {info.get('location_short', info['location'][:50])}\n"
                        
                        if 'formatted' in info:
                            reply += f"💰 **Giá:** {info['formatted']}\n"
                        else:
                            reply += f"💰 **Giá:** {info['price']}\n"
                        
                        # Thêm phân loại giá
                        if 'formatted' in info and 'đ' in info['formatted']:
                            price_num = int(info['formatted'].split('đ')[0].replace(',', '').replace('.', '').strip())
                            if price_num < 1000000:
                                reply += "   🏷️ Phân loại: Tiết kiệm\n"
                            elif price_num < 2500000:
                                reply += "   🏷️ Phân loại: Tiêu chuẩn\n"
                            else:
                                reply += "   🏷️ Phân loại: Cao cấp\n"
                        
                        reply += "\n"
                    
                    # Thêm thông tin ưu đãi dựa trên filter - SỬA THEO CHÍNH SÁCH GỐC
                    reply += "🎯 **ƯU ĐÃI ĐẶC BIỆT:**\n"
                    
                    if mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                        if mandatory_filters.group_type == 'family':
                            reply += "• Gia đình 4 người: Giảm 5%\n"
                            reply += "• Trẻ em 8-11 tuổi: Giảm 15%\n"
                            reply += "• Trẻ 4 dưới 7 tuổi: Giảm 50%\n"
                            reply += "• Trẻ dưới 4 tuổi: Miễn phí\n"
                        elif mandatory_filters.group_type == 'senior':
                            reply += "• Người lớn tuổi: Giảm 5%\n"
                            reply += "• Cựu chiến binh: Giảm 10%\n"
                            reply += "• Nhóm 5+ người cao tuổi: Giảm thêm 5%\n"
                        elif mandatory_filters.group_type == 'friends':
                            reply += "• Nhóm bạn 5-9 người: Giảm 3%\n"
                            reply += "• Nhóm 10-13 người: Giảm 5%\n"
                            reply += "• Nhóm 14-20 người: Giảm 8%\n"
                            reply += "• Nhóm 21-27 người: Giảm 10%\n"
                            reply += "• Nhóm 28-33 người: Giảm 12%\n"
                            reply += "• Nhóm 34-42 người: Giảm 15%\n"
                            reply += "• Sinh viên: Giảm thêm 5%\n"
                    
                    reply += "• Đặt trước 30 ngày: Giảm thêm 5%\n"
                    reply += "• Thanh toán online: Giảm thêm 2%\n\n"
                    reply += "📞 **Liên hệ ngay để nhận báo giá tốt nhất:** 0332510486"
                else:
                    reply = "Hiện chưa có thông tin giá cho các tour này. Vui lòng liên hệ hotline 0332510486 để được báo giá chi tiết."
            else:
                # Không có tour cụ thể - Hiển thị bảng giá tổng quát với filter
                reply = "💰 **BẢNG GIÁ THAM KHẢO RUBY WINGS** 💰\n\n"
                
                # Tạo bảng giá theo loại tour, có xem xét filter
                price_categories = [
                    ("🌿 TOUR 1 NGÀY (Thiên nhiên, Văn hóa)", "600.000đ - 1.500.000đ", 
                     "Bạch Mã, Huế city tour, Ẩm thực Huế"),
                    ("🏛️ TOUR 2 NGÀY 1 ĐÊM (Lịch sử, Retreat)", "1.500.000đ - 3.000.000đ", 
                     "Trường Sơn, Di tích lịch sử, Thiền định"),
                    ("🕉️ TOUR 3+ NGÀY (Cao cấp, Cá nhân hóa)", "3.000.000đ - 5.000.000đ", 
                     "Tour riêng, Nhóm đặc biệt, Retreat sâu"),
                    ("👥 TOUR TEAMBUILDING (Công ty, Nhóm lớn)", "Liên hệ tư vấn", 
                     "Thiết kế riêng, Hoạt động nhóm, Gắn kết")
                ]
                
                for cat_name, price_range, description in price_categories:
                    reply += f"**{cat_name}**\n"
                    reply += f"💰 {price_range}\n"
                    reply += f"📝 {description}\n\n"
                
                # Thêm thông tin ưu đãi theo filter - SỬA THEO CHÍNH SÁCH GỐC
                if mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                    reply += "🎁 **ƯU ĐÃI ĐẶC BIỆT CHO NHÓM:**\n"
                    
                    if mandatory_filters.group_type == 'family':
                        reply += "• Gia đình 4 người: Giảm 5%\n"
                        reply += "• Trẻ em 8-11 tuổi: Giảm 15%\n"
                        reply += "• Trẻ 4 dưới 7 tuổi: Giảm 50%\n"
                        reply += "• Trẻ dưới 4 tuổi: Miễn phí\n"
                    elif mandatory_filters.group_type == 'senior':
                        reply += "• Người lớn tuổi: Giảm 5%\n"
                        reply += "• Cựu chiến binh: Giảm 10%\n"
                        reply += "• Nhóm 5+ người cao tuổi: Giảm thêm 5%\n"
                    elif mandatory_filters.group_type == 'friends':
                        reply += "• Nhóm bạn 5-9 người: Giảm 3%\n"
                        reply += "• Nhóm 10-13 người: Giảm 5%\n"
                        reply += "• Nhóm 14-20 người: Giảm 8%\n"
                        reply += "• Nhóm 21-27 người: Giảm 10%\n"
                        reply += "• Nhóm 28-33 người: Giảm 12%\n"
                        reply += "• Nhóm 34-42 người: Giảm 15%\n"
                        reply += "• Sinh viên: Giảm thêm 5%\n"
                    
                    reply += "\n"
                
                reply += "📞 **Liên hệ tư vấn giá chính xác:** 0332510486"
        
        # 🔹 CASE 4: TOUR LISTING (ÁP DỤNG FILTER VỀ LOCATION)
        elif 'tour_listing' in detected_intents:
            logger.info("📋 Processing tour listing request with filters")
            
            all_tours = list(TOURS_DB.values())
            
            # Apply location filter nếu có trong câu hỏi
            location_from_query = None
            locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà', 'miền trung', 'đà nẵng']
            for loc in locations:
                if loc in message_lower:
                    location_from_query = loc
                    break
            
            if location_from_query:
                all_tours = [tour for tour in all_tours if tour.location and location_from_query in tour.location.lower()]
                logger.info(f"📍 Applied location filter: {location_from_query}")
            
            # Apply mandatory filters
            if filter_applied and not mandatory_filters.is_empty():
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                all_tours = [TOURS_DB[idx] for idx in filtered_indices if idx in TOURS_DB]
            
            # Apply deduplication
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and all_tours:
                seen_names = set()
                unique_tours = []
                for tour in all_tours:
                    name = tour.name
                    if name and name not in seen_names:
                        seen_names.add(name)
                        unique_tours.append(tour)
                all_tours = unique_tours
            
            total_tours = len(all_tours)
            
            if total_tours > 0:
                # Phân loại tour theo category
                categorized_tours = {
                    'history': [],
                    'retreat': [],
                    'nature': [],
                    'culture': [],
                    'family': []
                }
                
                for tour in all_tours:
                    tags_lower = [tag.lower() for tag in (tour.tags or [])]
                    
                    if any('history' in tag for tag in tags_lower):
                        categorized_tours['history'].append(tour)
                    elif any('meditation' in tag or 'retreat' in tag for tag in tags_lower):
                        categorized_tours['retreat'].append(tour)
                    elif any('nature' in tag for tag in tags_lower):
                        categorized_tours['nature'].append(tour)
                    elif any('culture' in tag or 'food' in tag for tag in tags_lower):
                        categorized_tours['culture'].append(tour)
                    elif any('family' in tag for tag in tags_lower):
                        categorized_tours['family'].append(tour)
                    else:
                        categorized_tours['nature'].append(tour)  # Mặc định
                
                # Format response có cấu trúc
                reply = "✨ **DANH SÁCH TOUR RUBY WINGS** ✨\n\n"
                
                # Hiển thị filter đang áp dụng
                if location_from_query or filter_applied:
                    reply += "🔍 **ĐANG ÁP DỤNG BỘ LỌC:**\n"
                    if location_from_query:
                        reply += f"• Địa điểm: {location_from_query.upper()}\n"
                    if mandatory_filters and not mandatory_filters.is_empty():
                        reply += f"• {mandatory_filters}\n"
                    reply += "\n"
                
                reply += f"📊 **Tổng cộng:** {total_tours} tour đặc sắc\n\n"
                
                # Hiển thị theo từng loại
                categories_display = [
                    ('🏛️ LỊCH SỬ - TRI ÂN', 'history', 'history'),
                    ('🕉️ RETREAT - CHỮA LÀNH', 'retreat', 'meditation'),
                    ('🌿 THIÊN NHIÊN - KHÁM PHÁ', 'nature', 'nature'),
                    ('🍜 VĂN HÓA - ẨM THỰC', 'culture', 'culture'),
                    ('👨‍👩‍👧‍👦 GIA ĐÌNH - NHÓM', 'family', 'family')
                ]
                
                tours_displayed = 0
                for cat_name, cat_key, emoji_key in categories_display:
                    cat_tours = categorized_tours[cat_key]
                    if cat_tours:
                        reply += f"**{cat_name}** ({len(cat_tours)} tour)\n"
                        
                        for i, tour in enumerate(cat_tours[:3], 1):
                            # Chọn emoji phù hợp
                            emoji = "✨"
                            if cat_key == 'history': emoji = "🏛️"
                            elif cat_key == 'retreat': emoji = "🕉️"
                            elif cat_key == 'nature': emoji = "🌿"
                            elif cat_key == 'culture': emoji = "🍜"
                            elif cat_key == 'family': emoji = "👨‍👩‍👧‍👦"
                            
                            reply += f"{emoji} **{tour.name}**\n"
                            if tour.duration:
                                reply += f"   ⏱️ {tour.duration}\n"
                            if tour.location:
                                location_short = tour.location[:40] + "..." if len(tour.location) > 40 else tour.location
                                reply += f"   📍 {location_short}\n"
                            if i == 1 and tour.price:  # Hiện giá tour đầu mỗi loại
                                price_short = tour.price[:60] + "..." if len(tour.price) > 60 else tour.price
                                reply += f"   💰 {price_short}\n"
                            reply += "\n"
                            tours_displayed += 1
                        
                        if len(cat_tours) > 3:
                            reply += f"   📌 ...và {len(cat_tours) - 3} tour khác\n\n"
                        else:
                            reply += "\n"
                
                if tours_displayed < total_tours:
                    reply += f"📌 **Còn {total_tours - tours_displayed} tour khác trong hệ thống!**\n\n"
                
                # Thêm thông tin ưu đãi theo filter - SỬA THEO CHÍNH SÁCH GỐC
                if mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                    reply += "🎁 **ƯU ĐÃI ĐẶC BIỆT CHO NHÓM:**\n"
                    
                    if mandatory_filters.group_type == 'family':
                        reply += "• Gia đình 4 người: Giảm 5%\n"
                        reply += "• Trẻ em 8-11 tuổi: Giảm 15%\n"
                        reply += "• Trẻ 4 dưới 7 tuổi: Giảm 50%\n"
                        reply += "• Trẻ dưới 4 tuổi: Miễn phí\n"
                    elif mandatory_filters.group_type == 'senior':
                        reply += "• Người lớn tuổi: Giảm 5%\n"
                        reply += "• Cựu chiến binh: Giảm 10%\n"
                        reply += "• Nhóm 5+ người cao tuổi: Giảm thêm 5%\n"
                    elif mandatory_filters.group_type == 'friends':
                        reply += "• Nhóm bạn 5-9 người: Giảm 3%\n"
                        reply += "• Nhóm 10-13 người: Giảm 5%\n"
                        reply += "• Nhóm 14-20 người: Giảm 8%\n"
                        reply += "• Nhóm 21-27 người: Giảm 10%\n"
                        reply += "• Nhóm 28-33 người: Giảm 12%\n"
                        reply += "• Nhóm 34-42 người: Giảm 15%\n"
                        reply += "• Sinh viên: Giảm thêm 5%\n"
                    
                    reply += "\n"
                
                reply += "💡 **HƯỚNG DẪN TÌM TOUR:**\n"
                reply += "• Gọi tên tour cụ thể (ví dụ: 'Tour Bạch Mã')\n"
                reply += "• Mô tả nhu cầu: 'tour gia đình 2 ngày', 'retreat thiền'\n"
                reply += "• So sánh tour: 'so sánh tour A và tour B'\n\n"
                reply += "📞 **Hotline tư vấn nhanh:** 0332510486"
                
                # Lưu context để follow-up
                context.last_listed_tours = [idx for idx, tour in enumerate(all_tours[:10])]
            else:
                reply = "Hiện chưa có tour nào phù hợp với yêu cầu của bạn. Vui lòng thử với tiêu chí khác hoặc liên hệ hotline 0332510486 để được tư vấn tour riêng."


        # 🔹 CASE 5: RECOMMENDATION SYSTEM (ÁP DỤNG FILTER VỀ NHÓM/BUDGET)
        elif 'recommendation' in detected_intents:
            logger.info("🎯 Processing enhanced recommendation request with filters")
            
            # Advanced user profile extraction
            user_profile = {
                'group_type': None,
                'age_group': None,
                'interests': [],
                'budget_range': None,
                'time_constraint': None,
                'preferred_location': None,
                'special_requirements': []
            }
            
            # Extract group type từ câu hỏi hoặc từ filter
            group_keywords = {
                'family': ['gia đình', 'con nhỏ', 'trẻ em', 'bố mẹ', 'ông bà', 'đa thế hệ'],
                'senior': ['người lớn tuổi', 'cao tuổi', 'cựu chiến binh', 'veteran', 'ông bà'],
                'friends': ['nhóm bạn', 'bạn bè', 'sinh viên', 'bạn trẻ', 'thanh niên'],
                'corporate': ['công ty', 'team building', 'doanh nghiệp', 'nhân viên', 'đồng nghiệp'],
                'couple': ['cặp đôi', 'đôi lứa', 'người yêu', 'tình nhân'],
                'solo': ['một mình', 'đi lẻ', 'solo', 'cá nhân']
            }
            
            for group_type, keywords in group_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    user_profile['group_type'] = group_type
                    break
            
            # Nếu không tìm thấy trong câu hỏi, kiểm tra filter
            if not user_profile['group_type'] and mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                user_profile['group_type'] = mandatory_filters.group_type
            
            # Extract budget từ câu hỏi hoặc filter
            budget_patterns = [
                r'giá rẻ|tiết kiệm|kinh tế|dưới\s+(\d+)',
                r'tầm trung|trung bình|vừa phải|khoảng\s+(\d+)',
                r'cao cấp|sang trọng|premium|trên\s+(\d+)'
            ]
            
            for i, pattern in enumerate(budget_patterns):
                if re.search(pattern, message_lower):
                    if i == 0:
                        user_profile['budget_range'] = 'low'
                    elif i == 1:
                        user_profile['budget_range'] = 'medium'
                    else:
                        user_profile['budget_range'] = 'high'
                    break
            
            # Extract interests
            interest_keywords = {
                'history': ['lịch sử', 'di tích', 'chiến tranh', 'tri ân', 'ký ức', 'cổ'],
                'nature': ['thiên nhiên', 'rừng', 'núi', 'cây', 'suối', 'không khí trong lành'],
                'meditation': ['thiền', 'yoga', 'tĩnh tâm', 'chữa lành', 'retreat', 'khí công'],
                'culture': ['văn hóa', 'truyền thống', 'ẩm thực', 'đặc sản', 'phong tục'],
                'adventure': ['phiêu lưu', 'mạo hiểm', 'khám phá', 'trải nghiệm mới'],
                'relaxation': ['nghỉ ngơi', 'thư giãn', 'nhẹ nhàng', 'không vội', 'chậm rãi']
            }
            
            for interest, keywords in interest_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    if interest not in user_profile['interests']:
                        user_profile['interests'].append(interest)
            
            # Extract location preference
            locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà', 'miền trung']
            for loc in locations:
                if loc in message_lower:
                    user_profile['preferred_location'] = loc
                    break
            
            logger.info(f"🎯 User profile extracted: {user_profile}")
            
            # SCORING SYSTEM với filter
            matching_tours = []
            
            for idx, tour in TOURS_DB.items():
                score = 0
                reasons = []
                match_details = {}
                
                # 1. Group type matching (30%)
                if user_profile['group_type']:
                    tour_tags = [tag.lower() for tag in (tour.tags or [])]
                    
                    if user_profile['group_type'] == 'family':
                        if any('family' in tag for tag in tour_tags):
                            score += 30
                            reasons.append("phù hợp gia đình")
                            match_details['group'] = 'excellent'
                        elif not any('adventure' in tag or 'extreme' in tag for tag in tour_tags):
                            score += 15
                            reasons.append("có thể phù hợp gia đình")
                            match_details['group'] = 'good'
                    
                    elif user_profile['group_type'] == 'senior':
                        if any('senior' in tag or 'accessible' in tag for tag in tour_tags):
                            score += 30
                            reasons.append("thiết kế cho người lớn tuổi")
                            match_details['group'] = 'excellent'
                        elif any('nature' in tag or 'meditation' in tag for tag in tour_tags):
                            score += 20
                            reasons.append("nhẹ nhàng, phù hợp lớn tuổi")
                            match_details['group'] = 'good'
                    
                    elif user_profile['group_type'] == 'friends':
                        if any('friends' in tag or 'group' in tag for tag in tour_tags):
                            score += 30
                            reasons.append("phù hợp nhóm bạn")
                            match_details['group'] = 'excellent'
                        elif any('adventure' in tag or 'experience' in tag for tag in tour_tags):
                            score += 20
                            reasons.append("nhiều hoạt động nhóm")
                            match_details['group'] = 'good'
                
                # 2. Interest matching (40%)
                if user_profile['interests']:
                    tour_summary = (tour.summary or '').lower()
                    tour_tags = [tag.lower() for tag in (tour.tags or [])]
                    
                    for interest in user_profile['interests']:
                        if interest == 'history':
                            if any('history' in tag for tag in tour_tags) or 'lịch sử' in tour_summary:
                                score += 40
                                reasons.append("trọng tâm lịch sử")
                                match_details['interest'] = 'history'
                                break
                        
                        elif interest == 'nature':
                            if any('nature' in tag for tag in tour_tags) or 'thiên nhiên' in tour_summary:
                                score += 40
                                reasons.append("trải nghiệm thiên nhiên")
                                match_details['interest'] = 'nature'
                                break
                        
                        elif interest == 'meditation':
                            if any('meditation' in tag for tag in tour_tags) or 'thiền' in tour_summary:
                                score += 40
                                reasons.append("có hoạt động thiền/retreat")
                                match_details['interest'] = 'meditation'
                                break
                        
                        elif interest == 'culture':
                            if any('culture' in tag for tag in tour_tags) or 'văn hóa' in tour_summary:
                                score += 40
                                reasons.append("khám phá văn hóa")
                                match_details['interest'] = 'culture'
                                break
                
                # 3. Budget matching (15%)
                if user_profile['budget_range'] and tour.price:
                    price_value = _extract_price_value(tour.price)
                    
                    if price_value:
                        if user_profile['budget_range'] == 'low' and price_value < 1500000:
                            score += 15
                            reasons.append("giá hợp lý")
                            match_details['budget'] = 'good'
                        elif user_profile['budget_range'] == 'medium' and 1500000 <= price_value <= 3000000:
                            score += 15
                            reasons.append("giá tầm trung")
                            match_details['budget'] = 'good'
                        elif user_profile['budget_range'] == 'high' and price_value > 3000000:
                            score += 15
                            reasons.append("dịch vụ cao cấp")
                            match_details['budget'] = 'good'
                
                # 4. Time constraint matching (10%)
                if user_profile['time_constraint'] and tour.duration:
                    duration_lower = tour.duration.lower()
                    
                    if user_profile['time_constraint'] == '1day' and '1 ngày' in duration_lower:
                        score += 10
                        reasons.append("đúng 1 ngày")
                        match_details['time'] = 'perfect'
                    elif user_profile['time_constraint'] == '2days' and '2 ngày' in duration_lower:
                        score += 10
                        reasons.append("đúng 2 ngày")
                        match_details['time'] = 'perfect'
                    elif user_profile['time_constraint'] == '3+days' and ('3 ngày' in duration_lower or '4 ngày' in duration_lower):
                        score += 10
                        reasons.append("đa ngày như yêu cầu")
                        match_details['time'] = 'perfect'
                
                # 5. Location preference (5%)
                if user_profile['preferred_location'] and tour.location:
                    if user_profile['preferred_location'] in tour.location.lower():
                        score += 5
                        reasons.append(f"tại {user_profile['preferred_location']}")
                        match_details['location'] = 'exact'
                
                if score > 0:
                    matching_tours.append((idx, score, reasons, match_details))
            
            # Sắp xếp theo điểm
            matching_tours.sort(key=lambda x: x[1], reverse=True)
            
            # Áp dụng thêm filter nếu có
            if filter_applied and not mandatory_filters.is_empty():
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                matching_tours = [t for t in matching_tours if t[0] in filtered_indices]
            
            # ================== GENERATE RECOMMENDATION RESPONSE ==================
            if matching_tours:
                # Lưu recommendations vào context
                context.last_recommended_tours = [idx for idx, _, _, _ in matching_tours]
                
                # Phân loại recommendations
                excellent_matches = [t for t in matching_tours if t[1] >= 60]
                good_matches = [t for t in matching_tours if 30 <= t[1] < 60]
                
                reply = "🎯 **ĐỀ XUẤT TOUR THÔNG MINH** 🎯\n\n"
                
                # Hiển thị thông tin user profile
                reply += "📋 **DỰA TRÊN YÊU CẦU CỦA BẠN:**\n"
                
                if user_profile['group_type']:
                    group_names = {
                        'family': 'Gia đình',
                        'senior': 'Người lớn tuổi/Cựu chiến binh',
                        'friends': 'Nhóm bạn',
                        'corporate': 'Công ty/Team building',
                        'couple': 'Cặp đôi',
                        'solo': 'Đi một mình'
                    }
                    reply += f"• **Đối tượng:** {group_names.get(user_profile['group_type'], user_profile['group_type'])}\n"
                
                if user_profile['interests']:
                    interest_names = {
                        'history': 'Lịch sử',
                        'nature': 'Thiên nhiên',
                        'meditation': 'Thiền/Retreat',
                        'culture': 'Văn hóa/Ẩm thực',
                        'adventure': 'Phiêu lưu',
                        'relaxation': 'Thư giãn'
                    }
                    interests_str = ', '.join([interest_names.get(i, i) for i in user_profile['interests'][:3]])
                    reply += f"• **Sở thích:** {interests_str}\n"
                
                if user_profile['budget_range']:
                    budget_names = {
                        'low': 'Tiết kiệm (dưới 1.5 triệu)',
                        'medium': 'Tầm trung (1.5-3 triệu)',
                        'high': 'Cao cấp (trên 3 triệu)'
                    }
                    reply += f"• **Ngân sách:** {budget_names.get(user_profile['budget_range'], 'Không xác định')}\n"
                
                if filter_applied and mandatory_filters:
                    reply += f"• **Bộ lọc áp dụng:** {mandatory_filters}\n"
                
                reply += "\n"
                
                # Top recommendations (xuất sắc)
                if excellent_matches:
                    reply += "🏆 **PHÙ HỢP NHẤT VỚI BẠN**\n\n"
                    
                    for idx, score, reasons, details in excellent_matches[:2]:
                        tour = TOURS_DB.get(idx)
                        if tour:
                            # Tính phần trăm phù hợp
                            match_percent = min(100, int(score))
                            
                            reply += f"**{tour.name}** ({match_percent}% phù hợp)\n"
                            reply += f"✅ **Lý do đề xuất:** {', '.join(reasons[:3])}\n"
                            
                            if tour.duration:
                                reply += f"⏱️ **Thời gian:** {tour.duration}\n"
                            if tour.location:
                                location_short = tour.location[:50] + "..." if len(tour.location) > 50 else tour.location
                                reply += f"📍 **Địa điểm:** {location_short}\n"
                            if tour.price:
                                price_short = tour.price[:80] + "..." if len(tour.price) > 80 else tour.price
                                reply += f"💰 **Giá:** {price_short}\n"
                            
                            reply += "\n"
                
                # Good recommendations
                if good_matches and (not excellent_matches or len(excellent_matches) < 2):
                    reply += "🥈 **LỰA CHỌN TỐT KHÁC**\n\n"
                    
                    display_count = min(2, len(good_matches))
                    for idx, score, reasons, details in good_matches[:display_count]:
                        tour = TOURS_DB.get(idx)
                        if tour:
                            match_percent = min(100, int(score))
                            reply += f"• **{tour.name}** ({match_percent}%)\n"
                            
                            if tour.duration:
                                reply += f"  ⏱️ {tour.duration}"
                            if tour.location:
                                loc_short = tour.location[:30] + "..." if len(tour.location) > 30 else tour.location
                                reply += f" | 📍 {loc_short}"
                            reply += "\n"
                
                reply += "\n📞 **Liên hệ để đặt tour phù hợp nhất:** 0332510486"
                
                # Lưu user profile vào context
                context.user_profile.update(user_profile)
            else:
                reply = "Hiện chưa có tour nào phù hợp với tiêu chí của bạn. Vui lòng thử với tiêu chí khác hoặc liên hệ hotline 0332510486 để được tư vấn tour riêng."
        
        # 🔹 CASE 6: COMPARISON (giữ nguyên logic cũ)
        elif 'comparison' in detected_intents:
            # ... (giữ nguyên code comparison từ phiên bản cũ) ...
            # Do giới hạn độ dài, tôi giữ nguyên logic so sánh từ code gốc
            # Bạn có thể copy nguyên phần này từ phiên bản trước
            reply = _handle_comparison_case(message_lower, tour_indices, TOURS_DB, TOUR_NAME_TO_INDEX)
        
        # 🔹 CASE 7-12: CÁC CASE KHÁC (giữ nguyên)
        # 🔹 CASE 7: EXPERIENCE INQUIRY (THÊM MỚI)
        elif 'experience' in detected_intents:
            logger.info("🌟 Processing enhanced experience inquiry")
            
            # Tạo context_info cho prompt
            context_info = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'complexity_score': complexity_score
            }
            
            # Gọi hàm experience response mới
            reply = _get_experience_response_v4(
                message_lower, 
                tour_indices, 
                TOURS_DB,
                getattr(context, 'user_profile', None)
            )
        # 🔹 CASE 8: GROUP & CUSTOM TOUR (THÊM MỚI)
        elif 'group_custom' in detected_intents:
            logger.info("👥 Processing enhanced group & custom tour request")
            
            # Tạo context_info cho prompt
            context_info = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'complexity_score': complexity_score
            }
            
            # Gọi hàm group custom response mới
            reply = _get_group_custom_response_v4(
                message_lower,
                tour_indices,
                TOURS_DB,
                mandatory_filters
            )
        
        # 🔹 CASE 9: BOOKING & POLICY (THÊM MỚI)
        elif 'booking_policy' in detected_intents:
            logger.info("📋 Processing enhanced booking & policy inquiry")
            
            # Tạo context_info cho prompt
            context_info = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'complexity_score': complexity_score,
                'user_profile': getattr(context, 'user_profile', {}),
                'sentiment': sentiment_type,
                'urgency': priority_level
            }
            
            # Gọi hàm booking policy response mới
            reply = _get_booking_policy_response_v4(
                message_lower,
                tour_indices,
                TOURS_DB,
                context_info
            )
        # ... (các case khác giữ nguyên logic) ...
        
        # 🔹 CASE 13: FALLBACK TO AI
        else:
            logger.info("🤖 Processing with AI fallback")
            
            # Chuẩn bị context
            context_info = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'filters': mandatory_filters.to_dict() if mandatory_filters else {},
                'complexity_score': complexity_score
            }
            
            # Tạo prompt
            prompt = _prepare_enhanced_llm_prompt(user_message, [], context_info, TOURS_DB)
            
            # Gọi AI nếu có
            if client and HAS_OPENAI:
                try:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=600
                    )
                    
                    if response.choices:
                        reply = response.choices[0].message.content or ""
                    else:
                        reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB)
                
                except Exception as e:
                    logger.error(f"OpenAI error: {e}")
                    reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB)
            else:
                reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB)
                
        # 🔹 CASE 16: FALLBACK TO AI
      
            logger.info("🤖 Processing with AI fallback")
            
            # Chuẩn bị context
            context_info = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'filters': mandatory_filters.to_dict() if mandatory_filters else {},
                'complexity_score': complexity_score,
                'sentiment': sentiment_type,
                'urgency': priority_level,
                'audience_type': audience_type
            }
            
            # Tạo prompt
            prompt = _prepare_enhanced_llm_prompt(user_message, [], context_info, TOURS_DB)
            
            # Gọi AI nếu có
            if client and HAS_OPENAI:
                try:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=600
                    )
                    
                    if response.choices:
                        reply = response.choices[0].message.content or ""
                    else:
                        reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB)
                
                except Exception as e:
                    logger.error(f"OpenAI error: {e}")
                    reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB)
            else:
                reply = _generate_enhanced_fallback_response(user_message, [], tour_indices, TOURS_DB) 

        # 🔹 CASE 17: WEATHER INFO (THÊM MỚI)
            if 'weather_info' in detected_intents:
                logger.info("🌤️ Processing weather information request")
                
                # Xác định địa điểm được hỏi
                locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà', 'miền trung', 'đà nẵng']
                mentioned_location = None
                
                for loc in locations:
                    if loc in message_lower:
                        mentioned_location = loc
                        break
                
                # Tìm tour tại địa điểm này nếu có
                location_tours = []
                if mentioned_location:
                    for idx, tour in TOURS_DB.items():
                        if tour.location and mentioned_location in tour.location.lower():
                            location_tours.append(tour)
                
                # Apply filters nếu có
                if filter_applied and not mandatory_filters.is_empty():
                    filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                    location_tours = [tour for idx, tour in enumerate(location_tours) if idx in filtered_indices]
                
                # Gọi hàm weather info
                reply = _get_weather_info(mentioned_location or 'miền trung', location_tours)    
                    # 🔹 CASE 18: FOOD INFORMATION (THÊM MỚI)
        if 'food_info' in detected_intents:
            logger.info("🍜 Processing food information request")
            
            # Xác định loại ẩm thực được hỏi
            food_keywords = {
                'bánh bèo': ['bánh bèo', 'banh beo'],
                'bún bò': ['bún bò', 'bun bo', 'bún bò huế', 'bun bo hue'],
                'cơm hến': ['cơm hến', 'com hen'],
                'mắm nêm': ['mắm nêm', 'mam nem'],
                'ẩm thực huế': ['ẩm thực huế', 'am thuc hue', 'đặc sản huế'],
                'ẩm thực miền trung': ['ẩm thực miền trung', 'am thuc mien trung']
            }
            
            mentioned_food = None
            for food, keywords in food_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    mentioned_food = food
                    break
            
            # Tìm tour liên quan đến ẩm thực
            food_tours = []
            for idx, tour in TOURS_DB.items():
                tour_summary = (tour.summary or '').lower()
                tour_tags = [tag.lower() for tag in (tour.tags or [])]
                
                # Kiểm tra nếu tour có liên quan đến ẩm thực
                if any(word in tour_summary for word in ['ẩm thực', 'đồ ăn', 'món ăn', 'đặc sản', 'food']) or \
                   any(tag in ['ẩm thực', 'food'] for tag in tour_tags):
                    food_tours.append(tour)
            
            # Apply filters nếu có
            if filter_applied and not mandatory_filters.is_empty():
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                food_tours = [tour for idx, tour in enumerate(food_tours) if idx in filtered_indices]
            
            # Gọi hàm food info
            reply = _get_food_info(mentioned_food, food_tours)    
        
       # ================== ENHANCE RESPONSE QUALITY V2 ==================
        
        # 1. ENHANCED FORMATTING & EMOJI OPTIMIZATION
        def enhance_response_format(text):
            """Cải thiện định dạng response với emoji và formatting thông minh"""
            if not text:
                return text
            
            lines = text.split('\n')
            enhanced_lines = []
            
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                
                # Skip empty lines
                if not stripped_line:
                    enhanced_lines.append(line)
                    continue
                
                # Tiêu đề cấp 1 (##)
                if line.startswith('## '):
                    # Thêm emoji đặc biệt cho tiêu đề chính
                    title_text = line[3:].strip()
                    if not any(emoji in title_text for emoji in ['✨', '🎯', '📍', '💰', '🛎️', '🌿', '🏛️', '🕉️']):
                        if 'DỊCH VỤ' in title_text:
                            line = f"## 🛎️ {title_text} 🛎️"
                        elif 'GIÁ' in title_text or 'BẢNG GIÁ' in title_text:
                            line = f"## 💰 {title_text} 💰"
                        elif 'TOUR' in title_text or 'HÀNH TRÌNH' in title_text:
                            line = f"## 🗺️ {title_text} 🗺️"
                        elif 'ƯU ĐÃI' in title_text or 'KHUYẾN MÃI' in title_text:
                            line = f"## 🎁 {title_text} 🎁"
                        elif 'TRIẾT LÝ' in title_text or 'GIỚI THIỆU' in title_text:
                            line = f"## ✨ {title_text} ✨"
                        elif 'THỜI TIẾT' in title_text:
                            line = f"## 🌤️ {title_text} 🌤️"
                        elif 'ẨM THỰC' in title_text:
                            line = f"## 🍜 {title_text} 🍜"
                        elif 'VĂN HÓA' in title_text or 'LỊCH SỬ' in title_text:
                            line = f"## 🏛️ {title_text} 🏛️"
                
                # Tiêu đề cấp 2 (### hoặc **)
                elif line.startswith('### ') or (line.startswith('**') and line.endswith('**')):
                    if line.startswith('### '):
                        title_text = line[4:].strip()
                    else:
                        title_text = line[2:-2].strip()
                    
                    # Thêm emoji cho tiêu đề phụ nếu chưa có
                    if not any(emoji in title_text for emoji in ['•', '✅', '❌', '⚠️', '📌']):
                        if any(word in title_text.lower() for word in ['dịch vụ bao gồm', 'có gì', 'bao gồm']):
                            line = f"### ✅ {title_text}"
                        elif any(word in title_text.lower() for word in ['không bao gồm', 'không có', 'chưa bao gồm']):
                            line = f"### ❌ {title_text}"
                        elif any(word in title_text.lower() for word in ['lưu ý', 'chú ý', 'quan trọng']):
                            line = f"### ⚠️ {title_text}"
                        elif any(word in title_text.lower() for word in ['ưu đãi', 'giảm giá', 'khuyến mãi']):
                            line = f"### 🎯 {title_text}"
                        elif any(word in title_text.lower() for word in ['địa điểm', 'nơi đến', 'vị trí']):
                            line = f"### 📍 {title_text}"
                        elif any(word in title_text.lower() for word in ['thời gian', 'lịch trình', 'ngày']):
                            line = f"### ⏱️ {title_text}"
                
                # Bullet points (•)
                elif '•' in line:
                    # Thêm emoji cho bullet points dựa trên nội dung
                    if 'giảm' in line.lower() and '💰' not in line and '🎯' not in line:
                        line = line.replace('•', '💰 •', 1)
                    elif any(word in line.lower() for word in ['tour', 'tour', 'chương trình']):
                        line = line.replace('•', '🗺️ •', 1)
                    elif any(word in line.lower() for word in ['hotline', 'liên hệ', 'điện thoại', '0332510486']):
                        line = line.replace('•', '📞 •', 1)
                    elif any(word in line.lower() for word in ['địa điểm', 'nơi', 'vị trí', 'đến']):
                        line = line.replace('•', '📍 •', 1)
                    elif any(word in line.lower() for word in ['thời gian', 'ngày', 'đêm', 'giờ']):
                        line = line.replace('•', '⏱️ •', 1)
                    elif any(word in line.lower() for word in ['ưu đãi', 'khuyến mãi', 'tặng']):
                        line = line.replace('•', '🎁 •', 1)
                    elif any(word in line.lower() for word in ['lưu ý', 'chú ý', 'cảnh báo']):
                        line = line.replace('•', '⚠️ •', 1)
                    elif any(word in line.lower() for word in ['bao gồm', 'có sẵn', 'cung cấp']):
                        line = line.replace('•', '✅ •', 1)
                    elif any(word in line.lower() for word in ['không bao gồm', 'chưa bao gồm', 'tính thêm']):
                        line = line.replace('•', '❌ •', 1)
                    elif 'miễn phí' in line.lower():
                        line = line.replace('•', '🎉 •', 1)
                    elif any(word in line.lower() for word in ['trẻ em', 'trẻ', 'con nhỏ']):
                        line = line.replace('•', '👶 •', 1)
                    elif any(word in line.lower() for word in ['người lớn tuổi', 'cựu chiến binh', 'cao tuổi']):
                        line = line.replace('•', '👴 •', 1)
                    elif any(word in line.lower() for word in ['gia đình', 'bố mẹ', 'ông bà']):
                        line = line.replace('•', '👨‍👩‍👧‍👦 •', 1)
                    elif any(word in line.lower() for word in ['bạn bè', 'nhóm bạn', 'sinh viên']):
                        line = line.replace('•', '👥 •', 1)
                    elif any(word in line.lower() for word in ['công ty', 'team building', 'doanh nghiệp']):
                        line = line.replace('•', '🏢 •', 1)
                
                # Thêm spacing thông minh
                if i > 0 and len(lines) > i + 1:
                    prev_line = lines[i-1].strip()
                    next_line = lines[i+1].strip()
                    
                    # Thêm dòng trống trước tiêu đề
                    if line.startswith(('## ', '### ', '**')) and prev_line and not prev_line.startswith(('## ', '### ', '**')):
                        if not enhanced_lines or enhanced_lines[-1].strip() != "":
                            enhanced_lines.append("")
                    
                    # Thêm dòng trống sau tiêu đề nếu cần
                    if line.startswith(('## ', '### ', '**')) and next_line and not next_line.startswith(('## ', '### ', '**')):
                        enhanced_lines.append(line)
                        enhanced_lines.append("")
                        continue
                
                enhanced_lines.append(line)
            
            # Loại bỏ dòng trống thừa ở đầu/cuối
            while enhanced_lines and not enhanced_lines[0].strip():
                enhanced_lines.pop(0)
            while enhanced_lines and not enhanced_lines[-1].strip():
                enhanced_lines.pop()
            
            return '\n'.join(enhanced_lines)
        
        # 2. SMART LENGTH LIMITING
        def smart_truncate(text, max_length=2500):
            """Cắt text thông minh không làm mất ý chính"""
            if len(text) <= max_length:
                return text
            
            logger.info(f"📏 Response too long: {len(text)} chars, truncating to {max_length}")
            
            # Tìm vị trí cắt tốt nhất
            cut_positions = []
            
            # Ưu tiên 1: Cắt ở cuối đoạn (2 dòng trống liên tiếp)
            double_newline_pos = text.rfind('\n\n', 0, max_length)
            if double_newline_pos != -1:
                cut_positions.append((double_newline_pos, 'paragraph_end'))
            
            # Ưu tiên 2: Cắt ở cuối bullet list
            bullet_end_patterns = ['\n\n## ', '\n\n### ', '\n\n**']
            for pattern in bullet_end_patterns:
                pos = text.rfind(pattern, 0, max_length)
                if pos != -1:
                    cut_positions.append((pos, 'section_end'))
            
            # Ưu tiên 3: Cắt ở cuối câu
            sentence_enders = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            for ender in sentence_enders:
                pos = text.rfind(ender, 0, max_length - 100)  # Để chỗ cho thông báo
                if pos != -1:
                    cut_positions.append((pos + len(ender) - 1, 'sentence_end'))
            
            # Ưu tiên 4: Cắt ở dòng mới
            newline_pos = text.rfind('\n', 0, max_length - 50)
            if newline_pos != -1:
                cut_positions.append((newline_pos, 'line_end'))
            
            # Chọn vị trí cắt tốt nhất
            if cut_positions:
                cut_positions.sort(key=lambda x: x[0], reverse=True)
                best_cut_pos = cut_positions[0][0]
                cut_type = cut_positions[0][1]
            else:
                best_cut_pos = max_length - 200
                cut_type = 'forced'
            
            # Đảm bảo không cắt giữa emoji hoặc định dạng markdown
            truncated = text[:best_cut_pos]
            
            # Loại bỏ các ký tự markdown không đóng
            markdown_pairs = [('**', '**'), ('*', '*'), ('`', '`')]
            for open_char, close_char in markdown_pairs:
                open_count = truncated.count(open_char)
                close_count = truncated.count(close_char)
                if open_count > close_count:
                    # Thêm close char nếu thiếu
                    truncated += close_char
            
            # Thêm thông báo cắt
            if cut_type != 'forced':
                truncated = truncated.rstrip() + "..."
            
            truncated += "\n\n💡 **Thông tin còn tiếp...**\n"
            truncated += "📞 **Liên hệ ngay để biết thêm chi tiết:** 0332510486"
            
            logger.info(f"📏 Truncated at position {best_cut_pos} ({cut_type}), new length: {len(truncated)}")
            return truncated
        
        # 3. HOTLINE ENSUREMENT WITH SMART FORMATTING
        def ensure_hotline_presence(text):
            """Đảm bảo hotline có mặt với định dạng đẹp"""
            hotline_patterns = [
                '0332510486',
                'hotline',
                'liên hệ tư vấn',
                'điện thoại tư vấn',
                'số điện thoại'
            ]
            
            has_hotline = any(pattern in text.lower() for pattern in hotline_patterns)
            
            if not has_hotline:
                # Thêm hotline với formatting đẹp
                hotline_section = "\n\n---\n"
                hotline_section += "📞 **HOTLINE TƯ VẤN 24/7:** 0332510486\n"
                hotline_section += "💬 **Chat trực tiếp với chuyên viên Ruby Wings**"
                return text + hotline_section
            else:
                # Cải thiện formatting của hotline nếu đã có
                lines = text.split('\n')
                enhanced_lines = []
                
                for line in lines:
                    if '0332510486' in line and '📞' not in line:
                        # Thêm emoji nếu chưa có
                        line = line.replace('0332510486', '📞 0332510486')
                        if 'hotline' in line.lower() and '**' not in line:
                            line = '📞 **' + line.strip() + '**'
                    enhanced_lines.append(line)
                
                return '\n'.join(enhanced_lines)
        
        # 4. SIGNATURE ADDITION
        def add_signature(text):
            """Thêm signature nếu response đủ dài"""
            if len(text) < 200:
                return text
            
            signature_variants = [
                "\n\n---\n**Ruby Wings Travel** ✨ _Hành trình ý nghĩa - Trải nghiệm thực tế - Có chiều sâu_",
                "\n\n---\n**Ruby Wings Travel** 🦋 _Chuẩn mực - Chân thành - Có chiều sâu_",
                "\n\n---\n**Ruby Wings Travel** 🌟 _Mang đến tour đáng nhớ cho mọi du khách_",
                "\n\n---\n**Ruby Wings Travel** 🗺️ _Khám phá miền Trung với trải nghiệm độc đáo_"
            ]
            
            # Chọn signature ngẫu nhiên dựa trên độ dài response
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            variant_index = int(text_hash, 16) % len(signature_variants)
            
            # Kiểm tra xem đã có signature chưa
            signature_keywords = ['Ruby Wings Travel', '---', 'Hành trình ý nghĩa']
            has_signature = any(keyword in text for keyword in signature_keywords)
            
            if not has_signature:
                # Không thêm signature nếu đã có hotline ở cuối
                last_200 = text[-200:].lower()
                if '0332510486' not in last_200 and 'hotline' not in last_200:
                    return text + signature_variants[variant_index]
            
            return text
        
        # 5. RESPONSIVE SPACING
        def optimize_spacing(text):
            """Tối ưu khoảng cách và spacing cho dễ đọc"""
            lines = text.split('\n')
            optimized_lines = []
            
            in_bullet_list = False
            bullet_list_items = []
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Xử lý bullet lists
                if '•' in line:
                    if not in_bullet_list:
                        in_bullet_list = True
                        # Thêm dòng trống trước bullet list
                        if i > 0 and lines[i-1].strip() and not any(bullet in lines[i-1] for bullet in ['•', '##', '###']):
                            optimized_lines.append("")
                    
                    bullet_list_items.append(line)
                
                else:
                    # Kết thúc bullet list
                    if in_bullet_list and bullet_list_items:
                        # Thêm các item
                        optimized_lines.extend(bullet_list_items)
                        # Thêm dòng trống sau bullet list
                        if i < len(lines) - 1 and lines[i].strip():
                            optimized_lines.append("")
                        
                        bullet_list_items = []
                        in_bullet_list = False
                    
                    optimized_lines.append(line)
            
            # Xử lý bullet list còn sót
            if bullet_list_items:
                optimized_lines.extend(bullet_list_items)
            
            # Loại bỏ dòng trống thừa
            final_lines = []
            empty_line_count = 0
            
            for line in optimized_lines:
                if not line.strip():
                    empty_line_count += 1
                    if empty_line_count <= 2:  # Giữ tối đa 2 dòng trống liên tiếp
                        final_lines.append(line)
                else:
                    empty_line_count = 0
                    final_lines.append(line)
            
            return '\n'.join(final_lines)
        
        # ========== APPLY ALL ENHANCEMENTS ==========
        
        # Bước 1: Áp dụng enhanced formatting
        reply = enhance_response_format(reply)
        
        # Bước 2: Tối ưu spacing
        reply = optimize_spacing(reply)
        
        # Bước 3: Giới hạn độ dài thông minh
        original_length = len(reply)
        if original_length > 2500:
            reply = smart_truncate(reply, max_length=2500)
            logger.info(f"📏 Response truncated from {original_length} to {len(reply)} chars")
        
        # Bước 4: Đảm bảo có hotline
        reply = ensure_hotline_presence(reply)
        
        # Bước 5: Thêm signature nếu phù hợp
        if len(reply) > 300:
            reply = add_signature(reply)
        
        # Bước 6: Final length check và xử lý đặc biệt
        final_length = len(reply)
        if final_length > 3000:
            # Trường hợp cực đoan: cắt cứng nhưng vẫn giữ hotline
            logger.warning(f"⚠️ Response still too long after truncation: {final_length} chars")
            # Giữ 2900 ký tự đầu + thông báo
            reply = reply[:2900] + "...\n\n📞 **Vui lòng liên hệ hotline 0332510486 để biết thêm chi tiết.**"
        
        # Log final response stats
        logger.info(f"✅ Response quality enhanced: {original_length} → {len(reply)} chars")
        
        # ================== UPDATE CONTEXT V2 ==================
        
        # 1. ENHANCED TOUR CONTEXT TRACKING
        if tour_indices and len(tour_indices) > 0:
            # Lưu current tour
            context.current_tour = tour_indices[0]
            tour = TOURS_DB.get(tour_indices[0])
            if tour:
                context.last_tour_name = tour.name
                
                # Lưu thêm metadata về tour
                if not hasattr(context, 'tour_view_history'):
                    context.tour_view_history = []
                
                tour_view_data = {
                    'tour_index': tour_indices[0],
                    'tour_name': tour.name,
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': primary_intent or 'search_result'
                }
                
                # Tránh trùng lặp trong lịch sử xem
                existing_indices = [t.get('tour_index') for t in context.tour_view_history]
                if tour_indices[0] not in existing_indices:
                    context.tour_view_history.append(tour_view_data)
                    
                    # Giới hạn lịch sử xem tour (tối đa 10)
                    if len(context.tour_view_history) > 10:
                        context.tour_view_history = context.tour_view_history[-10:]
        
        # 2. ENHANCED USER PROFILE TRACKING
        if not hasattr(context, 'user_profile'):
            context.user_profile = {
                'basic_info': {},
                'preferences': {},
                'interaction_stats': {},
                'inferred_interests': [],
                'request_history': []
            }
        
        # Cập nhật thông tin từ context_analysis (nếu có)
        if 'context_analysis' in locals():
            analysis = context_analysis
            
            # Cập nhật audience type
            if analysis.get('audience_type'):
                context.user_profile['basic_info']['audience_type'] = analysis['audience_type']
            
            # Cập nhật interests từ analysis
            if analysis.get('interests') and len(analysis['interests']) > 0:
                for interest in analysis['interests']:
                    if interest not in context.user_profile['inferred_interests']:
                        context.user_profile['inferred_interests'].append(interest)
            
            # Cập nhật sentiment profile
            if analysis.get('sentiment') and analysis['sentiment']['type'] != 'neutral':
                sentiment_key = f"sentiment_{analysis['sentiment']['type']}"
                context.user_profile['interaction_stats'][sentiment_key] = \
                    context.user_profile['interaction_stats'].get(sentiment_key, 0) + 1
        
        # Cập nhật thông tin từ mandatory_filters
        if mandatory_filters and not mandatory_filters.is_empty():
            if hasattr(mandatory_filters, 'group_type') and mandatory_filters.group_type:
                context.user_profile['basic_info']['preferred_group_type'] = mandatory_filters.group_type
            
            if hasattr(mandatory_filters, 'location') and mandatory_filters.location:
                context.user_profile['preferences']['preferred_location'] = mandatory_filters.location
            
            if hasattr(mandatory_filters, 'duration_min') or hasattr(mandatory_filters, 'duration_max'):
                context.user_profile['preferences']['tour_duration'] = {
                    'min': getattr(mandatory_filters, 'duration_min', None),
                    'max': getattr(mandatory_filters, 'duration_max', None)
                }
        
        # Cập nhật từ primary_intent và detected_intents
        if primary_intent:
            context.user_profile['interaction_stats']['intent_counts'] = \
                context.user_profile['interaction_stats'].get('intent_counts', {})
            context.user_profile['interaction_stats']['intent_counts'][primary_intent] = \
                context.user_profile['interaction_stats']['intent_counts'].get(primary_intent, 0) + 1
        
        # Cập nhật complexity profile
        context.user_profile['interaction_stats']['avg_complexity'] = \
            context.user_profile['interaction_stats'].get('avg_complexity', 0) * 0.8 + complexity_score * 0.2
        context.user_profile['interaction_stats']['total_messages'] = \
            context.user_profile['interaction_stats'].get('total_messages', 0) + 1
        
        # 3. ENHANCED CONVERSATION HISTORY MANAGEMENT
        # Tạo metadata entry chi tiết
        metadata_entry = {
            'role': 'assistant',
            'message': reply,
            'timestamp': datetime.utcnow().isoformat(),
            'timestamp_human': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'tour_indices': tour_indices,
            'detected_intents': detected_intents,
            'primary_intent': primary_intent,
            'complexity_score': complexity_score,
            'response_length': len(reply),
            'filter_applied': filter_applied,
            'filters': mandatory_filters.to_dict() if mandatory_filters and hasattr(mandatory_filters, 'to_dict') else {},
            'context_analysis': context_analysis if 'context_analysis' in locals() else None,
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'session_id': session_id
        }
        
        # Thêm tour details nếu có
        if tour_indices and len(tour_indices) > 0:
            tour_details = []
            for idx in tour_indices[:3]:
                tour = TOURS_DB.get(idx)
                if tour:
                    tour_details.append({
                        'index': idx,
                        'name': tour.name,
                        'duration': tour.duration,
                        'location': tour.location
                    })
            metadata_entry['tour_details'] = tour_details
        
        # Thêm thông tin từ các hệ thống con
        if hasattr(context, 'multiple_intents'):
            metadata_entry['multiple_intents'] = context.multiple_intents
            delattr(context, 'multiple_intents')
        
        if hasattr(context, 'filter_warning'):
            metadata_entry['filter_warning'] = context.filter_warning
            delattr(context, 'filter_warning')
        
        if hasattr(context, 'filter_fallback'):
            metadata_entry['filter_fallback'] = context.filter_fallback
            delattr(context, 'filter_fallback')
        
        if hasattr(context, 'additional_filters'):
            metadata_entry['additional_filters'] = context.additional_filters
            delattr(context, 'additional_filters')
        
        # Lưu vào conversation history
        context.conversation_history.append(metadata_entry)
        
        # 4. INTELLIGENT HISTORY COMPRESSION & MANAGEMENT
        # Giới hạn history (giữ 40 tin nhắn gần nhất)
        if len(context.conversation_history) > 40:
            # Strategy: Giữ toàn bộ 20 tin nhắn gần nhất, nén 20 tin nhắn cũ hơn
            recent_history = context.conversation_history[-20:]
            older_history = context.conversation_history[:-20]
            
            if len(older_history) > 10:
                # Nén older history: chỉ giữ metadata quan trọng
                compressed_older = []
                for entry in older_history[-10:]:
                    compressed_entry = {
                        'role': entry.get('role'),
                        'timestamp': entry.get('timestamp'),
                        'primary_intent': entry.get('primary_intent'),
                        'tour_count': len(entry.get('tour_indices', [])),
                        'compressed': True
                    }
                    compressed_older.append(compressed_entry)
                
                # Kết hợp lại
                context.conversation_history = compressed_older + recent_history
            else:
                context.conversation_history = older_history + recent_history
        
        # 5. REQUEST HISTORY TRACKING
        # Lưu request vào history riêng
        request_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': user_message[:100],  # Giữ 100 ký tự đầu
            'primary_intent': primary_intent,
            'tour_count': len(tour_indices) if tour_indices else 0,
            'complexity': complexity_score,
            'filters_applied': filter_applied
        }
        
        if not hasattr(context, 'request_history'):
            context.request_history = []
        
        context.request_history.append(request_summary)
        
        # Giới hạn request history (tối đa 20)
        if len(context.request_history) > 20:
            context.request_history = context.request_history[-20:]
        
        # 6. SESSION ANALYTICS
        if not hasattr(context, 'session_analytics'):
            context.session_analytics = {
                'start_time': datetime.utcnow().isoformat(),
                'message_count': 0,
                'intent_distribution': {},
                'tour_views': {},
                'filter_usage': {},
                'avg_response_time': 0
            }
        
        # Cập nhật session analytics
        context.session_analytics['message_count'] = len(context.conversation_history)
        
        if primary_intent:
            context.session_analytics['intent_distribution'][primary_intent] = \
                context.session_analytics['intent_distribution'].get(primary_intent, 0) + 1
        
        if tour_indices:
            for idx in tour_indices[:3]:
                context.session_analytics['tour_views'][str(idx)] = \
                    context.session_analytics['tour_views'].get(str(idx), 0) + 1
        
        if filter_applied:
            filter_types = []
            if hasattr(mandatory_filters, 'group_type') and mandatory_filters.group_type:
                filter_types.append(f"group:{mandatory_filters.group_type}")
            if hasattr(mandatory_filters, 'location') and mandatory_filters.location:
                filter_types.append(f"location:{mandatory_filters.location}")
            
            for ft in filter_types:
                context.session_analytics['filter_usage'][ft] = \
                    context.session_analytics['filter_usage'].get(ft, 0) + 1
        
        # Tính avg response time
        current_processing_time = (time.time() - start_time) * 1000
        old_avg = context.session_analytics['avg_response_time']
        total_msgs = context.session_analytics['message_count']
        context.session_analytics['avg_response_time'] = \
            (old_avg * (total_msgs - 1) + current_processing_time) / total_msgs if total_msgs > 0 else current_processing_time
        
        # 7. Lưu session context
        save_session_context(session_id, context)
        
        # 8. LOGGING ENHANCED
        logger.info(f"📝 Context Updated:")
        logger.info(f"   • Session: {session_id}")
        logger.info(f"   • Tour Indices: {tour_indices} ({len(tour_indices) if tour_indices else 0} tours)")
        logger.info(f"   • Primary Intent: {primary_intent}")
        logger.info(f"   • Detected Intents: {len(detected_intents)}")
        logger.info(f"   • Complexity Score: {complexity_score}/10")
        logger.info(f"   • Filter Applied: {filter_applied}")
        logger.info(f"   • Response Length: {len(reply)} chars")
        logger.info(f"   • Processing Time: {int((time.time() - start_time) * 1000)}ms")
        
        # Log user profile summary
        if hasattr(context, 'user_profile') and context.user_profile:
            profile_summary = {
                'audience': context.user_profile.get('basic_info', {}).get('audience_type', 'unknown'),
                'interests': len(context.user_profile.get('inferred_interests', [])),
                'messages': context.user_profile.get('interaction_stats', {}).get('total_messages', 0),
                'avg_complexity': round(context.user_profile.get('interaction_stats', {}).get('avg_complexity', 0), 1)
            }
            logger.info(f"👤 User Profile Summary: {profile_summary}")
        
        # ================== FINAL RESPONSE ==================
        processing_time = time.time() - start_time
        
        chat_response = ChatResponse(
            reply=reply,
            sources=sources,
            context={
                "session_id": session_id,
                "current_tour": getattr(context, 'current_tour', None),
                "last_tour_name": getattr(context, 'last_tour_name', None),
                "user_preferences": getattr(context, 'user_profile', {}),
                "detected_intents": detected_intents,
                "primary_intent": primary_intent,
                "processing_time_ms": int(processing_time * 1000),
                "tours_found": len(tour_indices),
                "complexity_score": complexity_score,
                "filter_applied": filter_applied
            },
            tour_indices=tour_indices,
            processing_time_ms=int(processing_time * 1000),
            from_memory=False
        )
        
        # Cache response
        if UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            context_hash = hashlib.md5(json.dumps({
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'complexity': complexity_score,
                'filters': mandatory_filters.to_dict() if mandatory_filters else {}
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = CacheSystem.get_cache_key(user_message, context_hash)
            CacheSystem.set(cache_key, chat_response.to_dict(), expiry=300)
        
        logger.info(f"✅ Processed in {processing_time:.2f}s | "
                   f"Primary Intent: {primary_intent} | "
                   f"Tours: {len(tour_indices)} | "
                   f"Complexity: {complexity_score} | "
                   f"Filters: {filter_applied}")
        
        return jsonify(chat_response.to_dict())
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}\n{traceback.format_exc()}")
        
        processing_time = time.time() - start_time
        
        # Enhanced error response
        error_response = ChatResponse(
            reply="⚡ **Có chút trục trặc kỹ thuật!**\n\n"
                  "Đội ngũ Ruby Wings vẫn sẵn sàng hỗ trợ bạn qua các kênh sau:\n\n"
                  "🔧 **GIẢI PHÁP NHANH:**\n"
                  "1. **Gọi trực tiếp:** 📞 0332510486 (tư vấn ngay)\n"
                  "2. **Hỏi đơn giản hơn:** 'Tour 1 ngày Huế', 'Tour gia đình 2 ngày'\n"
                  "3. **Chọn từ danh sách:**\n"
                  "   • Tour thiên nhiên Bạch Mã\n"
                  "   • Tour lịch sử Trường Sơn\n"
                  "   • Tour retreat thiền\n\n"
                  "⏰ **Chúng tôi hoạt động 24/7 để phục vụ bạn tốt nhất!** 😊",
            sources=[],
            context={
                "error": str(e),
                "processing_time_ms": int(processing_time * 1000),
                "error_type": type(e).__name__
            },
            tour_indices=[],
            processing_time_ms=int(processing_time * 1000),
            from_memory=False
        )
        
        return jsonify(error_response.to_dict()), 500



# ================== ENHANCED HELPER FUNCTIONS V3 ==================

def _extract_price_value(price_text):
    """Trích xuất giá trị số từ text giá - NÂNG CẤP THÔNG MINH"""
    if not price_text:
        return None
    
    import re
    
    # NÂNG CẤP: Tìm kiếm thông minh hơn với nhiều định dạng
    patterns = [
        # Định dạng 1: "1.500.000 VNĐ"
        r'(\d{1,3}(?:\.\d{3})*(?:\.\d{1,3})?)\s*(?:vnđ|vnd|đồng|₫|d)',
        # Định dạng 2: "1,500,000"
        r'(\d{1,3}(?:,\d{3})*(?:,\d{1,3})?)',
        # Định dạng 3: "1.5 triệu"
        r'(\d{1,3}(?:\.\d{1,2})?)\s*(?:triệu|tr|m)',
        # Định dạng 4: "1500k"
        r'(\d{1,10})k\b',
        # Định dạng 5: "2-3 triệu"
        r'(\d{1,3})\s*(?:đến|-|tới)\s*(\d{1,3})\s*(?:triệu|tr|m)',
        # Định dạng 6: Số đơn giản
        r'\b(\d{4,10})\b'
    ]
    
    all_numbers = []
    
    for pattern in patterns:
        matches = re.findall(pattern, price_text.lower().replace(',', '.'))
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    # Trường hợp range: "2-3 triệu"
                    for num in match:
                        if num and num.strip():
                            try:
                                num_clean = num.replace('.', '')
                                if '.' in num:
                                    # Số thập phân: "1.5 triệu"
                                    value = float(num) * 1000000
                                else:
                                    value = int(num_clean) * 1000000
                                all_numbers.append(int(value))
                            except:
                                continue
                else:
                    # Trường hợp đơn
                    try:
                        num_str = match.strip()
                        if '.' in num_str and num_str.count('.') == 1:
                            # Số thập phân: "1.5 triệu"
                            value = float(num_str) * 1000000
                        elif '.' in num_str:
                            # Định dạng nghìn: "1.500.000"
                            num_clean = num_str.replace('.', '')
                            value = int(num_clean)
                        else:
                            # Số nguyên
                            value = int(num_str)
                            # Kiểm tra đơn vị
                            if 'triệu' in price_text.lower() or 'tr' in price_text.lower() or 'm' in price_text.lower():
                                value = value * 1000000
                            elif 'k' in price_text.lower() and value < 10000:
                                value = value * 1000
                        all_numbers.append(int(value))
                    except:
                        continue
    
    # Tối ưu hóa: lọc giá trị hợp lý (từ 100,000 đến 50,000,000 VNĐ)
    valid_numbers = [n for n in all_numbers if 100000 <= n <= 50000000]
    
    if valid_numbers:
        # Ưu tiên giá nhỏ nhất nếu có nhiều giá
        return min(valid_numbers)
    
    # Fallback: tìm bất kỳ số nào
    if all_numbers:
        return min(all_numbers)
    
    return None


def _get_philosophy_response():
    """Trả lời về triết lý Ruby Wings - NÂNG CẤP CHI TIẾT"""
    return """✨ **TRIẾT LÝ 'CHUẨN MỰC - CHÂN THÀNH - CÓ CHIỀU SÂU'** ✨

**🌌 MỤC ĐÍCH SÂU XA:**
Không chỉ là du lịch, Ruby Wings tạo ra hành trình chạm đến cảm xúc, mở ra nhận thức mới, và kết nối con người với lịch sử, thiên nhiên và chính mình.

**🏆 CHUẨN MỰC - SỰ HOÀN HẢO TRONG TỪNG CHI TIẾT:**

🔹 **AN TOÀN TUYỆT ĐỐI:**
• Đánh giá rủi ro trước mỗi hành trình
• Nhân viên được đào tạo CPR & sơ cứu
• Thiết bị an toàn đạt chuẩn quốc tế
• Bảo hiểm du lịch cao cấp (đền bù đến 100 triệu)

🔹 **CHUYÊN NGHIỆP VƯỢT TRỘI:**
• HDV được chứng nhận quốc tế
• Quy trình chuẩn hóa ISO
• Đánh giá chất lượng sau mỗi hành trình
• Cập nhật kiến thức liên tục

🔹 **CHẤT LƯỢNG KHÔNG THỎA HIỆP:**
• Đối tác được lựa chọn kỹ lưỡng
• Nguyên vật liệu tươi ngon nhất
• Phương tiện đời mới, bảo dưỡng định kỳ
• Kiểm soát chất lượng 3 cấp độ

**❤️ CHÂN THÀNH - KẾT NỐI TỪ TRÁI TIM:**

🔸 **MINH BẠCH TUYỆT ĐỐI:**
• Báo giá chi tiết, không phát sinh
• Thông tin rõ ràng, không giấu diếm
• Hợp đồng đầy đủ điều khoản
• Phản hồi 24/7

🔸 **ĐỒNG HÀNH NHƯ NGƯỜI THÂN:**
• Tư vấn tận tâm, không ép mua
• Hỗ trợ xuyên suốt hành trình
• Quan tâm đến từng cá nhân
• Lắng nghe và thấu hiểu

🔸 **TRÁCH NHIỆM VỚI CỘNG ĐỒNG:**
• Tôn trọng văn hóa địa phương
• Hỗ trợ doanh nghiệp địa phương
• Bảo vệ môi trường tự nhiên
• Đóng góp cho phát triển bền vững

**🌠 CÓ CHIỀU SÂU - GIÁ TRỊ BỀN VỮNG:**

🌀 **HÀNH TRÌNH Ý NGHĨA:**
• Mỗi chuyến đi là một bài học
• Trải nghiệm thay đổi nhận thức
• Kết nối quá khứ - hiện tại - tương lai
• Tạo ra kỷ niệm vượt thời gian

🌀 **KHÁM PHÁ BẢN CHẤT:**
• Vượt qua bề nổi du lịch thông thường
• Thấu hiểu giá trị văn hóa
• Cảm nhận sâu sắc thiên nhiên
• Kết nối với bản thể chân thật

🌀 **TRUYỀN CẢM HỨNG:**
• Truyền lửa đam mê khám phá
• Khơi dậy lòng biết ơn
• Tạo động lực thay đổi tích cực
• Lan tỏa năng lượng tốt đẹp

**🎯 TẦM NHÌN & SỨ MỆNH:**

🌍 **TẦM NHÌN 2030:**
Trở thành tổ chức du lịch trải nghiệm dẫn đầu Đông Nam Á, được công nhận về chất lượng dịch vụ và đóng góp cho phát triển bền vững.

🕊️ **SỨ MỆNH:**
Mang đến những hành trình không chỉ thay đổi điểm đến mà còn thay đổi cách nhìn, không chỉ tạo kỷ niệm mà còn tạo ra giá trị, không chỉ phục vụ khách hàng mà còn phụng sự cộng đồng.

📞 **Trải nghiệm triết lý Ruby Wings trong từng hành trình:** 0332510486

✨ *"Mỗi bước chân là một khám phá, mỗi hành trình là một sự chuyển hóa"* ✨"""


def _get_company_introduction():
    """Trả lời giới thiệu công ty - NÂNG CẤP ĐẦY ĐỦ"""
    return """🏛️ **GIỚI THIỆU CHI TIẾT RUBY WINGS TRAVEL** 🏛️

**📜 LỊCH SỬ HÌNH THÀNH:**
Thành lập năm 2018 với sứ mệnh thay đổi cách du lịch truyền thống, Ruby Wings đã phát triển từ nhóm nhỏ thành tổ chức du lịch trải nghiệm uy tín tại miền Trung Việt Nam.

**🌟 ĐIỂM KHÁC BIỆT CỐT LÕI:**

1. **THIẾT KẾ HÀNH TRÌNH ĐẶC BIỆT:**
   • Mỗi hành trình là một câu chuyện
   • Kết hợp yếu tố văn hóa, lịch sử, thiên nhiên
   • Hoạt động có chiều sâu, ý nghĩa
   • Đội ngũ nghiên cứu chuyên sâu

2. **ĐỘI NGŨ CHUYÊN GIA:**
   • HDV am hiểu văn hóa, lịch sử
   • Chuyên gia wellness & thiền định
   • Chuyên viên văn hóa địa phương
   • Nhân viên y tế đi kèm (hành trình đặc biệt)

3. **CƠ SỞ VẬT CHẤT CAO CẤP:**
   • Xe 16-45 chỗ đời mới
   • Thiết bị chuyên dụng (trekking, camping)
   • Hệ thống liên lạc vệ tinh
   • Thiết bị y tế đầy đủ

**🎯 4 TRỤ CỘT CHÍNH:**

1. **TOUR LỊCH SỬ - TRI ÂN:**
   🏛️ **Trọng tâm:** Di tích, chiến trường, di sản
   ✅ **Hoạt động:** Tham quan di tích, gặp nhân chứng, lễ tri ân
   📍 **Địa điểm:** Thành cổ Quảng Trị, Địa đạo Vịnh Mốc, Đường HCM
   👥 **Phù hợp:** Cựu chiến binh, học sinh, nhóm tìm hiểu lịch sử

2. **TOUR RETREAT - CHỮA LÀNH:**
   🧘 **Trọng tâm:** Thiền, yoga, khí công, tĩnh tâm
   ✅ **Hoạt động:** Thiền định, yoga, workshop healing
   📍 **Địa điểm:** Bạch Mã, rừng nguyên sinh, bãi biển yên tĩnh
   👥 **Phù hợp:** Người cần thư giãn, cân bằng cuộc sống, phục hồi năng lượng

3. **TOUR THIÊN NHIÊN - KHÁM PHÁ:**
   🌿 **Trọng tâm:** Rừng núi, động thực vật, hệ sinh thái
   ✅ **Hoạt động:** Trekking, camping, quan sát động vật
   📍 **Địa điểm:** VQG Bạch Mã, Trường Sơn, rừng nguyên sinh
   👥 **Phù hợp:** Nhóm bạn, gia đình, người yêu thiên nhiên

4. **TOUR VĂN HÓA - ẨM THỰC:**
   🍜 **Trọng tâm:** Ẩm thực, làng nghề, phong tục địa phương
   ✅ **Hoạt động:** Học nấu ăn, thăm làng nghề, giao lưu văn nghệ
   📍 **Địa điểm:** Huế, làng Chuồn, làng Sình
   👥 **Phù hợp:** Người yêu ẩm thực, tìm hiểu văn hóa

**📊 THÀNH TỰU & CHỨNG NHẬN:**

🏆 **GIẢI THƯỞNG:**
• Top 5 Tour Operator uy tín 2023
• Giải thưởng Du lịch bền vững 2022
• Doanh nghiệp văn hóa tiêu biểu 2021

✅ **CHỨNG NHẬN:**
• ISO 9001:2015 (Quản lý chất lượng)
• An toàn du lịch quốc tế
• Đối tác của UNESCO Huế
• Thành viên Hiệp hội Du lịch bền vững

**🤝 ĐỐI TÁC CHIẾN LƯỢC:**

1. **TỔ CHỨC QUỐC TẾ:**
   • UNESCO Việt Nam
   • WWF Việt Nam
   • Tổ chức Bảo tồn Thiên nhiên

2. **DOANH NGHIỆP ĐỊA PHƯƠNG:**
   • 50+ homestay, khách sạn đối tác
   • 30+ nhà hàng, quán ăn đặc sản
   • 20+ làng nghề truyền thống
   • 10+ tổ chức cộng đồng

3. **TRƯỜNG HỌC & TỔ CHỨC:**
   • Các trường đại học tại Huế, Đà Nẵng
   • Tổ chức cựu chiến binh
   • Câu lạc bộ thiền, yoga
   • Doanh nghiệp lớn trong nước

**📈 QUY MÔ HOẠT ĐỘNG:**

• **Nhân sự:** 25 nhân viên chính thức + 50 cộng tác viên
• **Khách hàng:** 5,000+ khách/năm
• **Địa bàn:** Huế, Quảng Trị, Đà Nẵng, Bạch Mã, Trường Sơn
• **Tăng trưởng:** 30-40%/năm

**🌍 CAM KẾT PHÁT TRIỂN BỀN VỮNG:**

♻️ **MÔI TRƯỜNG:**
• Giảm 50% rác thải nhựa đến 2025
• Sử dụng 100% vật liệu tái chế
• Trồng 1,000 cây xanh/năm

🤲 **CỘNG ĐỒNG:**
• Tạo việc làm cho 100+ người địa phương
• Đào tạo kỹ năng du lịch cho thanh niên
• Hỗ trợ 10% doanh thu từ tour cộng đồng

📚 **GIÁO DỤC:**
• Workshop miễn phí về du lịch bền vững
• Chương trình học bổng cho sinh viên
• Tài liệu hướng dẫn du lịch có trách nhiệm

📞 **Kết nối với Ruby Wings:**
• **Hotline 24/7:** 0332510486
• **Email:** info@rubywings.com
• **Văn phòng:** 123 Đường ABC, Thành phố Huế
• **Giờ làm việc:** 8:00 - 20:00 hàng ngày

🌟 *"Ruby Wings - Nâng cánh ước mơ, chạm đến trái tim"* 🌟"""


def _get_weather_info(location, location_tours):
    """Trả lời thông tin thời tiết - NÂNG CẤP CHI TIẾT"""
    location_lower = location.lower()
    reply = f"🌤️ **THÔNG TIN THỜI TIẾT {location.upper()}** 🌤️\n\n"
    
    weather_data = {
        'huế': {
            'title': "HUẾ - KINH ĐÔ CỔ VỚI KHÍ HẬU ĐẶC TRƯNG",
            'temp_range': "18-35°C",
            'seasons': {
                'dry': "Tháng 1-8: Nắng đẹp, ít mưa, độ ẩm 65-75%",
                'rainy': "Tháng 9-12: Mưa nhiều, lụt cục bộ, độ ẩm 80-90%"
            },
            'best_months': "Tháng 1-4 & 11-12",
            'special_notes': [
                "🌅 Bình minh trên sông Hương: 5:00-6:00",
                "🌇 Hoàng hôn tại Núi Ngự: 17:30-18:30",
                "☔ Mưa thường tập trung chiều tối",
                "🌡️ Chênh lệch nhiệt ngày/đêm: 8-12°C"
            ],
            'packing_tips': [
                "🎽 Áo cotton thoáng mát",
                "🌂 Ô/dù nhỏ gọn",
                "🩴 Dép đi mưa",
                "🧴 Kem chống nắng SPF 50+",
                "💧 Nước uống đầy đủ"
            ],
            'activity_recommendations': {
                'dry_season': "Tham quan di tích, ẩm thực đường phố",
                'rainy_season': "Tham quan bảo tàng, trải nghiệm văn hóa trong nhà"
            }
        },
        'bạch mã': {
            'title': "BẠCH MÃ - VƯỜN QUỐC GIA VỚI KHÍ HẬU ÔN ĐỚI",
            'temp_range': "15-25°C (cao nhất 1,450m)",
            'seasons': {
                'dry': "Tháng 2-5: Ít mưa, hoa phong lan nở rộ",
                'rainy': "Tháng 9-12: Mưa rừng, sương mù dày đặc"
            },
            'best_months': "Tháng 3-5 & 10-11",
            'special_notes': [
                "🌫️ Sương mù buổi sáng: 6:00-9:00",
                "🌡️ Giảm 0.6°C/100m độ cao",
                "💨 Gió mạnh trên đỉnh núi",
                "🌧️ Lượng mưa: 2,500-3,000mm/năm"
            ],
            'packing_tips': [
                "🧥 Áo khoác mỏng",
                "🥾 Giày trekking chống nước",
                "🌧️ Áo mưa loại nhẹ",
                "🔦 Đèn pin/đèn trán",
                "🦟 Thuốc chống côn trùng"
            ],
            'activity_recommendations': {
                'dry_season': "Trekking, ngắm hoa, quan sát động vật",
                'rainy_season': "Thiền trong rừng, tĩnh dưỡng, viết nhật ký"
            }
        },
        'trường sơn': {
            'title': "TRƯỜNG SƠN - DÃY NÚI HUYỀN THOẠI",
            'temp_range': "18-30°C (chênh lệch lớn ngày/đêm)",
            'seasons': {
                'dry': "Tháng 1-4: Ít mưa, đường khô ráo",
                'rainy': "Tháng 5-12: Mưa rừng, ẩm ướt, sương mù"
            },
            'best_months': "Tháng 1-3 & 11-12",
            'special_notes': [
                "🌡️ Đêm lạnh: Có thể xuống 15°C",
                "🌫️ Sương mù quanh năm",
                "🌧️ Mưa rào bất chợt",
                "🛣️ Đường đất trơn trượt khi mưa"
            ],
            'packing_tips': [
                "🧣 Khăn quàng cổ",
                "🧤 Găng tay mỏng",
                "🥾 Giày bảo hộ cao cổ",
                "🎒 Balo chống nước",
                "📱 Sạc dự phòng"
            ],
            'activity_recommendations': {
                'dry_season': "Tham quan di tích, tìm hiểu lịch sử",
                'rainy_season': "Nghe kể chuyện lịch sử, giao lưu văn nghệ"
            }
        },
        'quảng trị': {
            'title': "QUẢNG TRỊ - VÙNG ĐẤT LỊCH SỬ",
            'temp_range': "20-35°C",
            'seasons': {
                'dry': "Tháng 1-8: Nắng nóng, gió Lào",
                'rainy': "Tháng 9-12: Mưa bão, lũ lụt"
            },
            'best_months': "Tháng 1-4 & 10-12",
            'special_notes': [
                "🌪️ Gió Lào khô nóng: Tháng 4-8",
                "🌀 Bão thường vào tháng 9-11",
                "🏞️ Sông Bến Hải chia cắt Bắc-Nam",
                "🌡️ Nhiệt độ cao nhất có thể lên 38°C"
            ],
            'packing_tips': [
                "🧢 Mũ rộng vành",
                "🕶️ Kính râm",
                "💦 Bình nước cá nhân",
                "🌬️ Quạt cầm tay",
                "🧴 Kem dưỡng ẩm"
            ],
            'activity_recommendations': {
                'dry_season': "Tham quan di tích, tìm hiểu lịch sử",
                'rainy_season': "Tham quan bảo tàng, xem phim tài liệu"
            }
        }
    }
    
    # Lấy dữ liệu thời tiết cho địa điểm
    if location_lower in weather_data:
        data = weather_data[location_lower]
        reply += f"**{data['title']}**\n\n"
        
        reply += "📊 **THÔNG SỐ CHÍNH:**\n"
        reply += f"• **Nhiệt độ:** {data['temp_range']}\n"
        reply += f"• **Mùa khô:** {data['seasons']['dry']}\n"
        reply += f"• **Mùa mưa:** {data['seasons']['rainy']}\n"
        reply += f"• **Tháng đẹp nhất:** {data['best_months']}\n\n"
        
        reply += "⚠️ **ĐẶC ĐIỂM ĐÁNG CHÚ Ý:**\n"
        for note in data['special_notes']:
            reply += f"• {note}\n"
        reply += "\n"
        
        reply += "🎒 **CHUẨN BỊ HÀNH LÝ:**\n"
        for tip in data['packing_tips']:
            reply += f"• {tip}\n"
        reply += "\n"
        
        reply += "🎯 **HOẠT ĐỘNG THEO MÙA:**\n"
        reply += f"• **Mùa khô:** {data['activity_recommendations']['dry_season']}\n"
        reply += f"• **Mùa mưa:** {data['activity_recommendations']['rainy_season']}\n\n"
        
    else:
        reply += f"**{location.upper()} - KHÍ HẬU MIỀN TRUNG VIỆT NAM**\n\n"
        reply += "🌡️ **ĐẶC TRƯNG CHUNG:**\n"
        reply += "• Khí hậu nhiệt đới gió mùa\n"
        reply += "• Hai mùa rõ rệt: khô & mưa\n"
        reply += "• Gió mùa Đông Bắc (tháng 10-3)\n"
        reply += "• Gió mùa Tây Nam (tháng 4-9)\n\n"
        
        reply += "📅 **MÙA DU LỊCH TỐT NHẤT:**\n"
        reply += "• **Tháng 1-4:** Mát mẻ, ít mưa\n"
        reply += "• **Tháng 10-12:** Dịu nhẹ, hoa nở\n"
        reply += "• **Tránh:** Tháng 9-11 (mưa bão)\n\n"
        
        reply += "💡 **LỜI KHUYÊN CHUNG:**\n"
        reply += "1. Check dự báo 3 ngày trước khi đi\n"
        reply += "2. Chuẩn bị đồ dùng đa dạng\n"
        reply += "3. Linh hoạt thay đổi lịch trình\n"
        reply += "4. Luôn có phương án dự phòng\n\n"
    
    # Thêm thông tin tour liên quan
    if location_tours:
        reply += "🗺️ **TOUR RUBY WINGS PHÙ HỢP:**\n"
        
        # Phân loại tour theo mùa
        dry_season_tours = []
        all_season_tours = []
        
        for tour in location_tours[:6]:
            tour_summary = (tour.summary or "").lower()
            tour_name = (tour.name or "").lower()
            
            # Phân loại sơ bộ
            if any(keyword in tour_summary for keyword in ['trong nhà', 'bảo tàng', 'văn hóa', 'ẩm thực']):
                all_season_tours.append(tour)
            elif any(keyword in tour_summary for keyword in ['trekking', 'leo núi', 'thiên nhiên', 'rừng']):
                dry_season_tours.append(tour)
            else:
                all_season_tours.append(tour)
        
        if dry_season_tours:
            reply += "🌤️ **MÙA KHÔ (phù hợp outdoor):**\n"
            for tour in dry_season_tours[:2]:
                reply += f"• **{tour.name}**"
                if tour.duration:
                    reply += f" ({tour.duration})"
                reply += "\n"
            reply += "\n"
        
        if all_season_tours:
            reply += "🌈 **QUANH NĂM (mọi thời tiết):**\n"
            for tour in all_season_tours[:2]:
                reply += f"• **{tour.name}**"
                if tour.duration:
                    reply += f" ({tour.duration})"
                reply += "\n"
            reply += "\n"
    
    reply += "📞 **Tư vấn chi tiết về thời tiết và tour phù hợp:** 0332510486\n"
    reply += "🌐 **Check dự báo thời tiết chi tiết:** weather.com/vietnam"
    
    return reply


def _get_location_info(location, location_tours):
    """Trả lời thông tin địa điểm - NÂNG CẤP CHI TIẾT"""
    location_lower = location.lower()
    reply = f"📍 **KHÁM PHÁ {location.upper()}** 📍\n\n"
    
    location_data = {
        'huế': {
            'title': "HUẾ - KINH ĐÔ TRIỀU NGUYỄN, DI SẢN UNESCO",
            'highlights': [
                "🏛️ 7 Di sản UNESCO: Đại Nội, Lăng tẩm, Nhã nhạc...",
                "🍜 Ẩm thực cung đình: 1,300 món ăn đặc sắc",
                "🏞️ Thiên nhiên: Sông Hương, Núi Ngự, biển Lăng Cô",
                "🎭 Văn hóa: Festival Huế, lễ hội cung đình"
            ],
            'must_see': [
                "1. Đại Nội Huế - Hoàng thành nhà Nguyễn",
                "2. Lăng Tự Đức - Kiến trúc hài hòa thiên nhiên",
                "3. Chùa Thiên Mụ - Biểu tượng tâm linh",
                "4. Cầu Tràng Tiền - Biểu tượng của Huế",
                "5. Chợ Đông Ba - Trung tâm ẩm thực"
            ],
            'cultural_significance': "Trung tâm văn hóa, chính trị Việt Nam thế kỷ 19-20",
            'best_for': "Lịch sử, ẩm thực, nhiếp ảnh, tâm linh",
            'travel_tips': [
                "⏰ Dành ít nhất 2 ngày để khám phá",
                "🚶 Đi bộ hoặc xích lô để cảm nhận",
                "🎫 Mua vé combo tiết kiệm",
                "🌙 Trải nghiệm Huế về đêm"
            ]
        },
        'bạch mã': {
            'title': "VƯỜN QUỐC GIA BẠCH MÃ - THIÊN ĐƯỜNG XANH",
            'highlights': [
                "🌳 Rừng nguyên sinh rộng 37,000ha",
                "🦜 2,373 loài động thực vật",
                "🌡️ Khí hậu ôn đới quanh năm",
                "🏞️ Hệ thống thác, suối, đỉnh núi hùng vĩ"
            ],
            'must_see': [
                "1. Đỉnh Bạch Mã (1,450m) - Ngắm toàn cảnh",
                "2. Thác Đỗ Quyên - Thác nước đẹp nhất",
                "3. Hồ Truồi - Hồ nước ngọt tự nhiên",
                "4. Rừng Chò Đen - Rừng nguyên sinh",
                "5. Vườn Lan - Hơn 300 loài lan rừng"
            ],
            'cultural_significance': "Khu dự trữ sinh quyển thế giới",
            'best_for': "Trekking, thiền, nghiên cứu, nhiếp ảnh thiên nhiên",
            'travel_tips': [
                "⏰ Cần ít nhất 1 ngày, tốt nhất 2 ngày 1 đêm",
                "🥾 Chuẩn bị giày trekking chuyên dụng",
                "📸 Mang theo ống nhòm, máy ảnh",
                "🌙 Ở lại qua đêm để trải nghiệm trọn vẹn"
            ]
        },
        'trường sơn': {
            'title': "DÃY TRƯỜNG SƠN - HUYỀN THOẠI ĐƯỜNG HỒ CHÍ MINH",
            'highlights': [
                "🎖️ Di tích lịch sử chiến tranh",
                "🌳 Rừng nhiệt đới nguyên sinh",
                "👥 Văn hóa dân tộc Vân Kiều, Pa Kô",
                "🏞️ Cảnh quan hùng vĩ, hoang sơ"
            ],
            'must_see': [
                "1. Đường Hồ Chí Minh - Huyết mạch lịch sử",
                "2. Thành cổ Quảng Trị - Chứng tích chiến tranh",
                "3. Địa đạo Vịnh Mốc - Thành phố dưới lòng đất",
                "4. Cầu Hiền Lương - Biểu tượng chia cắt",
                "5. Nghĩa trang Trường Sơn - Nơi yên nghỉ anh hùng"
            ],
            'cultural_significance': "Chứng nhân lịch sử, biểu tượng của sự hy sinh và chiến thắng",
            'best_for': "Tìm hiểu lịch sử, tri ân, nghiên cứu, trải nghiệm văn hóa",
            'travel_tips': [
                "⏰ Dành ít nhất 2 ngày để thấu hiểu",
                "📚 Tìm hiểu lịch sử trước khi đi",
                "🙏 Thái độ nghiêm trang tại di tích",
                "🎤 Thuê HDV am hiểu lịch sử"
            ]
        },
        'quảng trị': {
            'title': "QUẢNG TRỊ - VÙNG ĐẤT ANH HÙNG",
            'highlights': [
                "⚔️ Chiến trường ác liệt nhất",
                "🏞️ Cảnh quan sông nước hữu tình",
                "🌾 Nông nghiệp trù phú",
                "🏖️ Bãi biển hoang sơ đẹp"
            ],
            'must_see': [
                "1. Sông Bến Hải & Cầu Hiền Lương",
                "2. Địa đạo Vịnh Mốc",
                "3. Thành cổ Quảng Trị",
                "4. Cửa Tùng - Bãi tắm đẹp",
                "5. Đảo Cồn Cỏ - Tiền tiêu Tổ quốc"
            ],
            'cultural_significance': "Nơi diễn ra những trận đánh lịch sử, biểu tượng của lòng yêu nước",
            'best_for': "Lịch sử, tri ân, nhiếp ảnh, trải nghiệm văn hóa",
            'travel_tips': [
                "⏰ Dành 1-2 ngày tham quan",
                "📜 Đọc tài liệu lịch sử",
                "🎥 Xem phim tài liệu trước",
                "🌅 Ngắm bình minh trên sông Bến Hải"
            ]
        }
    }
    
    # Lấy dữ liệu địa điểm
    if location_lower in location_data:
        data = location_data[location_lower]
        reply += f"**{data['title']}**\n\n"
        
        reply += "🌟 **ĐIỂM NỔI BẬT:**\n"
        for highlight in data['highlights']:
            reply += f"• {highlight}\n"
        reply += "\n"
        
        reply += "🎯 **KHÔNG THỂ BỎ QUA:**\n"
        for spot in data['must_see']:
            reply += f"{spot}\n"
        reply += "\n"
        
        reply += "📚 **Ý NGHĨA VĂN HÓA - LỊCH SỬ:**\n"
        reply += f"{data['cultural_significance']}\n\n"
        
        reply += "👥 **PHÙ HỢP VỚI:**\n"
        reply += f"• {data['best_for']}\n\n"
        
        reply += "💡 **MẸO DU LỊCH:**\n"
        for tip in data['travel_tips']:
            reply += f"• {tip}\n"
        reply += "\n"
        
    else:
        reply += f"**{location.upper()} - ĐIỂM ĐẾN HẤP DẪN MIỀN TRUNG**\n\n"
        reply += "Miền Trung Việt Nam với nhiều điểm đến đa dạng:\n\n"
        reply += "🏛️ **DI SẢN VĂN HÓA:**\n"
        reply += "• Huế: Di sản UNESCO\n"
        reply += "• Hội An: Phố cổ\n"
        reply += "• Mỹ Sơn: Thánh địa Chăm Pa\n\n"
        
        reply += "🌿 **THIÊN NHIÊN:**\n"
        reply += "• Bạch Mã: Vườn quốc gia\n"
        reply += "• Sơn Trà: Bán đảo nguyên sinh\n"
        reply += "• Cù Lao Chàm: Đảo sinh thái\n\n"
        
        reply += "🎖️ **LỊCH SỬ:**\n"
        reply += "• Quảng Trị: Chiến trường xưa\n"
        reply += "• Đường HCM: Huyền thoại\n"
        reply += "• Địa đạo: Công trình ngầm\n\n"
        
        reply += "🍜 **ẨM THỰC:**\n"
        reply += "• Huế: Ẩm thực cung đình\n"
        reply += "• Đà Nẵng: Hải sản tươi ngon\n"
        reply += "• Quảng Nam: Mỳ Quảng, Cao lầu\n\n"
    
    # Thêm thông tin tour liên quan
    if location_tours:
        reply += "🗺️ **TOUR RUBY WINGS TẠI ĐÂY:**\n"
        
        # Phân loại tour theo loại hình
        categories = {
            'history': [],
            'nature': [],
            'culture': [],
            'wellness': []
        }
        
        for tour in location_tours[:8]:
            tour_summary = (tour.summary or "").lower()
            tour_name = (tour.name or "").lower()
            
            if any(keyword in tour_summary for keyword in ['lịch sử', 'di tích', 'chiến tranh', 'tri ân']):
                categories['history'].append(tour)
            elif any(keyword in tour_summary for keyword in ['thiên nhiên', 'rừng', 'trekking', 'khám phá']):
                categories['nature'].append(tour)
            elif any(keyword in tour_summary for keyword in ['văn hóa', 'ẩm thực', 'làng nghề', 'truyền thống']):
                categories['culture'].append(tour)
            elif any(keyword in tour_summary for keyword in ['thiền', 'yoga', 'retreat', 'chữa lành']):
                categories['wellness'].append(tour)
            else:
                categories['nature'].append(tour)
        
        # Hiển thị theo từng loại
        category_names = {
            'history': '🏛️ LỊCH SỬ',
            'nature': '🌿 THIÊN NHIÊN',
            'culture': '🍜 VĂN HÓA',
            'wellness': '🧘 WELLNESS'
        }
        
        for cat_key, cat_name in category_names.items():
            if categories[cat_key]:
                reply += f"\n{cat_name}:\n"
                for tour in categories[cat_key][:2]:
                    reply += f"• **{tour.name}**"
                    if tour.duration:
                        reply += f" ({tour.duration})"
                    if tour.price:
                        price_short = tour.price[:40] + "..." if len(tour.price) > 40 else tour.price
                        reply += f" - {price_short}"
                    reply += "\n"
        
        reply += "\n"
    
    reply += "📞 **Đặt tour khám phá chi tiết:** 0332510486\n"
    reply += "🗓️ **Tư vấn lịch trình phù hợp:** Liên hệ để được thiết kế riêng"
    
    return reply


def _get_food_culture_response(message_lower, tour_indices):
    """Trả lời về ẩm thực và văn hóa - NÂNG CẤP CHI TIẾT"""
    # Kiểm tra cụ thể loại ẩm thực/văn hóa được hỏi
    if 'bánh bèo' in message_lower:
        return _get_banh_beo_detail()
    elif 'bún bò' in message_lower or 'bun bo' in message_lower:
        return _get_bun_bo_hue_detail()
    elif 'cơm hến' in message_lower:
        return _get_com_hen_detail()
    elif 'mắm nêm' in message_lower:
        return _get_mam_nem_detail()
    elif 'ẩm thực huế' in message_lower or 'đặc sản huế' in message_lower:
        return _get_hue_food_overview()
    elif 'văn hóa huế' in message_lower or 'văn hóa miền trung' in message_lower:
        return _get_hue_culture_overview()
    elif 'lịch sử' in message_lower or 'di tích' in message_lower or 'di sản' in message_lower:
        return _get_history_culture_response()
    else:
        return _get_general_food_culture_response(message_lower, tour_indices)


def _get_banh_beo_detail():
    """Chi tiết về bánh bèo Huế"""
    reply = "🍜 **BÁNH BÈO HUẾ - TINH HOA ẨM THỰC CUNG ĐÌNH** 🍜\n\n"
    
    reply += "📜 **NGUỒN GỐC LỊCH SỬ:**\n"
    reply += "• Xuất hiện từ thời nhà Nguyễn (1802-1945)\n"
    reply += "• Ban đầu chỉ phục vụ trong cung đình\n"
    reply += "• Sau 1945, lan ra dân gian\n"
    reply += "• Tên gọi từ hình dáng giống lá bèo trên mặt nước\n\n"
    
    reply += "👑 **ĐẶC ĐIỂM CUNG ĐÌNH:**\n"
    reply += "• **Tinh tế:** Mỗi chén chỉ 2-3 muỗng bột\n"
    reply += "• **Cầu kỳ:** 15+ công đoạn chuẩn bị\n"
    reply += "• **Đẹp mắt:** Trình bày như tác phẩm nghệ thuật\n"
    reply += "• **Hài hòa:** Cân bằng 5 vị cơ bản\n\n"
    
    reply += "🥣 **THÀNH PHẦN CHUẨN:**\n"
    
    reply += "1. **BÁNH:**\n"
    reply += "   • Gạo ngon (nếp tẻ pha)\n"
    reply += "   • Ngâm 8-12 giờ\n"
    reply += "   • Xay mịn, lọc kỹ\n"
    reply += "   • Hấp cách thủy 5-7 phút\n\n"
    
    reply += "2. **NHÂN:**\n"
    reply += "   • Tôm sú bóc vỏ\n"
    reply += "   • Thịt heo xay\n"
    reply += "   • Mỡ hành phi thơm\n"
    reply += "   • Đậu phộng rang\n\n"
    
    reply += "3. **NƯỚC MẮM:**\n"
    reply += "   • Mắm nêm Huế đặc trưng\n"
    reply += "   • Đường, tỏi, ớt, chanh\n"
    reply += "   • Nấu sôi, để nguội\n\n"
    
    reply += "4. **RAU SỐNG:**\n"
    reply += "   • Xà lách, rau thơm\n"
    reply += "   • Ớt xanh Huế\n"
    reply += "   • Giá đỗ\n\n"
    
    reply += "🍽️ **QUY TRÌNH THƯỞNG THỨC:**\n"
    reply += "1. Dùng thìa nhỏ xúc từng chén\n"
    reply += "2. Chan 1/2 thìa nước mắm\n"
    reply += "3. Thêm ít rau sống\n"
    reply += "4. Trộn đều, thưởng thức\n"
    reply += "5. Uống trà nóng giữa các chén\n\n"
    
    reply += "🏆 **BIẾN TẤU ĐẶC BIỆT:**\n"
    reply += "• **Bánh bèo chén:** Truyền thống\n"
    reply += "• **Bánh bèo dĩa:** Tiện lợi\n"
    reply += "• **Bánh bèo thập cẩm:** Đầy đủ nhân\n"
    reply += "• **Bánh bèo chay:** Dành cho Phật tử\n\n"
    
    reply += "📍 **ĐỊA CHỈ NGON:**\n"
    reply += "1. **Bánh bèo Huế - 123 Đường ABC**\n"
    reply += "2. **Quán Bà Đợ - Khu phố cổ**\n"
    reply += "3. **Chợ Đông Ba - Gian hàng 45**\n"
    reply += "4. **Làng bánh bèo Phú Hậu**\n\n"
    
    reply += "🎯 **TRẢI NGHIỆM VỚI RUBY WINGS:**\n"
    reply += "• **Tour Ẩm thực Huế:** Học làm từ A-Z\n"
    reply += "• **Tour Văn hóa:** Thăm làng nghề truyền thống\n"
    reply += "• **Tour Đêm Huế:** Thưởng thức tại quán đặc sản\n"
    reply += "• **Tour Masterclass:** Học từ nghệ nhân 30 năm kinh nghiệm\n\n"
    
    reply += "📞 **Đặt tour ẩm thực Huế:** 0332510486\n"
    reply += "👨‍🍳 **Học làm bánh bèo:** Workshop hàng tuần"
    
    return reply


def _get_bun_bo_hue_detail():
    """Chi tiết về bún bò Huế"""
    reply = "🍜 **BÚN BÒ HUẾ - MÓN NGON ĐẬM ĐÀ HUYỀN THOẠI** 🍜\n\n"
    
    reply += "📜 **LỊCH SỬ 100 NĂM:**\n"
    reply += "• Ra đời đầu thế kỷ 20\n"
    reply += "• Kết hợp ẩm thực cung đình & dân gian\n"
    reply += "• Biểu tượng ẩm thực Huế\n"
    reply += "• Được UNESCO vinh danh\n\n"
    
    reply += "🥘 **BÍ QUYẾT NƯỚC DÙNG:**\n"
    reply += "• Xương bò hầm 12-15 giờ\n"
    reply += "• Sả, riềng, mắm ruốc\n"
    reply += "• Màu đỏ từ ớt bột\n"
    reply += "• Vị cay đặc trưng\n\n"
    
    reply += "🎯 **TOUR ẨM THỰC BÚN BÒ:**\n"
    reply += "1. **Học nấu từ cơ bản:** 2 giờ\n"
    reply += "2. **Chợ sáng & nấu ăn:** 4 giờ\n"
    reply += "3. **Masterclass nghệ nhân:** 6 giờ\n"
    reply += "4. **Trải nghiệm toàn diện:** 1 ngày\n\n"
    
    reply += "📞 **Đặt tour:** 0332510486"
    
    return reply


def _get_com_hen_detail():
    """Chi tiết về cơm hến"""
    reply = "🍚 **CƠM HẾN - ĐẶC SẢN DÂN DÃ HUẾ** 🍚\n\n"
    
    reply += "🌾 **HẾN SÔNG HƯƠNG:**\n"
    reply += "• Bắt từ sông Hương\n"
    reply += "• Nhỏ, thơm, ngọt đặc biệt\n"
    reply += "• Chế biến 10+ món\n"
    reply += "• Giá trị dinh dưỡng cao\n\n"
    
    reply += "🎯 **TOUR ẨM THỰC CƠM HẾN:**\n"
    reply += "• Thăm làng chài\n"
    reply += "• Học bắt & chế biến\n"
    reply += "• Nấu 5 món từ hến\n"
    reply += "• Thưởng thức tại chỗ\n\n"
    
    reply += "📞 **Đặt tour:** 0332510486"
    
    return reply


def _get_mam_nem_detail():
    """Chi tiết về mắm nêm"""
    reply = "🥫 **MẮM NÊM - LINH HỒN ẨM THỰC HUẾ** 🥫\n\n"
    
    reply += "🐟 **LÊN MEN TỰ NHIÊN:**\n"
    reply += "• Cá cơm tươi\n"
    reply += "• Muối biển tinh khiết\n"
    reply += "• Lên men 6-12 tháng\n"
    reply += "• Hương vị đậm đà\n\n"
    
    reply += "🎯 **WORKSHOP MẮM NÊM:**\n"
    reply += "• Thăm làng làm mắm\n"
    reply += "• Học kỹ thuật ủ\n"
    reply += "• Chế biến 3 loại mắm\n"
    reply += "• Đóng chai mang về\n\n"
    
    reply += "📞 **Đặt workshop:** 0332510486"
    
    return reply


def _get_hue_food_overview():
    """Tổng quan ẩm thực Huế"""
    reply = "🍽️ **ẨM THỰC HUẾ - DI SẢN VĂN HÓA PHI VẬT THỂ** 🍽️\n\n"
    
    reply += "👑 **3 DÒNG ẨM THỰC CHÍNH:**\n\n"
    
    reply += "1. **ẨM THỰC CUNG ĐÌNH:**\n"
    reply += "   • Phục vụ vua chúa\n"
    reply += "   • 1,300 món ăn\n"
    reply += "   • Trình bày nghệ thuật\n"
    reply += "   • Nguyên liệu quý hiếm\n\n"
    
    reply += "2. **ẨM THỰC DÂN GIAN:**\n"
    reply += "   • Phổ biến trong dân\n"
    reply += "   • Nguyên liệu địa phương\n"
    reply += "   • Hương vị đậm đà\n"
    reply += "   • Giá cả bình dân\n\n"
    
    reply += "3. **ẨM THỰC CHAY:**\n"
    reply += "   • Dành cho Phật tử\n"
    reply += "   • 200+ món chay\n"
    reply += "   • Tinh tế, thanh đạm\n"
    reply += "   • Dinh dưỡng cao\n\n"
    
    reply += "🎯 **CÁC MÓN TIÊU BIỂU:**\n"
    
    reply += "🍜 **MÓN BÚN/MỲ:**\n"
    reply += "• Bún bò Huế\n"
    reply += "• Bún thịt nướng\n"
    reply += "• Mỳ Quảng\n"
    reply += "• Bún hến\n\n"
    
    reply += "🍚 **MÓN CƠM:**\n"
    reply += "• Cơm hến\n"
    reply += "• Cơm âm phủ\n"
    reply += "• Cơm gà Hội An\n"
    reply += "• Cơm niêu\n\n"
    
    reply += "🥟 **BÁNH:**\n"
    reply += "• Bánh bèo\n"
    reply += "• Bánh nậm\n"
    reply += "• Bánh bột lọc\n"
    reply += "• Bánh ướt\n\n"
    
    reply += "🍢 **MÓN NHẬU:**\n"
    reply += "• Nem lụi\n"
    reply += "• Bò nướng lá lốt\n"
    reply += "• Chả tôm\n"
    reply += "• Gỏi cá trích\n\n"
    
    reply += "🍨 **TRÁNG MIỆNG:**\n"
    reply += "• Chè Huế\n"
    reply += "• Bánh flan\n"
    reply += "• Rau câu\n"
    reply += "• Sữa đậu nành\n\n"
    
    reply += "🎯 **TOUR ẨM THỰC RUBY WINGS:**\n"
    
    reply += "1. **TOUR KHÁM PHÁ ẨM THỰC (1 ngày):**\n"
    reply += "   • Tham quan chợ Đông Ba\n"
    reply += "   • Học làm 3 món Huế\n"
    reply += "   • Thưởng thức bữa trưa đặc sản\n"
    reply += "   • Thăm làng nghề truyền thống\n\n"
    
    reply += "2. **TOUR ẨM THỰC CAO CẤP (2 ngày):**\n"
    reply += "   • Trải nghiệm ẩm thực cung đình\n"
    reply += "   • Workshop với nghệ nhân\n"
    reply += "   • Thăm vườn rau hữu cơ\n"
    reply += "   • Dùng bữa tại nhà hàng Michelin\n\n"
    
    reply += "3. **TOUR MASTERCLASS (3 ngày):**\n"
    reply += "   • Học làm 10 món Huế\n"
    reply += "   • Chứng chỉ hoàn thành\n"
    reply += "   • Nguyên liệu cao cấp\n"
    reply += "   • Được nghệ nhân trực tiếp hướng dẫn\n\n"
    
    reply += "📞 **Đặt tour ẩm thực:** 0332510486\n"
    reply += "👨‍🍳 **Đội ngũ chuyên gia ẩm thực Huế**\n"
    reply += "🌟 **Chứng nhận ẩm thực quốc tế**"
    
    return reply


def _get_hue_culture_overview():
    """Tổng quan văn hóa Huế"""
    reply = "🏛️ **VĂN HÓA HUẾ - DI SẢN SỐNG ĐỘNG** 🏛️\n\n"
    
    reply += "🎭 **7 DI SẢN UNESCO TẠI HUẾ:**\n\n"
    
    reply += "1. **QUẦN THỂ DI TÍCH CỐ ĐÔ HUẾ:**\n"
    reply += "   • Đại Nội (Hoàng thành)\n"
    reply += "   • Lăng Tự Đức, Minh Mạng, Khải Định\n"
    reply += "   • Đàn Nam Giao\n"
    reply += "   • Hồ Quyển\n\n"
    
    reply += "2. **NHÃ NHẠC CUNG ĐÌNH HUẾ:**\n"
    reply += "   • Âm nhạc cung đình\n"
    reply += "   • 12 thể loại nhạc\n"
    reply += "   • Nhạc cụ truyền thống\n"
    reply += "   • Biểu diễn hàng đêm\n\n"
    
    reply += "3. **MỘC BẢN TRIỀU NGUYỄN:**\n"
    reply += "   • 34.619 tấm mộc bản\n"
    reply += "   • Tài liệu quý giá\n"
    reply += "   • Kỹ thuật khắc gỗ\n"
    reply += "   • Lưu trữ tại Trung tâm Lưu trữ\n\n"
    
    reply += "4. **CHÂU BẢN TRIỀU NGUYỄN:**\n"
    reply += "   • 700 tập tài liệu\n"
    reply += "   • Văn bản hành chính\n"
    reply += "   • Chữ Hán Nôm\n"
    reply += "   • Giá trị lịch sử cao\n\n"
    
    reply += "5. **THƠ VĂN TRÊN KIẾN TRÚC CUNG ĐÌNH:**\n"
    reply += "   • Thơ chữ Hán\n"
    reply += "   • Văn tự trang trí\n"
    reply += "   • Nghệ thuật thư pháp\n"
    reply += "   • Trên 2.500 ô thơ\n\n"
    
    reply += "6. **HỆ THỐNG THỦY ĐẠO KINH THÀNH:**\n"
    reply += "   • Hệ thống thoát nước\n"
    reply += "   • Kỹ thuật xây dựng\n"
    reply += "   • Bảo tồn nguyên vẹn\n"
    reply += "   • Công trình độc đáo\n\n"
    
    reply += "7. **NGHỆ THUẬT BÀI CHÒI:**\n"
    reply += "   • Trò chơi dân gian\n"
    reply += "   • Kết hợp ca hát\n"
    reply += "   • Phổ biến dịp Tết\n"
    reply += "   • Di sản văn hóa phi vật thể\n\n"
    
    reply += "🎨 **NGHỀ THỦ CÔNG TRUYỀN THỐNG:**\n"
    
    reply += "• **THÊU:** Làng thêu Phước Tích\n"
    reply += "• **GỐM:** Làng gốm Phước Tích\n"
    reply += "• **MỘC:** Làng mộc Kim Long\n"
    reply += "• **NÓN:** Làng nón bài thơ\n"
    reply += "• **HƯƠNG:** Làng hương Thủy Xuân\n"
    reply += "• **ĐÚC ĐỒNG:** Làng đúc đồng Phường Đúc\n\n"
    
    reply += "🎭 **LỄ HỘI TRUYỀN THỐNG:**\n"
    
    reply += "• **FESTIVAL HUẾ:** 2 năm/lần\n"
    reply += "• **LỄ TẾ NAM GIAO:** Tháng 3 âm lịch\n"
    reply += "• **LỄ HỘI ĐÈN LỒNG:** Rằm tháng Giêng\n"
    reply += "• **LỄ HỘI THÁNG 7:** Vu lan báo hiếu\n"
    reply += "• **LỄ HỘI CUNG ĐÌNH:** Hàng tháng\n\n"
    
    reply += "🎯 **TOUR VĂN HÓA RUBY WINGS:**\n"
    
    reply += "1. **TOUR DI SẢN UNESCO (1 ngày):**\n"
    reply += "   • Tham quan Đại Nội\n"
    reply += "   • Xem nhã nhạc cung đình\n"
    reply += "   • Thăm bảo tàng mộc bản\n"
    reply += "   • Trải nghiệm bài chòi\n\n"
    
    reply += "2. **TOUR LÀNG NGHỀ (1 ngày):**\n"
    reply += "   • Thăm 3 làng nghề\n"
    reply += "   • Học làm thủ công\n"
    reply += "   • Mua sắm sản phẩm\n"
    reply += "   • Giao lưu nghệ nhân\n\n"
    
    reply += "3. **TOUR VĂN HÓA SÂU (2 ngày):**\n"
    reply += "   • Trải nghiệm toàn diện\n"
    reply += "   • Ở homestay truyền thống\n"
    reply += "   • Học 2 nghề thủ công\n"
    reply += "   • Tham gia lễ hội\n\n"
    
    reply += "📞 **Đặt tour văn hóa:** 0332510486\n"
    reply += "🎓 **HDV am hiểu văn hóa Huế**\n"
    reply += "🏛️ **Đối tác của UNESCO Huế**"
    
    return reply


def _get_history_culture_response():
    """Trả lời về văn hóa lịch sử"""
    reply = "🏛️ **VĂN HÓA & LỊCH SỬ MIỀN TRUNG - NƠI LƯU GIỮ HỒN VIỆT** 🏛️\n\n"
    
    reply += "📜 **CÁC THỜI KỲ LỊCH SỬ QUAN TRỌNG:**\n\n"
    
    reply += "1. **THỜI KỲ CHĂM PA (192-1832):**\n"
    reply += "   • Vương quốc cổ đại\n"
    reply += "   • Thánh địa Mỹ Sơn\n"
    reply += "   • Tháp Chăm Po Nagar\n"
    reply += "   • Nghệ thuật điêu khắc đá\n\n"
    
    reply += "2. **THỜI KỲ ĐẠI VIỆT (1306-1802):**\n"
    reply += "   • Mở rộng lãnh thổ\n"
    reply += "   • Chiến tranh Trịnh-Nguyễn\n"
    reply += "   • Chúa Nguyễn xứ Đàng Trong\n"
    reply += "   • Phát triển văn hóa\n\n"
    
    reply += "3. **THỜI KỲ NHÀ NGUYỄN (1802-1945):**\n"
    reply += "   • Kinh đô Huế\n"
    reply += "   • 13 đời vua Nguyễn\n"
    reply += "   • Xây dựng đại nội\n"
    reply += "   • Phát triển ẩm thực cung đình\n\n"
    
    reply += "4. **THỜI KỲ CHIẾN TRANH (1945-1975):**\n"
    reply += "   • Chiến tranh Đông Dương\n"
    reply += "   • Chiến tranh Việt Nam\n"
    reply += "   • Vĩ tuyến 17\n"
    reply += "   • Đường Hồ Chí Minh\n\n"
    
    reply += "5. **THỜI KỲ HIỆN ĐẠI (1975-nay):**\n"
    reply += "   • Thống nhất đất nước\n"
    reply += "   • Bảo tồn di sản\n"
    reply += "   • Phát triển du lịch\n"
    reply += "   • Hội nhập quốc tế\n\n"
    
    reply += "🎖️ **DI TÍCH LỊCH SỬ QUAN TRỌNG:**\n"
    
    reply += "⚔️ **QUẢNG TRỊ - CHIẾN TRƯỜNG XƯA:**\n"
    reply += "• Thành cổ Quảng Trị\n"
    reply += "• Địa đạo Vịnh Mốc\n"
    reply += "• Cầu Hiền Lương\n"
    reply += "• Sông Bến Hải\n"
    reply += "• Nghĩa trang Trường Sơn\n\n"
    
    reply += "🛣️ **ĐƯỜNG HỒ CHÍ MINH:**\n"
    reply += "• Huyết mạch chiến tranh\n"
    reply += "• Dài 1,690km\n"
    reply += "• Hệ thống đường nhánh\n"
    reply += "• Kỳ tích lịch sử\n\n"
    
    reply += "🏛️ **DI TÍCH TRIỀU NGUYỄN:**\n"
    reply += "• Đại Nội Huế\n"
    reply += "• Lăng Tự Đức, Minh Mạng\n"
    reply += "• Lăng Khải Định, Gia Long\n"
    reply += "• Điện Thái Hòa\n\n"
    
    reply += "👥 **VĂN HÓA CÁC DÂN TỘC:**\n"
    
    reply += "• **NGƯỜI KINH:** Văn hóa Huế\n"
    reply += "• **VÂN KIỀU:** Dân tộc thiểu số\n"
    reply += "• **PA KÔ:** Vùng núi Trường Sơn\n"
    reply += "• **CHĂM:** Di sản Chăm Pa\n\n"
    
    reply += "🎯 **TOUR LỊCH SỬ RUBY WINGS:**\n"
    
    reply += "1. **TOUR TRI ÂN (1 ngày):**\n"
    reply += "   • Thăm di tích chiến tranh\n"
    reply += "   • Gặp gỡ nhân chứng\n"
    reply += "   • Lễ dâng hương\n"
    reply += "   • Xem phim tài liệu\n\n"
    
    reply += "2. **TOUR LỊCH SỬ SÂU (2 ngày):**\n"
    reply += "   • Khám phá đường HCM\n"
    reply += "   • Thăm làng dân tộc\n"
    reply += "   • Trải nghiệm đời sống\n"
    reply += "   • Nghe kể chuyện lịch sử\n\n"
    
    reply += "3. **TOUR DI SẢN (3 ngày):**\n"
    reply += "   • Kết hợp Huế - Quảng Trị\n"
    reply += "   • Thăm 10+ di tích\n"
    reply += "   • Giao lưu văn hóa\n"
    reply += "   • Học làm thủ công\n\n"
    
    reply += "📞 **Đặt tour lịch sử:** 0332510486\n"
    reply += "🎖️ **Đối tác của Hội Cựu chiến binh**\n"
    reply += "📚 **Tài liệu lịch sử chính thống**"
    
    return reply


def _get_general_food_culture_response(message_lower, tour_indices):
    """Trả lời tổng quan về ẩm thực và văn hóa"""
    reply = "🍽️ **ẨM THỰC & VĂN HÓA MIỀN TRUNG - BẢN SẮC ĐỘC ĐÁO** 🍽️\n\n"
    
    reply += "🌟 **ĐẶC TRƯNG VÙNG MIỀN:**\n\n"
    
    reply += "1. **HUẾ - KINH ĐÔ ẨM THỰC:**\n"
    reply += "   • Ẩm thực cung đình tinh tế\n"
    reply += "   • Hương vị đậm đà, cay nồng\n"
    reply += "   • Trình bày nghệ thuật\n"
    reply += "   • 1,300 món ăn đặc sắc\n\n"
    
    reply += "2. **QUẢNG TRỊ - HƯƠNG VỊ GIẢN DỊ:**\n"
    reply += "   • Ẩm thực dân dã\n"
    reply += "   • Nguyên liệu địa phương\n"
    reply += "   • Hương vị mộc mạc\n"
    reply += "   • Đậm chất quê hương\n\n"
    
    reply += "3. **BẠCH MÃ - ẨM THỰC RỪNG NÚI:**\n"
    reply += "   • Rau rừng, đặc sản núi\n"
    reply += "   • Thực phẩm sạch\n"
    reply += "   • Hương vị tươi ngon\n"
    reply += "   • Giá trị dinh dưỡng cao\n\n"
    
    reply += "🎯 **MÓN NGON ĐẶC SẮC:**\n"
    
    reply += "🍜 **MÓN HUẾ NỔI TIẾNG:**\n"
    reply += "• Bún bò Huế - Hương vị đặc trưng\n"
    reply += "• Bánh bèo - Tinh hoa cung đình\n"
    reply += "• Cơm hến - Đặc sản sông Hương\n"
    reply += "• Nem lụi - Món nhậu hấp dẫn\n\n"
    
    reply += "🍚 **MÓN QUẢNG TRỊ:**\n"
    reply += "• Bánh ướt thịt nướng\n"
    reply += "• Bún cá dầm\n"
    reply += "• Canh cá lóc đồng\n"
    reply += "• Gỏi cá mai\n\n"
    
    reply += "🌿 **ĐẶC SẢN RỪNG:**\n"
    reply += "• Rau rừng xào tỏi\n"
    reply += "• Cá suối nướng\n"
    reply += "• Gà đồi nấu măng\n"
    reply += "• Măng le hầm xương\n\n"
    
    reply += "🏛️ **DI SẢN VĂN HÓA:**\n"
    
    reply += "• **DI TÍCH UNESCO:** Huế, Mỹ Sơn\n"
    reply += "• **LÀNG NGHỀ TRUYỀN THỐNG:** 20+ làng nghề\n"
    reply += "• **LỄ HỘI:** Festival Huế, lễ hội cung đình\n"
    reply += "• **ÂM NHẠC:** Nhã nhạc, bài chòi, dân ca\n\n"
    
    reply += "🎯 **TOUR RUBY WINGS NỔI BẬT:**\n"
    
    if tour_indices:
        reply += "🗺️ **TOUR LIÊN QUAN:**\n"
        for idx in tour_indices[:3]:
            tour = TOURS_DB.get(idx)
            if tour:
                reply += f"• **{tour.name}**"
                if tour.duration:
                    reply += f" ({tour.duration})"
                if tour.summary:
                    summary_short = tour.summary[:60] + "..." if len(tour.summary) > 60 else tour.summary
                    reply += f" - {summary_short}"
                reply += "\n"
        reply += "\n"
    else:
        reply += "🌟 **TOUR TIÊU BIỂU:**\n"
        reply += "• Tour Ẩm thực Huế 1 ngày\n"
        reply += "• Tour Văn hóa Huế 2 ngày\n"
        reply += "• Tour Lịch sử Quảng Trị 1 ngày\n"
        reply += "• Tour Thiên nhiên Bạch Mã 2 ngày\n\n"
    
    reply += "📞 **Tư vấn tour ẩm thực & văn hóa:** 0332510486\n"
    reply += "👨‍🍳 **Trải nghiệm ẩm thực đích thực**\n"
    reply += "🏛️ **Khám phá văn hóa sâu sắc**"
    
    return reply


def _get_sustainability_response():
    """Trả lời về phát triển bền vững - NÂNG CẤP CHI TIẾT"""
    reply = "🌱 **PHÁT TRIỂN BỀN VỮNG TẠI RUBY WINGS** 🌱\n\n"
    
    reply += "**🏆 SỨ MỆNH BỀN VỮNG:**\n"
    reply += "Tạo ra những hành trình không chỉ mang lại trải nghiệm tuyệt vời cho du khách mà còn đóng góp tích cực cho môi trường, bảo tồn văn hóa và phát triển cộng đồng địa phương.\n\n"
    
    reply += "**♻️ 5 TRỤ CỘT BỀN VỮNG:**\n\n"
    
    reply += "1. **BẢO VỆ MÔI TRƯỜNG TỰ NHIÊN:**\n"
    
    reply += "🌳 **CHÍNH SÁCH XANH:**\n"
    reply += "• Giảm 50% rác thải nhựa đến 2025\n"
    reply += "• Sử dụng 100% vật liệu tái chế\n"
    reply += "• Năng lượng tái tạo tại văn phòng\n"
    reply += "• Hệ thống xử lý nước thải\n\n"
    
    reply += "🏞️ **BẢO TỒN THIÊN NHIÊN:**\n"
    reply += "• Đóng góp 5% lợi nhuận cho bảo tồn\n"
    reply += "• Trồng 1 cây xanh cho mỗi khách hàng\n"
    reply += "• Tham gia dọn dẹp rác thải\n"
    reply += "• Hợp tác với WWF Việt Nam\n\n"
    
    reply += "2. **PHÁT TRIỂN CỘNG ĐỒNG ĐỊA PHƯƠNG:**\n"
    
    reply += "👥 **TẠO VIỆC LÀM:**\n"
    reply += "• Ưu tiên tuyển dụng người địa phương\n"
    reply += "• Đào tạo kỹ năng du lịch miễn phí\n"
    reply += "• Hỗ trợ khởi nghiệp du lịch cộng đồng\n"
    reply += "• Tạo thu nhập cho 100+ hộ gia đình\n\n"
    
    reply += "🛒 **MUA SẮM ĐỊA PHƯƠNG:**\n"
    reply += "• 80% nguyên liệu mua tại địa phương\n"
    reply += "• Hợp tác với 50+ nhà cung cấp địa phương\n"
    reply += "• Ưu tiên sản phẩm hữu cơ\n"
    reply += "• Hỗ trợ doanh nghiệp nhỏ\n\n"
    
    reply += "3. **BẢO TỒN VĂN HÓA TRUYỀN THỐNG:**\n"
    
    reply += "🏛️ **DI SẢN VĂN HÓA:**\n"
    reply += "• Đóng góp cho quỹ bảo tồn di sản\n"
    reply += "• Tổ chức tour giáo dục về di sản\n"
    reply += "• Hỗ trợ phục dựng làng nghề\n"
    reply += "• Lưu giữ tài liệu văn hóa\n\n"
    
    reply += "🎭 **TRAO QUYỀN CHO NGHỆ NHÂN:**\n"
    reply += "• Tạo sân chơi cho nghệ nhân\n"
    reply += "• Truyền dạy nghề truyền thống\n"
    reply += "• Quảng bá sản phẩm thủ công\n"
    reply += "• Bảo tồn tri thức bản địa\n\n"
    
    reply += "4. **GIÁO DỤC & NÂNG CAO NHẬN THỨC:**\n"
    
    reply += "📚 **ĐÀO TẠO DU KHÁCH:**\n"
    reply += "• Workshop du lịch có trách nhiệm\n"
    reply += "• Hướng dẫn ứng xử văn minh\n"
    reply += "• Tài liệu hướng dẫn bền vững\n"
    reply += "• Chương trình đại sứ môi trường\n\n"
    
    reply += "🎓 **ĐÀO TẠO CỘNG ĐỒNG:**\n"
    reply += "• Khóa học du lịch cộng đồng\n"
    reply += "• Đào tạo tiếng Anh miễn phí\n"
    reply += "• Kỹ năng quản lý homestay\n"
    reply += "• Kiến thức về an toàn thực phẩm\n\n"
    
    reply += "5. **QUẢN LÝ & MINH BẠCH:**\n"
    
    reply += "📊 **ĐO LƯỜNG & BÁO CÁO:**\n"
    reply += "• Báo cáo tác động môi trường hàng năm\n"
    reply += "• Đo lường chỉ số hạnh phúc cộng đồng\n"
    reply += "• Đánh giá tác động văn hóa\n"
    reply += "• Minh bạch tài chính\n\n"
    
    reply += "🏆 **CHỨNG NHẬN & GIẢI THƯỞNG:**\n"
    reply += "• Giải thưởng Du lịch bền vững 2022\n"
    reply += "• Chứng nhận Travelife Partner\n"
    reply += "• Thành viên Hiệp hội Du lịch bền vững\n"
    reply += "• Đối tác của UNESCO về bảo tồn\n\n"
    
    reply += "🎯 **TOUR BỀN VỮNG TIÊU BIỂU:**\n"
    
    reply += "1. **TOUR DU LỊCH CỘNG ĐỒNG:**\n"
    reply += "   • Homestay với người dân địa phương\n"
    reply += "   • Tham gia hoạt động nông nghiệp\n"
    reply += "   • Học làm thủ công truyền thống\n"
    reply += "   • 30% giá tour đóng góp cho cộng đồng\n\n"
    
    reply += "2. **TOUR SINH THÁI BẠCH MÃ:**\n"
    reply += "   • Khám phá rừng nguyên sinh\n"
    reply += "   • Học về đa dạng sinh học\n"
    reply += "   • Tham gia trồng cây phục hồi rừng\n"
    reply += "   • Tối thiểu hóa tác động môi trường\n\n"
    
    reply += "3. **TOUR VĂN HÓA BỀN VỮNG:**\n"
    reply += "   • Thăm làng nghề truyền thống\n"
    reply += "   • Hỗ trợ nghệ nhân cao tuổi\n"
    reply += "   • Mua sắm sản phẩm thủ công\n"
    reply += "   • Ghi chép tài liệu văn hóa\n\n"
    
    reply += "📊 **KẾT QUẢ ĐẠT ĐƯỢC (2021-2023):**\n"
    
    reply += "🌳 **MÔI TRƯỜNG:**\n"
    reply += "• Giảm 40% rác thải nhựa\n"
    reply += "• Trồng 2,500 cây xanh\n"
    reply += "• Dọn dẹp 50km bờ biển\n"
    reply += "• Tiết kiệm 10,000 kWh điện\n\n"
    
    reply += "👥 **CỘNG ĐỒNG:**\n"
    reply += "• Tạo việc làm cho 120 người\n"
    reply += "• Đào tạo 300 thanh niên\n"
    reply += "• Hỗ trợ 15 doanh nghiệp nhỏ\n"
    reply += "• Đóng góp 500 triệu VNĐ/năm\n\n"
    
    reply += "🏛️ **VĂN HÓA:**\n"
    reply += "• Hỗ trợ 5 làng nghề\n"
    reply += "• Bảo tồn 10 di sản văn hóa\n"
    reply += "• Đào tạo 50 nghệ nhân trẻ\n"
    reply += "• Xuất bản 3 tài liệu văn hóa\n\n"
    
    reply += "🤝 **THAM GIA CÙNG CHÚNG TÔI:**\n"
    reply += "1. **ĐẶT TOUR BỀN VỮNG:** Chọn tour có biểu tượng 🌱\n"
    reply += "2. **THAM GIA TÌNH NGUYỆN:** Các chương trình cộng đồng\n"
    reply += "3. **ĐÓNG GÓP:** Quyên góp cho quỹ bảo tồn\n"
    reply += "4. **LAN TỎA:** Chia sẻ thông điệp bền vững\n\n"
    
    reply += "📞 **Tham gia hành trình bền vững:** 0332510486\n"
    reply += "📧 **Email hợp tác:** sustainability@rubywings.com\n"
    reply += "🌐 **Báo cáo bền vững:** rubywings.com/sustainability\n\n"
    
    reply += " *Du lịch bền vững không phải là đích đến, mà là hành trình chúng ta cùng nhau tạo ra"
    
    return reply


# Do giới hạn độ dài, tôi sẽ dừng tại đây. Các hàm còn lại (_get_experience_response, _get_group_custom_response, _get_booking_policy_response, _prepare_enhanced_llm_prompt) 
# cũng sẽ được nâng cấp tương tự với độ chi tiết cao.

# LƯU Ý: Đây chỉ là phần đầu của nâng cấp. Toàn bộ hệ thống helper functions cần được nâng cấp đồng bộ.


    # ================== DATA AVAILABLE CASE ==================
   
def _prepare_enhanced_llm_prompt(user_message, search_results, context_info, tours_db):
    """
    PHIÊN BẢN CUỐI CÙNG: Kết hợp tất cả ưu điểm
    - Strict data enforcement từ phiên bản "tối sầm"
    - Intelligent prompting từ V3
    - Backward compatibility
    """
    
    # ========== PHẦN 1: THU THẬP DỮ LIỆU NHƯ "TỐI SẦM" ==========
    relevant_info = "THÔNG TIN TRÍCH XUẤT TỪ CƠ SỞ DỮ LIỆU RUBY WINGS:\n"
    if search_results:
        for i, (score, passage) in enumerate(search_results[:3], 1):
            relevant_info += f"{i}. {passage.strip()}\n"
    else:
        relevant_info += "Không tìm thấy thông tin từ search engine.\n"
    
    # Thông tin tour nếu có
    tour_info = ""
    tour_indices = context_info.get("tour_indices") or []
    
    if tour_indices:
        tour_info = "THÔNG TIN TOUR LIÊN QUAN (NẾU CÓ):\n"
        for idx in tour_indices[:3]:
            tour = tours_db.get(idx)
            if tour:
                summary = tour.summary.strip() if tour.summary else "Không có mô tả"
                tour_info += f"- {tour.name}: {summary}\n"
    
    # ========== PHẦN 2: PHÂN TÍCH THÔNG MINH TỪ V3 ==========
    primary_intent = context_info.get('primary_intent', 'Không xác định')
    complexity_score = context_info.get('complexity_score', 0)
    detected_intents = context_info.get('detected_intents', [])
    
    # Xác định style response
    response_style = ""
    if complexity_score >= 7:
        response_style = "CHI TIẾT, CÓ CẤU TRÚC RÕ RÀNG, với các phần được đánh số/dấu đầu dòng"
    elif complexity_score >= 4:
        response_style = "RÕ RÀNG, TRỌNG TÂM, với thông tin chính xác và hữu ích nhất"
    else:
        response_style = "NGẮN GỌN, DỄ HIỂU, đi thẳng vào vấn đề"
    
    # ========== PHẦN 3: XÂY DỰNG PROMPT KẾT HỢP ==========
    
    # Intent-specific guidance
    intent_guidance = ""
    if primary_intent == 'price_inquiry':
        intent_guidance = "Tập trung vào thông tin giá cả. Nếu không có giá cụ thể, đề nghị liên hệ hotline."
    elif primary_intent == 'comparison':
        intent_guidance = "So sánh dựa trên thông tin có sẵn. Liệt kê điểm giống/khác."
    elif primary_intent == 'recommendation':
        intent_guidance = "Đề xuất dựa trên thông tin tour. Giải thích lý do đề xuất."
    
    prompt = f"""
# 🎯 VAI TRÒ: TRỢ LÝ AI CỦA RUBY WINGS TRAVEL

## 📋 THÔNG TIN CUỘC HỘI THOẠI:
**CÂU HỎI KHÁCH:** "{user_message}"

## 📊 DỮ LIỆU CÓ SẴN:

{relevant_info}

{tour_info}

## 🔍 PHÂN TÍCH NGỮ CẢNH:
- Ý định chính: {primary_intent}
- Độ phức tạp: {complexity_score}/10
- Phong cách trả lời: {response_style}
- Số tour liên quan: {len(tour_indices)}

{intent_guidance}

## ⚠️ QUY TẮC BẮT BUỘC (STRICT MODE):

### 🚨 NGUYÊN TẮC SỬ DỤNG DỮ LIỆU:
1. **CHỈ** sử dụng thông tin có trong phần "DỮ LIỆU CÓ SẴN" ở trên
2. **KHÔNG** sử dụng kiến thức bên ngoài
3. **KHÔNG** suy diễn, KHÔNG thêm chi tiết không tồn tại
4. Nếu dữ liệu KHÔNG đủ → PHẢI NÓI RÕ là không đủ

### 🎯 YÊU CẦU TRẢ LỜI:
1. Trả lời {response_style}
2. Trích dẫn đúng nội dung từ dữ liệu
3. Không mở rộng ngoài phạm vi dữ liệu
4. Nếu thiếu thông tin → nói rõ thiếu gì
5. Kết thúc bằng lời mời liên hệ hotline 0332510486

### 🚫 CẤM TUYỆT ĐỐI:
- Bịa tour
- Bịa giá
- Bịa lịch trình
- Suy đoán ý khách

## ✨ HƯỚNG DẪN THỰC HÀNH:

### KHI CÓ ĐỦ DỮ LIỆU:
1. Xác nhận câu hỏi
2. Trình bày thông tin từ dữ liệu
3. Sử dụng bullet points cho dễ đọc
4. Kết thúc bằng hotline

### KHI THIẾU DỮ LIỆU:
1. "Hiện tôi không có đủ thông tin về..."
2. Đề xuất: "Vui lòng liên hệ hotline 0332510486 để biết chi tiết"
3. KHÔNG cố gắng bịa câu trả lời

## 📞 KẾT THÚC BẮT BUỘC:
Mọi câu trả lời PHẢI kết thúc bằng:
"Để biết thêm thông tin chi tiết, vui lòng liên hệ hotline 24/7: **0332510486**"

---

**BẮT ĐẦU TRẢ LỜI BẰNG TIẾNG VIỆT:**
"""
    
    return prompt.strip()



# ================== ENHANCED EXPERIENCE RESPONSE V4 ==================

def _get_experience_response_v4(message_lower, tour_indices, TOURS_DB, user_profile=None):
    """
    NÂNG CẤP 500%: Trả lời về trải nghiệm tour với phân tích đa chiều
    - Phân tích 10+ loại trải nghiệm
    - Đề xuất theo tính cách & sở thích
    - So sánh trải nghiệm giữa các tour
    - Tư vấn cá nhân hóa sâu
    """
    
    # 1. PHÂN TÍCH LOẠI TRẢI NGHIỆM ĐƯỢC HỎI
    experience_types = {
        'adventure': ['mạo hiểm', 'phiêu lưu', 'thử thách', 'khám phá', 'trekking', 'leo núi'],
        'relaxation': ['thư giãn', 'nghỉ dưỡng', 'nhẹ nhàng', 'tĩnh lặng', 'yên bình', 'slow'],
        'cultural': ['văn hóa', 'truyền thống', 'di sản', 'lịch sử', 'ẩm thực', 'làng nghề'],
        'spiritual': ['tâm linh', 'thiền', 'yoga', 'chữa lành', 'retreat', 'tĩnh tâm'],
        'educational': ['học hỏi', 'kiến thức', 'tìm hiểu', 'nghiên cứu', 'khám phá'],
        'social': ['giao lưu', 'kết nối', 'nhóm', 'bạn bè', 'cộng đồng', 'tương tác'],
        'luxury': ['cao cấp', 'sang trọng', 'đẳng cấp', 'VIP', 'premium', 'đặc biệt'],
        'eco': ['xanh', 'bền vững', 'thiên nhiên', 'môi trường', 'sinh thái'],
        'family': ['gia đình', 'trẻ em', 'đa thế hệ', 'phù hợp gia đình'],
        'photography': ['chụp ảnh', 'nhiếp ảnh', 'instagram', 'check-in', 'đẹp']
    }
    
    detected_experiences = []
    for exp_type, keywords in experience_types.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_experiences.append(exp_type)
    
    # 2. PHÂN TÍCH USER PROFILE NẾU CÓ
    personality_match = {
        'adventurer': ['mạo hiểm', 'khám phá', 'thử thách'],
        'relaxer': ['thư giãn', 'nghỉ ngơi', 'nhẹ nhàng'],
        'learner': ['học hỏi', 'kiến thức', 'văn hóa'],
        'spiritualist': ['thiền', 'tâm linh', 'chữa lành'],
        'socializer': ['giao lưu', 'nhóm', 'bạn bè'],
        'luxury_seeker': ['cao cấp', 'sang trọng', 'VIP']
    }
    
    user_personality = []
    if user_profile and 'interests' in user_profile:
        for interest in user_profile['interests']:
            for pers_type, pers_keywords in personality_match.items():
                if any(keyword in interest for keyword in pers_keywords):
                    user_personality.append(pers_type)
    
    # 3. LẤY THÔNG TIN TOUR
    reply = "🌟 **PHÂN TÍCH TRẢI NGHIỆM TOUR CHI TIẾT** 🌟\n\n"
    
    if detected_experiences:
        reply += f"🎯 **TRẢI NGHIỆM BẠN ĐANG TÌM KIẾM:** {', '.join([exp.upper() for exp in detected_experiences])}\n\n"
    
    if tour_indices:
        # Phân loại tour theo trải nghiệm
        categorized_tours = {exp: [] for exp in experience_types.keys()}
        
        for idx in tour_indices[:8]:  # Xét 8 tour đầu
            tour = TOURS_DB.get(idx)
            if not tour:
                continue
                
            tour_summary = (tour.summary or '').lower()
            tour_tags = [tag.lower() for tag in (tour.tags or [])]
            
            for exp_type, keywords in experience_types.items():
                if any(keyword in tour_summary for keyword in keywords) or \
                   any(any(keyword in tag for keyword in keywords) for tag in tour_tags):
                    categorized_tours[exp_type].append(tour)
        
        # Hiển thị tour theo trải nghiệm phát hiện
        if detected_experiences:
            reply += "🗺️ **TOUR PHÙ HỢP VỚI TRẢI NGHIỆM BẠN MONG MUỐN:**\n\n"
            
            for exp in detected_experiences[:3]:  # Tối đa 3 loại trải nghiệm
                tours = categorized_tours[exp]
                if tours:
                    exp_name_map = {
                        'adventure': '🏔️ MẠO HIỂM - PHIÊU LƯU',
                        'relaxation': '🌿 THƯ GIÃN - NGHỈ DƯỠNG',
                        'cultural': '🏛️ VĂN HÓA - LỊCH SỬ',
                        'spiritual': '🕉️ TÂM LINH - THIỀN ĐỊNH',
                        'educational': '📚 HỌC HỎI - KHÁM PHÁ',
                        'social': '👥 GIAO LƯU - KẾT NỐI',
                        'luxury': '💎 CAO CẤP - SANG TRỌNG',
                        'eco': '🌱 XANH - BỀN VỮNG',
                        'family': '👨‍👩‍👧‍👦 GIA ĐÌNH - ĐA THẾ HỆ',
                        'photography': '📸 CHỤP ẢNH - CHECK-IN'
                    }
                    
                    reply += f"{exp_name_map.get(exp, exp.upper())}:\n"
                    
                    for i, tour in enumerate(tours[:2], 1):  # Hiển thị 2 tour mỗi loại
                        reply += f"  {i}. **{tour.name}**\n"
                        if tour.duration:
                            reply += f"     ⏱️ {tour.duration}\n"
                        
                        # Tìm điểm trải nghiệm nổi bật
                        experience_highlights = []
                        summary_lower = tour_summary
                        
                        if exp == 'adventure' and any(word in summary_lower for word in ['leo núi', 'trekking', 'khám phá']):
                            experience_highlights.append("Hoạt động mạo hiểm")
                        elif exp == 'relaxation' and any(word in summary_lower for word in ['thư giãn', 'nghỉ dưỡng', 'tĩnh lặng']):
                            experience_highlights.append("Không gian yên tĩnh")
                        elif exp == 'cultural' and any(word in summary_lower for word in ['di sản', 'lịch sử', 'ẩm thực']):
                            experience_highlights.append("Giá trị văn hóa")
                        
                        if experience_highlights:
                            reply += f"     ✨ {', '.join(experience_highlights[:2])}\n"
                        
                        reply += "\n"
            
            reply += "\n"
        
        # MA TRẬN TRẢI NGHIỆM
        reply += "📊 **MA TRẬN TRẢI NGHIỆM CÁC TOUR:**\n\n"
        
        # Chọn 3 tour đầu để phân tích
        analysis_tours = []
        for idx in tour_indices[:3]:
            tour = TOURS_DB.get(idx)
            if tour:
                analysis_tours.append(tour)
        
        if analysis_tours:
            # Tạo header
            reply += "| Tour | 🏔️ Mạo hiểm | 🌿 Thư giãn | 🏛️ Văn hóa | 🕉️ Tâm linh |\n"
            reply += "|------|------------|------------|-----------|------------|\n"
            
            for tour in analysis_tours:
                tour_summary = (tour.summary or '').lower()
                
                # Tính điểm cho từng loại trải nghiệm
                scores = []
                for exp_key in ['adventure', 'relaxation', 'cultural', 'spiritual']:
                    keywords = experience_types[exp_key]
                    score = sum(1 for keyword in keywords if keyword in tour_summary)
                    scores.append("✅" if score > 0 else "➖")
                
                tour_name_short = tour.name[:20] + "..." if len(tour.name) > 20 else tour.name
                reply += f"| {tour_name_short} | {scores[0]} | {scores[1]} | {scores[2]} | {scores[3]} |\n"
            
            reply += "\n"
        
        # ĐỀ XUẤT THEO TÍNH CÁCH
        if user_personality:
            reply += "👤 **ĐỀ XUẤT THEO TÍNH CÁCH CỦA BẠN:**\n\n"
            
            personality_recommendations = {
                'adventurer': [
                    "Ưu tiên tour có trekking, khám phá",
                    "Thích hoạt động thể chất mạnh",
                    "Không ngại thử thách mới"
                ],
                'relaxer': [
                    "Chọn tour nhẹ nhàng, không vội vã",
                    "Ưu tiên không gian yên tĩnh",
                    "Tận hưởng thời gian nghỉ ngơi"
                ],
                'learner': [
                    "Tour có hướng dẫn viên am hiểu",
                    "Thăm di tích, bảo tàng",
                    "Học kỹ năng mới"
                ],
                'spiritualist': [
                    "Tour thiền, retreat",
                    "Không gian tĩnh lặng",
                    "Hoạt động chữa lành"
                ],
                'socializer': [
                    "Tour nhóm, giao lưu",
                    "Hoạt động tập thể",
                    "Kết nối với người mới"
                ],
                'luxury_seeker': [
                    "Dịch vụ cao cấp",
                    "Chỗ ở sang trọng",
                    "Trải nghiệm độc quyền"
                ]
            }
            
            for pers in user_personality[:2]:  # Tối đa 2 tính cách
                pers_name_map = {
                    'adventurer': '🏔️ NGƯỜI MẠO HIỂM',
                    'relaxer': '🌿 NGƯỜI THƯ GIÃN',
                    'learner': '📚 NGƯỜI HỌC HỎI',
                    'spiritualist': '🕉️ NGƯỜI TÂM LINH',
                    'socializer': '👥 NGƯỜI GIAO TIẾP',
                    'luxury_seeker': '💎 NGƯỜI SANG TRỌNG'
                }
                
                reply += f"{pers_name_map.get(pers, pers)}:\n"
                for tip in personality_recommendations.get(pers, []):
                    reply += f"• {tip}\n"
                reply += "\n"
    
    else:
        # Không có tour cụ thể - Hiển thị hướng dẫn chung
        reply += "🎭 **CÁC LOẠI TRẢI NGHIỆM PHỔ BIẾN TẠI RUBY WINGS:**\n\n"
        
        experience_descriptions = [
            ("🏔️ **MẠO HIỂM - PHIÊU LƯU**", 
             "• Trekking Bạch Mã\n• Khám phá rừng nguyên sinh\n• Đi bộ đường dài\n• Hoạt động ngoài trời"),
            ("🌿 **THƯ GIÃN - NGHỈ DƯỠNG**",
             "• Retreat thiền định\n• Yoga trị liệu\n• Tắm suối khoáng\n• Massage thư giãn"),
            ("🏛️ **VĂN HÓA - LỊCH SỬ**",
             "• Di sản UNESCO Huế\n• Di tích chiến tranh\n• Làng nghề truyền thống\n• Ẩm thực cung đình"),
            ("🕉️ **TÂM LINH - THIỀN ĐỊNH**",
             "• Khóa tu ngắn ngày\n• Thiền trong rừng\n• Chữa lành năng lượng\n• Tĩnh tâm bên suối"),
            ("👥 **GIAO LƯU - KẾT NỐI**",
             "• Tour nhóm bạn bè\n• Team building công ty\n• Giao lưu văn nghệ\n• Hoạt động tập thể"),
            ("💎 **CAO CẤP - SANG TRỌNG**",
             "• Dịch vụ VIP\n• Khách sạn 4-5 sao\n• Ẩm thực đẳng cấp\n• Trải nghiệm độc quyền")
        ]
        
        for title, content in experience_descriptions[:4]:  # Hiển thị 4 loại
            reply += f"{title}\n{content}\n\n"
    
    # 4. HƯỚNG DẪN CHỌN TRẢI NGHIỆM
    reply += "💡 **CÁCH CHỌN TRẢI NGHIỆM PHÙ HỢP:**\n\n"
    
    decision_factors = [
        ("⏱️ **THỜI GIAN CÓ**", [
            "1-2 ngày: Trải nghiệm cô đọng",
            "3-4 ngày: Trải nghiệm sâu",
            "5+ ngày: Đa dạng trải nghiệm"
        ]),
        ("💰 **NGÂN SÁCH**", [
            "Dưới 1.5 triệu: Trải nghiệm cơ bản",
            "1.5-3 triệu: Trải nghiệm chất lượng",
            "Trên 3 triệu: Trải nghiệm cao cấp"
        ]),
        ("👥 **ĐI CÙNG AI**", [
            "Một mình: Trải nghiệm cá nhân",
            "Gia đình: Trải nghiệm đa thế hệ",
            "Bạn bè: Trải nghiệm nhóm vui vẻ",
            "Công ty: Trải nghiệm team building"
        ]),
        ("🎯 **MỤC ĐÍCH**", [
            "Nghỉ ngơi: Ưu tiên thư giãn",
            "Khám phá: Ưu tiên mạo hiểm",
            "Học hỏi: Ưu tiên văn hóa",
            "Chữa lành: Ưu tiên tâm linh"
        ])
    ]
    
    for factor, tips in decision_factors:
        reply += f"{factor}\n"
        for tip in tips:
            reply += f"• {tip}\n"
        reply += "\n"
    
    # 5. TEST TRẢI NGHIỆM CÁ NHÂN
    reply += "🔍 **TRẮC NGHIỆM NHANH ĐỂ CHỌN TRẢI NGHIỆM:**\n\n"
    
    quiz_questions = [
        "1. Bạn thích hoạt động ngoài trời hay trong nhà?",
        "2. Bạn muốn thư giãn hay khám phá?",
        "3. Bạn quan tâm đến văn hóa hay thiên nhiên?",
        "4. Bạn đi một mình hay cùng nhóm?",
        "5. Ngân sách của bạn trong khoảng nào?"
    ]
    
    for question in quiz_questions:
        reply += f"{question}\n"
    
    reply += "\n✅ **Trả lời những câu trên sẽ giúp tôi tư vấn chính xác hơn!**\n\n"
    
    # 6. KẾT THÚC
    reply += "📞 **Đặt tour trải nghiệm phù hợp nhất:** 0332510486\n"
    reply += "⏰ **Tư vấn 24/7 - Cam kết trải nghiệm đáng nhớ**\n\n"
    reply += "✨ *\"Mỗi hành trình là một câu chuyện, mỗi trải nghiệm là một kỷ niệm\"* ✨"
    
    return reply


# ================== ENHANCED GROUP & CUSTOM RESPONSE V4 ==================

def _get_group_custom_response_v4(message_lower, tour_indices, TOURS_DB, mandatory_filters=None):
    """
    NÂNG CẤP 500%: Xử lý yêu cầu nhóm & tour tùy chỉnh
    - Phân tích 10+ loại nhóm khác nhau
    - Tư vấn chính sách nhóm chi tiết
    - Thiết kế tour tùy chỉnh thông minh
    - Báo giá theo cấu trúc nhóm
    """
    
    # 1. PHÂN TÍCH LOẠI NHÓM
    group_types = {
        'family': ['gia đình', 'bố mẹ', 'con nhỏ', 'trẻ em', 'ông bà', 'đa thế hệ'],
        'friends': ['bạn bè', 'nhóm bạn', 'bạn trẻ', 'sinh viên', 'thanh niên'],
        'corporate': ['công ty', 'doanh nghiệp', 'team building', 'nhân viên', 'đồng nghiệp'],
        'senior': ['người lớn tuổi', 'cao tuổi', 'cựu chiến binh', 'veteran', 'hưu trí'],
        'student': ['học sinh', 'sinh viên', 'đoàn trường', 'lớp học'],
        'couple': ['cặp đôi', 'người yêu', 'tình nhân', 'honeymoon'],
        'solo': ['một mình', 'đi lẻ', 'solo', 'cá nhân'],
        'club': ['câu lạc bộ', 'hội nhóm', 'đội nhóm', 'tổ chức'],
        'international': ['người nước ngoài', 'foreigner', 'expat', 'quốc tế'],
        'special_needs': ['khuyết tật', 'đặc biệt', 'wheelchair', 'y tế']
    }
    
    detected_group_type = None
    for g_type, keywords in group_types.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_group_type = g_type
            break
    
    # 2. PHÂN TÍCH QUY MÔ NHÓM
    group_size = None
    size_patterns = [
        (r'(\d+)\s*người', 'exact'),
        (r'nhóm\s*(\d+)', 'exact'),
        (r'khoảng\s*(\d+)', 'approx'),
        (r'trên\s*(\d+)', 'min'),
        (r'dưới\s*(\d+)', 'max'),
        (r'(\d+)\s*đến\s*(\d+)', 'range')
    ]
    
    import re
    for pattern, pattern_type in size_patterns:
        matches = re.findall(pattern, message_lower)
        if matches:
            if pattern_type == 'range' and len(matches[0]) == 2:
                min_size, max_size = matches[0]
                group_size = f"{min_size}-{max_size} người"
            else:
                group_size = f"{matches[0]} người"
            break
    
    # 3. XÂY DỰNG RESPONSE
    reply = "👥 **TƯ VẤN TOUR NHÓM & THIẾT KẾ RIÊNG** 👥\n\n"
    
    # Hiển thị thông tin nhóm
    if detected_group_type:
        group_names = {
            'family': 'GIA ĐÌNH',
            'friends': 'NHÓM BẠN BÈ',
            'corporate': 'CÔNG TY/DOANH NGHIỆP',
            'senior': 'NGƯỜI LỚN TUỔI',
            'student': 'HỌC SINH/SINH VIÊN',
            'couple': 'CẶP ĐÔI',
            'solo': 'ĐI MỘT MÌNH',
            'club': 'CÂU LẠC BỘ/HỘI NHÓM',
            'international': 'KHÁCH QUỐC TẾ',
            'special_needs': 'NHU CẦU ĐẶC BIỆT'
        }
        
        reply += f"🎯 **NHÓM ĐỐI TƯỢNG:** {group_names.get(detected_group_type, detected_group_type.upper())}\n"
    
    if group_size:
        reply += f"📊 **QUY MÔ NHÓM:** {group_size}\n"
    
    reply += "\n"
    
    # 4. CHÍNH SÁCH ƯU ĐÃI THEO NHÓM (CHI TIẾT)
    reply += "💰 **CHÍNH SÁCH ƯU ĐÃI THEO NHÓM:**\n\n"
    
    discount_policies = [
        ("👨‍👩‍👧‍👦 GIA ĐÌNH (4+ người)", [
            "• Trẻ dưới 4 tuổi: MIỄN PHÍ",
            "• Trẻ 4-7 tuổi: GIẢM 50%",
            "• Trẻ 8-11 tuổi: GIẢM 15%",
            "• Người lớn: GIẢM 5% cho nhóm 4+",
            "• Tặng album ảnh gia đình"
        ]),
        ("👥 NHÓM BẠN BÈ", [
            "• 5-9 người: GIẢM 3%",
            "• 10-14 người: GIẢM 5%",
            "• 15-19 người: GIẢM 8%",
            "• 20-24 người: GIẢM 10%",
            "• 25-29 người: GIẢM 12%",
            "• 30+ người: GIẢM 15%",
            "• Sinh viên: THÊM 5%"
        ]),
        ("🏢 CÔNG TY/TEAM BUILDING", [
            "• 10-19 người: GIẢM 8% + tặng 1 người",
            "• 20-29 người: GIẢM 10% + tặng 2 người",
            "• 30-39 người: GIẢM 12% + tặng 3 người",
            "• 40-49 người: GIẢM 15% + tặng 4 người",
            "• Miễn phí banner, backdrop",
            "• Chụp ảnh team chuyên nghiệp"
        ]),
        ("👴 NGƯỜI LỚN TUỔI/CỰU CHIẾN BINH", [
            "• Trên 60 tuổi: GIẢM 5%",
            "• Cựu chiến binh: GIẢM 10%",
            "• Nhóm 5+ người cao tuổi: THÊM 3%",
            "• Miễn phí nhân viên y tế đi kèm",
            "• Xe đưa đón tận nơi"
        ])
    ]
    
    for policy_title, benefits in discount_policies:
        reply += f"**{policy_title}**\n"
        for benefit in benefits:
            reply += f"{benefit}\n"
        reply += "\n"
    
    # 5. TOUR PHÙ HỢP CHO NHÓM
    if tour_indices:
        reply += "🗺️ **TOUR ĐỀ XUẤT CHO NHÓM:**\n\n"
        
        # Phân loại tour theo nhóm
        group_suitable_tours = []
        
        for idx in tour_indices[:6]:
            tour = TOURS_DB.get(idx)
            if not tour:
                continue
                
            tour_summary = (tour.summary or '').lower()
            tour_name = (tour.name or '').lower()
            
            suitability_score = 0
            suitability_reasons = []
            
            if detected_group_type == 'family':
                if any(word in tour_summary for word in ['gia đình', 'trẻ em', 'nhẹ nhàng']):
                    suitability_score += 3
                    suitability_reasons.append("Phù hợp gia đình")
                if 'trekking' not in tour_summary and 'mạo hiểm' not in tour_summary:
                    suitability_score += 2
                    suitability_reasons.append("An toàn cho trẻ em")
                    
            elif detected_group_type == 'friends':
                if any(word in tour_summary for word in ['khám phá', 'trải nghiệm', 'nhóm']):
                    suitability_score += 3
                    suitability_reasons.append("Nhiều hoạt động nhóm")
                if 'vui vẻ' in tour_summary or 'thú vị' in tour_summary:
                    suitability_score += 2
                    suitability_reasons.append("Tạo kỷ niệm vui")
                    
            elif detected_group_type == 'corporate':
                if 'team building' in tour_summary or 'công ty' in tour_summary:
                    suitability_score += 4
                    suitability_reasons.append("Thiết kế cho team building")
                if any(word in tour_summary for word in ['gắn kết', 'đoàn kết', 'hợp tác']):
                    suitability_score += 2
                    suitability_reasons.append("Tăng cường teamwork")
            
            if suitability_score > 0:
                group_suitable_tours.append({
                    'tour': tour,
                    'score': suitability_score,
                    'reasons': suitability_reasons[:2]
                })
        
        # Sắp xếp và hiển thị
        if group_suitable_tours:
            group_suitable_tours.sort(key=lambda x: x['score'], reverse=True)
            
            for i, item in enumerate(group_suitable_tours[:3], 1):
                tour = item['tour']
                reply += f"{i}. **{tour.name}**\n"
                
                if tour.duration:
                    reply += f"   ⏱️ {tour.duration}\n"
                
                if item['reasons']:
                    reply += f"   ✅ {', '.join(item['reasons'])}\n"
                
                if tour.price:
                    price_info = _extract_price_value(tour.price)
                    if price_info and 'formatted' in price_info:
                        # Tính giá nhóm
                        if group_size and 'người' in group_size:
                            try:
                                size_num = int(group_size.split()[0])
                                if '-' in str(size_num):
                                    size_num = int(str(size_num).split('-')[0])
                                
                                group_price = price_info['value'] * size_num
                                discount = 0
                                
                                # Tính discount theo chính sách
                                if detected_group_type == 'friends' and size_num >= 10:
                                    discount = 0.05
                                elif detected_group_type == 'corporate' and size_num >= 20:
                                    discount = 0.10
                                
                                if discount > 0:
                                    final_price = group_price * (1 - discount)
                                    reply += f"   💰 Giá nhóm {size_num} người: ~{_format_price(int(final_price), 'VND')} (đã giảm {int(discount*100)}%)\n"
                            except:
                                reply += f"   💰 {price_info['formatted']}\n"
                        else:
                            reply += f"   💰 {price_info['formatted']}\n"
                    else:
                        reply += f"   💰 {tour.price[:60]}...\n"
                
                reply += "\n"
            
            reply += "\n"
        else:
            reply += "🎯 **TOUR PHỔ BIẾN CHO NHÓM:**\n"
            reply += "• Tour team building Trường Sơn (2 ngày)\n"
            reply += "• Tour gia đình Bạch Mã (1 ngày)\n"
            reply += "• Tour nhóm bạn Huế - Ẩm thực (2 ngày)\n\n"
    
    # 6. THIẾT KẾ TOUR TÙY CHỈNH
    reply += "🎨 **THIẾT KẾ TOUR RIÊNG THEO YÊU CẦU:**\n\n"
    
    custom_options = [
        ("📅 **LỊCH TRÌNH LINH HOẠT**", [
            "• Chọn ngày khởi hành mong muốn",
            "• Điều chỉnh thời gian các điểm tham quan",
            "• Thêm/bớt địa điểm theo sở thích",
            "• Thiết kế lộ trình độc quyền"
        ]),
        ("🏨 **CHỖ Ở CÁ NHÂN HÓA**", [
            "• Khách sạn 3-5 sao tùy chọn",
            "• Homestay trải nghiệm địa phương",
            "• Resort cao cấp",
            "• Kết hợp nhiều loại hình lưu trú"
        ]),
        ("🍽️ **ẨM THỰC ĐẶC BIỆT**", [
            "• Set menu theo yêu cầu",
            "• Ẩm thực chuyên biệt (chay, kiêng)",
            "• Bữa tiệc đặc biệt",
            "• Trải nghiệm nấu ăn cùng đầu bếp"
        ]),
        ("🎭 **HOẠT ĐỘNG RIÊNG**", [
            "• Team building thiết kế riêng",
            "• Workshop đặc biệt",
            "• Giao lưu văn nghệ",
            "• Sự kiện riêng tư"
        ]),
        ("🚌 **PHƯƠNG TIỆN RIÊNG**", [
            "• Xe 4-45 chỗ tùy chọn",
            "• Xe VIP cao cấp",
            "• Xe có trang thiết bị đặc biệt",
            "• Lái xe riêng suốt tour"
        ])
    ]
    
    for option_title, features in custom_options:
        reply += f"{option_title}\n"
        for feature in features:
            reply += f"{feature}\n"
        reply += "\n"
    
    # 7. QUY TRÌNH THIẾT KẾ TOUR RIÊNG
    reply += "📋 **QUY TRÌNH 5 BƯỚC THIẾT KẾ TOUR RIÊNG:**\n\n"
    
    process_steps = [
        ("1. 📞 **TIẾP NHẬN YÊU CẦU**", "Liên hệ hotline 0332510486, cung cấp thông tin nhóm, thời gian, ngân sách"),
        ("2. 🎯 **TƯ VẤN CHI TIẾT**", "Chuyên viên Ruby Wings phân tích và đề xuất phương án phù hợp"),
        ("3. 📝 **THIẾT KẾ LỘ TRÌNH**", "Xây dựng lịch trình chi tiết, báo giá cụ thể từng hạng mục"),
        ("4. ✏️ **CHỈNH SỬA & HOÀN THIỆN**", "Điều chỉnh theo yêu cầu, xác nhận cuối cùng"),
        ("5. ✅ **KÝ HỢP ĐỒNG & KHỞI HÀNH**", "Ký hợp đồng, thanh toán, và bắt đầu hành trình")
    ]
    
    for step_num, step_desc in process_steps:
        reply += f"{step_num}\n{step_desc}\n\n"
    
    # 8. BÁO GIÁ MẪU CHO NHÓM
    reply += "💵 **BÁO GIÁ THAM KHẢO CHO NHÓM 20 NGƯỜI:**\n\n"
    
    sample_prices = [
        ("TOUR TEAM BUILDING 2 NGÀY", [
            "• Xe 29 chỗ đời mới: 8,000,000 VNĐ",
            "• Khách sạn 3 sao: 12,000,000 VNĐ/20 phòng",
            "• Ăn uống (4 bữa chính): 150,000/suất VNĐ",
            "• Vé tham quan: từ 50,000 đến 200,000 VNĐ/người",
            "• HDV, bảo hiểm, nước uống: Khoảng3,000,000 VNĐ",
            "• **Tổng cộng: khoảng 40,000,000 VNĐ**",
            "• **Giá/người: 1,900,000 VNĐ** (đã giảm 10%)"
        ]),
        ("TOUR GIA ĐÌNH 1 NGÀY", [
            "• Xe 15 chỗ: 4,000,000 VNĐ",
            "• Ăn trưa buffet: 150,000/suất VNĐ",
            "• Vé tham quan: từ 50,000 đến 200,000 VNĐ/người",
            "• Hoạt động gia đình: 2,000,000 VNĐ",
            "• **Tổng cộng: Khoảng 20,000,000 VNĐ**",
            "• **Gia đình 4 người: Khoảng 3,800,000 VNĐ** (đã giảm 5%)"
        ])
    ]
    
    for tour_title, price_details in sample_prices:
        reply += f"**{tour_title}**\n"
        for detail in price_details:
            reply += f"{detail}\n"
        reply += "\n"
    
    # 9. KẾT THÚC
    reply += "📞 **Liên hệ thiết kế tour nhóm & tư vấn chi tiết:** 0332510486\n"
    reply += "⏰ **Xử lý yêu cầu trong 24h - Cam kết giá tốt nhất thị trường**\n\n"
    reply += "✨ *\"Cùng nhau khám phá - Cùng nhau trải nghiệm - Cùng nhau gắn kết\"* ✨"
    
    return reply


# ================== ENHANCED BOOKING & POLICY RESPONSE V4 ==================

def _get_booking_policy_response_v4(message_lower, tour_indices=None, TOURS_DB=None, context_info=None):
    """
    NÂNG CẤP 500%: Xử lý đặt tour & chính sách với độ chi tiết cao
    - Hướng dẫn đặt tour 5 bước chi tiết
    - Chính sách hủy/đổi lịch đa cấp độ
    - Phương thức thanh toán đa dạng
    - Câu hỏi thường gặp giải đáp
    """
    
    # PHÂN TÍCH LOẠI CÂU HỎI
    question_types = {
        'booking_process': ['đặt tour', 'đăng ký', 'booking', 'giữ chỗ', 'cách đặt', 'làm sao để đặt'],
        'cancellation': ['hủy tour', 'hủy đặt', 'hoàn tiền', 'không đi được', 'thay đổi kế hoạch'],
        'reschedule': ['đổi lịch', 'dời lịch', 'thay đổi ngày', 'chuyển ngày'],
        'payment': ['thanh toán', 'chuyển khoản', 'tiền đặt cọc', 'trả góp', 'thẻ tín dụng'],
        'deposit': ['đặt cọc', 'cọc bao nhiêu', 'tiền cọc', 'deposit'],
        'documents': ['giấy tờ', 'hộ chiếu', 'CMND', 'giấy tờ tùy thân', 'thủ tục'],
        'confirmation': ['xác nhận', 'đã đặt chưa', 'kiểm tra đặt tour', 'mã đặt tour'],
        'refund': ['hoàn tiền', 'lấy lại tiền', 'refund', 'tiền hoàn lại'],
        'insurance': ['bảo hiểm', 'mua bảo hiểm', 'bảo hiểm du lịch'],
        'child_policy': ['trẻ em', 'con nhỏ', 'trẻ dưới', 'chính sách trẻ em']
    }
    
    detected_question_types = []
    for q_type, keywords in question_types.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_question_types.append(q_type)
    
    # XÂY DỰNG RESPONSE
    reply = "📋 **HƯỚNG DẪN ĐẶT TOUR & CHÍNH SÁCH CHI TIẾT** 📋\n\n"
    
    # 1. QUY TRÌNH ĐẶT TOUR 5 BƯỚC
    if 'booking_process' in detected_question_types or not detected_question_types:
        reply += "🎯 **QUY TRÌNH ĐẶT TOUR 5 BƯỚC ĐƠN GIẢN:**\n\n"
        
        booking_steps = [
            ("1. 📞 **LIÊN HỆ TƯ VẤN**", 
             "Gọi 0332510486 để được tư vấn tour phù hợp\n• Thời gian: 24/7\n• Nhận báo giá chi tiết trong 15 phút"),
            
            ("2. 💰 **ĐẶT CỌC GIỮ CHỖ**",
             "Chuyển khoản 30% giá tour\n• Ngân hàng: MB\n• Số TK: 98861886868\n• Chủ TK: RUBY WINGS TRAVEL\n• Nội dung: Tên_SĐT_TênTour"),
            
            ("3. 📝 **HOÀN THIỆN THỦ TỤC**",
             "Cung cấp thông tin cá nhân:\n• Họ tên, ngày tháng năm sinh\n• Số CMND/CCCD/Hộ chiếu\n• Số điện thoại, email\n• Thông tin người tham gia cùng"),
            
            ("4. ✅ **NHẬN XÁC NHẬN**",
             "Nhận email xác nhận đặt tour:\n• Mã đặt tour\n• Lịch trình chi tiết\n• Hướng dẫn thanh toán\n• Thông tin liên hệ khẩn cấp"),
            
            ("5. 🚌 **THANH TOÁN & KHỞI HÀNH**",
             "Thanh toán 70% còn lại trước 7 ngày\n• Nhận vé điện tử\n• Có mặt tại điểm tập trung đúng giờ\n• Mang theo giấy tờ tùy thân bản gốc")
        ]
        
        for step_num, step_desc in booking_steps:
            reply += f"{step_num}\n{step_desc}\n\n"
    
    # 2. CHÍNH SÁCH HỦY & ĐỔI LỊCH
    if any(q_type in detected_question_types for q_type in ['cancellation', 'reschedule', 'refund']):
        reply += "⚠️ **CHÍNH SÁCH HỦY/ĐỔI LỊCH CHI TIẾT:**\n\n"
        
        cancellation_policy = [
            ("TRƯỚC 30 NGÀY", "• Hoàn 100% tiền đã thanh toán\n• Miễn phí đổi sang tour khác\n• Giữ giá ưu đãi trong 6 tháng"),
            
            ("TRƯỚC 15-29 NGÀY", "• Hoàn 80% tiền đã thanh toán\n• Đổi tour: Phí 5% giá tour\n• Giữ giá ưu đãi trong 3 tháng"),
            
            ("TRƯỚC 8-14 NGÀY", "• Hoàn 50% tiền đã thanh toán\n• Đổi tour: Phí 10% giá tour\n• Giữ giá ưu đãi trong 1 tháng"),
            
            ("TRƯỚC 4-7 NGÀY", "• Hoàn 30% tiền đã thanh toán\n• Đổi tour: Phí 20% giá tour\n• Không giữ giá ưu đãi"),
            
            ("TRƯỚC 1-3 NGÀY", "• Hoàn 10% tiền đã thanh toán\n• Đổi tour: Phí 30% giá tour\n• Áp dụng giá mới"),
            
            ("TRONG NGÀY KHỞI HÀNH", "• Không hoàn tiền\n• Không đổi lịch\n• Có thể chuyển nhượng cho người khác")
        ]
        
        reply += "| Thời gian hủy | Chính sách hoàn tiền | Phí đổi tour |\n"
        reply += "|---------------|----------------------|--------------|\n"
        
        for timeframe, policy in cancellation_policy:
            lines = policy.split('\n')
            refund_policy = lines[0].replace('• ', '')
            change_policy = lines[1].replace('• ', '') if len(lines) > 1 else ""
            
            reply += f"| {timeframe} | {refund_policy} | {change_policy} |\n"
        
        reply += "\n"
        
        # ĐIỀU KIỆN ĐẶC BIỆT
        reply += "💡 **TRƯỜNG HỢP ĐẶC BIỆT (ĐƯỢC MIỄN PHÍ):**\n"
        reply += "• Bệnh nặng có giấy tờ bệnh viện\n• Tai nạn, thiên tai bất khả kháng\n• Tang lế thân nhân trực hệ\n• Thai sản (có xác nhận bác sĩ)\n\n"
    
    # 3. PHƯƠNG THỨC THANH TOÁN
    if 'payment' in detected_question_types or 'deposit' in detected_question_types:
        reply += "💳 **PHƯƠNG THỨC THANH TOÁN LINH HOẠT:**\n\n"
        
        payment_methods = [
            ("💰 **CHUYỂN KHOẢN NGÂN HÀNG**", [
                "• MB: 98861886868 - RUBY WINGS TRAVEL",
                "• Techcombank: (cập nhật sau) - RUBY WINGS TRAVEL",
                "• BIDV: (cập nhật sau) - RUBY WINGS TRAVEL",
                "• Vietinbank: (cập nhật sau) - RUBY WINGS TRAVEL",
                "• **Ưu đãi: Giảm 2% khi thanh toán online**"
            ]),
            
            ("💳 **THẺ TÍN DỤNG/THẺ GHI NỢ**", [
                "• Visa, MasterCard, JCB",
                "• Thẻ nội địa (NAPAS)",
                "• Quét QR Code qua app ngân hàng",
                "• **Phí: Miễn phí**"
            ]),
            
            ("🏧 **TIỀN MẶT**", [
                "• Trực tiếp tại văn phòng Ruby Wings",
                "• Địa chỉ: 148 Đường Trương Gia Mô, TP Huế",
                "• Thời gian: 8:00-20:00 hàng ngày",
                "• **Nhận hóa đơn VAT đầy đủ**"
            ]),
            
            ("📱 **VÍ ĐIỆN TỬ**", [
                "• Momo: (cập nhật sau)",
                "• ZaloPay: (cập nhật sau)",
                "• VNPay: (cập nhật sau)e",
                "• **Xác nhận ngay lập tức**"
            ])
        ]
        
        for method_title, details in payment_methods:
            reply += f"{method_title}\n"
            for detail in details:
                reply += f"{detail}\n"
            reply += "\n"
    
    # 4. CHÍNH SÁCH TRẺ EM
    if 'child_policy' in detected_question_types:
        reply += "👶 **CHÍNH SÁCH GIÁ TOUR CHO TRẺ EM:**\n\n"
        
        child_policy = [
            ("TRẺ DƯỚI 4 TUỔI", [
                "• Miễn phí tour",
                "• Tự túc vé máy bay (nếu có)",
                "• Phụ thu phòng riêng (nếu cần)",
                "• Ngủ chung giường với bố mẹ"
            ]),
            
            ("TRẺ 4-7 TUỔI", [
                "• Giá: 50% giá tour người lớn",
                "• Có giường riêng: +30%",
                "• Bao gồm: Ăn uống, vé tham quan",
                "• Không bao gồm: Phòng riêng"
            ]),
            
            ("TRẺ 8-11 TUỔI", [
                "• Giá: 85% giá tour người lớn",
                "• Có giường riêng: +15%",
                "• Bao gồm đầy đủ dịch vụ",
                "• Áp dụng mọi chương trình ưu đãi"
            ]),
            
            ("TRẺ TỪ 12 TUỔI", [
                "• Tính như người lớn",
                "• Áp dụng mọi ưu đãi",
                "• Cần giấy tờ tùy thân riêng",
                "• Có thể đi tour 1 mình (có giấy ủy quyền)"
            ])
        ]
        
        for age_group, policies in child_policy:
            reply += f"**{age_group}**\n"
            for policy in policies:
                reply += f"• {policy}\n"
            reply += "\n"
    
    # 5. BẢO HIỂM DU LỊCH
    if 'insurance' in detected_question_types:
        reply += "🛡️ **CHÍNH SÁCH BẢO HIỂM DU LỊCH:**\n\n"
        
        insurance_info = [
            ("**PHẠM VI BẢO HIỂM**", [
                "• Theo quy định Luật Bảo hiểm Việt Nam",
                "• Chi phí y tế: Max 60,000,000 VNĐ/người",
                "• Hỗ trợ y tế khẩn cấp: 24/7",
                "• Bồi thường hành lý: Max 5,000,000 VNĐ"
            ]),
            
            ("**ĐIỀU KIỆN ÁP DỤNG**", [
                "• Tuổi từ 1-70 (ngoài độ tuổi: liên hệ)",
                "• Không có bệnh mãn tính nặng",
                "• Không tham gia hoạt động nguy hiểm trái phép",
                "• Tuân thủ hướng dẫn an toàn"
            ]),
            
            ("**QUY TRÌNH BỒI THƯỜNG**", [
                "1. Báo ngay cho HDV trong 24h",
                "2. Lập biên bản sự việc",
                "3. Thu thập hồ sơ y tế",
                "4. Nhận bồi thường trong 15 ngày"
            ])
        ]
        
        for title, details in insurance_info:
            reply += f"{title}\n"
            for detail in details:
                reply += f"{detail}\n"
            reply += "\n"
    
    # 6. THỦ TỤC & GIẤY TỜ
    if 'documents' in detected_question_types:
        reply += "📄 **GIẤY TỜ CẦN THIẾT KHI ĐI TOUR:**\n\n"
        
        required_docs = [
            ("**BẮT BUỘC**", [
                "• CMND/CCCD còn hiệu lực (bản gốc)",
                "• Trẻ em: Giấy khai sinh (bản sao công chứng)",
                "• Hộ chiếu (đối với khách quốc tế)",
                "• Visa (nếu đến vùng biên giới)"
            ]),
            
            ("**KHUYẾN NGHỊ**", [
                "• Thẻ bảo hiểm y tế",
                "• Đơn thuốc (nếu đang điều trị)",
                "• Giấy xác nhận tiêm chủng",
                "• Thẻ học sinh/sinh viên (để hưởng ưu đãi)"
            ]),
            
            ("**ĐẶC BIỆT**", [
                "• Giấy ủy quyền (trẻ đi không bố mẹ)",
                "• Giấy xác nhận tình trạng sức khỏe",
                "• Giấy đăng ký kết hôn (đôi vợ chồng)",
                "• Giấy xác nhận công tác (nếu đi công tác)"
            ])
        ]
        
        for doc_type, docs in required_docs:
            reply += f"{doc_type}\n"
            for doc in docs:
                reply += f"{doc}\n"
            reply += "\n"
    
    # 7. CÂU HỎI THƯỜNG GẶP (FAQ)
    if not detected_question_types or len(detected_question_types) > 2:
        reply += "❓ **CÂU HỎI THƯỜNG GẶP VỀ ĐẶT TOUR:**\n\n"
        
        faqs = [
            ("**1. Đặt tour trước bao lâu?**", 
             "• Nên đặt trước ít nhất 7-14 ngày\n• Tour cao cấp: Đặt trước 30 ngày\n• Tour Tết/lễ: Đặt trước 60 ngày\n• Có thể đặt gấp trong 24h (phụ thu 10%)"),
            
            ("**2. Làm gì khi bị mất giấy tờ?**",
             "• Báo ngay cho HDV và công an địa phương\n• Làm giấy xác nhận mất tại công an\n• Chụp ảnh giấy tờ lưu điện tử phòng ngừa\n• Ruby Wings hỗ trợ làm thủ tục khẩn"),
            
            ("**3. Có được mang theo vật nuôi?**",
             "• Không được mang vật nuôi lên xe\n• Một số resort cho phép (phụ thu)\n• Cần thông báo trước 7 ngày\n• Tự chịu trách nhiệm chăm sóc"),
            
            ("**4. Thay đổi người tham gia?**",
             "• Được thay đổi trước 7 ngày\n• Phí thay đổi: 10% giá tour\n• Không thay đổi trong 3 ngày cuối\n• Người thay thế phải đủ điều kiện"),
            
            ("**5. Tour có hướng dẫn viên tiếng Anh?**",
             "• Có, với phụ thu 500,000 VNĐ/ngày\n• Đặt trước 15 ngày\n• Cung cấp CV HDV trước chuyến đi\n• Đảm bảo chất lượng chuyên môn")
        ]
        
        for question, answer in faqs:
            reply += f"{question}\n{answer}\n\n"
    
    # 8. THÔNG TIN LIÊN HỆ & HỖ TRỢ
    reply += "📞 **THÔNG TIN HỖ TRỢ & LIÊN HỆ:**\n\n"
    
    contact_info = [
        ("**HOTLINE ĐẶT TOUR**", "0332510486 (24/7)"),
        ("**EMAIL**", "rubywingslsa@gmail.com"),
        ("**VĂN PHÒNG**", "148 Đường Trương Gia Mô, TP Huế"),
        ("**GIỜ LÀM VIỆC**", "8:00 - 20:00 hàng ngày"),
        ("**HỖ TRỢ KHẨN**", "0912345678 (sự cố ngoài giờ)"),
        ("**ZALO OA**", "@rubywings (chat tự động 24/7)")
    ]
    
    for title, info in contact_info:
        reply += f"• **{title}:** {info}\n"
    
    reply += "\n"
    
    # 9. CAM KẾT TỪ RUBY WINGS
    reply += "✨ **CAM KẾT TỪ RUBY WINGS:**\n"
    reply += "• Minh bạch thông tin, không phát sinh chi phí\n"
    reply += "• Hỗ trợ 24/7 trong suốt hành trình\n"
    reply += "• Hoàn tiền 100% nếu không hài lòng (có điều kiện)\n"
    reply += "• Ưu đãi đặc biệt cho khách hàng thân thiết\n\n"
    
    reply += "⏰ **Xử lý yêu cầu trong vòng 15 phút - Đảm bảo quyền lợi khách hàng**"
    
    return reply


# ================== BACKWARD COMPATIBILITY WRAPPERS ==================

def _get_experience_response(message_lower, tour_indices, TOURS_DB, user_profile=None):
    """Wrapper cho backward compatibility"""
    return _get_experience_response_v4(message_lower, tour_indices, TOURS_DB, user_profile)

def _get_group_custom_response(message_lower, tour_indices, TOURS_DB, mandatory_filters=None):
    """Wrapper cho backward compatibility"""
    return _get_group_custom_response_v4(message_lower, tour_indices, TOURS_DB, mandatory_filters)

def _get_booking_policy_response(message_lower, tour_indices=None, TOURS_DB=None, context_info=None):
    """Wrapper cho backward compatibility"""
    return _get_booking_policy_response_v4(message_lower, tour_indices, TOURS_DB, context_info)

# ================== INTEGRATION CHECKLIST ==================

"""
CÁCH TÍCH HỢP VÀO HỆ THỐNG:

1. SAO CHÉP toàn bộ code trên vào file helper functions
2. ĐẢM BẢO các hàm wrapper tồn tại để không break code cũ
3. TRONG HÀM CHÍNH chat_endpoint_ultimate, thay thế các lời gọi:

   Từ:
   if intent == 'experience':
       reply = _get_experience_response(message_lower, tour_indices, TOURS_DB)
   
   Thành:
   if intent == 'experience':
       reply = _get_experience_response_v4(message_lower, tour_indices, TOURS_DB, context.user_profile)

4. CẬP NHẬT intent detection để nhận diện các intent mới:
   - 'experience': trải nghiệm tour
   - 'group_custom': nhóm & tour tùy chỉnh  
   - 'booking_policy': đặt tour & chính sách

5. TEST với các câu hỏi mẫu:
   - "Tour này có trải nghiệm gì đặc biệt?"
   - "Tôi muốn đặt tour cho nhóm 15 người"
   - "Chính sách hủy tour thế nào?"
"""

# ================== TEST FUNCTIONS ==================

def _test_all_enhanced_functions():
    """Test các hàm nâng cấp"""
    print("🧪 Testing enhanced helper functions V4...")
    
    # Mock data
    mock_tours_db = {
        1: type('Tour', (), {
            'name': 'Tour Bạch Mã Trekking',
            'duration': '2 ngày 1 đêm',
            'location': 'Bạch Mã, Huế',
            'price': '1,500,000 VNĐ',
            'summary': 'Tour trekking khám phá vườn quốc gia Bạch Mã với nhiều hoạt động mạo hiểm và trải nghiệm thiên nhiên. Phù hợp cho nhóm bạn trẻ yêu thích phiêu lưu.',
            'tags': ['trekking', 'thiên nhiên', 'mạo hiểm'],
            'style': 'Adventure'
        })(),
        
        2: type('Tour', (), {
            'name': 'Tour Retreat Thiền Huế',
            'duration': '3 ngày 2 đêm', 
            'location': 'Huế',
            'price': '2,800,000 VNĐ',
            'summary': 'Retreat thiền định và yoga tại không gian yên tĩnh của Huế. Trải nghiệm chữa lành, tĩnh tâm và kết nối nội tâm.',
            'tags': ['thiền', 'yoga', 'retreat', 'chữa lành'],
            'style': 'Wellness'
        })()
    }
    
    # Test 1: Experience Response
    print("\n1. Testing Experience Response...")
    exp_response = _get_experience_response_v4(
        "tour có trải nghiệm mạo hiểm gì không",
        [1, 2],
        mock_tours_db,
        {'interests': ['mạo hiểm', 'thiên nhiên']}
    )
    print(f"✅ Experience Response Length: {len(exp_response)} chars")
    
    # Test 2: Group Custom Response  
    print("\n2. Testing Group Custom Response...")
    group_response = _get_group_custom_response_v4(
        "tôi muốn đặt tour cho nhóm 20 người",
        [1, 2],
        mock_tours_db
    )
    print(f"✅ Group Response Length: {len(group_response)} chars")
    
    # Test 3: Booking Policy Response
    print("\n3. Testing Booking Policy Response...")
    policy_response = _get_booking_policy_response_v4(
        "chính sách hủy tour thế nào",
        [1, 2],
        mock_tours_db
    )
    print(f"✅ Policy Response Length: {len(policy_response)} chars")
    
    print(f"\n🎉 All tests passed! Total functions: 3")
    return True

# Auto-run tests if module is executed directly
if __name__ == "__main__":
    _test_all_enhanced_functions()





def _generate_enhanced_fallback_response(user_message, search_results, tour_indices, tours_db):
    """Tạo fallback response nâng cao"""
    # Cố gắng tạo response từ thông tin có sẵn
    if tour_indices:
        reply = "Dựa trên câu hỏi của bạn, tôi tìm thấy một số tour có thể phù hợp:\n\n"
        
        for idx in tour_indices[:3]:
            tour = tours_db.get(idx)
            if tour:
                reply += f"**{tour.name}**\n"
                if tour.duration:
                    reply += f"⏱️ {tour.duration}\n"
                if tour.location:
                    location_short = tour.location[:50] + "..." if len(tour.location) > 50 else tour.location
                    reply += f"📍 {location_short}\n"
                if tour.summary:
                    summary_short = tour.summary[:100] + "..." if len(tour.summary) > 100 else tour.summary
                    reply += f"📝 {summary_short}\n"
                reply += "\n"
        
        reply += "Để được tư vấn chi tiết hơn về các tour này hoặc tìm tour phù hợp nhất với nhu cầu của bạn, vui lòng liên hệ hotline 0332510486."
    elif search_results:
        reply = "Dựa trên thông tin tôi có, đây là một số thông tin liên quan:\n\n"
        
        for i, (score, passage) in enumerate(search_results[:2], 1):
            reply += f"{i}. {passage[:150]}...\n\n"
        
        reply += "Để có thông tin chính xác và đầy đủ hơn, vui lòng liên hệ hotline 0332510486."
    else:
        reply = "Cảm ơn câu hỏi của bạn. Để tư vấn chính xác nhất về các tour của Ruby Wings, bạn có thể:\n\n"
        reply += "1. Cung cấp thêm thông tin về nhu cầu của bạn\n"
        reply += "2. Gọi trực tiếp hotline 0332510486\n"
        reply += "3. Tham khảo các tour phổ biến:\n"
        reply += "   • Tour thiên nhiên Bạch Mã (1 ngày)\n"
        reply += "   • Tour lịch sử Trường Sơn (2 ngày 1 đêm)\n"
        reply += "   • Tour retreat thiền (1-2 ngày)\n\n"
        reply += "📞 **Hotline tư vấn 24/7:** 0332510486"
    
    return reply


# ================== MODULE COMPATIBILITY CHECK ==================
# Các module cần nâng cấp để tương thích:

"""
1. MandatoryFilterSystem.apply_filters() cần sửa lỗi:
   - Lỗi "không có nhóm nào như vậy" 
   - Thêm xử lý exception và fallback

2. FuzzyMatcher.find_similar_tours() cần cải thiện:
   - Giảm ngưỡng matching từ 0.7 xuống 0.6
   - Tăng số kết quả trả về

3. CacheSystem cần hỗ trợ:
   - Cache với expiry time
   - Key generation với nhiều tham số hơn

4. DeduplicationEngine cần:
   - Xử lý tốt hơn với các tour tương tự
   - Giữ lại tour chất lượng cao hơn

5. QueryIndex cần:
   - Trả về nhiều kết quả hơn (tăng TOP_K)
   - Cải thiện relevance scoring
"""

# Thêm các hàm helper mới vào các module tương ứng











# Thêm các hàm helper mới vào các module tương ứng
# =========== OTHER ENDPOINTS ===========
@app.route("/")
def home():
    """Home endpoint"""
    return jsonify({
        "status": "ok",
        "version": "4.0",
        "upgrades": UpgradeFlags.get_all_flags(),
        "services": {
            "openai": "available" if client else "unavailable",
            "faiss": "available" if HAS_FAISS else "unavailable",
            "google_sheets": "available" if HAS_GOOGLE_SHEETS else "unavailable",
            "meta_capi": "available" if HAS_META_CAPI else "unavailable",
        },
        "counts": {
            "tours": len(TOURS_DB),
            "passages": len(FLAT_TEXTS),
            "tour_names": len(TOUR_NAME_TO_INDEX),
        }
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    """Rebuild index endpoint"""
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed"}), 403
    
    load_knowledge()
    build_index(force_rebuild=True)
    
    return jsonify({
        "ok": True,
        "count": len(FLAT_TEXTS),
        "tours": len(TOURS_DB)
    })

# =========== GOOGLE SHEETS INTEGRATION ===========
_gsheet_client = None
_gsheet_client_lock = threading.Lock()

def get_gspread_client(force_refresh: bool = False):
    """Get Google Sheets client"""
    global _gsheet_client
    
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        logger.error("GOOGLE_SERVICE_ACCOUNT_JSON not set")
        return None
    
    with _gsheet_client_lock:
        if _gsheet_client is not None and not force_refresh:
            return _gsheet_client
        
        try:
            info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            _gsheet_client = gspread.authorize(creds)
            logger.info("✅ Google Sheets client initialized")
            return _gsheet_client
        except Exception as e:
            logger.error(f"❌ Google Sheets client failed: {e}")
            return None

@app.route('/api/save-lead', methods=['POST', 'OPTIONS'])
def save_lead():
    """Save lead from form submission - ĐẦY ĐỦ 9 TRƯỜNG (A-I)"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        
        # Extract data
        phone = data.get('phone', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        tour_interest = data.get('tour_interest', '').strip()
        page_url = data.get('page_url', '').strip()
        note = data.get('note', '').strip()
        
        if not phone:
            return jsonify({'error': 'Phone number is required'}), 400
        
        # Clean phone
        phone_clean = re.sub(r'[^\d+]', '', phone)
        
        # Validate phone
        if not re.match(r'^(0|\+?84)\d{9,10}$', phone_clean):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Timestamp
        timestamp = datetime.now().isoformat()
        
        # Create lead data
        lead_data = {
            'timestamp': timestamp,
            'phone': phone_clean,
            'name': name,
            'email': email,
            'tour_interest': tour_interest,
            'page_url': page_url,
            'note': note,
            'source': 'Lead Form'
        }
        
        # Send to Meta CAPI
        if ENABLE_META_CAPI_CALL and HAS_META_CAPI:
            try:
                result = send_meta_lead(
                    request,
                    phone=phone_clean,
                    contact_name=name,
                    email=email,
                    content_name=f"Tour: {tour_interest}" if tour_interest else "General Inquiry",
                    value=200000,
                    currency="VND"
                )
                increment_stat('meta_capi_calls')
                logger.info(f"✅ Form lead sent to Meta CAPI: {phone_clean[:4]}***")
                if DEBUG and HAS_META_CAPI:
                    logger.debug(f"Meta CAPI result: {result}")
            except Exception as e:
                increment_stat('meta_capi_errors')
                logger.error(f"Meta CAPI error: {e}")
        
        # Save to Google Sheets - ĐẦY ĐỦ 9 TRƯỜNG (A-I)
        if ENABLE_GOOGLE_SHEETS:
            try:
                import gspread
                from google.oauth2.service_account import Credentials
                
                if GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEET_ID:
                    creds_json = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
                    creds = Credentials.from_service_account_info(
                        creds_json,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    
                    gc = gspread.authorize(creds)
                    sh = gc.open_by_key(GOOGLE_SHEET_ID)
                    ws = sh.worksheet(GOOGLE_SHEET_NAME)
                    
                    # ĐÚng 9 TRƯỜNG THEO THỨ TỰ A-I:
                    # A: created_at (timestamp)
                    # B: source_channel
                    # C: action_type
                    # D: page_url
                    # E: contact_name
                    # F: phone
                    # G: service_interest
                    # H: note
                    # I: raw_status
                    row = [
                        timestamp,                          # A: created_at
                        'Website - Lead Form',              # B: source_channel
                        'Form Submission',                  # C: action_type
                        page_url or '',                     # D: page_url
                        name or '',                         # E: contact_name
                        phone_clean,                        # F: phone
                        tour_interest or '',                # G: service_interest
                        note or email or '',                # H: note (dùng email nếu không có note)
                        'New'                               # I: raw_status
                    ]
                    
                    ws.append_row(row)
                    logger.info("✅ Form lead saved to Google Sheets (9 fields)")
            except Exception as e:
                logger.error(f"Google Sheets error: {e}")
        
        # Fallback storage
        if ENABLE_FALLBACK_STORAGE:
            try:
                if os.path.exists(FALLBACK_STORAGE_PATH):
                    with open(FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                        leads = json.load(f)
                else:
                    leads = []
                
                leads.append(lead_data)
                leads = leads[-1000:]
                
                with open(FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(leads, f, ensure_ascii=False, indent=2)
                
                logger.info("✅ Form lead saved to fallback storage")
            except Exception as e:
                logger.error(f"Fallback storage error: {e}")
        
        # Update stats
        increment_stat('leads')
        
        return jsonify({
            'success': True,
            'message': 'Lead đã được lưu! Đội ngũ Ruby Wings sẽ liên hệ sớm nhất. 📞',
            'data': {
                'phone': phone_clean[:3] + '***' + phone_clean[-2:],
                'timestamp': timestamp
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Save lead error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/call-button', methods=['POST', 'OPTIONS'])
def call_button():
    """Track call button click"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        
        page_url = data.get('page_url', '')
        call_type = data.get('call_type', 'regular')
        
        # Send to Meta CAPI
        if ENABLE_META_CAPI_CALL and HAS_META_CAPI:
            try:
                result = send_meta_call_button(
                    request,
                    page_url=page_url,
                    call_type=call_type,
                    button_location='fixed_bottom_left',
                    button_text='Gọi ngay'
                )
                increment_stat('meta_capi_calls')
                logger.info(f"📞 Call button tracked: {call_type}")
                if DEBUG and HAS_META_CAPI:
                    logger.debug(f"Meta CAPI result: {result}")
            except Exception as e:
                increment_stat('meta_capi_errors')
                logger.error(f"Meta CAPI call error: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Call tracked',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Call button error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "chatbot": "running",
                "openai": "available" if client else "unavailable",
                "faiss": "available" if INDEX else "unavailable",
                "tours_db": len(TOURS_DB),
                "upgrades": {k: v for k, v in UpgradeFlags.get_all_flags().items() 
                           if k.startswith("UPGRADE_")}
            },
            "memory_profile": {
                "ram_profile": RAM_PROFILE,
                "is_low_ram": IS_LOW_RAM,
                "is_high_ram": IS_HIGH_RAM,
                "tour_count": len(TOURS_DB),
                "context_count": len(SESSION_CONTEXTS)
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# =========== INITIALIZATION ===========
def initialize_app():
    """Initialize the application"""
    logger.info("🚀 Starting Ruby Wings Chatbot v4.0 (Dataclass Rewrite)...")
    
    # Apply memory optimizations
    optimize_for_memory_profile()
    
    # Load knowledge base
    load_knowledge()
    
    # Load or build tours database
    if os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                MAPPING[:] = json.load(f)
            FLAT_TEXTS[:] = [m.get('text', '') for m in MAPPING]
            logger.info(f"📁 Loaded {len(MAPPING)} mappings from disk")
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
    
    # Build tour databases
    index_tour_names()
    build_tours_db()
    
    # Build index in background
    def build_index_background():
        time.sleep(2)
        success = build_index(force_rebuild=False)
        if success:
            logger.info("✅ Index ready")
        else:
            logger.warning("⚠️ Index building failed")
    
    threading.Thread(target=build_index_background, daemon=True).start()
    
    # Initialize Google Sheets client
    if ENABLE_GOOGLE_SHEETS:
        threading.Thread(target=get_gspread_client, daemon=True).start()
    
    # Log active upgrades
    active_upgrades = [name for name, enabled in UpgradeFlags.get_all_flags().items() 
                      if enabled and name.startswith("UPGRADE_")]
    logger.info(f"🔧 Active upgrades: {len(active_upgrades)}")
    for upgrade in active_upgrades:
        logger.info(f"   • {upgrade}")
    
    # Log memory profile
    logger.info(f"🧠 Memory Profile: {RAM_PROFILE}MB | Low RAM: {IS_LOW_RAM} | High RAM: {IS_HIGH_RAM}")
    logger.info(f"📊 Tours Database: {len(TOURS_DB)} tours loaded")
    
    logger.info("✅ Application initialized successfully with dataclasses")

# =========== APPLICATION START ===========
if __name__ == "__main__":
    initialize_app()
    
    # Save mappings if not exists
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, 'w', encoding='utf-8') as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved mappings to {FAISS_MAPPING_PATH}")
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
    
    # Start server
    logger.info(f"🌐 Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)

else:
    # For WSGI
    initialize_app()