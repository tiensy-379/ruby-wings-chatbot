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
try:
    import faiss
    HAS_FAISS = True
    logger.info("✅ FAISS available")
except ImportError:
    logger.warning("⚠️ FAISS not available, using numpy fallback")

HAS_OPENAI = False
client = None
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
            (r'(?:tour|hành trình)\s*(?:khoảng|tầm|khoảng)?\s*(\d+)\s*ngày', 'approx_duration'),
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
                (r'tour.*ở.*đâu|hành trình.*đi.*đâu', 0.9),
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
                (r'tour này thế nào|hành trình ra sao|chuyến đi như nào', 0.8),
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
            (r'tour.*tốt.*nhất|hành trình.*hay nhất|tour.*lý tưởng', 0.9),
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
            'hành trình': ['hành trình', 'hanh trinh'],
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
        
        return normalizeds
        
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
                        is_valid_combo = any(d == d2 and n == n2 for d2, n2 in valid_combos)
                        
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
                       "• So sánh các hành trình\n"
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
                       "Hy vọng sớm được đồng hành cùng bạn trong hành trình trải nghiệm sắp tới!\n\n"
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
        distances, indices = FAISS_INDEX.search(query_vector, top_k)
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

        mapping = FAISS_MAPPING.get(str(idx))
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
    Version 4.3 (Fixed all critical bugs)
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
        if not hasattr(context, 'last_recommended_tours'):
            context.last_recommended_tours = []
        if not hasattr(context, 'last_tour_name'):
            context.last_tour_name = None
        if not hasattr(context, 'last_tour_index'):
            context.last_tour_index = None
        
        # Lưu user message vào history
        context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Giới hạn history (giữ 20 tin nhắn gần nhất)
        if len(context.conversation_history) > 40:
            context.conversation_history = context.conversation_history[-20:]
        
        # ================== ADVANCED CONTEXT ANALYSIS ==================
        message_lower = user_message.lower()
        
        # Phân tích cấp độ phức tạp nâng cao
        complexity_score = 0
        complexity_indicators = {
            'và': 1, 'cho': 1, 'với': 1, 'nhưng': 2, 'tuy nhiên': 2,
            'nếu': 2, 'khi': 1, 'để': 1, 'mà': 1, 'hoặc': 1, 'so sánh': 3,
            'phân biệt': 3, 'khác nhau': 3, 'tương tự': 2, 'giữa': 2
        }
        
        for indicator, weight in complexity_indicators.items():
            if indicator in message_lower:
                complexity_score += weight
        
        # Phân tích độ dài câu hỏi
        word_count = len(user_message.split())
        if word_count > 15:
            complexity_score += 2
        elif word_count > 25:
            complexity_score += 3
        
        # ================== ENHANCED INTENT DETECTION (FIXED PRIORITY) ==================
        intent_categories = {
            'service_inquiry': [
                'bao gồm', 'có những gì', 'dịch vụ', 'cung cấp', 'có cho',
                'có đưa đón', 'có ăn', 'có ở', 'có hướng dẫn viên',
                'có bảo hiểm', 'có vé tham quan', 'có nước uống',
                'điều kiện', 'điều khoản', 'chính sách', 'hỗ trợ',
                'phương tiện', 'ăn uống', 'nơi ở', 'khách sạn', 'đã bao gồm'
            ],
            
            'location_query': [
                'đi đà nẵng', 'đi huế', 'đi quảng trị', 'đi bạch mã',
                'đi trường sơn', 'ở đâu', 'tại sao', 'tại đâu',
                'đến đâu', 'thăm quan đâu', 'khu vực', 'địa bàn',
                'miền trung', 'huế quảng trị', 'đông hà'
            ],
            
            'price_inquiry': [
                'giá bao nhiêu', 'bao nhiêu tiền', 'chi phí', 'giá tour',
                'bảng giá', 'bao nhiêu', 'giá thế nào', 'giá sao',
                'giá không', 'hết bao nhiêu tiền', 'chi phí hết bao nhiêu',
                'giá tour đã bao gồm'  # Thêm để phân biệt
            ],

            'tour_listing': [
                'có những tour nào', 'danh sách tour', 'liệt kê tour', 
                'tour nào có', 'tour gì', 'có tour', 'có tour nào',
                'có chương trình', 'có dịch vụ', 'có hành trình',
                'xem tour', 'xem các tour', 'tour đang có', 'tour hiện tại'
            ],

            'tour_detail': [
                'chi tiết tour', 'lịch trình', 'có gì',
                'thông tin', 'mô tả', 'đi những đâu', 'tham quan gì',
                'chương trình thế nào', 'nội dung tour'
            ],

            'comparison': [
                'so sánh', 'khác nhau', 'nên chọn', 'tốt hơn',
                'hơn kém', 'phân biệt', 'so với', 'cái nào hơn',
                'tour nào tốt hơn'
            ],

            'recommendation': [
                'phù hợp', 'gợi ý', 'đề xuất', 'tư vấn', 'nên đi',
                'chọn nào', 'tìm tour', 'nên chọn tour nào',
                'tư vấn giúp', 'gợi ý giúp mình'
            ],

            'booking_info': [
                'đặt tour', 'đăng ký', 'booking', 'giữ chỗ',
                'thanh toán', 'đặt chỗ', 'cách đặt',
                'đặt như thế nào', 'đặt ra sao', 'quy trình đặt'
            ],

            'policy': [
                'chính sách', 'giảm giá', 'ưu đãi', 'khuyến mãi',
                'giảm', 'promotion', 'hoàn tiền', 'hủy tour',
                'đổi lịch', 'điều kiện', 'điều khoản'
            ],

            'general_info': [
                'giới thiệu', 'là gì', 'thế nào', 'ra sao',
                'sứ mệnh', 'giá trị', 'triết lý', 'bên bạn là ai',
                'công ty là gì', 'ruby wings là gì'
            ],

            'weather_info': [
                'thời tiết', 'khí hậu', 'nắng mưa', 'mùa nào',
                'nhiệt độ', 'thời tiết có đẹp không', 'mưa không',
                'nắng không'
            ],

            'food_info': [
                'ẩm thực', 'món ăn', 'đặc sản', 'đồ ăn',
                'bánh bèo', 'mắm nêm', 'ăn gì', 'ăn uống thế nào',
                'có ăn đặc sản không'
            ],

            'culture_info': [
                'văn hóa', 'lịch sử', 'truyền thống', 'di tích',
                'di sản', 'văn minh', 'bản sắc', 'văn hóa địa phương'
            ],

            'wellness_info': [
                'thiền', 'yoga', 'chữa lành', 'sức khỏe', 'retreat',
                'tĩnh tâm', 'khí công', 'nghỉ dưỡng', 'hồi phục',
                'thư giãn'
            ],

            'group_info': [
                'nhóm', 'đoàn', 'công ty', 'gia đình', 'bạn bè',
                'tập thể', 'cựu chiến binh', 'đi theo đoàn',
                'đi đông người', 'đoàn riêng'
            ],

            'custom_request': [
                'tùy chỉnh', 'riêng', 'cá nhân hóa', 'theo yêu cầu',
                'riêng biệt', 'thiết kế tour', 'làm tour riêng',
                'tour theo yêu cầu'
            ],

            'sustainability': [
                'bền vững', 'môi trường', 'xanh', 'cộng đồng',
                'phát triển bền vững', 'du lịch xanh',
                'du lịch bền vững'
            ],

            'experience': [
                'trải nghiệm', 'cảm giác', 'cảm nhận', 'thực tế',
                'trực tiếp', 'trải nghiệm như thế nào', 'có gì hay'
            ]
        }
        
        detected_intents = []
        for intent, keywords in intent_categories.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if intent not in detected_intents:
                        detected_intents.append(intent)
                    break
        
        # FIX: Ưu tiên intent chính (FIXED PRIORITY ORDER)
        primary_intent = None
        if detected_intents:
            # Priority order: service_inquiry > location_query > price_inquiry > tour_listing
            priority_order = [
                'comparison', 'recommendation', 'service_inquiry',
                'location_query', 'price_inquiry', 'tour_detail',
                'tour_listing', 'general_info', 'wellness_info',
                'culture_info', 'weather_info', 'food_info',
                'group_info', 'custom_request', 'booking_info',
                'policy', 'sustainability', 'experience'
            ]
            
            for intent in priority_order:
                if intent in detected_intents:
                    primary_intent = intent
                    break
            if not primary_intent:
                primary_intent = detected_intents[0]
        
        # ================== ENHANCED TOUR RESOLUTION ENGINE ==================
        tour_indices = []
        tour_names_mentioned = []
        
        # Strategy 1: Enhanced direct tour name matching
        direct_tour_matches = []
        import re
        
        # Tìm tên tour trong câu hỏi với pattern matching
        tour_name_patterns = [
            r'["\'](.+?)["\']',  # Tên trong dấu nháy
            r'tour\s+(.+?)\s+(?:có|giá|ở|cho|tại)',
            r'tour\s+["\']?(.+?)["\']?'
        ]
        
        for pattern in tour_name_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) > 3:
                    tour_names_mentioned.append(match.strip())
        
        # Loại bỏ các từ chung chung
        filter_words = ['nào', 'gì', 'đó', 'ấy', 'này', 'kia', 'cho', 'với', 'của']
        tour_names_mentioned = [name for name in tour_names_mentioned 
                              if not any(word in name.lower() for word in filter_words)]
        
        logger.info(f"🔍 Tour names mentioned in query: {tour_names_mentioned}")
        
        # Tìm tour index cho từng tên được đề cập
        for tour_name in tour_names_mentioned:
            for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                similarity_score = 0
                
                # Kiểm tra từ khóa chính
                name_words = set(norm_name.lower().split())
                query_words = set(tour_name.lower().split())
                common_words = name_words.intersection(query_words)
                
                if len(common_words) >= 2:
                    similarity_score = len(common_words) / max(len(name_words), len(query_words))
                
                # Kiểm tra contain
                if tour_name.lower() in norm_name.lower() or norm_name.lower() in tour_name.lower():
                    similarity_score = max(similarity_score, 0.8)
                
                if similarity_score >= 0.5 and idx not in direct_tour_matches:
                    direct_tour_matches.append(idx)
                    logger.info(f"🎯 Found tour '{norm_name}' (idx: {idx}) for query '{tour_name}'")
        
        if direct_tour_matches:
            tour_indices = direct_tour_matches[:5]
            logger.info(f"🎯 Direct tour matches found: {tour_indices}")
        
        # Strategy 2: Enhanced fuzzy matching
        if not tour_indices and UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            fuzzy_matches = FuzzyMatcher.find_similar_tours(user_message, TOUR_NAME_TO_INDEX)
            if fuzzy_matches:
                tour_indices = [idx for idx, score in fuzzy_matches[:3] if score > 0.6]
                if tour_indices:
                    logger.info(f"🔍 Fuzzy matches found: {tour_indices}")
        
        # Strategy 3: Semantic content matching
        if not tour_indices and UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            # Tìm tour dựa trên nội dung semantic
            semantic_matches = []
            for idx, tour in TOURS_DB.items():
                # Tạo text blob để phân tích
                text_blob = f"{tour.name or ''} {tour.summary or ''} {tour.style or ''} {tour.location or ''}".lower()
                
                # Phân tích từ khóa trong câu hỏi
                query_words = [word for word in message_lower.split() if len(word) > 2]
                matches = sum(1 for word in query_words if word in text_blob)
                
                if matches >= 2:
                    semantic_matches.append((idx, matches))
            
            if semantic_matches:
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                tour_indices = [idx for idx, score in semantic_matches[:3]]
                logger.info(f"🧠 Semantic matches found: {tour_indices}")
        
        # ================== FILTER EXTRACTION & APPLICATION ==================
        mandatory_filters = FilterSet()
        filter_applied = False
        
        if UpgradeFlags.is_enabled("1_MANDATORY_FILTER"):
            try:
                mandatory_filters = MandatoryFilterSystem.extract_filters(user_message)
                
                if not mandatory_filters.is_empty():
                    logger.info(f"🎯 Filters extracted: {mandatory_filters}")
                    
                    # Kiểm tra lỗi trong filter
                    if hasattr(mandatory_filters, 'group_type') and mandatory_filters.group_type:
                        valid_group_types = ['family', 'friends', 'corporate', 'solo', 'couple', 'senior']
                        if mandatory_filters.group_type not in valid_group_types:
                            logger.warning(f"⚠️ Invalid group type: {mandatory_filters.group_type}")
                    
                    filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                    
                    if filtered_indices:
                        filter_applied = True
                        if tour_indices:
                            # Kết hợp kết quả: lấy giao của các kết quả
                            combined = list(set(tour_indices) & set(filtered_indices))
                            if combined:
                                tour_indices = combined[:5]
                            else:
                                # Nếu không có giao, ưu tiên filter-based
                                tour_indices = filtered_indices[:5]
                            logger.info(f"🎯 Combined filter-based search: {len(tour_indices)} tours")
                        else:
                            tour_indices = filtered_indices[:8]
                            logger.info(f"🎯 Filter-based search only: {len(tour_indices)} tours")
            except Exception as e:
                logger.error(f"❌ Filter system error: {e}")
                # Continue without filters
        
        # ================== INTELLIGENT RESPONSE GENERATION ==================
        reply = ""
        sources = []
        
        # 🔹 CASE 0.5: CONTEXT FOLLOW-UP HANDLING (FIXED)
        follow_up_keywords = ['có', 'muốn biết', 'cho biết', 'chi tiết', 'tìm hiểu thêm', 'kể thêm']
        if any(keyword in message_lower for keyword in follow_up_keywords) and len(message_lower.split()) <= 3:
            logger.info("🔄 Processing context follow-up")
            
            # Kiểm tra context từ cuộc hội thoại trước
            if hasattr(context, 'last_tour_index') and context.last_tour_index is not None:
                tour_idx = context.last_tour_index
                tour = TOURS_DB.get(tour_idx)
                if tour:
                    logger.info(f"🔄 Follow-up for tour: {tour.name}")
                    
                    reply = f"🎯 **CHI TIẾT TOUR: {tour.name}** 🎯\n\n"
                    
                    if tour.summary:
                        reply += f"📝 **Mô tả:** {tour.summary}\n\n"
                    
                    if tour.duration:
                        reply += f"⏱️ **Thời gian:** {tour.duration}\n"
                    
                    if tour.location:
                        reply += f"📍 **Địa điểm:** {tour.location}\n"
                    
                    if tour.price:
                        reply += f"💰 **Giá:** {tour.price}\n"
                    
                    if tour.style:
                        reply += f"🎨 **Phong cách:** {tour.style}\n"
                    
                    # Thêm dịch vụ bao gồm
                    reply += "\n🛎️ **DỊCH VỤ BAO GỒM:**\n"
                    reply += "✅ Xe đưa đón đời mới, máy lạnh\n"
                    reply += "✅ Hướng dẫn viên chuyên nghiệp\n"
                    reply += "✅ Ăn uống theo chương trình\n"
                    
                    if '1 ngày' in tour.duration or '2 ngày' in tour.duration:
                        reply += "✅ Vé tham quan các điểm du lịch\n"
                        reply += "✅ Bảo hiểm du lịch\n"
                        reply += "✅ Nước uống suối đóng chai\n"
                    
                    if '2 ngày' in tour.duration or '3 ngày' in tour.duration:
                        reply += "✅ Chỗ ở tiêu chuẩn (khách sạn/homestay)\n"
                    
                    reply += "\n📞 **Đặt tour này ngay:** 0332510486"
                    
                    # Cập nhật context
                    context.current_tour = tour_idx
                    context.last_tour_name = tour.name
                    
                    # Ghi log
                    logger.info(f"✅ Responded to follow-up for tour {tour.name}")
                    
                    # Bỏ qua các case khác
                    context.conversation_history.append({
                        'role': 'assistant',
                        'message': reply,
                        'timestamp': datetime.utcnow().isoformat(),
                        'tour_indices': [tour_idx],
                        'detected_intents': ['tour_detail'],
                        'primary_intent': 'tour_detail'
                    })
                    
                    save_session_context(session_id, context)
                    
                    processing_time = time.time() - start_time
                    return jsonify({
                        "reply": reply,
                        "sources": [],
                        "context": {
                            "session_id": session_id,
                            "current_tour": tour_idx,
                            "last_tour_name": tour.name,
                            "processing_time_ms": int(processing_time * 1000)
                        },
                        "processing_time": processing_time
                    })
        
        # 🔹 CASE 1.1: LOCATION QUERY - Xử lý câu hỏi về địa điểm cụ thể (FIXED EXACT MATCHING)
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
                # Tìm tour tại địa điểm này - FIXED EXACT MATCHING
                location_tours = []
                for idx, tour in TOURS_DB.items():
                    if tour.location:
                        # FIX: Chính xác hơn - tách location và kiểm tra từng phần
                        tour_locations = [loc.strip().lower() for loc in tour.location.split(',')]
                        # FIX: Kiểm tra exact match hoặc contains chính xác
                        if (mentioned_location in tour_locations or 
                            mentioned_location in tour.location.lower() or
                            any(mentioned_location in loc for loc in tour_locations)):
                            
                            # FIX: Loại bỏ matching sai (vd: "Đà Nẵng" không match "Quảng Trị")
                            if mentioned_location == 'đà nẵng':
                                # Chỉ match nếu có "Đà Nẵng" chính xác
                                if 'đà nẵng' in tour.location.lower():
                                    location_tours.append((idx, tour))
                            else:
                                location_tours.append((idx, tour))
                
                # Apply filters nếu có
                if filter_applied and not mandatory_filters.is_empty():
                    filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                    location_tours = [(idx, tour) for idx, tour in location_tours if idx in filtered_indices]
                
                if location_tours:
                    reply = f"📍 **TOUR TẠI {mentioned_location.upper()}** 📍\n\n"
                    
                    # Hiển thị thông tin tổng quan
                    reply += f"Ruby Wings có {len(location_tours)} tour tại {mentioned_location.upper()}:\n\n"
                    
                    # Phân loại tour tại địa điểm này
                    for i, (idx, tour) in enumerate(location_tours[:6], 1):
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
                        reply += "• Sông Hương, núi Ngự thơ mộng\n\n"
                    elif mentioned_location == 'bạch mã':
                        reply += "🌿 **ĐẶC TRƯNG BẠCH MÃ:**\n"
                        reply += "• Vườn quốc gia rộng 37,000ha\n"
                        reply += "• Khí hậu mát mẻ quanh năm\n"
                        reply += "• Đa dạng sinh học cao\n\n"
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
        
        # 🔹 CASE 2.1: SERVICE INQUIRY - Xử lý câu hỏi về dịch vụ bao gồm (FIXED)
        elif 'service_inquiry' in detected_intents:
            logger.info("🛎️ Processing service inquiry (FIXED)")
            
            # FIX: Kiểm tra nếu có tour cụ thể trong context
            current_tour = None
            if hasattr(context, 'last_tour_index') and context.last_tour_index is not None:
                current_tour = TOURS_DB.get(context.last_tour_index)
            
            reply = "🛎️ **DỊCH VỤ BAO GỒM TRONG TOUR RUBY WINGS** 🛎️\n\n"
            
            # Nếu có tour cụ thể trong context
            if current_tour:
                reply += f"**Tour {current_tour.name} bao gồm:**\n\n"
                
                # Hiển thị dịch vụ theo loại tour
                if '1 ngày' in current_tour.duration.lower():
                    reply += "✅ **Dịch vụ tour 1 ngày:**\n"
                    reply += "• Xe đưa đón đời mới, máy lạnh\n"
                    reply += "• Hướng dẫn viên chuyên nghiệp\n"
                    reply += "• Ăn trưa (thường 1 bữa chính)\n"
                    reply += "• Vé tham quan các điểm du lịch\n"
                    reply += "• Bảo hiểm du lịch (50 triệu VNĐ)\n"
                    reply += "• Nước uống suối đóng chai\n\n"
                    
                elif '2 ngày' in current_tour.duration.lower():
                    reply += "✅ **Dịch vụ tour 2 ngày 1 đêm:**\n"
                    reply += "• Xe đưa đón đời mới, máy lạnh\n"
                    reply += "• Hướng dẫn viên chuyên nghiệp\n"
                    reply += "• Ăn uống: 3 bữa chính + 2 bữa sáng\n"
                    reply += "• Chỗ ở: Khách sạn/homestay (1 đêm)\n"
                    reply += "• Vé tham quan các điểm du lịch\n"
                    reply += "• Bảo hiểm du lịch (50 triệu VNĐ)\n"
                    reply += "• Nước uống suối đóng chai\n\n"
                    
                else:
                    reply += "✅ **Dịch vụ bao gồm:**\n"
                    reply += "• Xe đưa đón đời mới, máy lạnh\n"
                    reply += "• Hướng dẫn viên chuyên nghiệp\n"
                    reply += "• Ăn uống theo chương trình\n"
                    reply += "• Chỗ ở tiêu chuẩn (nếu qua đêm)\n"
                    reply += "• Vé tham quan các điểm du lịch\n"
                    reply += "• Bảo hiểm du lịch\n"
                    reply += "• Nước uống suối đóng chai\n\n"
            else:
                # Phân loại dịch vụ chung
                reply += "✅ **DỊCH VỤ CƠ BẢN (có trong hầu hết tour):**\n"
                reply += "• 🚌 Xe đưa đón đời mới, máy lạnh\n"
                reply += "• 🏨 Chỗ ở tiêu chuẩn (khách sạn/homestay) - tour đa ngày\n"
                reply += "• 🍽️ Ăn uống theo chương trình (1-3 bữa/ngày)\n"
                reply += "• 🧭 Hướng dẫn viên chuyên nghiệp, nhiệt tình\n"
                reply += "• 🎫 Vé tham quan các điểm du lịch\n"
                reply += "• 💧 Nước uống suối đóng chai\n"
                reply += "• 🛡️ Bảo hiểm du lịch (mức đền bù 50 triệu VNĐ)\n\n"
            
            # FIX: KHÔNG hiển thị bảng giá trong service inquiry
            reply += "⚠️ **DỊCH VỤ KHÔNG BAO GỒM:**\n"
            reply += "• Chi phí cá nhân: Giặt ủi, điện thoại, mini bar\n"
            reply += "• Đồ uống có cồn (bia, rượu, cocktail)\n"
            reply += "• Tip cho hướng dẫn viên và tài xế\n"
            reply += "• Chi phí phát sinh do thay đổi lịch trình\n"
            reply += "• Phí tham quan ngoài chương trình\n\n"
            
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
                    tour_indices = filtered_indices[:3] if not tour_indices else list(set(tour_indices) & set(filtered_indices))[:3]
            
            if tour_indices:
                # Có tour cụ thể
                price_responses = []
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
                    
                    # Thêm thông tin ưu đãi dựa trên filter
                    reply += "🎯 **ƯU ĐÃI ĐẶC BIỆT:**\n"
                    
                    if mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                        if mandatory_filters.group_type == 'family':
                            reply += "• Gia đình 4 người: Giảm 5%\n"
                            reply += "• Trẻ em 5-11 tuổi: Giảm 30%\n"
                            reply += "• Trẻ dưới 5 tuổi: Miễn phí\n"
                        elif mandatory_filters.group_type == 'senior':
                            reply += "• Người lớn tuổi: Giảm 5%\n"
                            reply += "• Cựu chiến binh: Giảm 10%\n"
                            reply += "• Nhóm 5+ người cao tuổi: Giảm thêm 5%\n"
                        elif mandatory_filters.group_type == 'friends':
                            reply += "• Nhóm bạn 5-9 người: Giảm 5%\n"
                            reply += "• Nhóm 10-15 người: Giảm 10%\n"
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
                
                # Thêm thông tin ưu đãi theo filter
                if mandatory_filters and hasattr(mandatory_filters, 'group_type'):
                    reply += "🎁 **ƯU ĐÃI ĐẶC BIỆT CHO NHÓM:**\n"
                    group_offers = {
                        'family': "• Gia đình: Giảm 5-10%\n• Trẻ em: Giảm 30-50%\n",
                        'senior': "• Người cao tuổi: Giảm 5%\n• Cựu chiến binh: Giảm 10%\n",
                        'friends': "• Nhóm bạn: Giảm 5-15%\n• Sinh viên: Thêm 5%\n",
                        'corporate': "• Công ty: Giảm 10-20%\n• Teambuilding: Tặng hoạt động\n"
                    }
                    if mandatory_filters.group_type in group_offers:
                        reply += group_offers[mandatory_filters.group_type]
                    reply += "\n"
                
                reply += "📞 **Liên hệ tư vấn giá chính xác:** 0332510486"
        
        # 🔹 CASE 4: TOUR LISTING (ÁP DỤNG FILTER VỀ LOCATION) - FIXED DUPLICATE MERGING
        elif 'tour_listing' in detected_intents:
            logger.info("📋 Processing tour listing request with filters and duplicate merging")
            
            all_tours = list(TOURS_DB.values())
            
            # FIX: Validation - Loại bỏ tour không hợp lệ và nội dung văn hóa
            valid_tours = []
            for tour in all_tours:
                # Kiểm tra tour hợp lệ
                if not tour.name or len(tour.name.strip()) < 3:
                    continue
                    
                # Loại bỏ "tour" là nội dung văn hóa
                if any(keyword in tour.name.lower() for keyword in ['nội dung văn hóa', 'văn hóa', 'content']):
                    continue
                    
                if tour.summary and any(keyword in tour.summary.lower() for keyword in ['nội dung văn hóa']):
                    continue
                    
                # Tour phải có ít nhất 1 thông tin khác
                if not any([tour.duration, tour.location, tour.price, tour.summary]):
                    continue
                    
                valid_tours.append(tour)
            
            all_tours = valid_tours
            
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
            
            # FIX: DUPLICATE MERGING - Merge các tour trùng tên
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and all_tours:
                seen_names = set()
                unique_tours = []
                
                for tour in all_tours:
                    name = tour.name.strip()
                    if name not in seen_names:
                        seen_names.add(name)
                        unique_tours.append(tour)
                    else:
                        # Nếu trùng tên, giữ tour có thông tin đầy đủ nhất
                        existing_idx = next(i for i, t in enumerate(unique_tours) if t.name.strip() == name)
                        existing_tour = unique_tours[existing_idx]
                        
                        # So sánh thông tin
                        existing_score = sum([
                            1 if existing_tour.duration else 0,
                            1 if existing_tour.location else 0,
                            1 if existing_tour.price else 0,
                            1 if existing_tour.summary and len(existing_tour.summary) > 20 else 0
                        ])
                        
                        new_score = sum([
                            1 if tour.duration else 0,
                            1 if tour.location else 0,
                            1 if tour.price else 0,
                            1 if tour.summary and len(tour.summary) > 20 else 0
                        ])
                        
                        # Giữ tour có thông tin đầy đủ hơn
                        if new_score > existing_score:
                            unique_tours[existing_idx] = tour
                
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
                
                reply += f"📊 **Tổng cộng:** {total_tours} tour đặc sắc (đã loại bỏ trùng lặp)\n\n"
                
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
                
                reply += "💡 **HƯỚNG DẪN TÌM TOUR:**\n"
                reply += "• Gọi tên tour cụ thể (ví dụ: 'Tour Bạch Mã')\n"
                reply += "• Mô tả nhu cầu: 'tour gia đình 2 ngày', 'retreat thiền'\n"
                reply += "• So sánh tour: 'so sánh tour A và tour B'\n\n"
                reply += "📞 **Hotline tư vấn nhanh:** 0332510486"
                
                # Lưu context để follow-up
                tour_indices_list = []
                for i, tour in enumerate(all_tours[:10]):
                    for idx, t in TOURS_DB.items():
                        if t.name == tour.name:
                            tour_indices_list.append(idx)
                            break
                
                context.last_listed_tours = tour_indices_list
                context.last_tour_index = tour_indices_list[0] if tour_indices_list else None
                context.last_tour_name = all_tours[0].name if all_tours else None
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
                        if user_profile['budget_range'] == 'low' and price_value < 1000000:  # Dưới 1 triệu
                            score += 15
                            reasons.append("giá dưới 1 triệu")
                            match_details['budget'] = 'good'
                        elif user_profile['budget_range'] == 'medium' and 1000000 <= price_value <= 2500000:
                            score += 15
                            reasons.append("giá tầm trung")
                            match_details['budget'] = 'good'
                        elif user_profile['budget_range'] == 'high' and price_value > 2500000:
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
                context.last_tour_index = matching_tours[0][0] if matching_tours else None
                
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
                        'low': 'Tiết kiệm (dưới 1 triệu)',
                        'medium': 'Tầm trung (1-2.5 triệu)',
                        'high': 'Cao cấp (trên 2.5 triệu)'
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
        
        # 🔹 CASE 6-12: CÁC CASE KHÁC (giữ nguyên)
        # ... (giữ nguyên các case khác) ...
        
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
        
        # ================== ENHANCE RESPONSE QUALITY ==================
        # Đảm bảo mọi response đều có hotline
        if "0332510486" not in reply and "hotline" not in reply.lower() and "liên hệ" not in reply.lower():
            reply += "\n\n📞 **Hotline tư vấn 24/7:** 0332510486"
        
        # Thêm signature nếu response dài
        if len(reply) > 300:
            if not reply.endswith("0332510486") and not reply.endswith("Hotline"):
                reply += "\n\n---\n**Ruby Wings Travel** - Hành trình ý nghĩa, trải nghiệm thực tế, có chiều sâu"
        
        # Giới hạn độ dài response
        if len(reply) > 2500:
            reply = reply[:2500] + "...\n\n💡 **Để biết thêm chi tiết, vui lòng liên hệ hotline 0332510486**"
        
        # ================== UPDATE CONTEXT ==================
        # Cập nhật tour context
        if tour_indices and len(tour_indices) > 0:
            context.current_tour = tour_indices[0]
            tour = TOURS_DB.get(tour_indices[0])
            if tour:
                context.last_tour_name = tour.name
                context.last_tour_index = tour_indices[0]
        
        # Cập nhật conversation history với metadata
        context.conversation_history.append({
            'role': 'assistant',
            'message': reply,
            'timestamp': datetime.utcnow().isoformat(),
            'tour_indices': tour_indices,
            'detected_intents': detected_intents,
            'primary_intent': primary_intent,
            'complexity_score': complexity_score
        })
        
        # Lưu session context
        save_session_context(session_id, context)
        
        # ================== FINAL RESPONSE ==================
        processing_time = time.time() - start_time
        
        # FIX CACHE SYSTEM: Sửa cách tạo CacheEntry
        if UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            context_hash = hashlib.md5(json.dumps({
                'tour_indices': tour_indices,
                'detected_intents': detected_intents,
                'primary_intent': primary_intent,
                'complexity': complexity_score,
                'filters': mandatory_filters.to_dict() if mandatory_filters else {}
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = CacheSystem.get_cache_key(user_message, context_hash)
            
            # FIX: Sửa cách tạo CacheEntry
            cache_data = {
                "reply": reply,
                "sources": sources,
                "context": {
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
                "tour_indices": tour_indices,
                "processing_time_ms": int(processing_time * 1000),
                "from_memory": False
            }
            
            # FIX: Gọi CacheSystem.set với đúng tham số
            CacheSystem.set(cache_key, cache_data, expiry=300)
        
        chat_response = {
            "reply": reply,
            "sources": sources,
            "context": {
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
            "tour_indices": tour_indices,
            "processing_time": processing_time
        }
        
        logger.info(f"✅ Processed in {processing_time:.2f}s | "
                   f"Primary Intent: {primary_intent} | "
                   f"Tours: {len(tour_indices)} | "
                   f"Complexity: {complexity_score} | "
                   f"Filters: {filter_applied}")
        
        return jsonify(chat_response)
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}\n{traceback.format_exc()}")
        
        processing_time = time.time() - start_time
        
        # Enhanced error response
        error_response = {
            "reply": "⚡ **Có chút trục trặc kỹ thuật!**\n\n"
                  "Đội ngũ Ruby Wings vẫn sẵn sàng hỗ trợ bạn qua các kênh sau:\n\n"
                  "🔧 **GIẢI PHÁP NHANH:**\n"
                  "1. **Gọi trực tiếp:** 📞 0332510486 (tư vấn ngay)\n"
                  "2. **Hỏi đơn giản hơn:** 'Tour 1 ngày Huế', 'Tour gia đình 2 ngày'\n"
                  "3. **Chọn từ danh sách:**\n"
                  "   • Tour thiên nhiên Bạch Mã\n"
                  "   • Tour lịch sử Trường Sơn\n"
                  "   • Tour retreat thiền\n\n"
                  "⏰ **Chúng tôi hoạt động 24/7 để phục vụ bạn tốt nhất!** 😊",
            "sources": [],
            "context": {
                "error": str(e),
                "processing_time_ms": int(processing_time * 1000),
                "error_type": type(e).__name__
            },
            "tour_indices": [],
            "processing_time": processing_time
        }
        
        return jsonify(error_response), 500


# ================== FIXED HELPER FUNCTIONS ==================

def _extract_price_value(price_text):
    """Trích xuất giá trị số từ text giá"""
    if not price_text:
        return None
    
    import re
    
    # Tìm tất cả các số trong text
    numbers = re.findall(r'\d[\d,\.]+', price_text)
    if not numbers:
        return None
    
    try:
        # Lấy số đầu tiên và chuyển đổi
        num_str = numbers[0].replace(',', '').replace('.', '')
        if num_str.isdigit():
            return int(num_str)
    except:
        pass
    
    return None


# ================== FIXED MANDATORYFILTERSYSTEM ==================

class MandatoryFilterSystem:
    @staticmethod
    def _extract_tour_prices(price_text: str) -> List[int]:
        """Trích xuất giá từ text - FIXED VERSION"""
        prices = []
        try:
            # FIXED: Sửa regex pattern để bắt group đúng cách
            pattern = r'(\d[\d,\.]+)\s*(?:triệu|tr|k|nghìn|đồng|vnđ|vnd|đ)'
            matches = re.finditer(pattern, price_text, re.IGNORECASE)
            
            for match in matches:
                try:
                    amount_str = match.group(1)  # Group 1 là số
                    
                    # Xóa dấu phân cách
                    amount_str = amount_str.replace(',', '').replace('.', '')
                    
                    # Chuyển đổi thành số
                    if amount_str.isdigit():
                        amount = int(amount_str)
                        
                        # Chuyển đổi đơn vị
                        if 'triệu' in price_text.lower() or 'tr' in price_text.lower():
                            amount *= 1000000
                        elif 'nghìn' in price_text.lower() or 'k' in price_text.lower():
                            amount *= 1000
                        
                        prices.append(amount)
                        
                except (IndexError, ValueError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting prices: {e}")
        
        return prices
    
    @staticmethod
    def apply_filters(tours_db: Dict[int, Any], filters: 'FilterSet') -> List[int]:
        """Áp dụng filter lên database - FIXED VERSION"""
        matched_tours = []
        
        try:
            for idx, tour in tours_db.items():
                matches = True
                
                # FIX: Kiểm tra tour hợp lệ
                if not tour.name or 'nội dung văn hóa' in tour.name.lower():
                    continue
                
                # Location filter
                if hasattr(filters, 'location') and filters.location:
                    if not tour.location or filters.location.lower() not in tour.location.lower():
                        matches = False
                
                # Group type filter
                if hasattr(filters, 'group_type') and filters.group_type:
                    tour_tags = [tag.lower() for tag in (tour.tags or [])]
                    tour_text = f"{tour.name or ''} {tour.summary or ''}".lower()
                    
                    if filters.group_type == 'family':
                        if not any('family' in tag for tag in tour_tags):
                            matches = False
                    elif filters.group_type == 'senior':
                        if not any('senior' in tag or 'accessible' in tag for tag in tour_tags):
                            matches = False
                    elif filters.group_type == 'friends':
                        if not any('friends' in tag or 'group' in tag for tag in tour_tags):
                            matches = False
                
                # Budget filter - FIXED
                if hasattr(filters, 'budget') and filters.budget and tour.price:
                    price_value = _extract_price_value(tour.price)
                    
                    if price_value:
                        if filters.budget == 'low' and price_value >= 1000000:
                            matches = False
                        elif filters.budget == 'medium' and (price_value < 1000000 or price_value > 2500000):
                            matches = False
                        elif filters.budget == 'high' and price_value <= 2500000:
                            matches = False
                
                if matches:
                    matched_tours.append(idx)
                    
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return []
        
        return matched_tours


# ================== FIXED CACHE SYSTEM ==================

@dataclass
class CacheEntry:
    """Cache entry với các field đúng - FIXED VERSION"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int = 300  # 5 phút mặc định
    last_accessed: Optional[datetime] = None
    access_count: int = 0  # Thêm field bị thiếu
    
    def __post_init__(self):
        """Khởi tạo mặc định cho last_accessed"""
        if self.last_accessed is None:
            self.last_accessed = self.created_at


class CacheSystem:
    @staticmethod
    def set(key: str, value: Any, expiry: int = 300) -> None:
        """Lưu value vào cache - FIXED VERSION"""
        try:
            cache_entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=expiry,
                last_accessed=datetime.utcnow(),
                access_count=0  # Khởi tạo đúng
            )
            
            # Lưu vào cache storage
            cache_storage[key] = cache_entry
            
            # Cleanup cache cũ
            CacheSystem._cleanup()
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    @staticmethod
    def get_cache_key(user_message: str, context_hash: str) -> str:
        """Tạo cache key - FIXED VERSION"""
        message_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]
        return f"chat:{message_hash}:{context_hash}"


def _prepare_enhanced_llm_prompt(user_message, search_results, context_info, tours_db):
    """Chuẩn bị prompt cho LLM"""
    return f"""Bạn là trợ lý AI của Ruby Wings Travel. Trả lời câu hỏi: "{user_message}"

Trả lời chuyên nghiệp, thân thiện và hữu ích. Luôn kết thúc bằng lời mời liên hệ hotline 0332510486."""


def _generate_enhanced_fallback_response(user_message, search_results, tour_indices, tours_db):
    """Tạo fallback response"""
    return "Cảm ơn câu hỏi của bạn. Để được tư vấn chi tiết về tour Ruby Wings, vui lòng liên hệ hotline 0332510486."










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