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
TOP_K = int(os.environ.get("TOP_K", "5"))

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
        """
        if filters.is_empty() or not tours_db:
            return list(tours_db.keys())
        
        passing_tours = []
        
        try:
            for tour_idx, tour in tours_db.items():
                passes_all = True
                
                # PRICE FILTERING
                if passes_all and (filters.price_max is not None or filters.price_min is not None):
                    tour_price_text = tour.price or ""
                    if not tour_price_text:
                        if filters.price_max is not None or filters.price_min is not None:
                            passes_all = False
                    else:
                        tour_prices = MandatoryFilterSystem._extract_tour_prices(tour_price_text)
                        if not tour_prices:
                            passes_all = False
                        else:
                            min_tour_price = min(tour_prices)
                            max_tour_price = max(tour_prices)
                            
                            if filters.price_max is not None and min_tour_price > filters.price_max:
                                passes_all = False
                            if filters.price_min is not None and max_tour_price < filters.price_min:
                                passes_all = False
                
                # DURATION FILTERING
                if passes_all and (filters.duration_min is not None or filters.duration_max is not None):
                    duration_text = (tour.duration or "").lower()
                    tour_duration = MandatoryFilterSystem._extract_duration_days(duration_text)
                    
                    if tour_duration is not None:
                        if filters.duration_min is not None and tour_duration < filters.duration_min:
                            passes_all = False
                        if filters.duration_max is not None and tour_duration > filters.duration_max:
                            passes_all = False
                    else:
                        if filters.duration_min is not None or filters.duration_max is not None:
                            passes_all = False
                
                # LOCATION FILTERING
                if passes_all and (filters.location is not None or filters.near_location is not None):
                    tour_location = (tour.location or "").lower()
                    if filters.location is not None:
                        filter_location = filters.location.lower()
                        if filter_location not in tour_location:
                            passes_all = False
                    if filters.near_location is not None:
                        near_location = filters.near_location.lower()
                        if near_location not in tour_location:
                            passes_all = False
                
                if passes_all:
                    passing_tours.append(tour_idx)
            
            logger.info(f"🔍 After mandatory filtering: {len(passing_tours)}/{len(tours_db)} tours pass")
        except Exception as e:
            logger.error(f"❌ Error in apply_filters: {e}")
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
        Find tours with names similar to query
        """
        if not query or not tour_names:
            return []
        
        query_norm = FuzzyMatcher.normalize_vietnamese(query)
        if not query_norm:
            return []
        
        matches = []
        
        for tour_name, tour_idx in tour_names.items():
            tour_norm = FuzzyMatcher.normalize_vietnamese(tour_name)
            if not tour_norm:
                continue
            
            similarity = SequenceMatcher(None, query_norm, tour_norm).ratio()
            
            if query_norm in tour_norm or tour_norm in query_norm:
                similarity = min(similarity + 0.2, 1.0)
            
            query_words = set(query_norm.split())
            tour_words = set(tour_norm.split())
            common_words = query_words.intersection(tour_words)
            
            if common_words:
                word_boost = len(common_words) * 0.1
                similarity = min(similarity + word_boost, 1.0)
            
            if similarity >= FuzzyMatcher.SIMILARITY_THRESHOLD:
                matches.append((tour_idx, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"🔍 Fuzzy matching: '{query}' → {len(matches)} matches")
        return matches
    
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
    def set(key: str, value: Any):
        """Set item in cache"""
        with _cache_lock:
            cache_entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=UpgradeFlags.get_all_flags().get("CACHE_TTL_SECONDS", 300)
            )
            _response_cache[key] = cache_entry
            
            if len(_response_cache) > 1000:
                sorted_items = sorted(_response_cache.items(), 
                                     key=lambda x: x[1].created_at)
                for old_key in [k for k, _ in sorted_items[:200]]:
                    if old_key in _response_cache:
                        del _response_cache[old_key]

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

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, Dict]]:
    """Query the index"""
    if not query or INDEX is None:
        return []
    
    emb, _ = embed_text(query)
    if not emb:
        return []
    
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    
    try:
        if HAS_FAISS and isinstance(INDEX, faiss.Index):
            D, I = INDEX.search(vec, top_k)
        else:
            D, I = INDEX.search(vec, top_k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(MAPPING):
                results.append((float(score), MAPPING[idx]))
        
        return results
    except Exception as e:
        logger.error(f"Index search error: {e}")
        return []

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

# =========== ULTIMATE CHAT ENDPOINT - SIÊU THÔNG MINH ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint_ultimate_v2():
    """
    Chat endpoint siêu thông minh với context-aware mạnh mẽ, xử lý đa nhiệm và trí nhớ phiên làm việc
    """
    start_time = time.time()
    
    try:
        # ================== KHỞI TẠO THÔNG MINH ==================
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        session_id = extract_session_id(data, request.remote_addr)
        
        if not user_message:
            return jsonify({
                "reply": "🌟 **XIN CHÀO! TÔI LÀ TRỢ LÝ AI RUBY WINGS** 🌟\n\n"
                        "Tôi được trang bị trí thông minh cao cấp để:\n"
                        "• Hiểu context & ghi nhớ tour đang thảo luận\n"
                        "• So sánh chi tiết 2-3 tour cùng lúc\n"
                        "• Đề xuất tour chính xác theo nhu cầu\n"
                        "• Cảnh báo tour không phù hợp với đối tượng\n"
                        "• Xử lý đa tương tác trong 1 phiên\n\n"
                        "🔮 **Hãy hỏi tôi bất kỳ điều gì về 32 tour đặc sắc!**\n"
                        "📞 **Hotline 24/7:** 0332510486",
                "sources": [],
                "context": {},
                "processing_time": 0
            })
        
        # ================== HỆ THỐNG CONTEXT THÔNG MINH ==================
        context = get_session_context(session_id)
        
        # Khởi tạo context nâng cao
        if not hasattr(context, 'conversation_history'):
            context.conversation_history = []
        if not hasattr(context, 'current_tour'):
            context.current_tour = None
        if not hasattr(context, 'user_profile'):
            context.user_profile = {
                'group_size': None,
                'ages': [],
                'preferences': [],
                'budget_range': None,
                'duration_pref': None,
                'special_needs': []
            }
        if not hasattr(context, 'discussed_tours'):
            context.discussed_tours = []
        if not hasattr(context, 'comparison_mode'):
            context.comparison_mode = False
        
        # Lưu lịch sử thông minh
        context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Giới hạn history (giữ 15 tin nhắn gần nhất)
        if len(context.conversation_history) > 30:
            context.conversation_history = context.conversation_history[-15:]
        
        # ================== PHÂN TÍCH NGỮ NGHĨA NÂNG CAO ==================
        message_lower = user_message.lower()
        
        # Trích xuất thông tin người dùng thông minh
        extracted_info = _extract_user_info(user_message, context.user_profile)
        context.user_profile.update(extracted_info)
        
        # Phát hiện intent với độ chính xác cao
        intents = _detect_intents_with_llm(user_message, context.conversation_history[-3:])
        
        # ================== HỆ THỐNG TRÍ NHỚ TOUR ==================
        # Xác định tour đang được thảo luận
        current_tour_indices = _resolve_tour_reference(
            user_message, 
            context.current_tour, 
            context.discussed_tours,
            TOUR_NAME_TO_INDEX
        )
        
        # Nếu tìm thấy tour cụ thể, cập nhật context
        if current_tour_indices:
            context.current_tour = current_tour_indices[0]
            if context.current_tour not in context.discussed_tours:
                context.discussed_tours.append(context.current_tour)
        
        # ================== XỬ LÝ ĐA TÁC VỤ THÔNG MINH ==================
        reply = ""
        sources = []
        tour_indices = []
        
        # 🔹 TRƯỜNG HỢP 1: HỎI VỀ TOUR CỤ THỂ ĐANG THẢO LUẬN
        if ('price_inquiry' in intents or 'tour_detail' in intents) and context.current_tour:
            logger.info(f"💎 Processing specific tour inquiry: {context.current_tour}")
            
            tour = TOURS_DB.get(context.current_tour)
            if tour:
                if 'price_inquiry' in intents:
                    reply = f"💰 **GIÁ TOUR {tour.name.upper()}** 💰\n\n"
                    if tour.price:
                        # Phân tích cấu trúc giá thông minh
                        price_analysis = _analyze_price_structure(tour.price)
                        reply += f"**Mức giá:** {price_analysis['range']}\n"
                        if price_analysis['per_person']:
                            reply += f"**Giá/người:** {price_analysis['per_person']}\n"
                        if price_analysis['group_discount']:
                            reply += f"**Chiết khấu nhóm:** {price_analysis['group_discount']}\n"
                        
                        # Tính toán cho nhóm cụ thể
                        if context.user_profile['group_size']:
                            group_price = _calculate_group_price(
                                tour.price, 
                                context.user_profile['group_size'],
                                context.user_profile.get('is_veteran', False)
                            )
                            if group_price:
                                reply += f"\n**Dự kiến cho nhóm {context.user_profile['group_size']} người:** {group_price}\n"
                    else:
                        reply += "Giá tour linh hoạt theo số lượng và dịch vụ.\n"
                    
                    reply += "\n📊 **CHI TIẾT GIÁ TRỊ:**\n"
                    if tour.includes:
                        for i, item in enumerate(tour.includes[:5], 1):
                            reply += f"{i}. {item[:80]}{'...' if len(item) > 80 else ''}\n"
                    
                    reply += "\n🎯 **ƯU ĐÃI ĐẶC BIỆT:**\n"
                    reply += "• Nhóm 10+ người: Giảm 5-20%\n"
                    reply += "• Cựu chiến binh: Thêm 5%\n"
                    reply += "• Đặt sớm trước 30 ngày: Giảm 5-8%\n"
                    reply += "\n📞 **Liên hệ 0332510486 để nhận báo giá chính xác!**"
                
                elif 'tour_detail' in intents:
                    reply = f"📋 **CHI TIẾT TOUR: {tour.name}** 📋\n\n"
                    
                    # Thông tin cơ bản
                    if tour.summary:
                        reply += f"**Tổng quan:** {tour.summary}\n\n"
                    
                    if tour.duration:
                        reply += f"⏱️ **Thời gian:** {tour.duration}\n"
                    
                    if tour.location:
                        reply += f"📍 **Địa điểm:** {tour.location}\n"
                    
                    # Điểm nổi bật
                    if tour.includes:
                        reply += "\n✨ **ĐIỂM NỔI BẬT:**\n"
                        for i, item in enumerate(tour.includes[:8], 1):
                            reply += f"• {item}\n"
                    
                    # Lịch trình chi tiết (nếu có trong includes)
                    schedule_items = [item for item in (tour.includes or []) 
                                    if any(keyword in item.lower() for keyword in ['ngày', 'buổi', 'sáng', 'trưa', 'chiều', 'tối'])]
                    if schedule_items:
                        reply += "\n🗓️ **LỊCH TRÌNH CHI TIẾT:**\n"
                        for item in schedule_items[:5]:
                            reply += f"• {item}\n"
                    
                    # Phù hợp với ai
                    suitability = _analyze_tour_suitability(tour, context.user_profile)
                    reply += f"\n🎯 **ĐỐI TƯỢNG PHÙ HỢP:**\n{suitability}\n"
                    
                    # Cảnh báo nếu không phù hợp
                    warnings = _check_tour_warnings(tour, context.user_profile)
                    if warnings:
                        reply += f"\n⚠️ **LƯU Ý QUAN TRỌNG:**\n{warnings}\n"
                    
                    reply += "\n📞 **Tư vấn chi tiết & đặt tour:** 0332510486"
                
                tour_indices = [context.current_tour]
        
        # 🔹 TRƯỜNG HỢP 2: SO SÁNH TOUR THÔNG MINH
        elif 'comparison' in intents:
            logger.info("⚖️ Processing intelligent tour comparison")
            
            # Trích xuất tour names từ câu hỏi với độ chính xác cao
            tour_names = _extract_tour_names_for_comparison(user_message, TOUR_NAME_TO_INDEX)
            
            if not tour_names and context.discussed_tours:
                # Nếu không tìm thấy tên tour, dùng các tour đã thảo luận
                tour_indices = context.discussed_tours[-2:]  # Lấy 2 tour gần nhất
            elif tour_names:
                # Map tên tour sang indices
                for name in tour_names:
                    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                        if name.lower() in norm_name.lower():
                            tour_indices.append(idx)
                            break
            
            if len(tour_indices) >= 2:
                # Kích hoạt chế độ so sánh
                context.comparison_mode = True
                
                # Tạo bảng so sánh chi tiết
                comparison_data = []
                for idx in tour_indices[:3]:  # Tối đa 3 tour
                    tour = TOURS_DB.get(idx)
                    if tour:
                        comparison_data.append({
                            'name': tour.name,
                            'duration': tour.duration or 'N/A',
                            'location': tour.location or 'N/A',
                            'price': _extract_price_range(tour.price) if tour.price else 'Liên hệ',
                            'style': tour.style[:100] + '...' if tour.style and len(tour.style) > 100 else tour.style or 'N/A',
                            'suitability': _analyze_tour_suitability(tour, context.user_profile),
                            'highlights': tour.includes[:3] if tour.includes else [],
                            'tags': tour.tags or []
                        })
                
                if len(comparison_data) >= 2:
                    reply = "📊 **SO SÁNH CHI TIẾT TOUR** 📊\n\n"
                    
                    # Tạo bảng so sánh
                    headers = ["TIÊU CHÍ"] + [tour['name'][:25] for tour in comparison_data]
                    
                    # Các tiêu chí so sánh
                    criteria = [
                        ('⏱️ Thời gian', lambda t: t['duration']),
                        ('📍 Địa điểm', lambda t: t['location'][:30] if t['location'] else 'N/A'),
                        ('💰 Giá cả', lambda t: t['price']),
                        ('🎯 Phong cách', lambda t: t['style'][:30] + '...' if t['style'] and len(t['style']) > 30 else t['style']),
                        ('👥 Phù hợp', lambda t: t['suitability'][:40] + '...' if len(t['suitability']) > 40 else t['suitability'])
                    ]
                    
                    for criterion, get_value in criteria:
                        row = [criterion]
                        for tour in comparison_data:
                            value = get_value(tour)
                            row.append(value or 'N/A')
                        
                        # Format hàng
                        row_str = " | ".join([str(cell).ljust(25) for cell in row])
                        reply += f"{row_str}\n"
                        reply += "-" * (len(row) * 27) + "\n"
                    
                    # Phân tích sâu
                    reply += "\n🔍 **PHÂN TÍCH CHUYÊN SÂU:**\n"
                    
                    # So sánh giá
                    prices = []
                    for tour in comparison_data:
                        if 'price' in tour and isinstance(tour['price'], str):
                            nums = re.findall(r'[\d,.]+', tour['price'])
                            if nums:
                                try:
                                    clean_num = int(nums[0].replace(',', '').replace('.', ''))
                                    prices.append((tour['name'], clean_num))
                                except:
                                    pass
                    
                    if prices:
                        min_tour = min(prices, key=lambda x: x[1])
                        max_tour = max(prices, key=lambda x: x[1])
                        if max_tour[1] > min_tour[1] * 1.3:
                            reply += f"• **Tiết kiệm nhất:** {min_tour[0]} ({min_tour[1]:,}đ)\n"
                            reply += f"• **Đầy đủ trải nghiệm:** {max_tour[0]}\n"
                    
                    # So sánh độ phù hợp
                    for tour in comparison_data:
                        if 'gia đình' in tour['suitability'].lower() and 'trẻ em' in tour['suitability'].lower():
                            reply += f"• **Cho gia đình:** {tour['name']}\n"
                        if 'lịch sử' in tour['style'].lower() or any('history' in tag for tag in tour.get('tags', [])):
                            reply += f"• **Yếu tố lịch sử:** {tour['name']}\n"
                        if 'thiền' in tour['style'].lower() or any('meditation' in tag for tag in tour.get('tags', [])):
                            reply += f"• **Thiền & chữa lành:** {tour['name']}\n"
                    
                    # Khuyến nghị dựa trên profile
                    if context.user_profile['ages']:
                        age_warnings = []
                        for tour in comparison_data:
                            if any('trekking' in str(tour['highlights']).lower() or 'đi bộ' in str(tour['highlights']).lower()):
                                if any(age > 60 for age in context.user_profile['ages']):
                                    age_warnings.append(f"• {tour['name']} có trekking, cân nhắc với người lớn tuổi\n")
                        
                        if age_warnings:
                            reply += "\n⚠️ **CẢNH BÁO PHÙ HỢP:**\n" + "".join(age_warnings)
                    
                    reply += "\n💡 **LỜI KHUYÊN:**\n"
                    reply += "1. Xem xét yếu tố ưu tiên (giá, thời gian, trải nghiệm)\n"
                    reply += "2. Kiểm tra kỹ độ phù hợp với thành viên\n"
                    reply += "3. Liên hệ tư vấn để hiểu rõ hơn về từng tour\n\n"
                    reply += "📞 **Tư vấn chọn tour phù hợp:** 0332510486"
            
            else:
                reply = "Để so sánh tour, vui lòng cung cấp tên 2-3 tour cụ thể. Ví dụ:\n"
                reply += "• 'So sánh tour Bạch Mã và tour Trường Sơn'\n"
                reply += "• 'Tour nào tốt hơn giữa Mưa Đỏ và Ký ức?'\n"
                reply += "• 'Phân biệt tour 1 ngày và 2 ngày của Ruby Wings'"
        
        # 🔹 TRƯỜNG HỢP 3: ĐỀ XUẤT TOUR THÔNG MINH
        elif 'recommendation' in intents or any(word in message_lower for word in ['phù hợp', 'gợi ý', 'tư vấn']):
            logger.info("🎯 Processing intelligent recommendation")
            
            # Phân tích nhu cầu chi tiết
            requirements = _analyze_user_requirements(user_message, context.user_profile)
            
            # Tìm tour phù hợp với điểm số chi tiết
            scored_tours = []
            for idx, tour in TOURS_DB.items():
                score, reasons, warnings = _calculate_tour_match_score(
                    tour, 
                    requirements, 
                    context.user_profile
                )
                
                if score > 0:
                    scored_tours.append({
                        'index': idx,
                        'score': score,
                        'tour': tour,
                        'reasons': reasons,
                        'warnings': warnings,
                        'match_percentage': min(100, int(score * 10))
                    })
            
            # Sắp xếp theo điểm
            scored_tours.sort(key=lambda x: x['score'], reverse=True)
            
            if scored_tours:
                reply = "🎯 **ĐỀ XUẤT TOUR THÔNG MINH** 🎯\n\n"
                
                # Tour phù hợp nhất
                top_tour = scored_tours[0]
                reply += f"🏆 **PHÙ HỢP NHẤT ({top_tour['match_percentage']}%)**\n"
                reply += f"**{top_tour['tour'].name}**\n"
                
                if top_tour['reasons']:
                    reply += f"✅ **Lý do:** {', '.join(top_tour['reasons'][:3])}\n"
                
                if top_tour['tour'].duration:
                    reply += f"⏱️ {top_tour['tour'].duration} | "
                if top_tour['tour'].location:
                    reply += f"📍 {top_tour['tour'].location[:40]}\n"
                
                # Thông tin quan trọng
                if top_tour['tour'].price:
                    price_summary = _summarize_price(top_tour['tour'].price)
                    reply += f"💰 {price_summary}\n"
                
                if top_tour['warnings']:
                    reply += f"⚠️ **Lưu ý:** {top_tour['warnings'][0]}\n"
                
                reply += "\n"
                
                # Các lựa chọn khác (2-3 tour)
                other_tours = scored_tours[1:4]
                if other_tours:
                    reply += "📋 **LỰA CHỌN KHÁC:**\n"
                    for t in other_tours:
                        reply += f"• **{t['tour'].name}** ({t['match_percentage']}%)\n"
                        if t['tour'].duration:
                            reply += f"  ⏱️ {t['tour'].duration}"
                        if t['tour'].location:
                            reply += f" | 📍 {t['tour'].location[:25]}"
                        reply += "\n"
                
                # Khuyến nghị dựa trên phân tích sâu
                reply += "\n🔍 **PHÂN TÍCH CHUYÊN SÂU:**\n"
                
                # Kiểm tra tính phù hợp với từng thành viên
                if requirements.get('has_elderly') and requirements.get('has_children'):
                    family_tours = [t for t in scored_tours[:3] 
                                  if 'gia đình' in str(t['reasons']).lower()]
                    if family_tours:
                        reply += "• ✅ **Rất phù hợp cho gia đình đa thế hệ**\n"
                
                if requirements.get('needs_gentle'):
                    gentle_tours = [t for t in scored_tours[:3] 
                                  if any('nhẹ nhàng' in reason.lower() for reason in t['reasons'])]
                    if gentle_tours:
                        reply += "• ✅ **Nhịp độ nhẹ nhàng, không vội vã**\n"
                
                if requirements.get('budget_conscious'):
                    budget_tours = [t for t in scored_tours[:3] 
                                  if _is_budget_friendly(t['tour'].price)]
                    if budget_tours:
                        reply += "• ✅ **Giá cả hợp lý, tiết kiệm**\n"
                
                # Cảnh báo nếu có
                if requirements.get('has_children') and requirements.get('children_ages'):
                    for t in scored_tours[:2]:
                        if 'trekking' in str(t['tour'].includes).lower():
                            reply += f"• ⚠️ **{t['tour'].name} có trekking, cân nhắc với trẻ nhỏ**\n"
                
                reply += "\n💡 **BẠN CẦN BIẾT:**\n"
                reply += "• Mỗi tour có điểm mạnh riêng, phù hợp với nhu cầu khác nhau\n"
                reply += "• Có thể kết hợp hoặc tùy chỉnh tour theo yêu cầu\n"
                reply += "• Liên hệ tư vấn để được phân tích chi tiết hơn\n\n"
                reply += "📞 **Tư vấn chọn tour hoàn hảo:** 0332510486"
                
                tour_indices = [t['index'] for t in scored_tours[:3]]
            
            else:
                # Sử dụng AI để đề xuất thông minh khi không tìm thấy
                if client and HAS_OPENAI:
                    try:
                        profile_summary = _summarize_user_profile(context.user_profile)
                        
                        prompt = f"""Bạn là chuyên gia tư vấn tour Ruby Wings. Khách hàng có nhu cầu nhưng chưa tìm thấy tour phù hợp ngay.

THÔNG TIN KHÁCH:
{profile_summary}

YÊU CẦU CỤ THỂ:
{user_message}

THÔNG TIN RUBY WINGS:
- 32 tour đa dạng: lịch sử, thiền, thiên nhiên, biển đảo
- Tour 1 ngày đến 6 ngày
- Phù hợp mọi đối tượng

YÊU CẦU TRẢ LỜI:
1. Thừa nhận nhu cầu phức tạp
2. Đề nghị chia nhỏ nhu cầu để tư vấn tốt hơn
3. Gợi ý các loại tour có thể kết hợp
4. Mời liên hệ chuyên gia

Giọng văn: Chuyên nghiệp, thấu hiểu, nhiệt tình"""

                        response = client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": user_message}
                            ],
                            temperature=0.7,
                            max_tokens=400
                        )
                        
                        reply = response.choices[0].message.content if response.choices else ""
                        
                    except Exception as e:
                        logger.error(f"OpenAI recommendation error: {e}")
                        reply = "Nhu cầu của bạn khá đặc biệt. Để tôi tư vấn chính xác hơn:\n\n"
                        reply += "1. **Ưu tiên hàng đầu** của nhóm bạn là gì? (giá cả, trải nghiệm, an toàn)\n"
                        reply += "2. **Hoạt động yêu thích** nhất? (thiền, trekking, văn hóa, ẩm thực)\n"
                        reply += "3. **Hạn chế** nào cần lưu ý? (sức khỏe, dị ứng, sở thích)\n\n"
                        reply += "Hoặc liên hệ trực tiếp chuyên gia: 📞 0332510486"
                else:
                    reply = "Để tư vấn tour phù hợp nhất, vui lòng cung cấp thêm thông tin:\n"
                    reply += "• Số lượng và độ tuổi thành viên\n"
                    reply += "• Sở thích chính của nhóm\n"
                    reply += "• Ngân sách dự kiến\n"
                    reply += "• Thời gian có thể đi\n\n"
                    reply += "📞 **Hoặc gọi ngay:** 0332510486"
        
        # 🔹 TRƯỜNG HỢP 4: TÌM TOUR THEO TIÊU CHÍ PHỨC TẠP
        elif any(keyword in intents for keyword in ['family', 'senior', 'children', 'veteran', 'retreat']):
            logger.info("🎯 Processing complex criteria search")
            
            # Áp dụng bộ lọc thông minh
            filtered_tours = _apply_intelligent_filters(TOURS_DB, context.user_profile, user_message)
            
            if filtered_tours:
                reply = f"🔍 **TÌM THẤY {len(filtered_tours)} TOUR PHÙ HỢP** 🔍\n\n"
                
                # Nhóm tour theo loại
                tour_categories = {
                    'Gia đình & Nhẹ nhàng': [],
                    'Lịch sử & Tri ân': [],
                    'Thiền & Retreat': [],
                    'Thiên nhiên & Khám phá': []
                }
                
                for idx, tour in filtered_tours[:10]:  # Giới hạn 10 tour
                    category = _categorize_tour(tour)
                    tour_categories[category].append((idx, tour))
                
                # Hiển thị theo nhóm
                displayed_count = 0
                for category, tours in tour_categories.items():
                    if tours:
                        reply += f"**{category}:**\n"
                        for idx, tour in tours[:3]:  # 3 tour mỗi nhóm
                            displayed_count += 1
                            emoji = "👨‍👩‍👧‍👦" if 'gia đình' in category.lower() else \
                                   "🏛️" if 'lịch sử' in category.lower() else \
                                   "🕉️" if 'thiền' in category.lower() else "🌿"
                            
                            reply += f"{emoji} **{tour.name}**\n"
                            if tour.duration:
                                reply += f"   ⏱️ {tour.duration}"
                            if tour.location:
                                reply += f" | 📍 {tour.location[:25]}\n"
                            
                            # Điểm phù hợp
                            score, reasons, _ = _calculate_tour_match_score(
                                tour, 
                                _analyze_user_requirements(user_message, context.user_profile),
                                context.user_profile
                            )
                            if reasons:
                                reply += f"   ✅ {reasons[0][:50]}...\n"
                            reply += "\n"
                
                if displayed_count < len(filtered_tours):
                    reply += f"📊 **Và {len(filtered_tours) - displayed_count} tour khác phù hợp...**\n\n"
                
                reply += "💡 **MẸO CHỌN TOUR:**\n"
                reply += "1. Tour gia đình: Ưu tiên nhịp độ chậm, ít trekking\n"
                reply += "2. Tour lịch sử: Phù hợp giáo dục và tri ân\n"
                reply += "3. Tour retreat: Tập trung thiền và chữa lành\n"
                reply += "4. Tour thiên nhiên: Khám phá và trải nghiệm mạnh\n\n"
                reply += "📞 **Tư vấn chọn tour theo tiêu chí:** 0332510486"
                
                tour_indices = [idx for idx, _ in filtered_tours[:5]]
            
            else:
                reply = "Hiện chưa có tour nào đáp ứng đầy đủ các tiêu chí của bạn. Tuy nhiên:\n\n"
                reply += "✅ **GIẢI PHÁP THAY THẾ:**\n"
                reply += "1. **Tour tùy chỉnh:** Ruby Wings có thể thiết kế tour riêng\n"
                reply += "2. **Điều chỉnh tiêu chí:** Mở rộng một số yêu cầu\n"
                reply += "3. **Kết hợp nhiều tour:** Chọn 2 tour ngắn thay vì 1 tour dài\n\n"
                reply += "📞 **Liên hệ thiết kế tour riêng:** 0332510486"
        
        # 🔹 TRƯỜNG HỢP 5: THÔNG TIN CHUNG & TRIẾT LÝ
        elif 'general_info' in intents:
            logger.info("🏛️ Processing enhanced general info")
            
            if 'triết lý' in message_lower or 'chuẩn mực' in message_lower:
                reply = "✨ **TRIẾT LÝ RUBY WINGS: CHUẨN MỰC - CHÂN THÀNH - CÓ CHIỀU SÂU** ✨\n\n"
                reply += "**1. CHUẨN MỰC TRONG TỪNG TOUR:**\n"
                reply += "• **An toàn tuyệt đối:** Tiêu chuẩn cao nhất về an toàn\n"
                reply += "• **Dịch vụ chuẩn:** Từ xe đời mới đến hướng dẫn viên chuyên nghiệp\n"
                reply += "• **Minh bạch:** Giá cả rõ ràng, không phát sinh\n\n"
                
                reply += "**2. CHÂN THÀNH TRONG TỪNG KẾT NỐI:**\n"
                reply += "• **Giao tiếp thật:** Tư vấn trung thực, không ép mua\n"
                reply += "• **Đồng hành thật:** HDV tận tâm, hiểu khách hàng\n"
                reply += "• **Trải nghiệm thật:** Khám phá bản chất, không chỉ bề nổi\n\n"
                
                reply += "**3. CÓ CHIỀU SÂU TRONG TỪNG TRẢI NGHIỆM:**\n"
                reply += "• **Ý nghĩa:** Mỗi tour mang thông điệp riêng (tri ân, chữa lành, khám phá)\n"
                reply += "• **Giá trị:** Đọng lại bài học, cảm xúc, kỷ niệm\n"
                reply += "• **Chuyển hóa:** Giúp khách hàng thay đổi tích cực sau tour\n\n"
                
                reply += "🏆 **THỂ HIỆN TRONG TOUR:**\n"
                reply += "• Tour lịch sử: Tri ân sâu sắc, không chỉ tham quan\n"
                reply += "• Tour thiền: Chữa lành thực sự, không chỉ nghỉ dưỡng\n"
                reply += "• Tour gia đình: Kết nối thực chất, không chỉ vui chơi\n\n"
                reply += "📞 **Trải nghiệm triết lý này:** 0332510486"
            
            elif 'khác biệt' in message_lower or 'điểm mạnh' in message_lower:
                reply = "🚀 **ĐIỂM KHÁC BIỆT CỦA RUBY WINGS** 🚀\n\n"
                reply += "**1. TOUR CÓ CHIỀU SÂU:**\n"
                reply += "• Không chỉ tham quan, mà còn trải nghiệm ý nghĩa\n"
                reply += "• Mỗi tour có thông điệp riêng (tri ân, chữa lành, khám phá)\n"
                reply += "• Kết hợp thiền, khí công, trị liệu thiên nhiên\n\n"
                
                reply += "**2. CHUYÊN GIA THỰC SỰ:**\n"
                reply += "• HDV am hiểu sâu về lịch sử, văn hóa, thiền\n"
                reply += "• Có chuyên gia sức khỏe đồng hành trong tour retreat\n"
                reply += "• Kết nối với nhân chứng lịch sử, nghệ nhân địa phương\n\n"
                
                reply += "**3. LINH HOẠT CAO:**\n"
                reply += "• Thiết kế tour theo yêu cầu\n"
                reply += "• Điều chỉnh lịch trình phù hợp với nhóm\n"
                reply += "• Hỗ trợ 24/7 trong suốt hành trình\n\n"
                
                reply += "**4. GIÁ TRỊ BỀN VỮNG:**\n"
                reply += "• Tôn trọng văn hóa bản địa\n"
                reply += "• Bảo vệ môi trường điểm đến\n"
                reply += "• Hỗ trợ cộng đồng địa phương\n\n"
                
                reply += "🎯 **SO SÁNH VỚI CÔNG TY KHÁC:**\n"
                reply += "| Tiêu chí | Ruby Wings | Công ty thường |\n"
                reply += "|----------|------------|----------------|\n"
                reply += "| Độ sâu | ⭐⭐⭐⭐⭐ | ⭐⭐ |\n"
                reply += "| Cá nhân hóa | ⭐⭐⭐⭐⭐ | ⭐⭐ |\n"
                reply += "| Chuyên môn | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |\n"
                reply += "| Linh hoạt | ⭐⭐⭐⭐⭐ | ⭐⭐ |\n"
                reply += "| Giá trị | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |\n\n"
                reply += "📞 **Trải nghiệm sự khác biệt:** 0332510486"
            
            else:
                reply = "🏛️ **RUBY WINGS TRAVEL - HÀNH TRÌNH Ý NGHĨA** 🏛️\n\n"
                reply += "**SỨ MỆNH:** Lan tỏa giá trị sống Chuẩn mực - Chân thành - Có chiều sâu\n\n"
                reply += "**3 TRỤ CỘT CHÍNH:**\n"
                reply += "1. **TOUR LỊCH SỬ - TRI ÂN**\n"
                reply += "   • Kết nối quá khứ - hiện tại\n"
                reply += "   • Gặp gỡ nhân chứng, cựu chiến binh\n"
                reply += "   • Tham quan di tích với góc nhìn sâu\n\n"
                
                reply += "2. **TOUR RETREAT - CHỮA LÀNH**\n"
                reply += "   • Thiền, khí công, yoga giữa thiên nhiên\n"
                reply += "   • Tĩnh tâm, giảm stress, cân bằng năng lượng\n"
                reply += "   • Kết hợp trị liệu thiên nhiên\n\n"
                
                reply += "3. **TOUR TRẢI NGHIỆM - KHÁM PHÁ**\n"
                reply += "   • Văn hóa bản địa, ẩm thực đặc sắc\n"
                reply += "   • Thiên nhiên nguyên sơ, biển đảo hoang sơ\n"
                reply += "   • Hoạt động đội nhóm, teambuilding\n\n"
                
                reply += "**32 TOUR ĐA DẠNG:**\n"
                reply += "• Thời gian: 1 ngày đến 6 ngày\n"
                reply += "• Địa điểm: Huế, Quảng Trị, Bạch Mã, Trường Sơn, Phú Quốc, Đà Lạt...\n"
                reply += "• Đối tượng: Gia đình, nhóm, cá nhân, cựu chiến binh, học sinh\n\n"
                reply += "📞 **Kết nối với chúng tôi:** 0332510486"
        
        # 🔹 TRƯỜNG HỢP 6: XỬ LÝ CÂU HỎI NGOÀI PHẠM VI THÔNG MINH
        elif _is_out_of_scope(user_message):
            logger.info("🌐 Processing out-of-scope intelligently")
            
            if 'chuyện cười' in message_lower:
                reply = "😊 **VUI VẺ CÙNG RUBY WINGS!** 😊\n\n"
                reply += "Tôi chuyên về tư vấn tour, nhưng có thể chia sẻ điều thú vị:\n\n"
                reply += "🎭 **CÂU CHUYỆN VUI VỀ DU LỊCH:**\n"
                reply += "Một khách hỏi: 'Tour này có nhiều muỗi không?'\n"
                reply += "HDV đáp: 'Không ạ, chúng tôi đã dạy chúng cách lịch sự rồi!' 🦟😄\n\n"
                reply += "🎯 **THAY VÀO ĐÓ, BẠN CÓ THỂ:**\n"
                reply += "• Nghe những câu chuyện thú vị từ HDV trong tour\n"
                reply += "• Trải nghiệm không khí vui vẻ tại các đêm lửa trại\n"
                reply += "• Thưởng thức các tiết mục văn nghệ đặc sắc\n\n"
                reply += "📞 **Đặt tour để có trải nghiệm vui vẻ thực sự:** 0332510486"
            
            else:
                reply = "🌿 **TÔI CHUYÊN VỀ DU LỊCH TRẢI NGHIỆM** 🌿\n\n"
                reply += "Tôi có thể giúp bạn với:\n\n"
                reply += "🎯 **32 TOUR ĐẶC SẮC:**\n"
                reply += "• Lịch sử, văn hóa, thiên nhiên, biển đảo\n"
                reply += "• Thiền, retreat, chữa lành\n"
                reply += "• Gia đình, nhóm, cá nhân hóa\n\n"
                reply += "🔍 **THÔNG TIN CHI TIẾT:**\n"
                reply += "• Giá cả, lịch trình, địa điểm\n"
                reply += "• So sánh các tour\n"
                reply += "• Tư vấn theo nhu cầu riêng\n\n"
                reply += "💡 **HÃY HỎI TÔI VỀ:**\n"
                reply += "• 'Tour nào phù hợp cho gia đình có trẻ nhỏ?'\n"
                reply += "• 'So sánh tour Bạch Mã và Trường Sơn'\n"
                reply += "• 'Tour retreat 1 ngày giá bao nhiêu?'\n\n"
                reply += "📞 **Hotline tư vấn mọi thắc mắc:** 0332510486"
        
        # 🔹 TRƯỜNG HỢP 7: MẶC ĐỊNH - XỬ LÝ THÔNG MINH TỔNG HỢP
        else:
            logger.info("🤖 Processing with intelligent synthesis")
            
            # Tìm kiếm semantic nâng cao
            search_results = query_index(user_message, TOP_K * 2)  # Tìm nhiều hơn
            
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and search_results:
                search_results = DeduplicationEngine.deduplicate_passages(search_results)
            
            # Áp dụng bộ lọc thông minh
            filtered_indices = _apply_smart_filters_to_search(search_results, context.user_profile)
            
            if filtered_indices:
                tour_indices = filtered_indices[:5]
            elif search_results:
                # Lấy tour từ kết quả tìm kiếm
                tour_indices = []
                for score, content in search_results:
                    # Tìm tour index từ content
                    for idx, tour in TOURS_DB.items():
                        if tour.name in content and idx not in tour_indices:
                            tour_indices.append(idx)
                            if len(tour_indices) >= 3:
                                break
                    if len(tour_indices) >= 3:
                        break
            
            # Tạo prompt thông minh với context đầy đủ
            context_info = {
                'user_message': user_message,
                'user_profile': context.user_profile,
                'current_tour': context.current_tour,
                'discussed_tours': context.discussed_tours[-3:],  # 3 tour gần nhất
                'comparison_mode': context.comparison_mode,
                'tour_indices': tour_indices,
                'intents': intents
            }
            
            prompt = _prepare_llm_prompt_v2(user_message, search_results, context_info)
            
            # Gọi AI với context phong phú
            if client and HAS_OPENAI:
                try:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    # Thêm lịch sử hội thoại gần nhất
                    for msg in context.conversation_history[-4:-1]:  # 3 tin nhắn gần nhất (trừ cái cuối)
                        if msg['role'] in ['user', 'assistant']:
                            messages.insert(-1, {"role": msg['role'], "content": msg['message'][:200]})
                    
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=600,
                        top_p=0.9,
                        frequency_penalty=0.1,
                        presence_penalty=0.1
                    )
                    
                    if response.choices:
                        reply = response.choices[0].message.content or ""
                    else:
                        reply = _generate_intelligent_fallback(user_message, search_results, tour_indices, context_info)
                
                except Exception as e:
                    logger.error(f"OpenAI synthesis error: {e}")
                    reply = _generate_intelligent_fallback(user_message, search_results, tour_indices, context_info)
            else:
                reply = _generate_intelligent_fallback(user_message, search_results, tour_indices, context_info)
            
            sources = [m for _, m in search_results[:3]]
        
        # ================== NÂNG CAO CHẤT LƯỢNG PHẢN HỒI ==================
        # Đảm bảo tính nhất quán và hữu ích
        reply = _enhance_response_quality(reply, context)
        
        # Thêm hotline nếu chưa có
        if "0332510486" not in reply and "hotline" not in reply.lower():
            reply += "\n\n📞 **Hotline tư vấn chuyên sâu:** 0332510486"
        
        # Giới hạn độ dài thông minh
        if len(reply) > 2500:
            # Cắt nhưng giữ phần quan trọng
            important_parts = reply.split('\n\n')
            if len(important_parts) > 3:
                reply = '\n\n'.join(important_parts[:4]) + "...\n\n💡 **Để biết thêm chi tiết, vui lòng liên hệ hotline 0332510486**"
            else:
                reply = reply[:2500] + "...\n\n💡 **Để biết thêm chi tiết, vui lòng liên hệ hotline 0332510486**"
        
        # ================== CẬP NHẬT CONTEXT THÔNG MINH ==================
        # Cập nhật tour đang thảo luận
        if tour_indices and not context.current_tour:
            context.current_tour = tour_indices[0]
        
        # Lưu các tour đã đề cập
        for idx in tour_indices:
            if idx not in context.discussed_tours:
                context.discussed_tours.append(idx)
        
        # Giới hạn số lượng tour đã thảo luận
        if len(context.discussed_tours) > 10:
            context.discussed_tours = context.discussed_tours[-5:]
        
        # Lưu phản hồi vào lịch sử
        context.conversation_history.append({
            'role': 'assistant',
            'message': reply[:500],  # Lưu một phần để tiết kiệm
            'timestamp': datetime.utcnow().isoformat(),
            'tour_indices': tour_indices,
            'intents': intents
        })
        
        # ================== PHẢN HỒI CUỐI CÙNG ==================
        processing_time = time.time() - start_time
        
        chat_response = ChatResponse(
            reply=reply,
            sources=sources,
            context={
                "session_id": session_id,
                "current_tour": context.current_tour,
                "discussed_tours": context.discussed_tours[-3:],
                "user_profile": context.user_profile,
                "detected_intents": intents,
                "processing_time_ms": int(processing_time * 1000),
                "tours_found": len(tour_indices),
                "comparison_mode": context.comparison_mode
            },
            tour_indices=tour_indices,
            processing_time_ms=int(processing_time * 1000),
            from_memory=False
        )
        
        # Cache thông minh với key phức tạp
        if UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            context_hash = hashlib.md5(json.dumps({
                'tour_indices': tour_indices,
                'user_profile': context.user_profile,
                'current_tour': context.current_tour,
                'intents': intents
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = CacheSystem.get_cache_key(user_message, context_hash)
            CacheSystem.set(cache_key, chat_response.to_dict(), ttl=3600)  # 1 giờ
        
        logger.info(f"✅ Processed in {processing_time:.2f}s | "
                   f"Intents: {intents} | "
                   f"Tours: {len(tour_indices)} | "
                   f"Profile: {context.user_profile}")
        
        return jsonify(chat_response.to_dict())
    
    except Exception as e:
        logger.error(f"❌ Ultimate chat error: {e}\n{traceback.format_exc()}")
        
        processing_time = time.time() - start_time
        
        # Error response thông minh
        error_response = ChatResponse(
            reply="⚡ **XIN LỖI, CÓ CHÚT TRỤC TRẶC!** ⚡\n\n"
                  "Nhưng đừng lo, Ruby Wings vẫn sẵn sàng hỗ trợ bạn:\n\n"
                  "🔧 **GIẢI PHÁP NGAY:**\n"
                  "1. **Gọi trực tiếp:** 📞 0332510486 (ưu tiên khẩn cấp)\n"
                  "2. **Thử cách khác:**\n"
                  "   • Hỏi ngắn gọn hơn\n"
                  "   • Chỉ định tên tour cụ thể\n"
                  "   • Mô tả 1-2 nhu cầu chính\n\n"
                  "🎯 **HOẶC HỎI TÔI NGAY:**\n"
                  "• 'Tour 1 ngày cho gia đình'\n"
                  "• 'So sánh 2 tour phổ biến nhất'\n"
                  "• 'Tour giá dưới 1 triệu'\n\n"
                  "⏰ **Chúng tôi luôn sẵn sàng 24/7!** 🌟",
            sources=[],
            context={
                "error": str(e),
                "processing_time_ms": int(processing_time * 1000)
            },
            tour_indices=[],
            processing_time_ms=int(processing_time * 1000),
            from_memory=False
        )
        
        return jsonify(error_response.to_dict()), 500


# =========== CÁC HÀM HỖ TRỢ THÔNG MINH ===========

def _extract_user_info(message: str, existing_profile: dict) -> dict:
    """Trích xuất thông tin người dùng thông minh từ message"""
    info = {}
    message_lower = message.lower()
    
    # Số người
    people_patterns = [
        r'(\d+)\s*người',
        r'nhóm\s*(\d+)',
        r'(\d+)\s*thành viên',
        r'(\d+)\s*người tham gia'
    ]
    for pattern in people_patterns:
        match = re.search(pattern, message_lower)
        if match:
            info['group_size'] = int(match.group(1))
            break
    
    # Độ tuổi
    if any(word in message_lower for word in ['trẻ em', 'con nhỏ', 'trẻ', 'bé']):
        info['has_children'] = True
        # Ước tính tuổi từ context
        if 'tuổi' in message_lower:
            age_match = re.search(r'(\d+)\s*tuổi', message_lower)
            if age_match:
                info['children_ages'] = [int(age_match.group(1))]
        else:
            info['children_ages'] = [5, 15]  # Mặc định
    
    if any(word in message_lower for word in ['người lớn tuổi', 'cao tuổi', 'ông bà', 'cụ']):
        info['has_elderly'] = True
        info['needs_gentle'] = True
    
    # Sở thích
    interests = []
    if any(word in message_lower for word in ['thiên nhiên', 'rừng', 'núi', 'cây']):
        interests.append('nature')
    if any(word in message_lower for word in ['lịch sử', 'di tích', 'chiến tranh', 'tri ân']):
        interests.append('history')
    if any(word in message_lower for word in ['thiền', 'tĩnh tâm', 'yoga', 'chữa lành']):
        interests.append('meditation')
    if any(word in message_lower for word in ['biển', 'đảo', 'bãi biển']):
        interests.append('beach')
    if any(word in message_lower for word in ['ẩm thực', 'đồ ăn', 'món ngon']):
        interests.append('food')
    
    if interests:
        info['preferences'] = interests
    
    # Ngân sách
    budget_patterns = [
        r'(\d[\d,\.]+)\s*đồng',
        r'(\d[\d,\.]+)\s*vnđ',
        r'giá\s*(\d[\d,\.]+)',
        r'tầm\s*(\d[\d,\.]+)'
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, message_lower)
        if match:
            try:
                budget_str = match.group(1).replace(',', '').replace('.', '')
                budget = int(budget_str)
                if budget < 2000000:
                    info['budget_range'] = 'low'
                elif budget < 5000000:
                    info['budget_range'] = 'medium'
                else:
                    info['budget_range'] = 'high'
                break
            except:
                pass
    
    # Thời gian
    if '1 ngày' in message_lower or 'ngày' in message_lower:
        info['duration_pref'] = '1_day'
    elif '2 ngày' in message_lower:
        info['duration_pref'] = '2_days'
    elif '3 ngày' in message_lower:
        info['duration_pref'] = '3_days'
    
    # Nhu cầu đặc biệt
    special_needs = []
    if any(word in message_lower for word in ['cựu chiến binh', 'ccb', 'veteran']):
        special_needs.append('veteran')
        info['is_veteran'] = True
    if any(word in message_lower for word in ['nhẹ nhàng', 'không vội', 'chậm rãi']):
        special_needs.append('gentle_pace')
        info['needs_gentle'] = True
    if any(word in message_lower for word in ['retreat', 'tĩnh dưỡng', 'nghỉ ngơi']):
        special_needs.append('retreat')
    
    if special_needs:
        info['special_needs'] = special_needs
    
    return info


def _detect_intents_with_llm(message: str, conversation_history: list) -> list:
    """Phát hiện intent với độ chính xác cao sử dụng LLM"""
    intents = []
    message_lower = message.lower()
    
    # Intent detection cơ bản
    intent_patterns = {
        'price_inquiry': ['giá', 'bao nhiêu tiền', 'chi phí', 'có đắt không'],
        'tour_detail': ['chi tiết', 'lịch trình', 'có gì', 'bao gồm'],
        'comparison': ['so sánh', 'khác nhau', 'nên chọn', 'tốt hơn'],
        'recommendation': ['phù hợp', 'gợi ý', 'đề xuất', 'tư vấn'],
        'booking_info': ['đặt tour', 'đăng ký', 'booking', 'giữ chỗ'],
        'general_info': ['giới thiệu', 'là gì', 'thế nào', 'triết lý'],
        'family': ['gia đình', 'trẻ em', 'con nhỏ', 'bố mẹ'],
        'senior': ['người lớn tuổi', 'cao tuổi', 'ông bà'],
        'veteran': ['cựu chiến binh', 'ccb', 'chiến sĩ'],
        'retreat': ['thiền', 'tĩnh tâm', 'retreat', 'chữa lành'],
        'history': ['lịch sử', 'di tích', 'tri ân', 'chiến tranh'],
        'nature': ['thiên nhiên', 'rừng', 'núi', 'trekking'],
        'beach': ['biển', 'đảo', 'bãi biển', 'cát']
    }
    
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if pattern in message_lower:
                intents.append(intent)
                break
    
    # Thêm intent dựa trên context
    if conversation_history:
        last_messages = [msg['message'].lower() for msg in conversation_history[-2:]]
        last_text = ' '.join(last_messages)
        
        # Nếu đang nói về giá
        if any(word in last_text for word in ['giá', 'tiền', 'chi phí']):
            intents.append('price_followup')
        
        # Nếu đang so sánh
        if any(word in last_text for word in ['so sánh', 'khác', 'giữa']):
            intents.append('comparison_context')
    
    return list(set(intents))  # Remove duplicates


def _resolve_tour_reference(message: str, current_tour: int, discussed_tours: list, tour_name_map: dict) -> list:
    """Xác định tour đang được thảo luận thông minh"""
    indices = []
    message_lower = message.lower()
    
    # Strategy 1: Từ khóa chỉ định tour hiện tại
    current_keywords = ['tour đó', 'cái đó', 'nó', 'cái kia', 'tour này']
    if any(keyword in message_lower for keyword in current_keywords) and current_tour:
        return [current_tour]
    
    # Strategy 2: Tên tour cụ thể
    for norm_name, idx in tour_name_map.items():
        # Kiểm tra tên tour có trong message không
        name_words = set(norm_name.lower().split())
        msg_words = set(message_lower.split())
        common_words = name_words.intersection(msg_words)
        
        if len(common_words) >= 2:  # Ít nhất 2 từ trùng
            indices.append(idx)
    
    # Strategy 3: Tour đã thảo luận gần đây
    if not indices and discussed_tours:
        # Kiểm tra các từ khóa liên quan đến tour đã thảo luận
        for idx in discussed_tours[-3:]:  # 3 tour gần nhất
            tour = TOURS_DB.get(idx)
            if tour:
                # Kiểm tra các từ khóa liên quan
                relevant_keywords = []
                if tour.location:
                    relevant_keywords.extend(tour.location.lower().split())
                if tour.tags:
                    for tag in tour.tags:
                        relevant_keywords.extend(tag.split(':'))
                
                if any(keyword in message_lower for keyword in relevant_keywords):
                    indices.append(idx)
    
    return indices[:3]  # Giới hạn 3 tour


def _analyze_price_structure(price_text: str) -> dict:
    """Phân tích cấu trúc giá thông minh"""
    analysis = {
        'range': price_text,
        'per_person': None,
        'group_discount': None,
        'notes': []
    }
    
    price_lower = price_text.lower()
    
    # Trích xuất khoảng giá
    ranges = re.findall(r'(\d[\d,\.]+)\s*–\s*(\d[\d,\.]+)', price_text)
    if ranges:
        analysis['range'] = f"{ranges[0][0]} – {ranges[0][1]} VNĐ/người"
    
    # Giá theo người
    if '/người' in price_lower or 'khách' in price_lower:
        analysis['per_person'] = 'Giá tính theo từng khách'
    
    # Chiết khấu nhóm
    if 'nhóm' in price_lower:
        group_matches = re.findall(r'(\d+)%\s*nhóm', price_lower)
        if group_matches:
            analysis['group_discount'] = f"Chiết khấu {group_matches[0]}% cho nhóm"
    
    # Ghi chú đặc biệt
    special_keywords = ['cựu chiến binh', 'học sinh', 'sinh viên', 'đặt sớm']
    for keyword in special_keywords:
        if keyword in price_lower:
            analysis['notes'].append(f"Có ưu đãi cho {keyword}")
    
    return analysis


def _calculate_group_price(price_text: str, group_size: int, is_veteran: bool = False) -> str:
    """Tính toán giá cho nhóm cụ thể"""
    try:
        # Trích xuất giá cơ bản
        numbers = re.findall(r'(\d[\d,\.]+)', price_text)
        if numbers:
            base_price = int(numbers[0].replace(',', '').replace('.', ''))
            
            # Áp dụng chiết khấu nhóm
            discount = 0
            if group_size >= 10 and group_size < 15:
                discount = 0.05
            elif group_size >= 15 and group_size < 20:
                discount = 0.08
            elif group_size >= 20:
                discount = 0.11
            
            # Thêm chiết khấu cựu chiến binh
            if is_veteran:
                discount += 0.05
            
            total = base_price * group_size * (1 - discount)
            
            # Format kết quả
            if total > 1000000:
                return f"Khoảng {total/1000000:.1f} triệu VNĐ cho cả nhóm"
            else:
                return f"Khoảng {total:,.0f} VNĐ cho cả nhóm"
    except:
        pass
    
    return None


def _analyze_tour_suitability(tour, user_profile: dict) -> str:
    """Phân tích độ phù hợp của tour với người dùng"""
    suitability = []
    tour_text = str(tour.includes or '') + ' ' + str(tour.style or '')
    tour_text_lower = tour_text.lower()
    
    # Kiểm tra cho gia đình
    if user_profile.get('has_children'):
        if any(word in tour_text_lower for word in ['gia đình', 'trẻ em', 'nhẹ nhàng']):
            suitability.append('Phù hợp gia đình')
        elif any(word in tour_text_lower for word in ['trekking', 'đi bộ', 'đường khó']):
            suitability.append('Hạn chế với trẻ nhỏ')
    
    # Kiểm tra cho người lớn tuổi
    if user_profile.get('has_elderly'):
        if any(word in tour_text_lower for word in ['nhịp chậm', 'nhẹ nhàng', 'không trekking']):
            suitability.append('Phù hợp người lớn tuổi')
        elif any(word in tour_text_lower for word in ['đường dốc', 'leo núi', 'vất vả']):
            suitability.append('Khó khăn với người lớn tuổi')
    
    # Kiểm tra cho cựu chiến binh
    if user_profile.get('is_veteran'):
        if any(word in tour_text_lower for word in ['lịch sử', 'tri ân', 'cựu chiến binh']):
            suitability.append('Rất phù hợp cựu chiến binh')
    
    # Kiểm tra cho retreat
    if 'retreat' in user_profile.get('special_needs', []):
        if any(word in tour_text_lower for word in ['thiền', 'tĩnh tâm', 'retreat']):
            suitability.append('Lý tưởng cho retreat')
    
    if not suitability:
        # Phân tích chung
        if any(word in tour_text_lower for word in ['lịch sử', 'di tích']):
            suitability.append('Phù hợp yêu thích lịch sử')
        elif any(word in tour_text_lower for word in ['thiên nhiên', 'rừng', 'núi']):
            suitability.append('Phù hợp yêu thiên nhiên')
        elif any(word in tour_text_lower for word in ['biển', 'đảo']):
            suitability.append('Phù hợp yêu biển')
        else:
            suitability.append('Đa dạng đối tượng')
    
    return ', '.join(suitability[:3])


def _check_tour_warnings(tour, user_profile: dict) -> str:
    """Kiểm tra và đưa ra cảnh báo về tour"""
    warnings = []
    tour_text = str(tour.includes or '') + ' ' + str(tour.notes or '')
    tour_text_lower = tour_text.lower()
    
    # Cảnh báo cho trẻ em
    if user_profile.get('has_children'):
        if any(word in tour_text_lower for word in ['không phù hợp trẻ', 'trẻ dưới', 'hạn chế trẻ']):
            warnings.append('Tour không khuyến khích cho trẻ em')
        elif any(word in tour_text_lower for word in ['trekking', 'đi bộ nhiều', 'đường khó']):
            warnings.append('Có hoạt động trekking, cân nhắc với trẻ nhỏ')
    
    # Cảnh báo cho người lớn tuổi
    if user_profile.get('has_elderly'):
        if any(word in tour_text_lower for word in ['đường dốc', 'leo núi', 'vất vả']):
            warnings.append('Có hoạt động thể chất mạnh')
        elif 'sức khỏe' in tour_text_lower:
            warnings.append('Yêu cầu sức khỏe tốt')
    
    # Cảnh báo chung
    if 'lưu ý' in tour_text_lower or 'chú ý' in tour_text_lower:
        # Trích xuất phần lưu ý
        notes_start = tour_text_lower.find('lưu ý')
        if notes_start != -1:
            note_text = tour_text[notes_start:notes_start+200]
            warnings.append(f"Lưu ý quan trọng: {note_text[:100]}...")
    
    return ', '.join(warnings[:2]) if warnings else "Không có cảnh báo đặc biệt"


def _extract_tour_names_for_comparison(message: str, tour_name_map: dict) -> list:
    """Trích xuất tên tour cho mục đích so sánh với độ chính xác cao"""
    tour_names = []
    message_lower = message.lower()
    
    # Pattern cho so sánh
    patterns = [
        r'tour\s+["\']?([^"\']+?)["\']?\s+và\s+tour\s+["\']?([^"\']+?)["\']?',
        r'tour\s+["\']?([^"\']+?)["\']?\s+so\s+sánh\s+với\s+tour\s+["\']?([^"\']+?)["\']?',
        r'giữa\s+tour\s+["\']?([^"\']+?)["\']?\s+và\s+tour\s+["\']?([^"\']+?)["\']?',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, message_lower, re.IGNORECASE)
        for match in matches:
            for name in match:
                if name.strip() and len(name.strip()) > 3:
                    tour_names.append(name.strip())
    
    # Nếu không tìm thấy bằng pattern, tìm các từ khóa tour
    if not tour_names:
        for norm_name in tour_name_map.keys():
            name_lower = norm_name.lower()
            # Kiểm tra nếu tên tour xuất hiện trong message
            if any(word in message_lower for word in name_lower.split()[:3]):
                tour_names.append(norm_name)
    
    return list(set(tour_names))[:3]  # Tối đa 3 tour


def _analyze_user_requirements(message: str, user_profile: dict) -> dict:
    """Phân tích nhu cầu người dùng chi tiết"""
    requirements = user_profile.copy()
    message_lower = message.lower()
    
    # Phân tích cường độ
    if any(word in message_lower for word in ['nhẹ nhàng', 'thư giãn', 'không vội']):
        requirements['intensity'] = 'low'
    elif any(word in message_lower for word in ['trung bình', 'vừa phải']):
        requirements['intensity'] = 'medium'
    elif any(word in message_lower for word in ['mạnh mẽ', 'khám phá', 'trekking']):
        requirements['intensity'] = 'high'
    
    # Phân tích mục tiêu
    goals = []
    if any(word in message_lower for word in ['học hỏi', 'giáo dục', 'kiến thức']):
        goals.append('education')
    if any(word in message_lower for word in ['nghỉ ngơi', 'thư giãn', 'xả stress']):
        goals.append('relaxation')
    if any(word in message_lower for word in ['trải nghiệm', 'khám phá', 'mới lạ']):
        goals.append('adventure')
    if any(word in message_lower for word in ['kết nối', 'gắn kết', 'đoàn viên']):
        goals.append('bonding')
    
    if goals:
        requirements['goals'] = goals
    
    # Phân tích ràng buộc
    constraints = []
    if any(word in message_lower for word in ['hạn chế đi lại', 'khó di chuyển', 'sức khỏe']):
        constraints.append('mobility_issues')
    if any(word in message_lower for word in ['dị ứng', 'kiêng kỵ', 'không ăn được']):
        constraints.append('dietary_restrictions')
    if any(word in message_lower for word in ['sợ độ cao', 'say sóng', 'say xe']):
        constraints.append('phobias')
    
    if constraints:
        requirements['constraints'] = constraints
    
    return requirements


def _calculate_tour_match_score(tour, requirements: dict, user_profile: dict) -> tuple:
    """Tính điểm phù hợp của tour với nhu cầu người dùng"""
    score = 0
    reasons = []
    warnings = []
    
    tour_text = (str(tour.includes or '') + ' ' + 
                 str(tour.style or '') + ' ' + 
                 str(tour.summary or '')).lower()
    
    # Điểm cơ bản
    base_score = 10
    
    # Kiểm tra độ phù hợp với gia đình
    if requirements.get('has_children'):
        if 'gia đình' in tour_text or 'trẻ em' in tour_text:
            score += 20
            reasons.append('phù hợp gia đình có trẻ em')
        elif 'không phù hợp trẻ' in tour_text:
            score -= 15
            warnings.append('không khuyến khích cho trẻ em')
    
    # Kiểm tra cho người lớn tuổi
    if requirements.get('has_elderly'):
        if any(word in tour_text for word in ['nhịp chậm', 'nhẹ nhàng', 'không vội']):
            score += 15
            reasons.append('nhịp độ phù hợp người lớn tuổi')
        elif any(word in tour_text for word in ['trekking', 'leo núi', 'đường khó']):
            score -= 10
            warnings.append('có hoạt động thể chất mạnh')
    
    # Kiểm tra cho cựu chiến binh
    if requirements.get('is_veteran'):
        if any(word in tour_text for word in ['lịch sử', 'tri ân', 'cựu chiến binh']):
            score += 25
            reasons.append('thiết kế riêng cho cựu chiến binh')
    
    # Kiểm tra sở thích
    if requirements.get('preferences'):
        for preference in requirements['preferences']:
            if preference == 'nature' and any(word in tour_text for word in ['thiên nhiên', 'rừng', 'núi']):
                score += 10
                reasons.append('trải nghiệm thiên nhiên')
            elif preference == 'history' and any(word in tour_text for word in ['lịch sử', 'di tích', 'tri ân']):
                score += 10
                reasons.append('yếu tố lịch sử')
            elif preference == 'meditation' and any(word in tour_text for word in ['thiền', 'tĩnh tâm', 'retreat']):
                score += 15
                reasons.append('có hoạt động thiền')
            elif preference == 'beach' and any(word in tour_text for word in ['biển', 'đảo', 'bãi biển']):
                score += 10
                reasons.append('trải nghiệm biển đảo')
    
    # Kiểm tra ngân sách
    if requirements.get('budget_range') and tour.price:
        price_lower = tour.price.lower()
        if requirements['budget_range'] == 'low':
            if '500' in price_lower or '600' in price_lower or '700' in price_lower:
                score += 10
                reasons.append('giá cả phải chăng')
        elif requirements['budget_range'] == 'medium':
            if '1.5' in price_lower or '2.0' in price_lower or '2.5' in price_lower:
                score += 10
                reasons.append('giá trị tương xứng')
    
    # Kiểm tra thời gian
    if requirements.get('duration_pref') and tour.duration:
        if requirements['duration_pref'] == '1_day' and '1 ngày' in tour.duration.lower():
            score += 10
            reasons.append('đúng thời lượng yêu cầu')
        elif requirements['duration_pref'] == '2_days' and '2 ngày' in tour.duration.lower():
            score += 10
            reasons.append('đúng thời lượng yêu cầu')
    
    # Điểm tổng
    total_score = base_score + score
    
    return total_score, reasons[:3], warnings[:2]


def _summarize_price(price_text: str) -> str:
    """Tóm tắt giá tour"""
    if not price_text:
        return "Giá linh hoạt theo yêu cầu"
    
    # Trích xuất số
    numbers = re.findall(r'(\d[\d,\.]+)', price_text)
    if numbers:
        try:
            first_num = int(numbers[0].replace(',', '').replace('.', ''))
            if first_num < 1000000:
                return f"Khoảng {first_num:,.0f} VNĐ/người"
            elif first_num < 3000000:
                return f"Khoảng {first_num/1000000:.1f} triệu VNĐ/người"
            else:
                return f"Từ {first_num/1000000:.1f} triệu VNĐ/người"
        except:
            pass
    
    return price_text[:80] + ('...' if len(price_text) > 80 else '')


def _apply_intelligent_filters(tours_db: dict, user_profile: dict, message: str) -> list:
    """Áp dụng bộ lọc thông minh cho tours"""
    filtered_tours = []
    message_lower = message.lower()
    
    for idx, tour in tours_db.items():
        include = True
        tour_text = (str(tour.includes or '') + ' ' + 
                     str(tour.style or '') + ' ' + 
                     str(tour.notes or '')).lower()
        
        # Lọc theo đối tượng
        if user_profile.get('has_children'):
            if 'không phù hợp trẻ' in tour_text:
                include = False
            elif 'trekking' in tour_text and 'đi bộ' in tour_text:
                include = False  # Hạn chế với trẻ nhỏ
        
        if user_profile.get('has_elderly'):
            if any(word in tour_text for word in ['leo núi', 'đường dốc', 'vất vả']):
                include = False
        
        # Lọc theo sở thích từ message
        if 'thiền' in message_lower and 'thiền' not in tour_text:
            include = False
        if 'lịch sử' in message_lower and not any(word in tour_text for word in ['lịch sử', 'tri ân', 'di tích']):
            include = False
        if 'biển' in message_lower and not any(word in tour_text for word in ['biển', 'đảo']):
            include = False
        
        # Lọc theo yêu cầu đặc biệt
        if 'nhẹ nhàng' in message_lower and any(word in tour_text for word in ['trekking', 'leo núi', 'vất vả']):
            include = False
        
        if include:
            # Tính điểm phù hợp
            score, _, _ = _calculate_tour_match_score(
                tour, 
                _analyze_user_requirements(message, user_profile),
                user_profile
            )
            filtered_tours.append((idx, tour, score))
    
    # Sắp xếp theo điểm
    filtered_tours.sort(key=lambda x: x[2], reverse=True)
    
    return [(idx, tour) for idx, tour, score in filtered_tours]


def _categorize_tour(tour) -> str:
    """Phân loại tour"""
    tour_text = (str(tour.style or '') + ' ' + str(tour.includes or '')).lower()
    
    if any(word in tour_text for word in ['gia đình', 'trẻ em', 'nhẹ nhàng']):
        return 'Gia đình & Nhẹ nhàng'
    elif any(word in tour_text for word in ['lịch sử', 'tri ân', 'cựu chiến binh']):
        return 'Lịch sử & Tri ân'
    elif any(word in tour_text for word in ['thiền', 'retreat', 'tĩnh tâm']):
        return 'Thiền & Retreat'
    elif any(word in tour_text for word in ['thiên nhiên', 'rừng', 'núi', 'khám phá']):
        return 'Thiên nhiên & Khám phá'
    elif any(word in tour_text for word in ['biển', 'đảo']):
        return 'Biển & Đảo'
    else:
        return 'Đa dạng trải nghiệm'


def _summarize_user_profile(user_profile: dict) -> str:
    """Tóm tắt thông tin người dùng"""
    summary = "**THÔNG TIN KHÁCH HÀNG:**\n"
    
    if user_profile.get('group_size'):
        summary += f"• Số người: {user_profile['group_size']}\n"
    
    if user_profile.get('has_children'):
        summary += "• Có trẻ em\n"
        if user_profile.get('children_ages'):
            ages = user_profile['children_ages']
            summary += f"• Độ tuổi trẻ: {min(ages)}-{max(ages)} tuổi\n"
    
    if user_profile.get('has_elderly'):
        summary += "• Có người lớn tuổi\n"
    
    if user_profile.get('is_veteran'):
        summary += "• Có cựu chiến binh\n"
    
    if user_profile.get('preferences'):
        pref_map = {
            'nature': 'Thiên nhiên',
            'history': 'Lịch sử',
            'meditation': 'Thiền',
            'beach': 'Biển',
            'food': 'Ẩm thực'
        }
        preferences = [pref_map.get(p, p) for p in user_profile['preferences']]
        summary += f"• Sở thích: {', '.join(preferences)}\n"
    
    if user_profile.get('budget_range'):
        budget_map = {
            'low': 'Tiết kiệm (< 2 triệu)',
            'medium': 'Trung bình (2-5 triệu)',
            'high': 'Cao cấp (> 5 triệu)'
        }
        summary += f"• Ngân sách: {budget_map.get(user_profile['budget_range'], user_profile['budget_range'])}\n"
    
    if user_profile.get('duration_pref'):
        duration_map = {
            '1_day': '1 ngày',
            '2_days': '2 ngày',
            '3_days': '3 ngày'
        }
        summary += f"• Thời gian: {duration_map.get(user_profile['duration_pref'], user_profile['duration_pref'])}\n"
    
    if user_profile.get('special_needs'):
        summary += f"• Nhu cầu đặc biệt: {', '.join(user_profile['special_needs'])}\n"
    
    return summary


def _is_out_of_scope(message: str) -> bool:
    """Kiểm tra câu hỏi ngoài phạm vi"""
    message_lower = message.lower()
    
    out_of_scope_keywords = [
        'chứng khoán', 'tỉ giá', 'thời sự', 'tin tức',
        'chuyện cười', 'đố vui', 'game', 'giải trí',
        'thể thao', 'bóng đá', 'ca nhạc', 'phim ảnh',
        'chính trị', 'tôn giáo', 'nhạy cảm',
        'thời trang', 'mua sắm', 'làm đẹp'
    ]
    
    return any(keyword in message_lower for keyword in out_of_scope_keywords)


def _is_budget_friendly(price_text: str) -> bool:
    """Kiểm tra giá có phải chăng không"""
    if not price_text:
        return False
    
    # Trích xuất số đầu tiên
    numbers = re.findall(r'(\d[\d,\.]+)', price_text)
    if numbers:
        try:
            price = int(numbers[0].replace(',', '').replace('.', ''))
            return price < 2000000
        except:
            pass
    
    return False


def _extract_price_range(price_text: str) -> str:
    """Trích xuất khoảng giá từ text"""
    if not price_text:
        return "Liên hệ"
    
    # Tìm khoảng giá
    range_match = re.search(r'(\d[\d,\.]+)\s*[–\-]\s*(\d[\d,\.]+)', price_text)
    if range_match:
        return f"{range_match.group(1)} – {range_match.group(2)} VNĐ"
    
    # Tìm giá đơn
    single_match = re.search(r'(\d[\d,\.]+)\s*VNĐ', price_text)
    if single_match:
        return f"{single_match.group(1)} VNĐ"
    
    return price_text[:50] + ('...' if len(price_text) > 50 else '')


def _apply_smart_filters_to_search(search_results: list, user_profile: dict) -> list:
    """Áp dụng bộ lọc thông minh cho kết quả tìm kiếm"""
    filtered_indices = []
    
    for score, content in search_results:
        # Tìm tour index từ content
        for idx, tour in TOURS_DB.items():
            if tour.name in content and idx not in filtered_indices:
                # Kiểm tra phù hợp cơ bản
                tour_text = (str(tour.includes or '') + ' ' + str(tour.style or '')).lower()
                
                # Lọc cho trẻ em
                if user_profile.get('has_children'):
                    if 'không phù hợp trẻ' in tour_text:
                        continue
                
                # Lọc cho người lớn tuổi
                if user_profile.get('has_elderly'):
                    if any(word in tour_text for word in ['leo núi', 'vất vả', 'đường dốc']):
                        continue
                
                filtered_indices.append(idx)
                break
    
    return filtered_indices[:5]


def _prepare_llm_prompt_v2(user_message: str, search_results: list, context_info: dict) -> str:
    """Chuẩn bị prompt thông minh cho LLM"""
    # Tập hợp thông tin context
    tours_info = []
    for idx in context_info.get('tour_indices', [])[:3]:
        tour = TOURS_DB.get(idx)
        if tour:
            tours_info.append(f"• {tour.name}: {tour.summary or 'No summary'}")
    
    user_profile = context_info.get('user_profile', {})
    profile_summary = _summarize_user_profile(user_profile)
    
    # Tập hợp search results
    search_context = "\n".join([content[:200] for _, content in search_results[:3]])
    
    prompt = f"""Bạn là chuyên gia tư vấn tour Ruby Wings - thông minh, am hiểu và nhiệt tình.

THÔNG TIN KHÁCH HÀNG:
{profile_summary}

CONTEXT HIỆN TẠI:
• Tour đang thảo luận: {context_info.get('current_tour', 'None')}
• Tour đã đề cập: {len(context_info.get('discussed_tours', []))} tour
• Đang so sánh: {'Có' if context_info.get('comparison_mode') else 'Không'}

TOUR CÓ LIÊN QUAN:
{chr(10).join(tours_info) if tours_info else 'Không có tour cụ thể'}

THÔNG TIN TÌM KIẾM:
{search_context}

CÂU HỎI: {user_message}

YÊU CẦU TRẢ LỜI:
1. HIỂU CONTEXT: Nhận biết tour đang thảo luận và nhu cầu người dùng
2. CHÍNH XÁC: Chỉ sử dụng thông tin từ Ruby Wings, không bịa
3. HỮU ÍCH: Cung cấp thông tin giá trị, có thể hành động được
4. CÁ NHÂN HÓA: Gợi ý dựa trên thông tin người dùng
5. KẾT NỐI: Đề cập đến hotline khi cần tư vấn sâu

QUAN TRỌNG:
- Nếu hỏi về giá/lịch trình tour cụ thể: cung cấp chi tiết từ thông tin tour
- Nếu so sánh: đưa ra phân tích khách quan
- Nếu đề xuất: giải thích lý do phù hợp
- Nếu không biết: thành thật và hướng dẫn liên hệ hotline

TRẢ LỜI BẰNG TIẾNG VIỆT, CHUYÊN NGHIỆP, THÂN THIỆN."""

    return prompt


def _generate_intelligent_fallback(user_message: str, search_results: list, tour_indices: list, context_info: dict) -> str:
    """Tạo phản hồi fallback thông minh"""
    if tour_indices:
        # Có tour liên quan
        reply = "🔍 **TÔI TÌM THẤY MỘT SỐ TOUR LIÊN QUAN** 🔍\n\n"
        
        for idx in tour_indices[:3]:
            tour = TOURS_DB.get(idx)
            if tour:
                reply += f"**{tour.name}**\n"
                if tour.summary:
                    reply += f"{tour.summary[:100]}...\n"
                if tour.duration:
                    reply += f"⏱️ {tour.duration} | "
                if tour.location:
                    reply += f"📍 {tour.location[:30]}\n"
                reply += "\n"
        
        reply += "💡 **BẠN MUỐN BIẾT GÌ VỀ CÁC TOUR NÀY?**\n"
        reply += "• 'Giá tour này bao nhiêu?'\n"
        reply += "• 'Tour này có gì đặc biệt?'\n"
        reply += "• 'Có phù hợp cho gia đình không?'\n\n"
        reply += "📞 **Hoặc gọi tư vấn trực tiếp:** 0332510486"
    
    elif search_results:
        # Có kết quả tìm kiếm
        reply = "📚 **THÔNG TIN LIÊN QUAN TỪ RUBY WINGS** 📚\n\n"
        
        for i, (score, content) in enumerate(search_results[:3], 1):
            # Trích xuất thông tin quan trọng
            lines = content.split('\n')
            for line in lines:
                if len(line) > 30 and any(keyword in line.lower() for keyword in ['tour', 'giá', 'bao gồm', 'thời gian']):
                    reply += f"{i}. {line[:150]}...\n"
                    break
        
        reply += "\n🎯 **ĐỂ ĐƯỢC TƯ VẤN CHÍNH XÁC:**\n"
        reply += "1. Cung cấp thêm thông tin về nhu cầu\n"
        reply += "2. Chỉ định tên tour cụ thể\n"
        reply += "3. Liên hệ chuyên gia Ruby Wings\n\n"
        reply += "📞 **Hotline 24/7:** 0332510486"
    
    else:
        # Không có thông tin
        reply = "🤔 **TÔI CHƯA HIỂU RÕ NHU CẦU CỦA BẠN** 🤔\n\n"
        reply += "Để tôi hỗ trợ tốt hơn, bạn có thể:\n\n"
        reply += "🎯 **HỎI CỤ THỂ HƠN:**\n"
        reply += "• 'Tour 1 ngày ở Huế giá bao nhiêu?'\n"
        reply += "• 'Tour nào có thiền và giá dưới 1 triệu?'\n"
        reply += "• 'So sánh tour Bạch Mã và Trường Sơn'\n\n"
        reply += "📋 **HOẶC CUNG CẤP THÔNG TIN:**\n"
        reply += "• Số người và độ tuổi\n"
        reply += "• Sở thích chính của nhóm\n"
        reply += "• Thời gian và ngân sách\n\n"
        reply += "📞 **Gọi ngay để được tư vấn nhanh:** 0332510486"
    
    return reply


def _enhance_response_quality(reply: str, context) -> str:
    """Nâng cao chất lượng phản hồi"""
    # Thêm thông tin context nếu có
    if context.current_tour and "tour" not in reply.lower()[:100]:
        tour = TOURS_DB.get(context.current_tour)
        if tour and tour.name not in reply:
            # Chèn thông tin tour ở đầu nếu phù hợp
            lines = reply.split('\n')
            if len(lines) > 2:
                lines.insert(1, f"\n📌 **Đang nói về:** {tour.name}")
                reply = '\n'.join(lines)
    
    # Đảm bảo có call-to-action
    if not any(word in reply.lower() for word in ['gọi', 'liên hệ', 'hotline', '0332510486']):
        if len(reply.split('\n')) > 5:
            reply += "\n\n📞 **Cần hỗ trợ thêm? Gọi ngay:** 0332510486"
    
    # Format lại cho đẹp
    lines = reply.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip() and not line.startswith(('•', '📌', '🎯', '💰', '📍', '⏱️', '✅', '⚠️', '🔍')):
            if len(line) > 80 and ' ' in line[60:]:
                # Tự động xuống dòng cho dòng dài
                words = line.split()
                new_line = ""
                current_length = 0
                for word in words:
                    if current_length + len(word) + 1 > 80:
                        formatted_lines.append(new_line)
                        new_line = word + " "
                        current_length = len(word) + 1
                    else:
                        new_line += word + " "
                        current_length += len(word) + 1
                if new_line:
                    formatted_lines.append(new_line.strip())
                continue
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

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