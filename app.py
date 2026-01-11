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
        
        # LISTING detection
        listing_patterns = [
            (r'liệt kê.*tour|danh sách.*tour|các tour', 0.95),
            (r'có những.*nào|kể tên.*nào|nêu tên.*nào', 0.9),
            (r'tất cả.*tour|mọi.*tour|mấy.*tour', 0.8),
            (r'bên bạn.*có.*tour|hiện có.*tour', 0.85),
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
            (r'phù hợp.*với|nên đi.*nào|gợi ý.*tour', 0.9),
            (r'tour nào.*tốt|hành trình nào.*hay', 0.85),
            (r'đề xuất.*tour|tư vấn.*tour|chọn.*nào', 0.8),
            (r'cho.*tôi|dành cho.*tôi|hợp với.*tôi', 0.7),
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
    """Prepare prompt for LLM"""
    prompt_parts = [
        "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.",
        "HƯỚNG DẪN QUAN TRỌNG:",
        "1. LUÔN sử dụng thông tin từ dữ liệu nội bộ được cung cấp bên dưới",
        "2. Nếu thiếu thông tin chi tiết, tổng hợp từ thông tin chung có sẵn",
        "3. KHÔNG BAO GIỜ nói 'không có thông tin', 'không biết', 'không rõ'",
        "4. Luôn giữ thái độ nhiệt tình, hữu ích, chuyên nghiệp",
        "5. Nếu không tìm thấy thông tin chính xác, đưa ra thông tin tổng quát",
        "6. KHÔNG tự ý bịa thông tin không có trong dữ liệu",
        "",
        "THÔNG TIN NGỮ CẢNH:",
    ]
    
    if context.get('user_preferences'):
        prefs = []
        if context['user_preferences'].get('duration_pref'):
            prefs.append(f"Thích tour {context['user_preferences']['duration_pref']}")
        if context['user_preferences'].get('interests'):
            prefs.append(f"Quan tâm: {', '.join(context['user_preferences']['interests'])}")
        if prefs:
            prompt_parts.append(f"- Sở thích người dùng: {'; '.join(prefs)}")
    
    if context.get('current_tours'):
        tours_info = []
        for tour in context['current_tours']:
            tours_info.append(f"{tour['name']} ({tour.get('duration', '?')})")
        if tours_info:
            prompt_parts.append(f"- Tour đang thảo luận: {', '.join(tours_info)}")
    
    if context.get('filters'):
        filters = context['filters']
        filter_strs = []
        if filters.get('price_max'):
            filter_strs.append(f"giá dưới {filters['price_max']:,} VND")
        if filters.get('price_min'):
            filter_strs.append(f"giá trên {filters['price_min']:,} VND")
        if filters.get('location'):
            filter_strs.append(f"địa điểm: {filters['location']}")
        if filter_strs:
            prompt_parts.append(f"- Bộ lọc: {', '.join(filter_strs)}")
    
    prompt_parts.append("")
    prompt_parts.append("DỮ LIỆU NỘI BỘ RUBY WINGS:")
    
    if search_results:
        for i, (score, passage) in enumerate(search_results[:5], 1):
            text = passage.get('text', '')[:300]
            prompt_parts.append(f"\n[{i}] (Độ liên quan: {score:.2f})")
            prompt_parts.append(f"{text}")
    else:
        prompt_parts.append("Không tìm thấy dữ liệu liên quan trực tiếp.")
    
    prompt_parts.append("")
    prompt_parts.append("TRẢ LỜI:")
    prompt_parts.append("1. Dựa trên dữ liệu trên, trả lời câu hỏi người dùng")
    prompt_parts.append("2. Nếu có thông tin từ dữ liệu, trích dẫn nó")
    prompt_parts.append("3. Giữ câu trả lời ngắn gọn, rõ ràng, hữu ích")
    prompt_parts.append("4. Kết thúc bằng lời mời liên hệ hotline 0332510486 nếu cần thêm thông tin")
    
    return "\n".join(prompt_parts)

def _generate_fallback_response(user_message: str, search_results: List, tour_indices: List[int] = None) -> str:
    """Generate fallback response when LLM is unavailable"""
    message_lower = user_message.lower()
    
    if 'dưới' in message_lower and ('triệu' in message_lower or 'tiền' in message_lower):
        if not tour_indices and TOURS_DB:
            all_tours = list(TOURS_DB.items())[:3]
            response = "Dựa trên yêu cầu của bạn, tôi đề xuất các tour có giá hợp lý:\n"
            for idx, tour in all_tours:
                tour_name = tour.name or f'Tour #{idx}'
                price = tour.price or 'Liên hệ để biết giá'
                response += f"• **{tour_name}**: {price}\n"
            response += "\n💡 *Liên hệ hotline 0332510486 để biết giá chính xác và ưu đãi*"
            return response
    
    if not search_results:
        if tour_indices and TOURS_DB:
            response = "Thông tin về tour bạn quan tâm:\n"
            for idx in tour_indices[:2]:
                tour = TOURS_DB.get(idx)
                if tour:
                    response += f"\n**{tour.name or f'Tour #{idx}'}**\n"
                    if tour.duration:
                        response += f"⏱️ {tour.duration}\n"
                    if tour.location:
                        response += f"📍 {tour.location}\n"
                    if tour.price:
                        response += f"💰 {tour.price}\n"
            response += "\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết*"
            return response
        else:
            return "Xin lỗi, hiện không tìm thấy thông tin liên quan trong dữ liệu. " \
                   "Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp."
    
    top_results = search_results[:3]
    response_parts = ["Tôi tìm thấy một số thông tin liên quan:"]
    
    for i, (score, passage) in enumerate(top_results, 1):
        text = passage.get('text', '')[:150]
        if text:
            response_parts.append(f"\n{i}. {text}")
    
    response_parts.append("\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết*")
    
    return "".join(response_parts)

# =========== MAIN CHAT ENDPOINT WITH ALL UPGRADES ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Main chat endpoint with all 10 upgrades integrated
    """
    start_time = time.time()
    
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        
        if not user_message:
            return jsonify({
                "reply": "Xin chào! Tôi có thể giúp gì cho bạn về các tour của Ruby Wings?",
                "sources": [],
                "context": {},
                "processing_time": 0
            })
        
        session_id = extract_session_id(data, request.remote_addr)
        context = get_session_context(session_id)
        
        # Check memory cache
        recent_response = None
        if hasattr(context, 'get_recent_response') and hasattr(context, 'check_recent_question'):
            recent_response = context.get_recent_response(user_message)
            if recent_response and context.check_recent_question(user_message):
                logger.info("💭 Using cached response from recent conversation")
                processing_time = time.time() - start_time
                chat_response = ChatResponse(
                    reply=recent_response,
                    sources=[],
                    context={
                        "session_id": session_id,
                        "from_memory": True,
                        "processing_time_ms": int(processing_time * 1000)
                    },
                    tour_indices=[],
                    processing_time_ms=int(processing_time * 1000),
                    from_memory=True
                )
                return jsonify(chat_response.to_dict())
        
        # Initialize state machine
        if UpgradeFlags.is_enabled("7_STATE_MACHINE"):
            if not hasattr(context, 'state_machine') or context.state_machine is None:
                context.state_machine = ConversationStateMachine(session_id)
        
        state_tour_indices = []
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            state_tour_indices = context.state_machine.extract_reference(user_message)
            if state_tour_indices:
                logger.info(f"🔄 State machine injected tours: {state_tour_indices}")
                context.last_tour_indices = state_tour_indices
        
        # UPGRADE 5: COMPLEX QUERY SPLITTER
        sub_queries = []
        if UpgradeFlags.is_enabled("5_QUERY_SPLITTER"):
            sub_queries = ComplexQueryProcessor.split_query(user_message)
        
        # UPGRADE 1: MANDATORY FILTER EXTRACTION
        mandatory_filters = FilterSet()
        if UpgradeFlags.is_enabled("1_MANDATORY_FILTER"):
            mandatory_filters = MandatoryFilterSystem.extract_filters(user_message)
            
            if not mandatory_filters.is_empty() and TOURS_DB:
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                if filtered_indices:
                    if state_tour_indices:
                        combined = [idx for idx in state_tour_indices if idx in filtered_indices]
                        context.last_tour_indices = combined if combined else filtered_indices
                    else:
                        context.last_tour_indices = filtered_indices
                    logger.info(f"🔍 Applied mandatory filters")
        
        # UPGRADE 6: FUZZY MATCHING
        fuzzy_matches = []
        if UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            fuzzy_matches = FuzzyMatcher.find_similar_tours(user_message, TOUR_NAME_TO_INDEX)
            if fuzzy_matches:
                fuzzy_indices = [idx for idx, _ in fuzzy_matches]
                logger.info(f"🔍 Fuzzy matches found: {fuzzy_indices}")
                
                if context.last_tour_indices:
                    context.last_tour_indices = list(set(context.last_tour_indices + fuzzy_indices))
                else:
                    context.last_tour_indices = fuzzy_indices
        
        # UPGRADE 3: ENHANCED FIELD DETECTION
        requested_field = None
        field_confidence = 0.0
        if UpgradeFlags.is_enabled("3_ENHANCED_FIELDS"):
            requested_field, field_confidence, _ = EnhancedFieldDetector.detect_field_with_confidence(user_message)
        
        # UPGRADE 4: QUESTION CLASSIFICATION
        question_type = QuestionType.INFORMATION
        question_confidence = 0.0
        question_metadata = {}
        
        if UpgradeFlags.is_enabled("4_QUESTION_PIPELINE"):
            question_type, question_confidence, question_metadata = QuestionPipeline.classify_question(user_message)
        
        # UPGRADE 8: SEMANTIC ANALYSIS
        user_profile = UserProfile()
        if UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            current_profile = getattr(context, 'user_profile', None)
            user_profile = SemanticAnalyzer.analyze_user_profile(user_message, current_profile)
            context.user_profile = user_profile
        
        # UPGRADE 7: STATE MACHINE PROCESSING
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            placeholder_response = "Processing your request..."
            context.state_machine.update(user_message, placeholder_response, tour_indices if 'tour_indices' in locals() else [])
        
        # TOUR RESOLUTION
        tour_indices = context.last_tour_indices or []
        
        # Handle comparison questions
        if question_type == QuestionType.COMPARISON and not tour_indices:
            comparison_tour_names = []
            name_patterns = [
                r'tour\s+([^\s,]+)\s+và\s+tour\s+([^\s,]+)',
                r'tour\s+([^\s,]+)\s+với\s+tour\s+([^\s,]+)',
            ]
            
            for pattern in name_patterns:
                matches = re.finditer(pattern, user_message.lower())
                for match in matches:
                    for i in range(1, 3):
                        if match.group(i):
                            tour_name = match.group(i).strip()
                            for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                                if tour_name in norm_name or FuzzyMatcher.normalize_vietnamese(tour_name) in norm_name:
                                    comparison_tour_names.append(idx)
                                    break
            
            if len(comparison_tour_names) >= 2:
                tour_indices = comparison_tour_names[:2]
                context.last_tour_indices = tour_indices
                logger.info(f"🔍 Extracted tours for comparison: {tour_indices}")
        
        # Check cache
        cache_key = None
        if UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            context_hash = hashlib.md5(json.dumps({
                'tour_indices': tour_indices,
                'field': requested_field,
                'question_type': question_type.value,
                'filters': mandatory_filters.to_dict()
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = CacheSystem.get_cache_key(user_message, context_hash)
            cached_response = CacheSystem.get(cache_key)
            
            if cached_response:
                logger.info("💾 Using cached response")
                return jsonify(cached_response)
        
        # PROCESS BY QUESTION TYPE
        reply = ""
        sources = []
        
        # GREETING
        if question_type == QuestionType.GREETING:
            reply = TemplateSystem.render('greeting')
        
        # FAREWELL
        elif question_type == QuestionType.FAREWELL:
            reply = TemplateSystem.render('farewell')
        
        # COMPARISON
        elif question_type == QuestionType.COMPARISON:
            if len(tour_indices) >= 2:
                comparison_result = QuestionPipeline.process_comparison_question(
                    tour_indices, TOURS_DB, "", question_metadata
                )
                reply = comparison_result
            else:
                if TOURS_DB:
                    all_tours = list(TOURS_DB.items())
                    if len(all_tours) >= 2:
                        tour1_idx, tour1 = all_tours[0]
                        tour2_idx, tour2 = all_tours[1]
                        reply = f"Bạn có thể so sánh:\n1. {tour1.name or f'Tour #{tour1_idx}'}\n2. {tour2.name or f'Tour #{tour2_idx}'}\n\nHãy cho tôi biết bạn muốn so sánh tour nào cụ thể."
                    else:
                        reply = "Hiện chỉ có 1 tour trong hệ thống, không thể so sánh."
                else:
                    reply = "Bạn muốn so sánh tour nào với nhau? Vui lòng nêu tên 2 tour trở lên."
        
        # RECOMMENDATION
        elif question_type == QuestionType.RECOMMENDATION:
            if 'so sánh' in user_message.lower() or 'với' in user_message.lower():
                question_type = QuestionType.COMPARISON
                if not tour_indices and TOURS_DB:
                    tour_indices = list(TOURS_DB.keys())[:2]
                    reply = f"Tôi thấy bạn muốn so sánh. Bạn có thể so sánh:\n1. {TOURS_DB[tour_indices[0]].name or f'Tour #{tour_indices[0]}'}\n2. {TOURS_DB[tour_indices[1]].name or f'Tour #{tour_indices[1]}'}"
                else:
                    reply = "Bạn muốn so sánh tour nào với nhau?"
            elif UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
                profile_matches = SemanticAnalyzer.match_tours_to_profile(
                    user_profile, TOURS_DB, max_results=3
                )
                
                if profile_matches:
                    recommendations = []
                    for idx, score, reasons in profile_matches:
                        tour = TOURS_DB.get(idx)
                        if tour:
                            recommendations.append({
                                'name': tour.name or f'Tour #{idx}',
                                'score': score,
                                'reasons': reasons,
                                'duration': tour.duration or '',
                                'location': tour.location or '',
                                'price': tour.price or '',
                            })
                    
                    if recommendations:
                        reply = TemplateSystem.render('recommendation',
                            top_tour=recommendations[0] if recommendations else None,
                            other_tours=recommendations[1:] if len(recommendations) > 1 else [],
                            criteria=user_profile.to_summary()
                        )
                    else:
                        reply = "Hiện chưa tìm thấy tour phù hợp với yêu cầu của bạn."
                else:
                    reply = "Xin lỗi, tôi chưa hiểu rõ bạn cần tour như thế nào. " \
                           "Bạn có thể nói cụ thể hơn về sở thích và yêu cầu của mình không?"
            else:
                if TOURS_DB:
                    top_tours = list(TOURS_DB.items())[:2]
                    reply = "Dựa trên thông tin hiện có, tôi đề xuất bạn tham khảo:\n"
                    for idx, tour in top_tours:
                        reply += f"• {tour.name or f'Tour #{idx}'}\n"
                    reply += "\n💡 Bạn có thể hỏi chi tiết về từng tour cụ thể."
                else:
                    reply = "Hiện chưa có thông tin tour để đề xuất."
        
        # LISTING
        elif question_type == QuestionType.LISTING or requested_field == "tour_name":
            all_tours = []
            for idx, tour in TOURS_DB.items():
                all_tours.append(tour)
            
            # UPGRADE 2: DEDUPLICATION
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and all_tours:
                seen_names = set()
                unique_tours = []
                for tour in all_tours:
                    name = tour.name
                    if name not in seen_names:
                        seen_names.add(name)
                        unique_tours.append(tour)
                all_tours = unique_tours
            
            all_tours = all_tours[:15]
            
            # UPGRADE 10: TEMPLATE SYSTEM
            if UpgradeFlags.is_enabled("10_TEMPLATE_SYSTEM"):
                reply = TemplateSystem.render('tour_list', tours=all_tours)
            else:
                if all_tours:
                    reply = "✨ **Danh sách tour Ruby Wings:** ✨\n\n"
                    for i, tour in enumerate(all_tours[:10], 1):
                        reply += f"{i}. **{tour.name or f'Tour #{i}'}**\n"
                        if tour.duration:
                            reply += f"   ⏱️ {tour.duration}\n"
                        if tour.location:
                            reply += f"   📍 {tour.location}\n"
                        reply += "\n"
                    reply += "💡 *Hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour*"
                else:
                    reply = "Hiện chưa có thông tin tour trong hệ thống."
        
        # FIELD-SPECIFIC QUERY
        elif requested_field and field_confidence > 0.3:
            if tour_indices:
                field_info = []
                for idx in tour_indices:
                    tour = TOURS_DB.get(idx)
                    if tour:
                        field_value = getattr(tour, requested_field, None)
                        if field_value:
                            if isinstance(field_value, list):
                                field_text = "\n".join([f"• {item}" for item in field_value])
                            else:
                                field_text = field_value
                            
                            tour_name = tour.name or f'Tour #{idx}'
                            field_info.append(f"**{tour_name}**:\n{field_text}")
                
                if field_info:
                    reply = "\n\n".join(field_info)
                    field_passages = get_passages_by_field(requested_field, tour_indices=tour_indices)
                    sources = [m for _, m in field_passages]
                else:
                    reply = f"Không tìm thấy thông tin về {requested_field} cho tour đã chọn."
            else:
                field_passages = get_passages_by_field(requested_field, limit=5)
                if field_passages:
                    field_texts = [m.get('text', '') for _, m in field_passages]
                    reply = "**Thông tin chung:**\n" + "\n".join([f"• {text}" for text in field_texts[:3]])
                    sources = [m for _, m in field_passages]
                else:
                    reply = f"Hiện không có thông tin về {requested_field} trong dữ liệu."
        
        # DEFAULT: SEMANTIC SEARCH + LLM
        else:
            search_results = query_index(user_message, TOP_K)
            
            # UPGRADE 2: DEDUPLICATION
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and search_results:
                search_results = DeduplicationEngine.deduplicate_passages(search_results)
            
            # Prepare context for LLM
            current_tours = []
            if tour_indices:
                for idx in tour_indices[:2]:
                    tour = TOURS_DB.get(idx)
                    if tour:
                        current_tours.append({
                            'index': idx,
                            'name': tour.name or f'Tour #{idx}',
                            'duration': tour.duration or '',
                            'location': tour.location or '',
                            'price': tour.price or '',
                        })
            
            # Prepare prompt
            prompt = _prepare_llm_prompt(user_message, search_results, {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'question_type': question_type.value,
                'requested_field': requested_field,
                'user_preferences': getattr(context, 'user_preferences', {}),
                'current_tours': current_tours,
                'filters': mandatory_filters.to_dict()
            })
            
            # Get LLM response
            if client and HAS_OPENAI:
                try:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=800,
                        top_p=0.95
                    )
                    
                    if response.choices and len(response.choices) > 0:
                        reply = response.choices[0].message.content or ""
                    else:
                        reply = "Xin lỗi, tôi không thể tạo phản hồi ngay lúc này."
                
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    reply = _generate_fallback_response(user_message, search_results, tour_indices)
            else:
                reply = _generate_fallback_response(user_message, search_results, tour_indices)
            
            sources = [m for _, m in search_results]
        
        # UPGRADE 9: AUTO-VALIDATION
        if UpgradeFlags.is_enabled("9_AUTO_VALIDATION"):
            reply = AutoValidator.validate_response(reply)
        
        # Update context
        context.last_action = "chat_response"
        context.timestamp = datetime.utcnow()
        
        if tour_indices and tour_indices[0] in TOURS_DB:
            tour = TOURS_DB[tour_indices[0]]
            context.last_tour_name = tour.name
        
        # Update state machine
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            context.state_machine.update(user_message, reply, tour_indices)
        
        # Add to memory
        if hasattr(context, 'add_to_history'):
            context.add_to_history(user_message, reply)
        
        # Prepare response
        processing_time = time.time() - start_time
        
        chat_response = ChatResponse(
            reply=reply,
            sources=sources,
            context={
                "session_id": session_id,
                "last_tour_name": getattr(context, 'last_tour_name', None),
                "user_preferences": getattr(context, 'user_preferences', {}),
                "question_type": question_type.value,
                "requested_field": requested_field,
                "processing_time_ms": int(processing_time * 1000),
                "from_memory": False
            },
            tour_indices=tour_indices,
            processing_time_ms=int(processing_time * 1000),
            from_memory=False
        )
        
        response_data = chat_response.to_dict()
        
        # Cache the response
        if cache_key and UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            CacheSystem.set(cache_key, response_data)
        
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s | "
                   f"Question: {question_type.value} | "
                   f"Tours: {len(tour_indices)} | "
                   f"Reply length: {len(reply)}")
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}\n{traceback.format_exc()}")
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "reply": "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. "
                    "Vui lòng thử lại sau hoặc liên hệ hotline 0332510486.",
            "sources": [],
            "context": {
                "error": str(e),
                "processing_time_ms": int(processing_time * 1000)
            }
        }), 500

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

@app.route('/api/save-lead', methods=['POST'])
def save_lead_to_sheet():
    """Save lead to Google Sheets using LeadData dataclass"""
    try:
        if not request.is_json:
            return jsonify({"error": "JSON required", "success": False}), 400
        
        data = request.get_json() or {}
        phone = (data.get("phone") or "").strip()
        
        if not phone:
            return jsonify({"error": "Phone required", "success": False}), 400
        
        # Create LeadData object
        lead_data = LeadData(
            timestamp=datetime.utcnow(),
            source_channel=data.get("source_channel", "Website"),
            action_type=data.get("action_type", "Click Call"),
            page_url=data.get("page_url", ""),
            contact_name=data.get("contact_name", ""),
            phone=phone,
            service_interest=data.get("service_interest", ""),
            note=data.get("note", ""),
            status="New"
        )
        
        logger.info(f"💾 Processing lead: {phone}")
        
        # Try Google Sheets
        sheets_success = False
        if ENABLE_GOOGLE_SHEETS and HAS_GOOGLE_SHEETS:
            try:
                gc = get_gspread_client()
                if gc:
                    sh = gc.open_by_key(GOOGLE_SHEET_ID)
                    ws = sh.worksheet(GOOGLE_SHEET_NAME)
                    
                    ws.append_row(lead_data.to_row())
                    sheets_success = True
                    logger.info(f"✅ Lead saved to Google Sheets: {phone}")
                    
                    # Meta CAPI
                    if HAS_META_CAPI and ENABLE_META_CAPI_CALL:
                        try:
                            send_meta_lead(
                                request=request,
                                event_name="Lead",
                                phone=phone,
                                value=200000,
                                currency="VND"
                            )
                        except Exception as e:
                            logger.warning(f"Meta CAPI error: {e}")
            except Exception as e:
                logger.error(f"Google Sheets error: {e}")
        
        # Fallback storage
        fallback_success = False
        if ENABLE_FALLBACK_STORAGE:
            try:
                leads = []
                if os.path.exists(FALLBACK_STORAGE_PATH):
                    with open(FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                        leads = json.load(f)
                
                leads.append(lead_data.to_dict())
                
                with open(FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(leads, f, ensure_ascii=False, indent=2)
                
                fallback_success = True
                logger.info(f"✅ Lead saved to fallback storage: {phone}")
            except Exception as e:
                logger.error(f"Fallback storage error: {e}")
        
        if sheets_success:
            return jsonify({
                "success": True,
                "message": "Lead saved successfully",
                "data": {"phone": phone}
            })
        elif fallback_success:
            return jsonify({
                "success": True,
                "message": "Lead saved to backup",
                "warning": "Google Sheets not available",
                "data": {"phone": phone}
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save lead"
            }), 500
    
    except Exception as e:
        logger.error(f"Save lead error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/api/track-call', methods=['POST'])
def track_call():
    """Track call button clicks"""
    try:
        data = request.get_json() or {}
        logger.info(f"📞 Call tracked: {data.get('phone')}")
        
        # Meta CAPI
        if HAS_META_CAPI and ENABLE_META_CAPI_CALL:
            try:
                send_meta_call_button(
                    request=request,
                    page_url=data.get('page_url'),
                    phone=data.get('phone'),
                    call_type=data.get('call_type', 'regular')
                )
            except Exception as e:
                logger.warning(f"Meta CAPI error: {e}")
        
        return jsonify({
            "success": True,
            "message": "Call tracked",
            "meta_capi_sent": HAS_META_CAPI
        })
    
    except Exception as e:
        logger.error(f"Track call error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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