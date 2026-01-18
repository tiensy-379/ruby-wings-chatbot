#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 6.0.0 (COMPLETE FIX + ENHANCED RESPONSES)
Created: 2025-01-17
Author: Ruby Wings AI Team

FIX V6.0.0: COMPREHENSIVE FIX FOR RESPONSE GENERATION
1. FIXED: OpenAI client initialization (removed proxies parameter)
2. FIXED: SearchEngine never returns empty results for tour-related intents
3. FIXED: ResponseGenerator always receives proper context (search_results or fallback)
4. ENHANCED: Intent detection with tour advisory keywords
5. ENHANCED: LLM prompt engineering for Ruby Wings brand voice
6. MAINTAINED: All existing features (lead capture, Meta CAPI, Google Sheets, etc.)
"""

# ==================== CORE IMPORTS ====================
import os
import sys
import json
import time
import threading
import logging
import re
import hashlib
import traceback
import random
import unicodedata
import warnings
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict, field

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ==================== PLATFORM DETECTION ====================
import platform
IS_WINDOWS = platform.system().lower().startswith("win")
IS_RENDER = "RENDER" in os.environ
IS_PRODUCTION = os.environ.get("FLASK_ENV", "production") == "production"

# ==================== FLASK & WEB ====================
from flask import Flask, request, jsonify, g, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ruby_wings.log') if IS_PRODUCTION else logging.NullHandler()
    ]
)
logger = logging.getLogger("ruby-wings-v6.0.0-complete-fix")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration - COMPLETE ENV MAPPING"""
    
    # RAM Profile
    RAM_PROFILE = os.getenv("RAM_PROFILE", "512")
    IS_LOW_RAM = RAM_PROFILE == "512"
    
    # Core API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    META_CAPI_TOKEN = os.getenv("META_CAPI_TOKEN", "").strip()
    META_PIXEL_ID = os.getenv("META_PIXEL_ID", "").strip()
    SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or os.urandom(24).hex()
    
    # File Paths
    KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "knowledge.json")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
    FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "faiss_mapping.json")
    FALLBACK_VECTORS_PATH = os.getenv("FALLBACK_VECTORS_PATH", "vectors.npz")
    TOUR_ENTITIES_PATH = os.getenv("TOUR_ENTITIES_PATH", "tour_entities.json")
    FALLBACK_STORAGE_PATH = os.getenv("FALLBACK_STORAGE_PATH", "leads_fallback.json")
    
    # OpenAI Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Feature Toggles
    FAISS_ENABLED = os.getenv("FAISS_ENABLED", "false").lower() == "true"
    ENABLE_INTENT_DETECTION = os.getenv("ENABLE_INTENT_DETECTION", "true").lower() == "true"
    ENABLE_PHONE_DETECTION = os.getenv("ENABLE_PHONE_DETECTION", "true").lower() == "true"
    ENABLE_LEAD_CAPTURE = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_LLM_FALLBACK = True
    ENABLE_CACHING = True
    ENABLE_GOOGLE_SHEETS = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_META_CAPI = os.getenv("ENABLE_META_CAPI_LEAD", "true").lower() == "true"
    ENABLE_META_CAPI_CALL = os.getenv("ENABLE_META_CAPI_CALL", "true").lower() == "true"
    ENABLE_FALLBACK_STORAGE = os.getenv("ENABLE_FALLBACK_STORAGE", "true").lower() == "true"
    ENABLE_TOUR_FILTERING = os.getenv("ENABLE_TOUR_FILTERING", "true").lower() == "true"
    ENABLE_COMPANY_INFO = os.getenv("ENABLE_COMPANY_INFO", "true").lower() == "true"
    ENABLE_META_CAPI_LEAD = os.getenv("ENABLE_META_CAPI_LEAD", "false").lower() == "true"
    ENABLE_ADVANCED_INTENT = True
    
    # State Machine
    STATE_MACHINE_ENABLED = True
    ENABLE_LOCATION_FILTER = True
    ENABLE_SEMANTIC_ANALYSIS = True
    
    # Performance Settings
    TOP_K = int(os.getenv("TOP_K", "5" if IS_LOW_RAM else "10"))
    MAX_TOURS_PER_RESPONSE = 3
    CACHE_TTL_SECONDS = 300
    MAX_SESSIONS = 50 if IS_LOW_RAM else 100
    MAX_EMBEDDING_CACHE = 30 if IS_LOW_RAM else 50
    CONVERSATION_HISTORY_LIMIT = 5 if IS_LOW_RAM else 10
    
    # Server Config
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "10000"))
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS
    CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")
    CORS_ORIGINS = CORS_ORIGINS_RAW if CORS_ORIGINS_RAW == "*" else [
        o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()
    ]
    
    # Google Sheets
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
    GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox")
    
    # Meta CAPI
    META_CAPI_ENDPOINT = os.getenv("META_CAPI_ENDPOINT", "https://graph.facebook.com")
    META_TEST_EVENT_CODE = os.getenv("META_TEST_EVENT_CODE", "")
    DEBUG_META_CAPI = os.getenv("DEBUG_META_CAPI", "false").lower() == "true"
    
    # LLM Settings
    ENABLE_LLM_ADVICE = True
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 800
    ENABLE_PROMPT_ENGINEERING = True
    
    @classmethod
    def log_config(cls):
        """Log configuration on startup"""
        logger.info("=" * 60)
        logger.info("🚀 RUBY WINGS CHATBOT v6.0.0 (COMPLETE FIX)")
        logger.info("=" * 60)
        logger.info(f"📊 RAM Profile: {cls.RAM_PROFILE}MB")
        logger.info(f"🌍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
        
        features = []
        if cls.STATE_MACHINE_ENABLED:
            features.append("State Machine")
        if cls.FAISS_ENABLED:
            features.append("FAISS")
        else:
            features.append("Numpy Fallback")
        if cls.ENABLE_META_CAPI:
            features.append("Meta CAPI")
        if cls.ENABLE_GOOGLE_SHEETS:
            features.append("Google Sheets")
        if cls.ENABLE_TOUR_FILTERING:
            features.append("Tour Filtering")
        if cls.ENABLE_COMPANY_INFO:
            features.append("Company Info")
        if cls.ENABLE_LLM_ADVICE:
            features.append("LLM Advisory")
        
        logger.info(f"🎯 Features: {', '.join(features)}")
        logger.info(f"🔑 OpenAI: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        logger.info(f"🌐 CORS: {cls.CORS_ORIGINS}")
        logger.info("=" * 60)

# ==================== ENUM INTENT FIX ====================
class Intent:
    """Fixed Intent Enum - Complete set"""
    # Core conversation intents
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    SMALLTALK = "SMALLTALK"
    UNKNOWN = "UNKNOWN"
    
    # Tour-related intents
    TOUR_INQUIRY = "TOUR_INQUIRY"
    TOUR_LIST = "TOUR_LIST"
    TOUR_FILTER = "TOUR_FILTER"
    TOUR_DETAIL = "TOUR_DETAIL"
    TOUR_COMPARE = "TOUR_COMPARE"
    TOUR_RECOMMEND = "TOUR_RECOMMEND"
    TOUR_ADVICE = "TOUR_ADVICE"
    
    # Price intents
    PRICE_ASK = "PRICE_ASK"
    PRICE_COMPARE = "PRICE_COMPARE"
    PRICE_RANGE = "PRICE_RANGE"
    
    # Booking intents
    BOOKING_REQUEST = "BOOKING_REQUEST"
    BOOKING_PROCESS = "BOOKING_PROCESS"
    BOOKING_CONDITION = "BOOKING_CONDITION"
    
    # Contact intents
    PROVIDE_PHONE = "PROVIDE_PHONE"
    CALLBACK_REQUEST = "CALLBACK_REQUEST"
    CONTACT_INFO = "CONTACT_INFO"
    
    # Company intents
    ABOUT_COMPANY = "ABOUT_COMPANY"
    COMPANY_SERVICE = "COMPANY_SERVICE"
    COMPANY_MISSION = "COMPANY_MISSION"
    
    # Lead capture
    LEAD_CAPTURED = "LEAD_CAPTURED"

class ConversationStage:
    """Conversation stages"""
    EXPLORE = "explore"
    SUGGEST = "suggest"
    COMPARE = "compare"
    SELECT = "select"
    BOOK = "book"
    LEAD = "lead"
    CALLBACK = "callback"

# ==================== INTENT UTILITIES ====================
def normalize_intent(intent_value: Any) -> str:
    """Normalize intent to string value"""
    if intent_value is None:
        return Intent.UNKNOWN
    
    if isinstance(intent_value, str):
        return intent_value
    
    if hasattr(intent_value, 'name'):
        return intent_value.name
    
    if hasattr(intent_value, 'value'):
        return intent_value.value
    
    return str(intent_value)

def get_intent_enum(intent_value: Any) -> str:
    """Get intent enum value from any input"""
    normalized = normalize_intent(intent_value)
    
    valid_intents = {
        # Core intents
        "GREETING": Intent.GREETING,
        "FAREWELL": Intent.FAREWELL,
        "SMALLTALK": Intent.SMALLTALK,
        "UNKNOWN": Intent.UNKNOWN,
        
        # Tour intents
        "TOUR_INQUIRY": Intent.TOUR_INQUIRY,
        "TOUR_LIST": Intent.TOUR_LIST,
        "TOUR_FILTER": Intent.TOUR_FILTER,
        "TOUR_DETAIL": Intent.TOUR_DETAIL,
        "TOUR_COMPARE": Intent.TOUR_COMPARE,
        "TOUR_RECOMMEND": Intent.TOUR_RECOMMEND,
        "TOUR_ADVICE": Intent.TOUR_ADVICE,
        
        # Price intents
        "PRICE_ASK": Intent.PRICE_ASK,
        "PRICE_COMPARE": Intent.PRICE_COMPARE,
        "PRICE_RANGE": Intent.PRICE_RANGE,
        
        # Booking intents
        "BOOKING_REQUEST": Intent.BOOKING_REQUEST,
        "BOOKING_PROCESS": Intent.BOOKING_PROCESS,
        "BOOKING_CONDITION": Intent.BOOKING_CONDITION,
        
        # Contact intents
        "PROVIDE_PHONE": Intent.PROVIDE_PHONE,
        "CALLBACK_REQUEST": Intent.CALLBACK_REQUEST,
        "CONTACT_INFO": Intent.CONTACT_INFO,
        
        # Company intents
        "ABOUT_COMPANY": Intent.ABOUT_COMPANY,
        "COMPANY_SERVICE": Intent.COMPANY_SERVICE,
        "COMPANY_MISSION": Intent.COMPANY_MISSION,
        
        # Lead capture
        "LEAD_CAPTURED": Intent.LEAD_CAPTURED
    }
    
    return valid_intents.get(normalized, Intent.UNKNOWN)

def is_intent_equal(intent1: Any, intent2: Any) -> bool:
    """Compare two intents safely"""
    intent1_str = normalize_intent(intent1)
    intent2_str = normalize_intent(intent2)
    
    return intent1_str == intent2_str

# ==================== LAZY IMPORTS ====================
def lazy_import_numpy():
    """Lazy import numpy"""
    try:
        import numpy as np
        return np, True
    except ImportError:
        logger.warning("⚠️ Numpy not available")
        return None, False

def lazy_import_faiss():
    """Lazy import FAISS"""
    if not Config.FAISS_ENABLED:
        return None, False
    try:
        import faiss
        return faiss, True
    except ImportError:
        logger.warning("⚠️ FAISS not available, using numpy fallback")
        return None, False

def lazy_import_openai():
    """Lazy import OpenAI - FIXED: No proxies parameter"""
    try:
        from openai import OpenAI
        return OpenAI, True
    except ImportError:
        logger.error("❌ OpenAI library not available")
        return None, False

# Initialize lazy imports
np, NUMPY_AVAILABLE = lazy_import_numpy()
faiss, FAISS_AVAILABLE = lazy_import_faiss()
OpenAI, OPENAI_AVAILABLE = lazy_import_openai()

# ==================== ENHANCED INTENT DETECTION ====================
def detect_intent(text): 
    """Enhanced intent detection with advisory keyword upgrade"""
    text_lower = text.lower().strip()
    metadata = {
        "duration_days": None,
        "location": None,
        "price_max": None,
        "tags": [],
        "region": None,
        "raw_query": text,
        "has_advisory_keywords": False
    }
    
    # ==================== ADVISORY KEYWORD DETECTION ====================
    # Enhanced list of advisory keywords that should trigger TOUR_INQUIRY/TOUR_ADVICE
    advisory_keywords = [
        # Tour/retreat related
        'tour', 'du lịch', 'retreat', 'hành trình', 'trải nghiệm',
        'chữa lành', 'thiền', 'khí công', 'tĩnh tâm', 'tâm linh',
        
        # Recommendation keywords
        'gợi ý', 'suggest', 'recommend', 'tư vấn', 'advice',
        'nên đi', 'nên chọn', 'phù hợp', 'hợp với',
        
        # Comparison keywords
        'so sánh', 'compare', 'khác nhau', 'giống nhau',
        'nào tốt hơn', 'nào hay hơn', 'nào phù hợp hơn',
        
        # Time-based
        'cuối tuần', 'weekend', 'ngắn ngày', 'dài ngày',
        '1 ngày', '2 ngày', '3 ngày', 'một ngày', 'hai ngày',
        
        # Audience-based
        'đi một mình', 'đi cùng bạn', 'đi với gia đình',
        'cho người lớn tuổi', 'cho thanh niên', 'cho doanh nghiệp',
        'team building', 'công ty', 'doanh nghiệp',
        
        # Activity-based
        'ít di chuyển', 'nhẹ nhàng', 'khám phá', 'mạo hiểm',
        'thiên nhiên', 'biển', 'núi', 'rừng',
        
        # General inquiry
        'có gì', 'làm gì', 'đi đâu', 'thế nào',
        'giới thiệu', 'tìm hiểu', 'khám phá'
    ]
    
    metadata['has_advisory_keywords'] = any(keyword in text_lower for keyword in advisory_keywords)
    
    # ==================== ENTITY EXTRACTION ====================
    # Extract duration days
    duration_patterns = [
        (r'(\d+)\s*ngày', 'duration_days'),
        (r'(\d+)\s*ngay', 'duration_days'),
        (r'(\d+)\s*day', 'duration_days'),
        (r'một\s*ngày', 'duration_days'),
        (r'hai\s*ngày', 'duration_days'),
        (r'ba\s*ngày', 'duration_days')
    ]
    
    for pattern, key in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if pattern.startswith(r'(\d+)'):
                metadata[key] = int(match.group(1))
            else:
                num_map = {'một': 1, 'hai': 2, 'ba': 3}
                for vn_num, num in num_map.items():
                    if vn_num in match.group(0):
                        metadata[key] = num
                        break
    
    # Extract location/region
    location_keywords = {
        'huế': 'Huế', 'hue': 'Huế',
        'đà nẵng': 'Đà Nẵng', 'da nang': 'Đà Nẵng',
        'hội an': 'Hội An', 'hoi an': 'Hội An',
        'quảng trị': 'Quảng Trị', 'quang tri': 'Quảng Trị',
        'bạch mã': 'Bạch Mã', 'bach ma': 'Bạch Mã',
        'phong nha': 'Phong Nha', 'quảng bình': 'Quảng Bình'
    }
    
    for keyword, location in location_keywords.items():
        if keyword in text_lower:
            metadata['location'] = location
            metadata['region'] = location
            break
    
    # Extract tags/keywords
    tag_keywords = {
        'thiền': 'thiền', 'meditation': 'thiền',
        'retreat': 'retreat',
        'chữa lành': 'chữa lành', 'healing': 'chữa lành',
        'trải nghiệm': 'trải nghiệm', 'experience': 'trải nghiệm',
        'lịch sử': 'lịch sử', 'history': 'lịch sử',
        'team building': 'team building', 'corporate': 'team building',
        'gia đình': 'gia đình', 'family': 'gia đình',
        'thiên nhiên': 'thiên nhiên', 'nature': 'thiên nhiên'
    }
    
    for keyword, tag in tag_keywords.items():
        if keyword in text_lower:
            metadata['tags'].append(tag)
    
    # ==================== INTENT CLASSIFICATION ====================
    # 1. GREETING & FAREWELL
    greeting_words = ['xin chào', 'chào', 'hello', 'hi', 'helo', 'chao']
    farewell_words = ['tạm biệt', 'bye', 'goodbye', 'cảm ơn']
    
    if any(word in text_lower for word in greeting_words):
        # Check if greeting contains advisory keywords (e.g., "Xin chào, tư vấn tour cho tôi")
        if metadata['has_advisory_keywords']:
            return Intent.TOUR_ADVICE, 0.90, metadata
        return Intent.GREETING, 0.95, metadata
    
    if any(word in text_lower for word in farewell_words):
        return Intent.FAREWELL, 0.95, metadata
    
    # 2. COMPANY INFO
    company_keywords = [
        'ruby wings', 'công ty', 'đơn vị', 'bạn là ai', 
        'giới thiệu', 'công ty bạn', 'doanh nghiệp',
        'tổ chức', 'rubywings', 'thương hiệu'
    ]
    
    if any(keyword in text_lower for keyword in company_keywords):
        return Intent.ABOUT_COMPANY, 0.92, metadata
    
    # 3. TOUR LIST
    tour_list_keywords = [
        'tour nào', 'tour gì', 'có những tour nào', 
        'danh sách tour', 'các tour', 'tour của bạn',
        'bạn có tour nào', 'dịch vụ nào', 'sản phẩm nào'
    ]
    
    if any(keyword in text_lower for keyword in tour_list_keywords):
        return Intent.TOUR_LIST, 0.90, metadata
    
    # 4. TOUR FILTER
    has_filter_criteria = (
        metadata['duration_days'] is not None or
        metadata['location'] is not None or
        metadata['price_max'] is not None or
        len(metadata['tags']) > 0
    )
    
    filter_words = ['tour', 'du lịch', 'trải nghiệm', 'retreat', 'hành trình']
    has_tour_word = any(word in text_lower for word in filter_words)
    
    if has_tour_word and has_filter_criteria:
        return Intent.TOUR_FILTER, 0.88, metadata
    
    # 5. TOUR ADVICE/RECOMMENDATION (NEW)
    advice_keywords = [
        'gợi ý', 'tư vấn', 'recommend', 'suggest',
        'nên đi', 'phù hợp', 'cho tôi', 'hợp với',
        'nào hay', 'nào tốt'
    ]
    
    if any(keyword in text_lower for keyword in advice_keywords) and metadata['has_advisory_keywords']:
        return Intent.TOUR_ADVICE, 0.85, metadata
    
    # 6. TOUR COMPARISON
    comparison_keywords = ['so sánh', 'compare', 'khác nhau', 'giống nhau']
    if any(keyword in text_lower for keyword in comparison_keywords):
        return Intent.TOUR_COMPARE, 0.85, metadata
    
    # 7. TOUR INQUIRY (default for tour-related queries)
    if has_tour_word or metadata['has_advisory_keywords']:
        return Intent.TOUR_INQUIRY, 0.85, metadata
    
    # 8. PRICE ASK
    price_words = ['giá', 'bao nhiêu tiền', 'cost', 'price', 'chi phí']
    if any(word in text_lower for word in price_words):
        return Intent.PRICE_ASK, 0.85, metadata
    
    # 9. BOOKING REQUEST
    booking_words = ['đặt', 'book', 'đăng ký', 'reserve', 'booking']
    if any(word in text_lower for word in booking_words):
        return Intent.BOOKING_REQUEST, 0.90, metadata
    
    # 10. PHONE PROVIDE
    phone = detect_phone_number(text)
    if phone:
        metadata['phone_number'] = phone
        return Intent.PROVIDE_PHONE, 0.98, metadata
    
    # 11. UNKNOWN with advisory keywords -> TOUR_ADVICE
    if metadata['has_advisory_keywords']:
        return Intent.TOUR_ADVICE, 0.75, metadata
    
    # Default to SMALLTALK
    return Intent.SMALLTALK, 0.70, metadata

def detect_phone_number(text):
    """Detect Vietnamese phone numbers"""
    patterns = [
        r'0\d{9,10}',
        r'\+84\d{9,10}',
        r'84\d{9,10}'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def extract_location_from_query(text): 
    """Extract location from query"""
    location_keywords = {
        'huế': 'Huế',
        'đà nẵng': 'Đà Nẵng', 
        'hội an': 'Hội An',
        'quảng trị': 'Quảng Trị',
        'bạch mã': 'Bạch Mã'
    }
    
    text_lower = text.lower()
    for keyword, location in location_keywords.items():
        if keyword in text_lower:
            return location
    return None

# ==================== GLOBAL STATE INIT ====================
class GlobalState:
    """Global state with enhanced intent tracking"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize state"""
        self.tours_db: Dict[int, Dict] = {}
        self.tour_name_index: Dict[str, int] = {}
        self.tour_entities: List[Dict] = []
        self.about_company: Dict = {}
        self.session_contexts: Dict[str, Dict] = {}
        self.mapping: List[Dict] = []
        self.index = None
        self.vectors = None
        
        # Enhanced data structures
        self.tour_entities_dict: Dict[str, Dict] = {}
        self.tour_tags_index: Dict[str, List[int]] = defaultdict(list)
        self.tour_region_index: Dict[str, List[int]] = defaultdict(list)
        
        self.response_cache: OrderedDict = OrderedDict()
        self.embedding_cache: OrderedDict = OrderedDict()
        
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "sessions": 0,
            "leads": 0,
            "errors": 0,
            "meta_capi_calls": 0,
            "meta_capi_errors": 0,
            "intent_counts": defaultdict(int),
            "llm_calls": 0,
            "llm_errors": 0,
            "start_time": datetime.now()
        }
        
        self._knowledge_loaded = False
        self._index_loaded = False
        self._tour_entities_loaded = False
        self._company_info_loaded = False
        
        self.search_engine = None
        self.response_generator = None
        self.chat_processor = None
        
        logger.info("🌐 Global state initialized")
    
    def init_components(self):
        """Initialize components after knowledge loaded"""
        with self._lock:
            if self.search_engine is None:
                try:
                    from app import SearchEngine, ResponseGenerator, ChatProcessor
                    self.search_engine = SearchEngine()
                    logger.info("✅ SearchEngine initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize SearchEngine: {e}")
            
            if self.response_generator is None and self.search_engine is not None:
                try:
                    from app import ResponseGenerator
                    self.response_generator = ResponseGenerator()
                    logger.info("✅ ResponseGenerator initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize ResponseGenerator: {e}")
            
            if self.chat_processor is None and self.search_engine is not None and self.response_generator is not None:
                try:
                    from app import ChatProcessor
                    self.chat_processor = ChatProcessor()
                    logger.info("✅ ChatProcessor initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize ChatProcessor: {e}")
    
    def get_search_engine(self):
        """Get or create search engine"""
        if self.search_engine is None:
            self.init_components()
        return self.search_engine
    
    def get_response_generator(self):
        """Get or create response generator"""
        if self.response_generator is None:
            self.init_components()
        return self.response_generator
    
    def get_chat_processor(self):
        """Get or create chat processor"""
        if self.chat_processor is None:
            self.init_components()
        return self.chat_processor
    
    def get_tour(self, index: int) -> Optional[Dict]:
        """Get tour by index"""
        return self.tours_db.get(index)
    
    def get_session(self, session_id: str) -> Dict:
        """Get or create session context"""
        with self._lock:
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = {
                    "session_id": session_id,
                    "stage": ConversationStage.EXPLORE,
                    "intent": Intent.UNKNOWN,
                    "intent_metadata": {},
                    "mentioned_tours": [],
                    "selected_tour_id": None,
                    "location_filter": None,
                    "lead_phone": None,
                    "conversation_history": [],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now(),
                    "llm_context": []  # Store LLM context for coherence
                }
                self.stats["sessions"] += 1
                
                if len(self.session_contexts) > Config.MAX_SESSIONS:
                    self._cleanup_sessions()
            
            return self.session_contexts[session_id]
    
    def _cleanup_sessions(self):
        """Remove old sessions"""
        with self._lock:
            sorted_sessions = sorted(
                self.session_contexts.items(),
                key=lambda x: x[1].get("last_updated", datetime.min)
            )
            
            remove_count = max(1, len(sorted_sessions) // 3)
            for sid, _ in sorted_sessions[:remove_count]:
                del self.session_contexts[sid]
            
            logger.info(f"🧹 Cleaned {remove_count} old sessions")
    
    def get_cached_response(self, key: str) -> Optional[Dict]:
        """Get cached response"""
        if not Config.ENABLE_CACHING:
            return None
        
        with self._lock:
            if key in self.response_cache:
                entry = self.response_cache[key]
                if time.time() - entry['ts'] < Config.CACHE_TTL_SECONDS:
                    self.response_cache.move_to_end(key)
                    self.stats["cache_hits"] += 1
                    return entry['value']
                else:
                    del self.response_cache[key]
            
            self.stats["cache_misses"] += 1
            return None
    
    def cache_response(self, key: str, value: Dict):
        """Cache response"""
        if not Config.ENABLE_CACHING:
            return
        
        with self._lock:
            self.response_cache[key] = {
                'value': value,
                'ts': time.time()
            }
            
            if len(self.response_cache) > Config.MAX_EMBEDDING_CACHE:
                self.response_cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        with self._lock:
            uptime = datetime.now() - self.stats["start_time"]
            
            intent_dist = {}
            total_intents = sum(self.stats["intent_counts"].values())
            if total_intents > 0:
                for intent, count in self.stats["intent_counts"].items():
                    intent_dist[intent] = {
                        "count": count,
                        "percentage": round(count / total_intents * 100, 1)
                    }
            
            return {
                **self.stats,
                "uptime_seconds": int(uptime.total_seconds()),
                "active_sessions": len(self.session_contexts),
                "tours_loaded": len(self.tours_db),
                "mapping_entries": len(self.mapping),
                "cache_size": len(self.response_cache),
                "knowledge_loaded": self._knowledge_loaded,
                "intent_distribution": intent_dist,
                "company_info_loaded": self._company_info_loaded,
                "components_initialized": self.search_engine is not None,
                "llm_available": OPENAI_AVAILABLE and Config.OPENAI_API_KEY != ""
            }

# Initialize global state
state = GlobalState()

# ==================== IMPORT CUSTOM MODULES ====================
try:
    from meta_capi import (
        send_meta_pageview,
        send_meta_lead,
        send_meta_lead_from_entities,
        send_meta_call_button,
        check_meta_capi_health,
        config as meta_config
    )
    META_CAPI_AVAILABLE = True
    logger.info("✅ Meta CAPI module loaded")
    logger.info(
        f"META FLAGS | CALL={Config.ENABLE_META_CAPI_CALL}, "
        f"LEAD={Config.ENABLE_META_CAPI_LEAD}, "
        f"PIXEL={'OK' if Config.META_PIXEL_ID else 'MISSING'}, "
        f"TOKEN={'OK' if Config.META_CAPI_TOKEN else 'MISSING'}"
    )
except ImportError as e:
    logger.warning(f"⚠️ meta_capi.py not available: {e}")
    META_CAPI_AVAILABLE = False
    
    def send_meta_pageview(request): 
        pass
    
    def send_meta_lead(*args, **kwargs): 
        return {"status": "unavailable"}
    
    def send_meta_lead_from_entities(*args, **kwargs): 
        return {"status": "unavailable"}
    
    def send_meta_call_button(*args, **kwargs): 
        return {"status": "unavailable"}
    
    def check_meta_capi_health(): 
        return {"status": "unavailable", "message": "Meta CAPI module not loaded"}

try:
    from response_guard import validate_and_format_answer
    RESPONSE_GUARD_AVAILABLE = True
    logger.info("✅ Response guard module loaded")
except ImportError as e:
    logger.warning(f"⚠️ response_guard.py not available: {e}")
    RESPONSE_GUARD_AVAILABLE = False
    
    def validate_and_format_answer(llm_text, top_passages, **kwargs):
        return {
            "answer": llm_text or "Tôi đang tìm hiểu thông tin cho bạn...",
            "sources": [],
            "guard_passed": True,
            "reason": "no_guard"
        }

# ==================== FLASK APP ====================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_LENGTH", "1048576"))
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# Apply ProxyFix for Render
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# CORS
if Config.CORS_ORIGINS == "*":
    CORS(app, 
         origins="*",
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "X-Admin-Key"],
         supports_credentials=True)
else:
    CORS(app, 
         origins=Config.CORS_ORIGINS,
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "X-Admin-Key"],
         supports_credentials=True)

logger.info(f"✅ CORS configured for: {Config.CORS_ORIGINS}")

# ==================== ENHANCED KNOWLEDGE LOADER ====================
def load_knowledge() -> bool:
    """Load knowledge base with tour entities"""
    
    if state._knowledge_loaded:
        logger.info("📚 Knowledge already loaded, skipping")
        return True
    
    try:
        logger.info(f"📚 Loading knowledge from {Config.KNOWLEDGE_PATH}")
        
        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.error(f"❌ Knowledge file not found: {Config.KNOWLEDGE_PATH}")
            return False
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        # Load company info
        state.about_company = knowledge.get('about_company', {})
        if state.about_company:
            logger.info(f"✅ Company info loaded")
            state._company_info_loaded = True
        
        # Load tours
        tours_data = knowledge.get('tours', [])
        
        for idx, tour_data in enumerate(tours_data):
            try:
                state.tours_db[idx] = tour_data
                name = tour_data.get('tour_name', '')
                if name:
                    state.tour_name_index[name.lower()] = idx
            except Exception as e:
                logger.error(f"❌ Error loading tour {idx}: {e}")
                continue
        
        logger.info(f"✅ Knowledge loaded: {len(state.tours_db)} tours")
        
        # Load or create mapping
        if os.path.exists(Config.FAISS_MAPPING_PATH):
            try:
                with open(Config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    state.mapping = json.load(f)
                logger.info(f"✅ Mapping loaded: {len(state.mapping)} entries")
            except Exception as e:
                logger.error(f"❌ Error loading mapping: {e}")
                state.mapping = []
        else:
            state.mapping = []
            for idx, tour in state.tours_db.items():
                if not tour:
                    continue
                    
                fields_to_map = ['tour_name', 'location', 'duration', 'price', 'summary']
                
                for field in fields_to_map:
                    value = tour.get(field, '')
                    if value:
                        if isinstance(value, list):
                            value = ' '.join(str(v) for v in value if v)
                        value_str = str(value).strip()
                        if value_str and len(value_str) > 3:
                            state.mapping.append({
                                "path": f"tours[{idx}].{field}",
                                "text": value_str,
                                "tour_index": idx,
                                "field": field
                            })
            
            logger.info(f"✅ Mapping created: {len(state.mapping)} entries")
        
        # Load tour entities if available
        if os.path.exists(Config.TOUR_ENTITIES_PATH):
            try:
                with open(Config.TOUR_ENTITIES_PATH, 'r', encoding='utf-8') as f:
                    state.tour_entities_dict = json.load(f)
                
                # Build indices
                for tour_id, entity in state.tour_entities_dict.items():
                    tour_idx = entity.get('index')
                    if tour_idx is not None:
                        # Tag index
                        for tag in entity.get('tags', []):
                            state.tour_tags_index[tag].append(tour_idx)
                        # Region index
                        region = entity.get('region', '')
                        if region:
                            state.tour_region_index[region].append(tour_idx)
                
                logger.info(f"✅ Tour entities loaded: {len(state.tour_entities_dict)} entities")
                logger.info(f"   - Tags indexed: {len(state.tour_tags_index)}")
                logger.info(f"   - Regions indexed: {len(state.tour_region_index)}")
                state._tour_entities_loaded = True
            except Exception as e:
                logger.error(f"❌ Error loading tour entities: {e}")
        
        state._knowledge_loaded = True
        
        # Initialize components after knowledge loaded
        state.init_components()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        traceback.print_exc()
        return False

# ==================== ENHANCED SEARCH ENGINE ====================
class SearchEngine:
    """Enhanced search engine with guaranteed results for advisory intents"""
    
    def __init__(self):
        logger.info("🧠 Initializing search engine")
        self.openai_client = None
        
        # FIXED: OpenAI client initialization without proxies
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY
                )
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")
    
    def load_index(self) -> bool:
        """Load search index"""
        if state._index_loaded:
            return True
        
        try:
            if Config.FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info(f"📦 Loading FAISS index")
                state.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS loaded: {state.index.ntotal} vectors")
                state._index_loaded = True
                return True
            
            if NUMPY_AVAILABLE and os.path.exists(Config.FALLBACK_VECTORS_PATH):
                logger.info(f"📦 Loading numpy vectors")
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                
                if 'mat' in data:
                    state.vectors = data['mat']
                elif 'vectors' in data:
                    state.vectors = data['vectors']
                
                if state.vectors is not None:
                    norms = np.linalg.norm(state.vectors, axis=1, keepdims=True)
                    state.vectors = state.vectors / (norms + 1e-12)
                
                logger.info(f"✅ Numpy loaded: {state.vectors.shape[0]} vectors")
                state._index_loaded = True
                return True
            
            logger.info("ℹ️ No vector index found, using text search")
            state._index_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            state._index_loaded = True
            return False
    
    def search(self, query: str, top_k: int = None, intent: Any = None, metadata: Dict = None) -> List[Tuple[float, Dict]]:
        """
        Search with guaranteed results for tour-related intents
        NEVER returns empty for TOUR_INQUIRY, TOUR_ADVICE, TOUR_FILTER, etc.
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        # FIXED: Normalize intent
        intent_str = normalize_intent(intent)
        
        # FIXED: Skip vector search for company info (handled separately)
        if is_intent_equal(intent_str, Intent.ABOUT_COMPANY):
            logger.debug("🔍 Skipping vector search for ABOUT_COMPANY intent")
            return self._get_fallback_results_for_intent(intent_str, metadata, top_k)
        
        # If no mapping, use fallback immediately
        if not state.mapping:
            logger.warning("⚠️ Mapping is empty, using fallback")
            return self._get_fallback_results_for_intent(intent_str, metadata, top_k)
        
        # Perform text search
        query_lower = query.lower().strip()
        if not query_lower:
            # Return random samples if query is empty
            results = []
            sample_size = min(len(state.mapping), top_k * 2)
            for entry in random.sample(state.mapping, sample_size):
                results.append((0.3, entry))
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:top_k]
        
        query_words = [w for w in query_lower.split() if len(w) > 2]
        
        if not query_words:
            # Return random samples
            results = []
            sample_size = min(len(state.mapping), top_k * 2)
            for entry in random.sample(state.mapping, sample_size):
                results.append((0.2, entry))
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:top_k]
        
        # Text search scoring
        results = []
        for entry in state.mapping[:1000]:  # Limit for performance
            text = entry.get('text', '').lower()
            
            score = 0
            for word in query_words:
                if word in text:
                    score += 1
                elif any(word in t for t in text.split()):
                    score += 0.5
            
            # Boost score for tour_name matches
            if entry.get('field') == 'tour_name' and any(word in text for word in query_words):
                score += 2
            
            if score > 0:
                results.append((float(score), entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        # FIXED: If no results, use fallback for tour-related intents
        if not results:
            logger.warning(f"⚠️ No search results for query: '{query}'")
            return self._get_fallback_results_for_intent(intent_str, metadata, top_k)
        
        logger.debug(f"🔍 Text search found {len(results[:top_k])} results for query: '{query}'")
        return results[:top_k]
    
    def _get_fallback_results_for_intent(self, intent: str, metadata: Dict, top_k: int) -> List[Tuple[float, Dict]]:
        """
        Generate fallback results for tour-related intents
        NEVER returns empty for advisory intents
        """
        # List of intents that should NEVER get empty results
        advisory_intents = [
            Intent.TOUR_INQUIRY,
            Intent.TOUR_ADVICE,
            Intent.TOUR_FILTER,
            Intent.TOUR_RECOMMEND,
            Intent.TOUR_COMPARE,
            Intent.TOUR_LIST,
            Intent.PRICE_ASK
        ]
        
        # If intent is not advisory, return empty (allowing default response)
        if not any(is_intent_equal(intent, adv_intent) for adv_intent in advisory_intents):
            return []
        
        logger.info(f"🔄 Generating fallback results for intent: {intent}")
        
        # Get available tours
        available_tours = []
        for idx, tour in state.tours_db.items():
            if tour:
                available_tours.append(idx)
        
        if not available_tours:
            logger.error("❌ No tours available for fallback")
            return []
        
        # Apply filters from metadata
        filtered_tours = available_tours.copy()
        
        # Filter by duration
        duration_days = metadata.get('duration_days')
        if duration_days is not None:
            filtered_tours = [
                idx for idx in filtered_tours 
                if self._tour_matches_duration(idx, duration_days)
            ]
        
        # Filter by location
        location = metadata.get('location')
        if location:
            filtered_tours = [
                idx for idx in filtered_tours 
                if self._tour_matches_location(idx, location)
            ]
        
        # Filter by tags
        tags = metadata.get('tags', [])
        if tags:
            filtered_tours = [
                idx for idx in filtered_tours 
                if any(self._tour_has_tag(idx, tag) for tag in tags)
            ]
        
        # If filtering removed all tours, use original list
        if not filtered_tours:
            filtered_tours = available_tours
        
        # Limit to top_k
        filtered_tours = filtered_tours[:top_k * 2]
        
        # Create fake mapping entries
        results = []
        for tour_idx in filtered_tours:
            tour = state.get_tour(tour_idx)
            if not tour:
                continue
                
            # Create entry from tour data
            text_parts = []
            if tour.get('tour_name'):
                text_parts.append(f"Tên tour: {tour['tour_name']}")
            if tour.get('summary'):
                text_parts.append(f"Tóm tắt: {tour['summary']}")
            if tour.get('location'):
                text_parts.append(f"Địa điểm: {tour['location']}")
            
            entry = {
                "path": f"tours[{tour_idx}]",
                "text": " | ".join(text_parts),
                "tour_index": tour_idx,
                "field": "fallback",
                "tour_data": tour
            }
            
            # Calculate score based on relevance
            score = 0.5  # Base score
            if duration_days and self._tour_matches_duration(tour_idx, duration_days):
                score += 0.2
            if location and self._tour_matches_location(tour_idx, location):
                score += 0.2
            if tags and any(self._tour_has_tag(tour_idx, tag) for tag in tags):
                score += 0.1
            
            results.append((score, entry))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"✅ Generated {len(results[:top_k])} fallback results")
        return results[:top_k]
    
    def _tour_matches_duration(self, tour_idx: int, duration_days: int) -> bool:
        """Check if tour matches duration"""
        tour = state.get_tour(tour_idx)
        if not tour:
            return False
        
        duration_text = tour.get('duration', '').lower()
        return str(duration_days) in duration_text or f"{duration_days} ngày" in duration_text
    
    def _tour_matches_location(self, tour_idx: int, location: str) -> bool:
        """Check if tour matches location"""
        tour = state.get_tour(tour_idx)
        if not tour:
            return False
        
        tour_location = tour.get('location', '').lower()
        location_lower = location.lower()
        return location_lower in tour_location
    
    def _tour_has_tag(self, tour_idx: int, tag: str) -> bool:
        """Check if tour has tag"""
        if not state._tour_entities_loaded:
            return False
        
        # Find tour entity
        for entity in state.tour_entities_dict.values():
            if entity.get('index') == tour_idx:
                return tag in entity.get('tags', [])
        return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        if not text:
            return None
        
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=text[:2000]
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
        
        # Fallback
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        
        dim = 1536
        embedding = []
        for i in range(dim):
            val = ((hash_int >> (i % 32)) & 0xFF) / 255.0
            val = (val + (i % 7) / 7.0) % 1.0
            embedding.append(float(val))
        
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding

# ==================== ENHANCED RESPONSE GENERATOR ====================
class ResponseGenerator:
    """Enhanced response generator with LLM advisory capabilities"""
    
    def __init__(self):
        self.llm_client = None
        
        # FIXED: OpenAI client initialization without proxies
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.llm_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY
                )
                logger.info("✅ LLM client initialized")
            except Exception as e:
                logger.error(f"LLM client init failed: {e}")
    
    def generate(self, user_message: str, search_results: List, context: Dict) -> str:
        """Generate response with LLM advisory when appropriate"""
        
        intent_value = context.get("intent", Intent.UNKNOWN)
        intent_str = normalize_intent(intent_value)
        metadata = context.get("intent_metadata", {})
        
        logger.info(f"🎯 Generating response for intent: {intent_str}")
        
        # ==================== ROUTING LOGIC ====================
        # Check if we should use LLM for detailed advice
        should_use_llm = (
            Config.ENABLE_LLM_ADVICE and
            self.llm_client is not None and
            Config.OPENAI_API_KEY and
            any(is_intent_equal(intent_str, adv_intent) for adv_intent in [
                Intent.TOUR_ADVICE,
                Intent.TOUR_RECOMMEND,
                Intent.TOUR_COMPARE,
                Intent.TOUR_INQUIRY,
                Intent.TOUR_FILTER
            ])
        )
        
        if should_use_llm:
            logger.info("🧠 Using LLM for advisory response")
            try:
                return self._generate_llm_advice(user_message, search_results, context)
            except Exception as e:
                logger.error(f"LLM advisory failed: {e}")
                # Fall back to rule-based response
        
        # ==================== RULE-BASED ROUTING ====================
        if is_intent_equal(intent_str, Intent.GREETING):
            return self._generate_greeting()
        
        if is_intent_equal(intent_str, Intent.FAREWELL):
            return self._generate_farewell()
        
        if is_intent_equal(intent_str, Intent.ABOUT_COMPANY):
            return self._generate_about_company(metadata)
        
        if is_intent_equal(intent_str, Intent.TOUR_LIST):
            return self._generate_tour_list(metadata)
        
        if is_intent_equal(intent_str, Intent.TOUR_FILTER):
            return self._generate_tour_filter(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.TOUR_INQUIRY):
            return self._generate_tour_inquiry(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.TOUR_ADVICE):
            return self._generate_tour_advice(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.TOUR_COMPARE):
            return self._generate_tour_compare(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.PRICE_ASK):
            return self._generate_price_info(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.BOOKING_REQUEST):
            return self._generate_booking_info(search_results, metadata)
        
        if is_intent_equal(intent_str, Intent.CONTACT_INFO):
            return self._generate_contact_info()
        
        if is_intent_equal(intent_str, Intent.CALLBACK_REQUEST):
            return self._generate_callback_request(metadata)
        
        if is_intent_equal(intent_str, Intent.PROVIDE_PHONE):
            return self._generate_lead_confirm(metadata)
        
        if is_intent_equal(intent_str, Intent.SMALLTALK):
            return self._generate_smalltalk(search_results, metadata)
        
        # Default fallback
        return self._generate_fallback(search_results, metadata)
    
    # ==================== LLM ADVISORY FUNCTIONS ====================
    
    def _generate_llm_advice(self, user_message: str, search_results: List, context: Dict) -> str:
        """Generate advisory response using LLM"""
        if not self.llm_client:
            return self._generate_tour_inquiry(search_results, context.get("intent_metadata", {}))
        
        # Prepare context from search results
        tour_contexts = []
        for score, entry in search_results[:5]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None:
                tour = state.get_tour(tour_idx)
                if tour:
                    tour_context = {
                        "name": tour.get('tour_name', ''),
                        "summary": tour.get('summary', ''),
                        "location": tour.get('location', ''),
                        "duration": tour.get('duration', ''),
                        "price": tour.get('price', ''),
                        "style": tour.get('style', ''),
                        "tags": self._extract_tour_tags(tour_idx)
                    }
                    tour_contexts.append(tour_context)
        
        # Prepare company info
        company_info = ""
        if state.about_company:
            company_info = f"""
Công ty: {state.about_company.get('name', 'Ruby Wings')}
Sứ mệnh: {state.about_company.get('mission', '')}
Triết lý: {state.about_company.get('philosophy', '')}
"""
        
        # Build prompt
        prompt = f"""Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.
TRẢ LỜI THEO CÁC NGUYÊN TẮC:
1. ƯU TIÊN CAO NHẤT: Luôn sử dụng thông tin từ dữ liệu nội bộ được cung cấp.
2. Nếu thiếu thông tin CHI TIẾT, hãy tổng hợp và trả lời dựa trên THÔNG TIN CHUNG có sẵn.
3. Đối với tour cụ thể: nếu tìm thấy bất kỳ dữ liệu nội bộ liên quan nào, PHẢI tổng hợp và trình bày rõ ràng.
4. TUYỆT ĐỐI KHÔNG nói rằng bạn không đọc được file, không truy cập dữ liệu.
5. Luôn giữ thái độ nhiệt tình, hữu ích, trả lời trực tiếp vào nội dung người dùng hỏi.

Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn ngành du lịch trải nghiệm, retreat, thiền, khí công, hành trình chữa lành và các hành trình tham quan linh hoạt theo nhu cầu. Trả lời ngắn gọn, chính xác, rõ ràng, tử tế và bám sát dữ liệu Ruby Wings.

THÔNG TIN CÔNG TY:
{company_info}

CÁC TOUR CÓ SẴN:
{json.dumps(tour_contexts, ensure_ascii=False, indent=2)}

CÂU HỎI CỦA NGƯỜI DÙNG: "{user_message}"

YÊU CẦU:
1. Trả lời bằng tiếng Việt, thân thiện, nhiệt tình
2. Tập trung vào giá trị chữa lành, trải nghiệm sâu
3. Nếu có tour phù hợp, giới thiệu 2-3 tour
4. Nếu không có tour phù hợp, đưa ra gợi ý chung
5. Kết thúc bằng lời mời liên hệ hotline 0332510486

TRẢ LỜI:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            state.stats["llm_calls"] += 1
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✅ LLM generated {len(answer)} chars")
            
            return answer
            
        except Exception as e:
            state.stats["llm_errors"] += 1
            logger.error(f"LLM error: {e}")
            return self._generate_tour_inquiry(search_results, context.get("intent_metadata", {}))
    
    def _extract_tour_tags(self, tour_idx: int) -> List[str]:
        """Extract tags for a tour"""
        if not state._tour_entities_loaded:
            return []
        
        for entity in state.tour_entities_dict.values():
            if entity.get('index') == tour_idx:
                return entity.get('tags', [])
        return []
    
    # ==================== RULE-BASED RESPONSES ====================
    
    def _generate_greeting(self) -> str:
        greetings = [
            "Xin chào! Tôi là trợ lý AI của Ruby Wings 🌿\n\nTôi có thể giúp bạn:\n• Tìm hiểu về các tour trải nghiệm, retreat\n• Tư vấn tour phù hợp với nhu cầu\n• Giải đáp thông tin về Ruby Wings\n• Hỗ trợ đặt tour\n\nBạn muốn tìm hiểu về điều gì ạ? 😊",
            "Chào bạn! Rất vui được đồng hành cùng bạn trên hành trình khám phá những trải nghiệm sâu sắc tại Ruby Wings 🌈\n\nTôi có thể tư vấn về:\n• Các tour thiền, chữa lành\n• Hành trình khám phá thiên nhiên\n• Retreat công ty, team building\n• Tour gia đình, cá nhân\n\nBạn quan tâm đến chủ đề nào? 🤗"
        ]
        return random.choice(greetings)
    
    def _generate_farewell(self) -> str:
        farewells = [
            "Cảm ơn bạn đã đồng hành cùng Ruby Wings! Chúc bạn một ngày tràn đầy năng lượng và bình an 🌿\n\nNếu cần hỗ trợ thêm, đừng ngần ngại liên hệ:\n📞 Hotline: 0332510486\n🌐 Website: rubywings.vn\n\nHẹn gặp lại bạn trên những hành trình ý nghĩa! ✨",
            "Tạm biệt bạn! Mong rằng những thông tin tôi cung cấp đã hữu ích với bạn 🌈\n\nRuby Wings luôn sẵn sàng đồng hành cùng bạn trên hành trình:\n• Khám phá bản thân\n• Kết nối thiên nhiên\n• Tìm lại sự cân bằng\n\nLiên hệ ngay 0332510486 khi bạn sẵn sàng! 🌿"
        ]
        return random.choice(farewells)
    
    def _generate_about_company(self, metadata: Dict) -> str:
        """Generate company information response"""
        if not state.about_company:
            base_response = """**Ruby Wings** - Hành trình chữa lành và trải nghiệm sâu 🌿

Ruby Wings là đơn vị tiên phong trong lĩnh vực du lịch trải nghiệm, retreat, và hành trình chữa lành tại Miền Trung Việt Nam. Chúng tôi không chỉ tổ chức tour mà còn kiến tạo những hành trình chuyển hóa nội tâm.

**Triết lý hoạt động:**
• **4 cánh xanh lá:** Thân - Tâm - Thiên nhiên - Niềm tin
• **Viên ruby hồng:** Trái tim, sự chữa lành, tình yêu thương
• **Vòng tròn kết nối:** Sự tái sinh, hoàn thiện bản thân

**Hệ sinh thái Ruby Wings:**
• **Travel:** Du lịch trải nghiệm, retreat, hành trình chữa lành
• **Learn:** Giáo dục nội tâm, thiền, khí công
• **Stay:** Lưu trú xanh, homestay cộng đồng
• **Auto:** Di chuyển cân bằng, xe điện, xe xanh

**Sứ mệnh:** Lan tỏa giá trị sống chuẩn mực - chân thành - có chiều sâu
**Tầm nhìn 2030:** Trở thành đơn vị dẫn đầu về du lịch cảm xúc & phát triển con người tại miền Trung

Khám phá các hành trình của chúng tôi hoặc liên hệ **0332510486** để được tư vấn chi tiết! 🌈"""
            return base_response
        
        overview = state.about_company.get('overview', '')
        mission = state.about_company.get('mission', '')
        philosophy = state.about_company.get('philosophy', '')
        
        response = "**Ruby Wings** - Tổ chức du lịch trải nghiệm & chữa lành 🌈\n\n"
        
        if overview:
            response += f"{overview}\n\n"
        else:
            response += "Ruby Wings là đơn vị tiên phong trong lĩnh vực du lịch trải nghiệm, retreat, và hành trình chữa lành tại Miền Trung Việt Nam.\n\n"
        
        if mission:
            response += f"**Sứ mệnh:** {mission}\n\n"
        
        if philosophy:
            response += f"**Triết lý:** {philosophy}\n\n"
        
        response += """**Giá trị cốt lõi:**
• **Chuẩn mực:** Trong dịch vụ, đào tạo, vận hành
• **Chân thành:** Trong giao tiếp, đồng hành, phục vụ
• **Có chiều sâu:** Trong trải nghiệm, nội dung, kết nối

**Hệ sinh thái Ruby:**
• **Travel:** Du lịch chữa lành, retreat, hành trình trải nghiệm
• **Learn:** Giáo dục nội tâm, thiền định, khí công
• **Stay:** Lưu trú xanh, homestay cộng đồng
• **Auto:** Di chuyển cân bằng, phương tiện xanh

👉 Khám phá các hành trình của chúng tôi hoặc liên hệ **0332510486** để được tư vấn! 🌿"""
        
        return response
    
    def _generate_tour_list(self, metadata: Dict) -> str:
        """Generate list of all tours"""
        if not state.tours_db:
            return "Hiện tại chúng tôi đang cập nhật thông tin tour. Vui lòng liên hệ **0332510486** để được tư vấn trực tiếp về các hành trình sắp tới! 📞"
        
        tours = list(state.tours_db.values())
        
        if len(tours) > Config.MAX_TOURS_PER_RESPONSE:
            response = f"Ruby Wings hiện có **{len(tours)}** hành trình đa dạng. Dưới đây là những tour tiêu biểu:\n\n"
            # Select diverse tours
            selected_tours = []
            # Try to get different types
            for tour in tours:
                if len(selected_tours) >= Config.MAX_TOURS_PER_RESPONSE:
                    break
                selected_tours.append(tour)
        else:
            response = f"Ruby Wings có **{len(tours)}** hành trình:\n\n"
            selected_tours = tours
        
        for idx, tour in enumerate(selected_tours, 1):
            response += f"{idx}. **{tour.get('tour_name', 'Hành trình Ruby Wings')}**\n"
            
            if tour.get('duration'):
                response += f"   ⏱️ {tour['duration']}\n"
            
            if tour.get('location'):
                response += f"   📍 {tour['location']}\n"
            
            if tour.get('style'):
                style = tour['style']
                if isinstance(style, list):
                    style = ', '.join(style)
                response += f"   🎨 {style[:50]}...\n"
            
            response += "\n"
        
        response += """**Phân loại tour theo nhu cầu:**
• **Retreat & Chữa lành:** Tour thiền, khí công, tĩnh tâm
• **Trải nghiệm văn hóa:** Khám phá di sản, giao lưu cộng đồng
• **Thiên nhiên & Mạo hiểm:** Trekking, khám phá rừng núi
• **Team building:** Tour công ty, gắn kết tập thể
• **Gia đình & Cá nhân:** Hành trình nhẹ nhàng, phù hợp mọi lứa tuổi

Bạn quan tâm đến loại hành trình nào? Tôi có thể tư vấn chi tiết hơn! 😊"""
        
        return response
    
    def _generate_tour_filter(self, search_results: List, metadata: Dict) -> str:
        """Generate filtered tour response"""
        if not search_results:
            return self._generate_tour_list(metadata)
        
        response = "Dựa trên tiêu chí của bạn, đây là những hành trình phù hợp:\n\n"
        
        tours_mentioned = set()
        tour_details = []
        
        for score, entry in search_results[:Config.MAX_TOURS_PER_RESPONSE]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour:
                    tours_mentioned.add(tour_idx)
                    tour_details.append(tour)
        
        if not tour_details:
            return self._generate_tour_list(metadata)
        
        for idx, tour in enumerate(tour_details, 1):
            response += f"{idx}. **{tour.get('tour_name', 'Hành trình Ruby Wings')}**\n"
            
            if tour.get('duration'):
                response += f"   ⏱️ {tour['duration']}\n"
            
            if tour.get('location'):
                response += f"   📍 {tour['location']}\n"
            
            if tour.get('summary'):
                summary = tour['summary']
                if isinstance(summary, list):
                    summary = ' '.join(summary)
                response += f"   📖 {summary[:80]}...\n"
            
            if tour.get('price'):
                price = tour['price']
                if isinstance(price, list):
                    price = price[0] if price else "Liên hệ"
                response += f"   💰 {price}\n"
            
            response += "\n"
        
        response += "Bạn muốn tìm hiểu chi tiết về hành trình nào? Tôi có thể cung cấp thêm thông tin về lịch trình, dịch vụ bao gồm, và những trải nghiệm đặc biệt! 📱"
        
        return response
    
    def _generate_tour_inquiry(self, search_results: List, metadata: Dict) -> str:
        """Generate response for general tour inquiry"""
        if not search_results:
            # Use LLM if available
            if self.llm_client and Config.ENABLE_LLM_ADVICE:
                try:
                    return self._generate_llm_advice(
                        metadata.get('raw_query', 'Tư vấn tour'),
                        [],
                        {"intent": Intent.TOUR_INQUIRY, "intent_metadata": metadata}
                    )
                except:
                    pass
            
            return self._generate_tour_list(metadata)
        
        response = "Dựa trên yêu cầu của bạn, đây là những hành trình Ruby Wings phù hợp:\n\n"
        
        tours_mentioned = set()
        
        for score, entry in search_results[:Config.MAX_TOURS_PER_RESPONSE]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour:
                    tours_mentioned.add(tour_idx)
                    
                    response += f"**{tour.get('tour_name', 'Hành trình Ruby Wings')}**\n"
                    
                    if tour.get('location'):
                        response += f"📍 {tour['location']}\n"
                    if tour.get('duration'):
                        response += f"⏱️ {tour['duration']}\n"
                    if tour.get('summary'):
                        summary = tour['summary']
                        if isinstance(summary, list):
                            summary = ' '.join(summary)
                        response += f"📖 {summary[:100]}...\n"
                    
                    response += "\n"
        
        if not tours_mentioned:
            return self._generate_tour_list(metadata)
        
        response += """**Tại sao chọn Ruby Wings?**
• 🧘‍♀️ **Trải nghiệm sâu:** Thiền, khí công, chánh niệm
• 🌿 **Kết nối thiên nhiên:** Rừng núi, biển cả, không gian xanh
• 👥 **Đồng hành chân thành:** Hướng dẫn viên tâm huyết
• 💖 **Giá trị nhân văn:** Hành trình chữa lành, phát triển bản thân

Bạn muốn biết thêm chi tiết về hành trình nào? Hoặc có tiêu chí cụ thể nào khác? 😊"""
        
        return response
    
    def _generate_tour_advice(self, search_results: List, metadata: Dict) -> str:
        """Generate tour advice response"""
        # Try LLM first
        if self.llm_client and Config.ENABLE_LLM_ADVICE:
            try:
                return self._generate_llm_advice(
                    metadata.get('raw_query', 'Tư vấn tour'),
                    search_results,
                    {"intent": Intent.TOUR_ADVICE, "intent_metadata": metadata}
                )
            except Exception as e:
                logger.error(f"LLM advice failed: {e}")
        
        # Fallback to rule-based
        return self._generate_tour_inquiry(search_results, metadata)
    
    def _generate_tour_compare(self, search_results: List, metadata: Dict) -> str:
        """Generate tour comparison response"""
        if not search_results:
            return "Để so sánh các tour, bạn có thể cho tôi biết cụ thể bạn muốn so sánh tour nào? Hoặc bạn có thể tham khảo:\n\n" + self._generate_tour_list(metadata)
        
        response = "**So sánh các hành trình Ruby Wings**\n\n"
        
        tours_to_compare = []
        for score, entry in search_results[:3]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None:
                tour = state.get_tour(tour_idx)
                if tour and tour not in tours_to_compare:
                    tours_to_compare.append(tour)
        
        if len(tours_to_compare) < 2:
            response += "Tôi cần thêm thông tin để so sánh. Bạn có thể:\n"
            response += "1. Chỉ rõ tên 2 tour muốn so sánh\n"
            response += "2. So sánh theo tiêu chí: giá, thời gian, địa điểm, hoạt động\n"
            response += "3. Tham khảo danh sách tour đầy đủ\n\n"
            response += self._generate_tour_list(metadata)
            return response
        
        # Create comparison table
        response += "| Tiêu chí | " + " | ".join([t.get('tour_name', 'Tour')[:15] for t in tours_to_compare]) + " |\n"
        response += "|" + "|---" * (len(tours_to_compare) + 1) + "|\n"
        
        # Duration
        durations = [t.get('duration', 'N/A') for t in tours_to_compare]
        response += f"| Thời lượng | " + " | ".join(durations) + " |\n"
        
        # Location
        locations = [t.get('location', 'N/A') for t in tours_to_compare]
        response += f"| Địa điểm | " + " | ".join(locations) + " |\n"
        
        # Price
        prices = []
        for t in tours_to_compare:
            price = t.get('price', 'Liên hệ')
            if isinstance(price, list):
                price = price[0] if price else "Liên hệ"
            prices.append(price[:20])
        response += f"| Giá | " + " | ".join(prices) + " |\n"
        
        # Style
        styles = []
        for t in tours_to_compare:
            style = t.get('style', 'N/A')
            if isinstance(style, list):
                style = ', '.join(style[:2])
            styles.append(style[:20])
        response += f"| Phong cách | " + " | ".join(styles) + " |\n"
        
        response += "\n**Gợi ý lựa chọn:**\n"
        
        if len(tours_to_compare) >= 2:
            tour1 = tours_to_compare[0]
            tour2 = tours_to_compare[1]
            
            response += f"• **{tour1.get('tour_name', 'Tour 1')}:** "
            if tour1.get('duration') and '1 ngày' in str(tour1.get('duration', '')):
                response += "Phù hợp cho trải nghiệm ngắn ngày, cuối tuần\n"
            elif tour1.get('style') and any(x in str(tour1.get('style', '')).lower() for x in ['thiền', 'chữa lành']):
                response += "Tập trung vào trải nghiệm nội tâm, chữa lành\n"
            else:
                response += "Hành trình đa dạng trải nghiệm\n"
            
            response += f"• **{tour2.get('tour_name', 'Tour 2')}:** "
            if tour2.get('duration') and any(x in str(tour2.get('duration', '')) for x in ['2', '3', 'nhiều']):
                response += "Phù hợp cho hành trình dài ngày, khám phá sâu\n"
            elif tour2.get('style') and any(x in str(tour2.get('style', '')).lower() for x in ['mạo hiểm', 'trekking']):
                response += "Dành cho người yêu thích khám phá, mạo hiểm\n"
            else:
                response += "Hành trình cân bằng giữa nghỉ ngơi và trải nghiệm\n"
        
        response += "\nĐể được tư vấn chi tiết hơn, vui lòng liên hệ **0332510486** hoặc cung cấp thêm tiêu chí bạn quan tâm! 📞"
        
        return response
    
    def _generate_price_info(self, search_results: List, metadata: Dict) -> str:
        """Generate price information response"""
        response = "💰 **Thông tin giá các hành trình Ruby Wings**\n\n"
        
        tours_mentioned = set()
        
        for score, entry in search_results[:3]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour and tour.get('price'):
                    tours_mentioned.add(tour_idx)
                    
                    response += f"**{tour.get('tour_name', 'Hành trình')}**\n"
                    price = tour['price']
                    if isinstance(price, list):
                        price = "\n".join([f"   • {p}" for p in price[:3]])
                    response += f"{price}\n\n"
        
        if not tours_mentioned:
            response += "Giá tour Ruby Wings dao động từ **890.000 VNĐ** đến **3.500.000 VNĐ** tùy theo:\n"
            response += "• Thời lượng (1 ngày, 2N1Đ, 3N2Đ)\n"
            response += "• Loại hình (retreat, trekking, văn hóa)\n"
            response += "• Dịch vụ bao gồm (ăn uống, lưu trú, vé tham quan)\n"
            response += "• Số lượng người tham gia\n\n"
        
        response += """**Chính sách giá ưu đãi:**
• Giảm 5% cho nhóm từ 5 người trở lên
• Giảm 10% cho đặt tour trước 15 ngày
• Ưu đãi đặc biệt cho công ty, đoàn thể
• Combo gia đình (2 người lớn + 1 trẻ em)

**Giá đã bao gồm:**
✓ Xe đưa đón đời mới
✓ Hướng dẫn viên chuyên nghiệp
✓ Bữa ăn theo chương trình
✓ Vé tham quan các điểm
✓ Bảo hiểm du lịch
✓ Nước uống, khăn lạnh

Liên hệ **0332510486** để biết giá chi tiết và nhận ưu đãi phù hợp với nhu cầu của bạn! 📞"""
        
        return response
    
    def _generate_booking_info(self, search_results: List, metadata: Dict) -> str:
        """Generate booking information response"""
        response = "🎯 **Đặt hành trình Ruby Wings - 4 bước đơn giản**\n\n"
        response += "**Bước 1:** Chọn hành trình phù hợp\n"
        response += "   • Dựa trên thời gian, ngân sách, sở thích\n"
        response += "   • Tham khảo ý kiến tư vấn viên\n\n"
        
        response += "**Bước 2:** Cung cấp thông tin\n"
        response += "   • Số lượng người tham gia\n"
        response += "   • Ngày khởi hành dự kiến\n"
        response += "   • Yêu cầu đặc biệt (ăn chay, phòng đơn, ...)\n\n"
        
        response += "**Bước 3:** Xác nhận & Thanh toán\n"
        response += "   • Nhận báo giá chi tiết\n"
        response += "   • Chọn phương thức thanh toán (chuyển khoản, tiền mặt)\n"
        response += "   • Nhận hợp đồng & hóa đơn\n\n"
        
        response += "**Bước 4:** Chuẩn bị hành trình\n"
        response += "   • Nhận thông tin chi tiết trước tour\n"
        response += "   • Checklist đồ dùng cần mang\n"
        response += "   • Liên hệ hỗ trợ 24/7\n\n"
        
        response += """**Cách thức đặt tour:**
1. 📞 **Gọi hotline:** 0332510486 (8:00 - 22:00)
2. 💬 **Nhắn tin Zalo:** 0332510486
3. 📧 **Email:** info@rubywings.vn
4. 🌐 **Website:** rubywings.vn
5. 👥 **Facebook:** Ruby Wings Travel

**Ưu đãi khi đặt tour:**
• Tặng voucher 200.000 VNĐ cho lần đặt tiếp theo
• Miễn phí tư vấn thiết kế tour riêng
• Hỗ trợ đón tiễn sân bay (với tour dài ngày)

Chúng tôi sẽ xác nhận trong vòng 30 phút và đồng hành cùng bạn suốt hành trình! 🌈"""
        
        return response
    
    def _generate_contact_info(self) -> str:
        """Generate contact information response"""
        response = "📞 **Liên hệ Ruby Wings - Kết nối từ trái tim**\n\n"
        response += "**Thông tin liên hệ chính thức:**\n"
        response += "• **Hotline:** 0332510486\n"
        response += "• **Zalo:** 0332510486\n"
        response += "• **Email:** info@rubywings.vn\n"
        response += "• **Website:** rubywings.vn\n"
        response += "• **Facebook:** Ruby Wings Travel\n"
        response += "• **Địa chỉ:** 148 Trương Gia Mô, Vỹ Dạ, TP. Huế\n\n"
        
        response += "⏰ **Thời gian làm việc:**\n"
        response += "• Thứ 2 - Thứ 6: 8:00 - 17:00\n"
        response += "• Thứ 7: 8:00 - 12:00\n"
        response += "• Chủ nhật: Hỗ trợ qua điện thoại\n"
        response += "• Hỗ trợ khẩn cấp: 24/7 qua hotline\n\n"
        
        response += """**Dịch vụ hỗ trợ:**
• 🗣️ **Tư vấn tour:** Miễn phí, chi tiết, nhiệt tình
• 📋 **Thiết kế tour riêng:** Theo nhu cầu cá nhân, công ty
• 🚗 **Đón tiễn sân bay:** Dịch vụ xe riêng
• 🏨 **Đặt phòng khách sạn:** Ưu đãi đối tác
• 🍽️ **Thực đơn đặc biệt:** Chay, kiêng khem, dị ứng
• 👨‍👩‍👧‍👦 **Hỗ trợ gia đình:** Trẻ em, người lớn tuổi

**Cam kết từ Ruby Wings:**
• Phản hồi trong 15 phút
• Tư vấn không mệt mỏi
• Hỗ trợ tận tâm suốt hành trình
• Giá trị thật - dịch vụ thật

Chúng tôi sẵn sàng lắng nghe và đồng hành cùng bạn! 🌿"""
        
        return response
    
    def _generate_callback_request(self, metadata: Dict) -> str:
        """Generate callback request response"""
        response = "✅ **Yêu cầu gọi lại đã được ghi nhận!**\n\n"
        response += "Đội ngũ Ruby Wings sẽ liên hệ với bạn trong thời gian sớm nhất (thường trong vòng 15-30 phút).\n\n"
        
        response += """**Thông tin chúng tôi sẽ trao đổi:**
1. Nhu cầu và mong muốn của bạn về hành trình
2. Gợi ý các tour phù hợp nhất
3. Thông tin về giá, lịch trình chi tiết
4. Ưu đãi đặc biệt dành riêng cho bạn
5. Giải đáp mọi thắc mắc liên quan\n\n"""
        
        response += "**Hoặc bạn có thể chủ động liên hệ ngay:**\n"
        response += "• 📞 **Gọi trực tiếp:** 0332510486\n"
        response += "• 💬 **Nhắn tin Zalo:** 0332510486\n"
        response += "• 📧 **Email:** info@rubywings.vn\n\n"
        
        response += "Cảm ơn bạn đã tin tưởng Ruby Wings! Hẹn sớm được đồng hành cùng bạn trên những hành trình ý nghĩa 🌈"
        
        return response
    
    def _generate_lead_confirm(self, metadata: Dict) -> str:
        """Generate lead confirmation response"""
        phone = metadata.get('phone_number', '')
        masked_phone = phone[:3] + '***' + phone[-2:] if phone else '***'
        
        response = "✅ **Thông tin của bạn đã được lưu thành công!**\n\n"
        response += f"**Số điện thoại:** {masked_phone}\n"
        response += "**Thời gian tiếp nhận:** " + datetime.now().strftime("%H:%M %d/%m/%Y") + "\n\n"
        
        response += """**Quy trình tiếp theo:**
1. **Trong 15 phút:** Chuyên viên Ruby Wings sẽ gọi điện tư vấn
2. **Tư vấn chi tiết:** Hiểu nhu cầu, gợi ý tour phù hợp
3. **Báo giá & Ưu đãi:** Nhận báo giá chi tiết và ưu đãi đặc biệt
4. **Xác nhận đặt tour:** Hướng dẫn các bước đặt tour\n\n"""
        
        def _generate_lead_confirm(self, metadata: Dict) -> str:
            """Generate lead confirmation response"""
        phone = metadata.get('phone_number', '')
        masked_phone = phone[:3] + '***' + phone[-2:] if phone else '***'
        
        response = "✅ **Thông tin của bạn đã được lưu thành công!**\n\n"
        response += f"**Số điện thoại:** {masked_phone}\n"
        response += "**Thời gian tiếp nhận:** " + datetime.now().strftime("%H:%M %d/%m/%Y") + "\n\n"
        
        response += """**Quy trình tiếp theo:**
1. **Trong 15 phút:** Chuyên viên Ruby Wings sẽ gọi điện tư vấn
2. **Tư vấn chi tiết:** Hiểu nhu cầu, gợi ý tour phù hợp
3. **Báo giá & Ưu đãi:** Nhận báo giá chi tiết và ưu đãi đặc biệt
4. **Xác nhận đặt tour:** Hướng dẫn các bước đặt tour\n\n"""
        
        response += """**Lưu ý quan trọng:**
• Vui lòng giữ máy điện thoại trong 15 phút tới
• Nếu không nhận được cuộc gọi, vui lòng gọi lại 0332510486
• Kiểm tra hộp thư rác nếu sử dụng email\n\n"""
        
        response += "Cảm ơn bạn đã tin tưởng Ruby Wings! Chúng tôi rất mong được đồng hành cùng bạn 🌿"
        
        return response
    
    def _generate_smalltalk(self, search_results: List, metadata: Dict) -> str:
        """Generate smalltalk response"""
        responses = [
            "Tôi có thể giúp gì cho bạn về các hành trình Ruby Wings? Bạn quan tâm đến tour retreat, trải nghiệm văn hóa, hay hành trình chữa lành? 😊",
            "Xin chào! Ruby Wings có nhiều hành trình ý nghĩa đang chờ bạn khám phá. Bạn muốn tìm hiểu về chủ đề nào? 🌿",
            "Chào bạn! Tôi sẵn sàng tư vấn về các tour trải nghiệm, retreat, và hành trình phát triển bản thân tại Ruby Wings. Bạn có quan tâm đến lĩnh vực nào? 🌈"
        ]
        return random.choice(responses)
    
    def _generate_fallback(self, search_results: List, metadata: Dict) -> str:
        """Generate fallback response"""
        responses = [
            "Tôi có thể giúp gì cho bạn về các hành trình Ruby Wings? Bạn có thể hỏi về tour, giá cả, đặt tour, hoặc liên hệ **0332510486** để được hỗ trợ trực tiếp! 📞",
            "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể hỏi về:\n• Các tour hiện có của Ruby Wings\n• Thông tin về công ty\n• Giá tour và ưu đãi\n• Cách đặt tour\n• Liên hệ hỗ trợ\n\nHoặc gọi ngay **0332510486** để được tư vấn nhanh nhất! 🌿",
            "Ruby Wings chuyên về du lịch trải nghiệm, retreat, và hành trình chữa lành. Bạn muốn tìm hiểu về:\n1. 🧘‍♀️ Tour thiền, khí công\n2. 🌳 Khám phá thiên nhiên\n3. 🏛️ Trải nghiệm văn hóa\n4. 👥 Team building công ty\n5. 👨‍👩‍👧‍👦 Tour gia đình\n\nCho tôi biết bạn quan tâm đến chủ đề nào nhé! 😊"
        ]
        return random.choice(responses)

# ==================== ENHANCED CHAT PROCESSOR ====================
class ChatProcessor:
    """Enhanced chat processor with guaranteed context"""
    
    def __init__(self):
        self.response_generator = state.get_response_generator()
        self.search_engine = state.get_search_engine()
    
    def ensure_knowledge_loaded(self):
        """Ensure knowledge is loaded"""
        if not state._knowledge_loaded:
            logger.warning("⚠️ Knowledge not initialized")
            if not load_knowledge():
                logger.error("❌ Failed to load knowledge")
                return False
            
            search_engine = state.get_search_engine()
            if search_engine:
                search_engine.load_index()
                logger.info("✅ Knowledge ready")
                return True
            else:
                logger.error("❌ Search engine not available")
                return False
        return True
    
    def process(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process user message with guaranteed context"""
        start_time = time.time()
        
        try:
            if not self.ensure_knowledge_loaded():
                return {
                    'reply': "Xin lỗi, hệ thống đang khởi tạo. Vui lòng thử lại sau 1 phút hoặc liên hệ **0332510486** để được hỗ trợ ngay! 🙏",
                    'session_id': session_id,
                    'error': 'knowledge_not_loaded',
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get session context
            context = state.get_session(session_id)
            context['last_updated'] = datetime.now()
            
            # Check cache
            cache_key = f"{session_id}:{hashlib.md5(user_message.encode()).hexdigest()[:12]}"
            cached = state.get_cached_response(cache_key)
            if cached:
                logger.info(f"💾 Cache hit: {session_id}")
                cached['processing_time_ms'] = int((time.time() - start_time) * 1000)
                cached['from_cache'] = True
                return cached
            
            # Detect intent with enhanced detection
            intent, confidence, metadata = detect_intent(user_message)
            
            # Store normalized intent
            context['intent'] = normalize_intent(intent)
            context['intent_metadata'] = metadata
            
            # Update intent statistics
            state.stats['intent_counts'][context['intent']] += 1
            
            # Detect phone number
            phone = metadata.get('phone_number') or detect_phone_number(user_message)
            if phone:
                context['lead_phone'] = phone
                context['stage'] = ConversationStage.LEAD.value
                
                if Config.ENABLE_LEAD_CAPTURE:
                    self._capture_lead(phone, session_id, user_message, context)
            
            # FIXED: Always get search results, search engine guarantees non-empty for advisory intents
            search_results = []
            if self.search_engine:
                search_results = self.search_engine.search(
                    user_message, 
                    Config.TOP_K, 
                    intent=context['intent'],
                    metadata=metadata
                )
                
                # Log search results
                logger.info(f"🔍 Search returned {len(search_results)} results for intent: {context['intent']}")
            
            # Extract mentioned tours
            mentioned_tours = []
            for score, entry in search_results:
                tour_idx = entry.get('tour_index')
                if tour_idx is not None and tour_idx not in mentioned_tours:
                    mentioned_tours.append(tour_idx)
            
            context['mentioned_tours'] = mentioned_tours
            
            # Generate response with guaranteed context
            response_text = ""
            if self.response_generator:
                response_text = self.response_generator.generate(
                    user_message,
                    search_results,
                    context
                )
            else:
                response_text = "Xin chào! Tôi có thể giúp gì cho bạn về các hành trình Ruby Wings? 🌿"
            
            # Apply response guard if available
            if RESPONSE_GUARD_AVAILABLE:
                try:
                    guarded = validate_and_format_answer(
                        response_text,
                        [(s, e) for s, e in search_results],
                        context=context
                    )
                    response_text = guarded.get('answer', response_text)
                except Exception as e:
                    logger.error(f"Response guard error: {e}")
            
            # Add conversation to history
            context.setdefault('conversation_history', []).append({
                'role': 'user',
                'content': user_message[:200],
                'timestamp': datetime.now().isoformat()
            })
            context['conversation_history'].append({
                'role': 'assistant',
                'content': response_text[:200],
                'timestamp': datetime.now().isoformat()
            })
            
            # Build result
            result = {
                'reply': response_text,
                'session_id': session_id,
                'session_state': {
                    'stage': context.get('stage'),
                    'intent': context.get('intent'),
                    'intent_metadata': metadata,
                    'mentioned_tours': mentioned_tours,
                    'has_phone': bool(phone)
                },
                'intent': {
                    'name': context['intent'],
                    'confidence': confidence,
                    'metadata': metadata
                },
                'search': {
                    'results_count': len(search_results),
                    'tours': mentioned_tours
                },
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'from_cache': False,
                'timestamp': datetime.now().isoformat(),
                'llm_used': 'llm' in response_text.lower()[:100]  # Simple detection
            }
            
            # Cache result
            state.cache_response(cache_key, result)
            
            # Update stats
            state.stats['requests'] += 1
            
            # Log
            processing_time = result['processing_time_ms']
            logger.info(f"⏱️ Processed in {processing_time}ms | "
                       f"Intent: {context['intent']} | "
                       f"Results: {len(search_results)} | "
                       f"Chars: {len(response_text)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            traceback.print_exc()
            
            state.stats['errors'] += 1
            
            return {
                'reply': "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau 1 phút hoặc liên hệ trực tiếp **0332510486** để được hỗ trợ ngay! 🙏",
                'session_id': session_id,
                'error': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }
    
    def _capture_lead(self, phone: str, session_id: str, message: str, context: Dict):
        """Capture lead data"""
        try:
            phone_clean = re.sub(r'[^\d+]', '', phone)
            
            lead_data = {
                'timestamp': datetime.now().isoformat(),
                'source_channel': 'Website',
                'action_type': 'Chatbot',
                'page_url': '',
                'contact_name': 'Khách hàng từ chatbot',
                'phone': phone_clean,
                'service_interest': ', '.join(map(str, context.get('mentioned_tours', []))),
                'note': message[:200],
                'status': 'New',
                'session_id': session_id,
                'intent': context.get('intent', ''),
                'tour_id': context.get('mentioned_tours', [None])[0] if context.get('mentioned_tours') else None,
                'stage': context.get('stage', '')
            }
            
            if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
                try:
                    result = send_meta_lead(
                        request,
                        phone=phone_clean,
                        content_name="Chatbot Lead Capture",
                        value=200000,
                        currency="VND"
                    )
                    state.stats['meta_capi_calls'] += 1
                    logger.info(f"✅ Lead sent to Meta CAPI: {phone_clean[:4]}***")
                except Exception as e:
                    state.stats['meta_capi_errors'] += 1
                    logger.error(f"Meta CAPI lead error: {e}")
            
            if Config.ENABLE_GOOGLE_SHEETS:
                self._save_to_sheets(lead_data)
            
            if Config.ENABLE_FALLBACK_STORAGE:
                self._save_to_fallback(lead_data)
            
            state.stats['leads'] += 1
            logger.info(f"📞 Lead captured: {phone_clean[:4]}***{phone_clean[-2:]}")
            
        except Exception as e:
            logger.error(f"Lead capture error: {e}")
    
    def _save_to_sheets(self, lead_data: Dict):
        """Save to Google Sheets"""
        try:
            if not Config.GOOGLE_SERVICE_ACCOUNT_JSON or not Config.GOOGLE_SHEET_ID:
                logger.warning("Google Sheets not configured")
                return
            
            import gspread
            from google.oauth2.service_account import Credentials
            
            creds_json = json.loads(Config.GOOGLE_SERVICE_ACCOUNT_JSON)
            creds = Credentials.from_service_account_info(
                creds_json,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(Config.GOOGLE_SHEET_ID)
            ws = sh.worksheet(Config.GOOGLE_SHEET_NAME)
            
            row = [
                str(lead_data.get('timestamp', '')),
                str(lead_data.get('source_channel', '')),
                str(lead_data.get('action_type', '')),
                str(lead_data.get('page_url', '')),
                str(lead_data.get('contact_name', '')),
                str(lead_data.get('phone', '')),
                str(lead_data.get('service_interest', '')),
                str(lead_data.get('note', '')),
                str(lead_data.get('status', '')),
                str(lead_data.get('session_id', '')),
                str(lead_data.get('intent', '')),
                str(lead_data.get('tour_id', '') if lead_data.get('tour_id') else ''),
                str(lead_data.get('stage', ''))
            ]
            
            ws.append_row(row, value_input_option='USER_ENTERED')
            logger.info(f"✅ Saved to Google Sheets: {len(row)} values")
            
        except Exception as e:
            logger.error(f"Google Sheets error: {e}")
    
    def _save_to_fallback(self, lead_data: Dict):
        """Save to fallback JSON file"""
        try:
            if os.path.exists(Config.FALLBACK_STORAGE_PATH):
                with open(Config.FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                    leads = json.load(f)
            else:
                leads = []
            
            leads.append(lead_data)
            leads = leads[-1000:]
            
            with open(Config.FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                json.dump(leads, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Saved to fallback storage")
            
        except Exception as e:
            logger.error(f"Fallback storage error: {e}")

# ==================== ROUTES ====================
@app.before_request
def before_request():
    """Before request handler"""
    g.start_time = time.time()
    
    if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
        try:
            if request.path not in ['/health', '/stats', '/favicon.ico']:
                send_meta_pageview(request)
                state.stats['meta_capi_calls'] += 1
        except Exception as e:
            state.stats['meta_capi_errors'] += 1
            logger.error(f"Meta CAPI pageview error: {e}")

@app.after_request
def after_request(response):
    """After request handler"""
    if hasattr(g, 'start_time'):
        elapsed = (time.time() - g.start_time) * 1000
        response.headers['X-Processing-Time'] = f"{elapsed:.2f}ms"
    
    return response

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '6.0.0-complete-fix',
        'timestamp': datetime.now().isoformat(),
        'knowledge': {
            'loaded': state._knowledge_loaded,
            'tours': len(state.tours_db),
            'company_info_loaded': state._company_info_loaded,
            'tour_entities_loaded': state._tour_entities_loaded
        },
        'modules': {
            'meta_capi': META_CAPI_AVAILABLE,
            'response_guard': RESPONSE_GUARD_AVAILABLE,
            'openai': OPENAI_AVAILABLE,
            'llm_enabled': Config.OPENAI_API_KEY != ""
        },
        'components': {
            'search_engine': state.search_engine is not None,
            'response_generator': state.response_generator is not None,
            'chat_processor': state.chat_processor is not None
        },
        'performance': {
            'cache_size': len(state.response_cache),
            'active_sessions': len(state.session_contexts),
            'requests_processed': state.stats['requests']
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Index route"""
    return jsonify({
        'service': 'Ruby Wings AI Chatbot',
        'version': '6.0.0 (Complete Fix)',
        'status': 'running',
        'tours_available': len(state.tours_db),
        'features': {
            'llm_advisory': Config.ENABLE_LLM_ADVICE,
            'intent_detection': Config.ENABLE_INTENT_DETECTION,
            'lead_capture': Config.ENABLE_LEAD_CAPTURE,
            'meta_capi': Config.ENABLE_META_CAPI
        },
        'endpoints': {
            'chat': '/api/chat',
            'save_lead': '/api/save-lead',
            'health': '/health',
            'stats': '/stats'
        }
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        chat_processor = state.get_chat_processor()
        if chat_processor is None:
            return jsonify({
                'error': 'Chat processor not initialized',
                'message': 'Xin lỗi, hệ thống đang khởi tạo. Vui lòng thử lại sau 1 phút hoặc liên hệ **0332510486**!'
            }), 503
        
        result = chat_processor.process(user_message, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        traceback.print_exc()
        state.stats['errors'] += 1
        
        return jsonify({
            'error': 'Internal server error',
            'message': 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ **0332510486**!'
        }), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_legacy():
    """Legacy /chat endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        chat_processor = state.get_chat_processor()
        if chat_processor is None:
            return jsonify({
                'error': 'Chat processor not initialized',
                'message': 'Xin lỗi, hệ thống đang khởi tạo. Vui lòng thử lại sau!'
            }), 503
        
        result = chat_processor.process(user_message, session_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ /chat error: {e}")
        traceback.print_exc()
        state.stats['errors'] += 1
        return jsonify({
            'error': 'Internal server error',
            'message': 'Xin lỗi, có lỗi xảy ra!'
        }), 500

@app.route('/api/save-lead', methods=['POST', 'OPTIONS'])
def save_lead():
    """Save lead from form submission"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        
        phone = data.get('phone', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        tour_interest = data.get('tour_interest', '').strip()
        page_url = data.get('page_url', '').strip()
        source_channel = data.get('source_channel', 'Website').strip()
        action_type = data.get('action_type', 'Lead Form').strip()
        note = data.get('note', '').strip()
        
        if not phone:
            return jsonify({'error': 'Phone number is required'}), 400
        
        phone_clean = re.sub(r'[^\d+]', '', phone)
        
        if not re.match(r'^(0|\+?84)\d{9,10}$', phone_clean):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        lead_data = {
            'timestamp': datetime.now().isoformat(),
            'source_channel': source_channel,
            'action_type': action_type,
            'page_url': page_url or request.referrer or '',
            'contact_name': name or 'Khách yêu cầu gọi lại',
            'phone': phone_clean,
            'service_interest': tour_interest,
            'note': note,
            'status': 'New',
            'session_id': '',
            'intent': '',
            'tour_id': None,
            'stage': ''
        }
        
        if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
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
                state.stats['meta_capi_calls'] += 1
                logger.info(f"✅ Form lead sent to Meta CAPI: {phone_clean[:4]}***")
            except Exception as e:
                state.stats['meta_capi_errors'] += 1
                logger.error(f"Meta CAPI error: {e}")
        
        if Config.ENABLE_GOOGLE_SHEETS:
            try:
                import gspread
                from google.oauth2.service_account import Credentials
                
                if Config.GOOGLE_SERVICE_ACCOUNT_JSON and Config.GOOGLE_SHEET_ID:
                    creds_json = json.loads(Config.GOOGLE_SERVICE_ACCOUNT_JSON)
                    creds = Credentials.from_service_account_info(
                        creds_json,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    
                    gc = gspread.authorize(creds)
                    sh = gc.open_by_key(Config.GOOGLE_SHEET_ID)
                    ws = sh.worksheet(Config.GOOGLE_SHEET_NAME)
                    
                    row = [
                        str(lead_data['timestamp']),
                        str(lead_data['source_channel']),
                        str(lead_data['action_type']),
                        str(lead_data['page_url']),
                        str(lead_data['contact_name']),
                        str(lead_data['phone']),
                        str(lead_data['service_interest']),
                        str(lead_data['note']),
                        str(lead_data['status']),
                        str(lead_data['session_id']),
                        str(lead_data['intent']),
                        str(lead_data['tour_id']) if lead_data['tour_id'] else '',
                        str(lead_data['stage'])
                    ]
                    
                    ws.append_row(row, value_input_option='USER_ENTERED')
                    logger.info(f"✅ Form lead saved to Google Sheets")
            except Exception as e:
                logger.error(f"Google Sheets error: {e}")
        
        if Config.ENABLE_FALLBACK_STORAGE:
            try:
                if os.path.exists(Config.FALLBACK_STORAGE_PATH):
                    with open(Config.FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                        leads = json.load(f)
                else:
                    leads = []
                
                leads.append(lead_data)
                leads = leads[-1000:]
                
                with open(Config.FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(leads, f, ensure_ascii=False, indent=2)
                
                logger.info("✅ Form lead saved to fallback storage")
            except Exception as e:
                logger.error(f"Fallback storage error: {e}")
        
        state.stats['leads'] += 1
        
        return jsonify({
            'success': True,
            'message': 'Thông tin của bạn đã được lưu thành công! Đội ngũ Ruby Wings sẽ liên hệ trong vòng 15 phút. Cảm ơn bạn đã tin tưởng! 🌿',
            'data': {
                'phone': phone_clean[:3] + '***' + phone_clean[-2:],
                'timestamp': lead_data['timestamp']
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Save lead error: {e}")
        traceback.print_exc()
        state.stats['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Statistics endpoint"""
    return jsonify(state.get_stats())

# ==================== INITIALIZATION ====================
def initialize_app():
    """Initialize application - ONLY CALLED IN MAIN"""
    try:
        logger.info("🚀 Initializing Ruby Wings Chatbot v6.0.0 (Complete Fix)...")
        
        Config.log_config()
        
        logger.info("🔍 Loading knowledge...")
        if not load_knowledge():
            logger.error("❌ Failed to load knowledge")
        else:
            logger.info("✅ Knowledge loaded")
        
        # Load search index through search engine
        search_engine = state.get_search_engine()
        if search_engine:
            logger.info("🔍 Loading search index...")
            search_engine.load_index()
        
        logger.info("=" * 60)
        logger.info("✅ RUBY WINGS CHATBOT READY!")
        logger.info(f"📊 Tours loaded: {len(state.tours_db)}")
        logger.info(f"🧠 LLM Advisory: {'✅' if Config.ENABLE_LLM_ADVICE and Config.OPENAI_API_KEY else '❌'}")
        logger.info(f"🌐 Server: {Config.HOST}:{Config.PORT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        traceback.print_exc()

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == '__main__':
    initialize_app()

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False
    )

__all__ = ["app"]