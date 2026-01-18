#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 6.0.1 (COMPLETE FIX)
Created: 2025-01-17
Author: Ruby Wings AI Team

FIX V6.0.1: URGENT FIX FOR OPENAI CLIENT INITIALIZATION
- FIXED: Client.__init__() không mong muốn 'proxies' parameter
- FIXED: OpenAI client initialization trong cả SearchEngine và ResponseGenerator
- ENHANCED: Logging bằng tiếng Việt cho dễ đọc
- OPTIMIZED: Performance với minimal initialization
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
import warnings
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict

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
logger = logging.getLogger("ruby-wings-v6.0.1-fixed")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration"""
    
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
        logger.info("🚀 RUBY WINGS CHATBOT v6.0.1 (Fixed OpenAI Client)")
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
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    SMALLTALK = "SMALLTALK"
    UNKNOWN = "UNKNOWN"
    TOUR_INQUIRY = "TOUR_INQUIRY"
    TOUR_LIST = "TOUR_LIST"
    TOUR_FILTER = "TOUR_FILTER"
    TOUR_DETAIL = "TOUR_DETAIL"
    TOUR_COMPARE = "TOUR_COMPARE"
    TOUR_RECOMMEND = "TOUR_RECOMMEND"
    TOUR_ADVICE = "TOUR_ADVICE"
    PRICE_ASK = "PRICE_ASK"
    PRICE_COMPARE = "PRICE_COMPARE"
    PRICE_RANGE = "PRICE_RANGE"
    BOOKING_REQUEST = "BOOKING_REQUEST"
    BOOKING_PROCESS = "BOOKING_PROCESS"
    BOOKING_CONDITION = "BOOKING_CONDITION"
    PROVIDE_PHONE = "PROVIDE_PHONE"
    CALLBACK_REQUEST = "CALLBACK_REQUEST"
    CONTACT_INFO = "CONTACT_INFO"
    ABOUT_COMPANY = "ABOUT_COMPANY"
    COMPANY_SERVICE = "COMPANY_SERVICE"
    COMPANY_MISSION = "COMPANY_MISSION"
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
        logger.warning("⚠️ Numpy không khả dụng")
        return None, False

def lazy_import_faiss():
    """Lazy import FAISS"""
    if not Config.FAISS_ENABLED:
        return None, False
    try:
        import faiss
        return faiss, True
    except ImportError:
        logger.warning("⚠️ FAISS không khả dụng, sử dụng numpy fallback")
        return None, False

def lazy_import_openai():
    """Lazy import OpenAI - FIXED: Không có proxies parameter"""
    try:
        from openai import OpenAI
        return OpenAI, True
    except ImportError:
        logger.error("❌ Thư viện OpenAI không khả dụng")
        return None, False

# Initialize lazy imports
np, NUMPY_AVAILABLE = lazy_import_numpy()
faiss, FAISS_AVAILABLE = lazy_import_faiss()
OpenAI, OPENAI_AVAILABLE = lazy_import_openai()

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
        
        logger.info("🌐 Global state đã được khởi tạo")
    
    def init_components(self):
        """Initialize components after knowledge loaded"""
        with self._lock:
            if self.search_engine is None:
                try:
                    from app import SearchEngine
                    self.search_engine = SearchEngine()
                    logger.info("✅ SearchEngine đã được khởi tạo")
                except Exception as e:
                    logger.error(f"❌ Không thể khởi tạo SearchEngine: {e}")
            
            if self.response_generator is None:
                try:
                    from app import ResponseGenerator
                    self.response_generator = ResponseGenerator()
                    logger.info("✅ ResponseGenerator đã được khởi tạo")
                except Exception as e:
                    logger.error(f"❌ Không thể khởi tạo ResponseGenerator: {e}")
            
            if self.chat_processor is None:
                try:
                    from app import ChatProcessor
                    self.chat_processor = ChatProcessor()
                    logger.info("✅ ChatProcessor đã được khởi tạo")
                except Exception as e:
                    logger.error(f"❌ Không thể khởi tạo ChatProcessor: {e}")
    
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
    logger.info("✅ Meta CAPI module đã được tải")
except ImportError as e:
    logger.warning(f"⚠️ meta_capi.py không khả dụng: {e}")
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
    logger.info("✅ Response guard module đã được tải")
except ImportError as e:
    logger.warning(f"⚠️ response_guard.py không khả dụng: {e}")
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

logger.info(f"✅ CORS đã được cấu hình cho: {Config.CORS_ORIGINS}")

# ==================== ENHANCED KNOWLEDGE LOADER ====================
def load_knowledge() -> bool:
    """Load knowledge base with tour entities"""
    
    if state._knowledge_loaded:
        logger.info("📚 Kiến thức đã được tải, bỏ qua")
        return True
    
    try:
        logger.info(f"📚 Đang tải kiến thức từ {Config.KNOWLEDGE_PATH}")
        
        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.error(f"❌ Không tìm thấy file kiến thức: {Config.KNOWLEDGE_PATH}")
            return False
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        # Load company info
        state.about_company = knowledge.get('about_company', {})
        if state.about_company:
            logger.info(f"✅ Thông tin công ty đã được tải")
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
                logger.error(f"❌ Lỗi khi tải tour {idx}: {e}")
                continue
        
        logger.info(f"✅ Kiến thức đã được tải: {len(state.tours_db)} tours")
        
        # Load or create mapping
        if os.path.exists(Config.FAISS_MAPPING_PATH):
            try:
                with open(Config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    state.mapping = json.load(f)
                logger.info(f"✅ Bản đồ đã được tải: {len(state.mapping)} mục")
            except Exception as e:
                logger.error(f"❌ Lỗi khi tải bản đồ: {e}")
                state.mapping = []
        
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
                
                logger.info(f"✅ Tour entities đã được tải: {len(state.tour_entities_dict)} entities")
                logger.info(f"   - Tags được lập chỉ mục: {len(state.tour_tags_index)}")
                logger.info(f"   - Regions được lập chỉ mục: {len(state.tour_region_index)}")
                state._tour_entities_loaded = True
            except Exception as e:
                logger.error(f"❌ Lỗi khi tải tour entities: {e}")
        
        state._knowledge_loaded = True
        
        # Initialize components after knowledge loaded
        state.init_components()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Không thể tải kiến thức: {e}")
        traceback.print_exc()
        return False

# ==================== FIXED SEARCH ENGINE ====================
class SearchEngine:
    """Search engine với OpenAI client được fix"""
    
    def __init__(self):
        logger.info("🧠 Khởi tạo công cụ tìm kiếm")
        self.openai_client = None
        
        # FIXED: Chỉ truyền api_key, không có proxies
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                # FIXED: Chỉ truyền api_key, không có base_url hoặc proxies
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("✅ OpenAI client đã được khởi tạo")
            except Exception as e:
                logger.error(f"❌ Khởi tạo OpenAI thất bại: {e}")
                logger.error(f"   Lỗi chi tiết: {str(e)}")
    
    def load_index(self) -> bool:
        """Load search index"""
        if state._index_loaded:
            return True
        
        try:
            if Config.FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info(f"📦 Đang tải FAISS index")
                state.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS đã được tải: {state.index.ntotal} vectors")
                state._index_loaded = True
                return True
            
            if NUMPY_AVAILABLE and os.path.exists(Config.FALLBACK_VECTORS_PATH):
                logger.info(f"📦 Đang tải numpy vectors")
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                
                if 'mat' in data:
                    state.vectors = data['mat']
                elif 'vectors' in data:
                    state.vectors = data['vectors']
                
                if state.vectors is not None:
                    norms = np.linalg.norm(state.vectors, axis=1, keepdims=True)
                    state.vectors = state.vectors / (norms + 1e-12)
                
                logger.info(f"✅ Numpy đã được tải: {state.vectors.shape[0]} vectors")
                state._index_loaded = True
                return True
            
            logger.info("ℹ️ Không tìm thấy vector index, sử dụng text search")
            state._index_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Không thể tải index: {e}")
            state._index_loaded = True
            return False

# ==================== FIXED RESPONSE GENERATOR ====================
class ResponseGenerator:
    """Response generator với LLM client được fix"""
    
    def __init__(self):
        self.llm_client = None
        
        # FIXED: Chỉ truyền api_key, không có proxies
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                # FIXED: Chỉ truyền api_key, không có base_url hoặc proxies
                self.llm_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("✅ LLM client đã được khởi tạo")
            except Exception as e:
                logger.error(f"❌ Khởi tạo LLM client thất bại: {e}")
                logger.error(f"   Lỗi chi tiết: {str(e)}")

# ==================== ENHANCED CHAT PROCESSOR ====================
class ChatProcessor:
    """Enhanced chat processor"""
    
    def __init__(self):
        self.response_generator = state.get_response_generator()
        self.search_engine = state.get_search_engine()
    
    def process(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process user message"""
        start_time = time.time()
        
        try:
            # Load knowledge if needed
            if not state._knowledge_loaded:
                if not load_knowledge():
                    return {
                        'reply': "Xin lỗi, hệ thống đang khởi tạo. Vui lòng thử lại sau! 🙏",
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
            
            # Detect intent (simplified)
            text_lower = user_message.lower().strip()
            
            # Simple intent detection
            if any(word in text_lower for word in ['xin chào', 'chào', 'hello', 'hi']):
                intent = Intent.GREETING
            elif 'tour' in text_lower or 'du lịch' in text_lower:
                intent = Intent.TOUR_INQUIRY
            elif 'giá' in text_lower or 'bao nhiêu' in text_lower:
                intent = Intent.PRICE_ASK
            elif 'đặt' in text_lower or 'book' in text_lower:
                intent = Intent.BOOKING_REQUEST
            elif 'ruby wings' in text_lower or 'công ty' in text_lower:
                intent = Intent.ABOUT_COMPANY
            else:
                intent = Intent.TOUR_INQUIRY  # Default to tour inquiry
            
            context['intent'] = normalize_intent(intent)
            
            # Generate response
            response_text = ""
            if self.response_generator and self.response_generator.llm_client and Config.ENABLE_LLM_ADVICE:
                # Try LLM first
                try:
                    response_text = self._generate_llm_response(user_message, intent)
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    response_text = self._generate_rule_based_response(intent)
            else:
                response_text = self._generate_rule_based_response(intent)
            
            # Build result
            result = {
                'reply': response_text,
                'session_id': session_id,
                'intent': {
                    'name': context['intent'],
                    'confidence': 0.9,
                    'metadata': {}
                },
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'from_cache': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            state.cache_response(cache_key, result)
            
            # Update stats
            state.stats['requests'] += 1
            state.stats['intent_counts'][context['intent']] += 1
            
            # Log
            processing_time = result['processing_time_ms']
            logger.info(f"⏱️ Đã xử lý trong {processing_time}ms | "
                       f"Ý định: {context['intent']} | "
                       f"Ký tự: {len(response_text)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi xử lý chat: {e}")
            traceback.print_exc()
            
            state.stats['errors'] += 1
            
            return {
                'reply': "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ **0332510486**! 🙏",
                'session_id': session_id,
                'error': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_llm_response(self, user_message: str, intent: str) -> str:
        """Generate response using LLM"""
        try:
            llm_client = self.response_generator.llm_client
            
            # Prepare prompt
            prompt = f"""Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm, retreat, thiền, khí công, hành trình chữa lành.

Hãy trả lời câu hỏi: "{user_message}"

Yêu cầu:
1. Trả lời bằng tiếng Việt, thân thiện, nhiệt tình
2. Tập trung vào giá trị chữa lành, trải nghiệm sâu
3. Nếu có tour phù hợp, giới thiệu 2-3 tour
4. Kết thúc bằng lời mời liên hệ hotline 0332510486

Trả lời:"""
            
            response = llm_client.chat.completions.create(
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
            return answer
            
        except Exception as e:
            state.stats["llm_errors"] += 1
            raise e
    
    def _generate_rule_based_response(self, intent: str) -> str:
        """Generate rule-based response"""
        intent_str = normalize_intent(intent)
        
        if is_intent_equal(intent_str, Intent.GREETING):
            return self._generate_greeting()
        elif is_intent_equal(intent_str, Intent.ABOUT_COMPANY):
            return self._generate_about_company()
        elif is_intent_equal(intent_str, Intent.TOUR_INQUIRY):
            return self._generate_tour_inquiry()
        elif is_intent_equal(intent_str, Intent.PRICE_ASK):
            return self._generate_price_info()
        elif is_intent_equal(intent_str, Intent.BOOKING_REQUEST):
            return self._generate_booking_info()
        else:
            return self._generate_tour_inquiry()
    
    def _generate_greeting(self) -> str:
        return """Xin chào! Tôi là trợ lý AI của Ruby Wings 🌿

Tôi có thể giúp bạn:
• Tìm hiểu về các tour trải nghiệm, retreat
• Tư vấn tour phù hợp với nhu cầu
• Giải đáp thông tin về Ruby Wings
• Hỗ trợ đặt tour

Bạn muốn tìm hiểu về điều gì ạ? 😊"""
    
    def _generate_about_company(self) -> str:
        return """**Ruby Wings** - Hành trình chữa lành và trải nghiệm sâu 🌿

Ruby Wings là đơn vị tiên phong trong lĩnh vực du lịch trải nghiệm, retreat, và hành trình chữa lành tại Miền Trung Việt Nam.

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

👉 Khám phá các hành trình của chúng tôi hoặc liên hệ **0332510486** để được tư vấn! 🌈"""
    
    def _generate_tour_inquiry(self) -> str:
        return """**Ruby Wings có các hành trình đa dạng:** 🏞️

1. **Tour Retreat & Chữa lành:**
   • Thiền, khí công, tĩnh tâm
   • Hành trình nội tâm, cân bằng cảm xúc
   • Khám phá bản thân, tìm lại sự bình an

2. **Tour Trải nghiệm Văn hóa:**
   • Khám phá di sản Huế, Hội An
   • Giao lưu cộng đồng bản địa
   • Trải nghiệm ẩm thực đặc sắc

3. **Tour Thiên nhiên & Mạo hiểm:**
   • Trekking rừng Bạch Mã
   • Khám phá Phong Nha - Kẻ Bàng
   • Hành trình xuyên rừng, vượt suối

4. **Tour Team Building:**
   • Gắn kết doanh nghiệp, công ty
   • Hoạt động teamwork sáng tạo
   • Phát triển kỹ năng lãnh đạo

**Ưu đãi đặc biệt:**
• Giảm 5% cho nhóm từ 5 người
• Giảm 10% cho đặt tour trước 15 ngày
• Voucher 200.000 VNĐ cho lần đặt tiếp theo

Liên hệ **0332510486** để được tư vấn tour phù hợp nhất với bạn! 📞"""
    
    def _generate_price_info(self) -> str:
        return """💰 **Thông tin giá các hành trình Ruby Wings**

Giá tour dao động từ **890.000 VNĐ** đến **3.500.000 VNĐ** tùy theo:
• Thời lượng (1 ngày, 2N1Đ, 3N2Đ)
• Loại hình (retreat, trekking, văn hóa)
• Dịch vụ bao gồm
• Số lượng người tham gia

**Giá đã bao gồm:**
✓ Xe đưa đón đời mới
✓ Hướng dẫn viên chuyên nghiệp
✓ Bữa ăn theo chương trình
✓ Vé tham quan các điểm
✓ Bảo hiểm du lịch
✓ Nước uống, khăn lạnh

**Chính sách giá ưu đãi:**
• Giảm 5% cho nhóm từ 5 người trở lên
• Giảm 10% cho đặt tour trước 15 ngày
• Ưu đãi đặc biệt cho công ty, đoàn thể
• Combo gia đình (2 người lớn + 1 trẻ em)

Liên hệ **0332510486** để biết giá chi tiết và nhận ưu đãi phù hợp! 📞"""
    
    def _generate_booking_info(self) -> str:
        return """🎯 **Đặt hành trình Ruby Wings - 4 bước đơn giản**

**Bước 1:** Chọn hành trình phù hợp
**Bước 2:** Cung cấp thông tin (số người, ngày đi, yêu cầu)
**Bước 3:** Xác nhận & Thanh toán
**Bước 4:** Chuẩn bị hành trình

**Cách thức đặt tour:**
1. 📞 **Gọi hotline:** 0332510486 (8:00 - 22:00)
2. 💬 **Nhắn tin Zalo:** 0332510486
3. 📧 **Email:** info@rubywings.vn
4. 🌐 **Website:** rubywings.vn

Chúng tôi sẽ xác nhận trong vòng 30 phút và đồng hành cùng bạn suốt hành trình! 🌈"""

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
            logger.error(f"Lỗi Meta CAPI pageview: {e}")

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
        'version': '6.0.1-fixed',
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
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Index route"""
    return jsonify({
        'service': 'Ruby Wings AI Chatbot',
        'version': '6.0.1 (Fixed OpenAI Client)',
        'status': 'running',
        'tours_available': len(state.tours_db),
        'features': {
            'llm_advisory': Config.ENABLE_LLM_ADVICE and Config.OPENAI_API_KEY != "",
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
                'message': 'Xin lỗi, hệ thống đang khởi tạo. Vui lòng thử lại sau!'
            }), 503
        
        result = chat_processor.process(user_message, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        traceback.print_exc()
        state.stats['errors'] += 1
        
        return jsonify({
            'error': 'Internal server error',
            'message': 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại!'
        }), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_legacy():
    """Legacy /chat endpoint"""
    return chat()

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
        note = data.get('note', '').strip()
        
        if not phone:
            return jsonify({'error': 'Phone number is required'}), 400
        
        phone_clean = re.sub(r'[^\d+]', '', phone)
        
        if not re.match(r'^(0|\+?84)\d{9,10}$', phone_clean):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        lead_data = {
            'timestamp': datetime.now().isoformat(),
            'contact_name': name or 'Khách yêu cầu gọi lại',
            'phone': phone_clean,
            'service_interest': tour_interest,
            'note': note,
            'status': 'New'
        }
        
        # Save to fallback storage
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
                
                logger.info("✅ Lead saved to fallback storage")
            except Exception as e:
                logger.error(f"Fallback storage error: {e}")
        
        state.stats['leads'] += 1
        
        return jsonify({
            'success': True,
            'message': 'Thông tin đã được lưu! Ruby Wings sẽ liên hệ sớm nhất. 📞',
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
    return jsonify({
        'status': 'ok',
        'requests': state.stats['requests'],
        'errors': state.stats['errors'],
        'leads': state.stats['leads'],
        'sessions': len(state.session_contexts),
        'tours': len(state.tours_db),
        'uptime_seconds': int((datetime.now() - state.stats['start_time']).total_seconds())
    })

# ==================== INITIALIZATION ====================
def initialize_app():
    """Initialize application"""
    try:
        logger.info("🚀 Khởi động Ruby Wings Chatbot v6.0.1...")
        
        Config.log_config()
        
        logger.info("🔍 Đang tải kiến thức...")
        if load_knowledge():
            logger.info("✅ Kiến thức đã sẵn sàng")
        else:
            logger.error("❌ Không thể tải kiến thức")
        
        logger.info("=" * 60)
        logger.info("✅ RUBY WINGS CHATBOT SẴN SÀNG!")
        logger.info(f"📊 Tours đã tải: {len(state.tours_db)}")
        logger.info(f"🧠 LLM Advisory: {'✅' if Config.ENABLE_LLM_ADVICE and Config.OPENAI_API_KEY else '❌'}")
        logger.info(f"🌐 Server: {Config.HOST}:{Config.PORT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Khởi tạo thất bại: {e}")
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
else:
    # For Gunicorn
    initialize_app()

__all__ = ["app"]