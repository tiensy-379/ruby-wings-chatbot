#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 5.2.3 (INTENT-DRIVEN FIX)
Created: 2025-01-17
Author: Ruby Wings AI Team

FIX V5.2.3: INTENT-DRIVEN RESPONSE SYSTEM
- Fixed intent classification (SMALLTALK -> TOUR_LIST, TOUR_FILTER, ABOUT_COMPANY)
- Added intent-based response routing
- Added tour filtering by duration, location, etc.
- Added company info response
- Fixed search results to be context-aware
- Preserved all existing features (lead capture, Meta CAPI, Google Sheets)
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
logger = logging.getLogger("ruby-wings-v5.2.3-intent")

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
    
    @classmethod
    def log_config(cls):
        """Log configuration on startup"""
        logger.info("=" * 60)
        logger.info("🚀 RUBY WINGS CHATBOT v5.2.3 (INTENT-DRIVEN FIX)")
        logger.info("=" * 60)
        logger.info(f"📊 RAM Profile: {cls.RAM_PROFILE}MB")
        logger.info(f"🌍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
        logger.info(f"🔧 Platform: {platform.system()}")
        
        features = []
        if cls.STATE_MACHINE_ENABLED:
            features.append("State Machine")
        if cls.ENABLE_LOCATION_FILTER:
            features.append("Location Filter")
        if cls.FAISS_ENABLED:
            features.append("FAISS")
        else:
            features.append("Numpy Fallback")
        if cls.ENABLE_META_CAPI:
            features.append("Meta CAPI (Lead)")
        if cls.ENABLE_META_CAPI_CALL:
            features.append("Meta CAPI (Call)")
        if cls.ENABLE_GOOGLE_SHEETS:
            features.append("Google Sheets")
        if cls.ENABLE_FALLBACK_STORAGE:
            features.append("Fallback Storage")
        if cls.ENABLE_TOUR_FILTERING:
            features.append("Tour Filtering")
        if cls.ENABLE_COMPANY_INFO:
            features.append("Company Info")
        
        logger.info(f"🎯 Features: {', '.join(features)}")
        logger.info(f"🔑 OpenAI: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        logger.info(f"🔍 FAISS enabled: {cls.FAISS_ENABLED}")
        
        if cls.META_PIXEL_ID and len(cls.META_PIXEL_ID) > 10:
            logger.info(f"📞 Meta Pixel: {cls.META_PIXEL_ID[:6]}...{cls.META_PIXEL_ID[-4:]}")
        else:
            logger.info(f"📞 Meta Pixel: {cls.META_PIXEL_ID}")
            
        logger.info(f"🌐 CORS: {cls.CORS_ORIGINS}")
        logger.info("=" * 60)

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
    """Lazy import OpenAI"""
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

# ==================== IMPORT CUSTOM MODULES ====================
try:
    from entities import (
        ConversationStage,
        Intent,
        Tour,
        ConversationContext,
        LeadData,
        detect_intent,
        detect_phone_number,
        extract_location_from_query,
        get_region_from_location
    )
    ENTITIES_AVAILABLE = True
    logger.info("✅ Entities module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import entities.py: {e}")
    ENTITIES_AVAILABLE = False
    
    # ==================== ENHANCED INTENT DETECTION SYSTEM ====================
    class Intent:
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
        EXPLORE = "explore"
        SUGGEST = "suggest"
        COMPARE = "compare"
        SELECT = "select"
        BOOK = "book"
        LEAD = "lead"
        CALLBACK = "callback"
    
    def detect_intent(text): 
        """Enhanced intent detection with entity extraction"""
        text_lower = text.lower().strip()
        metadata = {
            "duration_days": None,
            "location": None,
            "price_max": None,
            "tags": [],
            "region": None,
            "raw_query": text
        }
        
        # Extract duration days (e.g., "tour 1 ngày", "2 ngày")
        duration_patterns = [
            (r'(\d+)\s*ngày', 'duration_days'),
            (r'(\d+)\s*ngay', 'duration_days'),
            (r'(\d+)\s*day', 'duration_days'),
            (r'một\s*ngày', 'duration_days'),
            (r'hai\s*ngày', 'duration_days'),
            (r'ba\s*ngày', 'duration_days'),
            (r'bốn\s*ngày', 'duration_days'),
            (r'năm\s*ngày', 'duration_days')
        ]
        
        for pattern, key in duration_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if pattern.startswith(r'(\d+)'):
                    metadata[key] = int(match.group(1))
                else:
                    # Map Vietnamese numbers
                    num_map = {
                        'một': 1, 'hai': 2, 'ba': 3, 
                        'bốn': 4, 'năm': 5
                    }
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
            'hiền lương': 'Hiền Lương', 'hien luong': 'Hiền Lương',
            'miền trung': 'Miền Trung', 'mien trung': 'Miền Trung'
        }
        
        for keyword, location in location_keywords.items():
            if keyword in text_lower:
                metadata['location'] = location
                metadata['region'] = location
                break
        
        # Extract price mentions
        price_patterns = [
            r'giá\s+(\d+[\.,]?\d*)\s*(k|tr|triệu|triệu đồng|vnđ|vnd|đồng)',
            r'(\d+[\.,]?\d*)\s*(k|tr|triệu|triệu đồng|vnđ|vnd|đồng)',
            r'giá.*dưới\s+(\d+[\.,]?\d*)',
            r'giá.*khoảng\s+(\d+[\.,]?\d*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    price_num = float(match.group(1).replace(',', '.'))
                    # Convert to VND if needed
                    if 'k' in match.group(2).lower():
                        price_num *= 1000
                    elif 'tr' in match.group(2).lower() or 'triệu' in match.group(2).lower():
                        price_num *= 1000000
                    metadata['price_max'] = price_num
                    break
                except:
                    pass
        
        # Extract tags/keywords
        tag_keywords = {
            'thiền': 'thiền', 'meditation': 'thiền',
            'retreat': 'retreat',
            'chữa lành': 'chữa lành', 'healing': 'chữa lành',
            'trải nghiệm': 'trải nghiệm', 'experience': 'trải nghiệm',
            'thiên nhiên': 'thiên nhiên', 'nature': 'thiên nhiên',
            'lịch sử': 'lịch sử', 'history': 'lịch sử'
        }
        
        for keyword, tag in tag_keywords.items():
            if keyword in text_lower:
                metadata['tags'].append(tag)
        
        # ==================== INTENT CLASSIFICATION ====================
        # 1. GREETING & FAREWELL
        greeting_words = ['xin chào', 'chào', 'hello', 'hi', 'helo', 'chao']
        farewell_words = ['tạm biệt', 'bye', 'goodbye', 'cảm ơn', 'thank you', 'thanks']
        
        if any(word in text_lower for word in greeting_words):
            return Intent.GREETING, 0.95, metadata
        
        if any(word in text_lower for word in farewell_words):
            return Intent.FAREWELL, 0.95, metadata
        
        # 2. COMPANY INFO
        company_keywords = [
            'ruby wings', 'công ty', 'đơn vị', 'bạn là ai', 
            'giới thiệu', 'công ty bạn', 'cty', 'doanh nghiệp',
            'tổ chức', 'cty ruby', 'rubywings'
        ]
        
        if any(keyword in text_lower for keyword in company_keywords):
            return Intent.ABOUT_COMPANY, 0.92, metadata
        
        # 3. TOUR LIST (general inquiry about tours)
        tour_list_keywords = [
            'tour nào', 'tour gì', 'có những tour nào', 
            'danh sách tour', 'các tour', 'tour của bạn',
            'bạn có tour nào', 'dịch vụ nào', 'sản phẩm nào'
        ]
        
        if any(keyword in text_lower for keyword in tour_list_keywords):
            return Intent.TOUR_LIST, 0.90, metadata
        
        # 4. TOUR FILTER (specific filtering)
        filter_indicators = [
            'có tour', 'tour nào', 'tìm tour', 'lọc tour',
            'tour 1 ngày', 'tour 2 ngày', 'tour 3 ngày',
            'tour huế', 'tour đà nẵng', 'tour quảng trị',
            'tour giá', 'tour rẻ', 'tour thiền', 'tour retreat'
        ]
        
        # Check if any filter criteria are present
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
        
        # 5. TOUR INQUIRY (general tour question)
        if has_tour_word:
            return Intent.TOUR_INQUIRY, 0.85, metadata
        
        # 6. PRICE ASK
        price_words = ['giá', 'bao nhiêu tiền', 'cost', 'price', 'chi phí']
        if any(word in text_lower for word in price_words):
            return Intent.PRICE_ASK, 0.85, metadata
        
        # 7. BOOKING REQUEST
        booking_words = ['đặt', 'book', 'đăng ký', 'reserve', 'booking']
        if any(word in text_lower for word in booking_words):
            return Intent.BOOKING_REQUEST, 0.90, metadata
        
        # 8. PHONE PROVIDE
        phone = detect_phone_number(text)
        if phone:
            metadata['phone_number'] = phone
            return Intent.PROVIDE_PHONE, 0.98, metadata
        
        # 9. CALLBACK REQUEST
        callback_words = ['gọi lại', 'liên hệ lại', 'call back', 'tư vấn']
        if any(word in text_lower for word in callback_words):
            return Intent.CALLBACK_REQUEST, 0.87, metadata
        
        # 10. CONTACT INFO
        contact_words = ['số điện thoại', 'hotline', 'liên hệ', 'contact']
        if any(word in text_lower for word in contact_words):
            return Intent.CONTACT_INFO, 0.89, metadata
        
        # Default to SMALLTALK for other queries
        return Intent.SMALLTALK, 0.70, metadata
    
    def detect_phone_number(text):
        """Detect Vietnamese phone numbers"""
        patterns = [
            r'0\d{9,10}',
            r'\+84\d{9,10}',
            r'84\d{9,10}',
            r'\(\+84\)\d{9,10}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def extract_location_from_query(text): 
        """Extract location from query"""
        # This is a simple version - real implementation should be in entities.py
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
    
    def get_region_from_location(location): 
        """Get region from location"""
        region_map = {
            'Huế': 'Miền Trung',
            'Đà Nẵng': 'Miền Trung',
            'Hội An': 'Miền Trung',
            'Quảng Trị': 'Miền Trung',
            'Bạch Mã': 'Miền Trung'
        }
        return region_map.get(location, 'Miền Trung')
    
    class Tour:
        pass
    
    class ConversationContext:
        pass
    
    class LeadData:
        pass

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

# ==================== GLOBAL STATE ====================
class GlobalState:
    """Enhanced global state with company info and tour entities"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize state with all data structures"""
        # Core data
        self.tours_db: Dict[int, Dict] = {}
        self.tour_name_index: Dict[str, int] = {}
        self.tour_entities: List[Dict] = []
        
        # Company info
        self.about_company: Dict = {}
        
        # Session management
        self.session_contexts: Dict[str, Dict] = {}
        
        # Search data
        self.mapping: List[Dict] = []
        self.index = None
        self.vectors = None
        
        # Caching
        self.response_cache: OrderedDict = OrderedDict()
        self.embedding_cache: OrderedDict = OrderedDict()
        
        # Statistics
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
            "start_time": datetime.now()
        }
        
        # Track initialization status
        self._knowledge_loaded = False
        self._index_loaded = False
        self._tour_entities_loaded = False
        self._company_info_loaded = False
        
        logger.info("🌐 Global state initialized (enhanced with intent tracking)")
    
    def get_tour(self, index: int) -> Optional[Dict]:
        """Get tour by index"""
        return self.tours_db.get(index)
    
    def get_tour_by_name(self, name: str) -> Optional[Dict]:
        """Get tour by name (case-insensitive)"""
        idx = self.tour_name_index.get(name.lower())
        if idx is not None:
            return self.tours_db.get(idx)
        return None
    
    def filter_tours(self, **filters) -> List[Dict]:
        """Filter tours based on criteria"""
        if not self.tours_db:
            return []
        
        filtered = []
        for idx, tour in self.tours_db.items():
            match = True
            
            # Duration filter
            if 'duration_days' in filters and filters['duration_days'] is not None:
                tour_duration = self._extract_duration_days(tour)
                if tour_duration != filters['duration_days']:
                    match = False
            
            # Location filter
            if match and 'location' in filters and filters['location']:
                tour_location = tour.get('location', '').lower()
                filter_location = filters['location'].lower()
                if filter_location not in tour_location:
                    match = False
            
            # Price filter
            if match and 'price_max' in filters and filters['price_max'] is not None:
                tour_price = self._extract_price(tour)
                if tour_price is None or tour_price > filters['price_max']:
                    match = False
            
            # Tag filter
            if match and 'tags' in filters and filters['tags']:
                tour_tags = set(tag.lower() for tag in tour.get('tags', []))
                filter_tags = set(tag.lower() for tag in filters['tags'])
                if not filter_tags.intersection(tour_tags):
                    match = False
            
            if match:
                filtered.append(tour)
        
        return filtered
    
    def _extract_duration_days(self, tour: Dict) -> Optional[int]:
        """Extract duration in days from tour"""
        duration = tour.get('duration', '')
        if not duration:
            return None
        
        # Try to extract number from duration string
        match = re.search(r'(\d+)\s*ngày', duration.lower())
        if match:
            return int(match.group(1))
        
        # Check duration_days field if exists
        if 'duration_days' in tour:
            return tour['duration_days']
        
        return None
    
    def _extract_price(self, tour: Dict) -> Optional[float]:
        """Extract price from tour"""
        price_text = tour.get('price', '')
        if not price_text:
            return None
        
        # Try to extract price number
        patterns = [
            r'(\d+[\.,]?\d*)\s*(k|tr|triệu|triệu đồng|vnđ|vnd|đồng)',
            r'(\d+[\.,]?\d*)\s*-\s*(\d+[\.,]?\d*)',
            r'từ\s*(\d+[\.,]?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, price_text.lower())
            if match:
                try:
                    price_num = float(match.group(1).replace(',', '.'))
                    if len(match.groups()) > 1 and match.group(2):
                        unit = match.group(2).lower()
                        if 'k' in unit:
                            price_num *= 1000
                        elif 'tr' in unit or 'triệu' in unit:
                            price_num *= 1000000
                    return price_num
                except:
                    pass
        
        return None
    
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
                    "last_updated": datetime.now()
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
            
            # Calculate intent distribution
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
                "tour_entities_loaded": len(self.tour_entities),
                "mapping_entries": len(self.mapping),
                "cache_size": len(self.response_cache),
                "knowledge_loaded": self._knowledge_loaded,
                "intent_distribution": intent_dist,
                "company_info_loaded": self._company_info_loaded
            }

# Initialize global state FIRST
state = GlobalState()

# ==================== KNOWLEDGE LOADER ====================
def load_knowledge() -> bool:
    """Load knowledge base with company info"""
    
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
            logger.info(f"✅ Company info loaded: {len(state.about_company)} fields")
            state._company_info_loaded = True
        
        # Load tours
        tours_data = knowledge.get('tours', [])
        
        if not tours_data:
            logger.warning("⚠️ No tours found in knowledge.json")
        
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
        
        # Load tour entities if available
        if os.path.exists(Config.TOUR_ENTITIES_PATH):
            try:
                with open(Config.TOUR_ENTITIES_PATH, 'r', encoding='utf-8') as f:
                    state.tour_entities = json.load(f)
                logger.info(f"✅ Tour entities loaded: {len(state.tour_entities)} entities")
                state._tour_entities_loaded = True
            except Exception as e:
                logger.error(f"❌ Error loading tour entities: {e}")
                state.tour_entities = []
        
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
            logger.info("📝 Creating mapping from tours...")
            state.mapping = []
            
            for idx, tour in state.tours_db.items():
                if not tour:
                    continue
                    
                fields_to_map = ['tour_name', 'location', 'duration', 'price', 
                                'summary', 'includes', 'style', 'description']
                
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
            
            logger.info(f"✅ Mapping created: {len(state.mapping)} entries from tours")
        
        state._knowledge_loaded = True
        logger.info(f"✅ Knowledge initialization complete")
        logger.info(f"   - Tours: {len(state.tours_db)}")
        logger.info(f"   - Mapping: {len(state.mapping)} entries")
        logger.info(f"   - Company info: {'Yes' if state._company_info_loaded else 'No'}")
        logger.info(f"   - Tour entities: {'Yes' if state._tour_entities_loaded else 'No'}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error in knowledge file: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        traceback.print_exc()
        return False

# ==================== ENHANCED SEARCH ENGINE ====================
class SearchEngine:
    """Enhanced search engine with intent-aware searching"""
    
    def __init__(self):
        logger.info("🧠 Initializing enhanced search engine (intent-aware)")
        self.openai_client = None
        
        # Log search mode
        if Config.FAISS_ENABLED:
            logger.info("🧠 Search mode: FAISS (if available)")
        elif NUMPY_AVAILABLE:
            logger.info("🧠 Search mode: Numpy")
        else:
            logger.info("🧠 Search mode: Text-based (fallback)")
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY
                )
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")
        else:
            logger.info("ℹ️ OpenAI not available, using text search")
    
    def load_index(self) -> bool:
        """Load search index"""
        if state._index_loaded:
            logger.info("📦 Index already loaded, skipping")
            return True
        
        try:
            if Config.FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info(f"📦 Loading FAISS index from {Config.FAISS_INDEX_PATH}")
                state.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS loaded: {state.index.ntotal} vectors")
                state._index_loaded = True
                return True
            
            if NUMPY_AVAILABLE and os.path.exists(Config.FALLBACK_VECTORS_PATH):
                logger.info(f"📦 Loading numpy vectors from {Config.FALLBACK_VECTORS_PATH}")
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                
                if 'mat' in data:
                    state.vectors = data['mat']
                elif 'vectors' in data:
                    state.vectors = data['vectors']
                else:
                    first_key = list(data.keys())[0]
                    state.vectors = data[first_key]
                
                if state.vectors is not None:
                    norms = np.linalg.norm(state.vectors, axis=1, keepdims=True)
                    state.vectors = state.vectors / (norms + 1e-12)
                
                logger.info(f"✅ Numpy loaded: {state.vectors.shape[0]} vectors")
                state._index_loaded = True
                return True
            
            logger.info("ℹ️ No vector index found, will use text-based search")
            state._index_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            logger.info("⚠️ Continuing with text-based search only")
            state._index_loaded = True
            return False
    
    def search(self, query: str, top_k: int = None, intent: str = None, metadata: Dict = None) -> List[Tuple[float, Dict]]:
        """Enhanced search with intent awareness"""
        if top_k is None:
            top_k = Config.TOP_K
        
        if not state.mapping:
            logger.warning("⚠️ Search called but mapping is empty")
            return []
        
        # Intent-specific search optimization
        if intent == Intent.ABOUT_COMPANY:
            # Search for company info
            return self._search_company_info(query, top_k)
        
        # Get query embedding if available
        embedding = self.get_embedding(query)
        
        # FAISS search
        if embedding is not None and state.index is not None and FAISS_AVAILABLE:
            try:
                query_vec = np.array([embedding], dtype='float32')
                scores, indices = state.index.search(query_vec, top_k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(state.mapping):
                        results.append((float(score), state.mapping[idx]))
                
                if results:
                    logger.debug(f"🔍 FAISS search found {len(results)} results")
                    return results
            except Exception as e:
                logger.error(f"FAISS search error: {e}")
        
        # Numpy search
        if embedding is not None and state.vectors is not None and NUMPY_AVAILABLE:
            try:
                query_vec = np.array([embedding], dtype='float32')
                query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
                
                similarities = np.dot(state.vectors, query_norm.T).flatten()
                top_indices = np.argsort(-similarities)[:top_k]
                
                results = []
                for idx in top_indices:
                    if 0 <= idx < len(state.mapping):
                        results.append((float(similarities[idx]), state.mapping[idx]))
                
                if results:
                    logger.debug(f"🔍 Numpy search found {len(results)} results")
                    return results
            except Exception as e:
                logger.error(f"Numpy search error: {e}")
        
        # Text fallback
        return self._text_search(query, top_k, intent, metadata)
    
    def _search_company_info(self, query: str, top_k: int) -> List[Tuple[float, Dict]]:
        """Search for company information"""
        if not state.about_company:
            return []
        
        # Create mapping entries for company info
        company_mapping = []
        for key, value in state.about_company.items():
            if isinstance(value, str) and value.strip():
                company_mapping.append({
                    "path": f"about_company.{key}",
                    "text": value,
                    "type": "company_info",
                    "field": key
                })
        
        query_lower = query.lower()
        results = []
        
        for entry in company_mapping:
            text = entry.get('text', '').lower()
            score = 0
            
            # Simple keyword matching
            company_keywords = ['ruby wings', 'công ty', 'đơn vị', 'giới thiệu', 'sứ mệnh', 'tầm nhìn']
            for keyword in company_keywords:
                if keyword in query_lower:
                    score += 2
                if keyword in text:
                    score += 1
            
            if score > 0:
                results.append((float(score), entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _text_search(self, query: str, top_k: int, intent: str = None, metadata: Dict = None) -> List[Tuple[float, Dict]]:
        """Enhanced text-based search with intent awareness"""
        if not state.mapping:
            return []
        
        query_lower = query.lower().strip()
        if not query_lower:
            import random
            results = []
            for entry in random.sample(state.mapping, min(len(state.mapping), top_k)):
                results.append((0.5, entry))
            return results
        
        query_words = [w for w in query_lower.split() if len(w) > 2]
        
        if not query_words:
            import random
            results = []
            for entry in random.sample(state.mapping, min(len(state.mapping), top_k)):
                results.append((0.3, entry))
            return results
        
        results = []
        for entry in state.mapping[:500]:
            text = entry.get('text', '').lower()
            
            score = 0
            for word in query_words:
                if word in text:
                    score += 1
                elif any(word in t for t in text.split()):
                    score += 0.5
            
            # Boost score based on intent
            if intent:
                if intent == Intent.TOUR_FILTER and metadata:
                    # Check if this entry matches filter criteria
                    tour_idx = entry.get('tour_index')
                    if tour_idx is not None:
                        tour = state.get_tour(tour_idx)
                        if tour and self._matches_filter(tour, metadata):
                            score += 3
                
                # Boost tour_name matches for tour-related intents
                if intent in [Intent.TOUR_LIST, Intent.TOUR_INQUIRY, Intent.TOUR_FILTER]:
                    if entry.get('field') == 'tour_name':
                        score += 2
            
            if score > 0:
                results.append((float(score), entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        if not results and state.mapping:
            import random
            results = []
            for entry in random.sample(state.mapping, min(len(state.mapping), top_k)):
                results.append((0.1, entry))
        
        logger.debug(f"🔍 Text search found {len(results[:top_k])} results for query: '{query}'")
        return results[:top_k]
    
    def _matches_filter(self, tour: Dict, metadata: Dict) -> bool:
        """Check if tour matches filter criteria"""
        match = True
        
        # Duration filter
        if 'duration_days' in metadata and metadata['duration_days'] is not None:
            tour_duration = state._extract_duration_days(tour)
            if tour_duration != metadata['duration_days']:
                match = False
        
        # Location filter
        if match and 'location' in metadata and metadata['location']:
            tour_location = tour.get('location', '').lower()
            filter_location = metadata['location'].lower()
            if filter_location not in tour_location:
                match = False
        
        return match
    
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
        
        # Fallback: deterministic hash-based embedding
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

# Initialize search engine
search_engine = SearchEngine()

# ==================== INTENT-DRIVEN RESPONSE GENERATOR ====================
class ResponseGenerator:
    """Intent-driven response generator"""
    
    def __init__(self):
        self.llm_client = None
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.llm_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY
                )
            except Exception as e:
                logger.error(f"LLM client init failed: {e}")
    
    def generate(self, user_message: str, search_results: List, context: Dict) -> str:
        """Generate response based on intent"""
        
        intent = context.get("intent", Intent.UNKNOWN)
        metadata = context.get("intent_metadata", {})
        
        logger.info(f"🎯 Generating response for intent: {intent}")
        
        # ==================== INTENT ROUTING ====================
        # 1. GREETING
        if intent == Intent.GREETING:
            return self._generate_greeting()
        
        # 2. FAREWELL
        if intent == Intent.FAREWELL:
            return self._generate_farewell()
        
        # 3. ABOUT_COMPANY
        if intent == Intent.ABOUT_COMPANY:
            return self._generate_about_company(metadata)
        
        # 4. TOUR_LIST
        if intent == Intent.TOUR_LIST:
            return self._generate_tour_list(metadata)
        
        # 5. TOUR_FILTER
        if intent == Intent.TOUR_FILTER:
            return self._generate_tour_filter(metadata)
        
        # 6. TOUR_INQUIRY
        if intent == Intent.TOUR_INQUIRY:
            return self._generate_tour_inquiry(search_results, metadata)
        
        # 7. PRICE_ASK
        if intent in [Intent.PRICE_ASK, Intent.PRICE_COMPARE, Intent.PRICE_RANGE]:
            return self._generate_price_info(search_results, metadata)
        
        # 8. BOOKING_REQUEST
        if intent in [Intent.BOOKING_REQUEST, Intent.BOOKING_PROCESS]:
            return self._generate_booking_info(search_results, metadata)
        
        # 9. CONTACT_INFO
        if intent == Intent.CONTACT_INFO:
            return self._generate_contact_info()
        
        # 10. CALLBACK_REQUEST
        if intent == Intent.CALLBACK_REQUEST:
            return self._generate_callback_request(metadata)
        
        # 11. PROVIDE_PHONE / LEAD_CAPTURED
        if intent in [Intent.PROVIDE_PHONE, Intent.LEAD_CAPTURED]:
            return self._generate_lead_confirm(metadata)
        
        # 12. SMALLTALK
        if intent == Intent.SMALLTALK:
            return self._generate_smalltalk(search_results, metadata)
        
        # Default: UNKNOWN
        return self._generate_fallback(search_results, metadata)
    
    # ==================== INTENT HANDLERS ====================
    
    def _generate_greeting(self) -> str:
        """Generate greeting response"""
        greetings = [
            "Xin chào! Tôi là trợ lý AI của Ruby Wings. Rất vui được hỗ trợ bạn! 😊\n\nBạn muốn tìm hiểu về tour nào?",
            "Chào bạn! Tôi có thể giúp gì cho bạn về các tour Ruby Wings? 🌿",
            "Xin chào! Ruby Wings hân hạnh được phục vụ bạn. Bạn cần tư vấn về tour nào ạ? 🎒"
        ]
        return random.choice(greetings)
    
    def _generate_farewell(self) -> str:
        """Generate farewell response"""
        farewells = [
            "Cảm ơn bạn! Chúc một ngày tuyệt vời! ✨",
            "Tạm biệt! Liên hệ **0332510486** nếu cần hỗ trợ nhé! 👋",
            "Cảm ơn đã trò chuyện! Hy vọng sớm được đồng hành cùng bạn trên hành trình Ruby Wings! 🌈"
        ]
        return random.choice(farewells)
    
    def _generate_about_company(self, metadata: Dict) -> str:
        """Generate company information response"""
        if not state.about_company:
            return "Ruby Wings là đơn vị tổ chức du lịch trải nghiệm, retreat, và hành trình chữa lành tại Miền Trung Việt Nam. 🌿"
        
        overview = state.about_company.get('overview', '')
        mission = state.about_company.get('mission', '')
        vision = state.about_company.get('vision', '')
        
        response = "**Ruby Wings** - Tổ chức du lịch trải nghiệm & chữa lành 🌈\n\n"
        
        if overview:
            response += f"{overview}\n\n"
        
        if mission:
            response += f"**Sứ mệnh:** {mission}\n\n"
        
        if vision:
            response += f"**Tầm nhìn:** {vision}\n\n"
        
        response += "👉 Khám phá các hành trình của chúng tôi hoặc liên hệ **0332510486** để được tư vấn!"
        
        return response
    
    def _generate_tour_list(self, metadata: Dict) -> str:
        """Generate list of all tours"""
        if not state.tours_db:
            return "Hiện tại chưa có tour nào. Vui lòng liên hệ **0332510486** để biết thêm chi tiết! 📞"
        
        tours = list(state.tours_db.values())
        
        if len(tours) > Config.MAX_TOURS_PER_RESPONSE:
            response = f"Ruby Wings hiện có **{len(tours)}** tour đa dạng. Dưới đây là một số tour tiêu biểu:\n\n"
            tours = random.sample(tours, min(Config.MAX_TOURS_PER_RESPONSE, len(tours)))
        else:
            response = f"Ruby Wings có **{len(tours)}** tour:\n\n"
        
        for idx, tour in enumerate(tours[:Config.MAX_TOURS_PER_RESPONSE], 1):
            response += f"{idx}. **{tour.get('tour_name', 'Tour')}**\n"
            
            if tour.get('duration'):
                response += f"   ⏱️ {tour['duration']}\n"
            
            if tour.get('location'):
                response += f"   📍 {tour['location']}\n"
            
            if tour.get('price'):
                price = tour['price']
                if len(price) > 50:
                    price = price[:50] + "..."
                response += f"   💰 {price}\n"
            
            response += "\n"
        
        response += "Bạn muốn tìm hiểu chi tiết về tour nào? Hoặc có thể lọc tour theo thời gian/địa điểm nữa nhé! 😊"
        
        return response
    
    def _generate_tour_filter(self, metadata: Dict) -> str:
        """Generate filtered tour response"""
        if not state.tours_db:
            return "Hiện tại chưa có tour nào phù hợp. Vui lòng liên hệ **0332510486** để được tư vấn cụ thể! 📞"
        
        # Extract filter criteria
        duration_days = metadata.get('duration_days')
        location = metadata.get('location')
        price_max = metadata.get('price_max')
        tags = metadata.get('tags', [])
        
        # Build filter description
        filter_desc = []
        if duration_days:
            filter_desc.append(f"{duration_days} ngày")
        if location:
            filter_desc.append(f"địa điểm {location}")
        if price_max:
            filter_desc.append(f"giá dưới {price_max:,} VNĐ")
        if tags:
            filter_desc.append(f"chủ đề {', '.join(tags)}")
        
        # Filter tours
        filtered_tours = state.filter_tours(
            duration_days=duration_days,
            location=location,
            price_max=price_max,
            tags=tags
        )
        
        if not filtered_tours:
            filter_text = " và ".join(filter_desc) if filter_desc else "theo yêu cầu của bạn"
            return f"Hiện chưa có tour {filter_text}. Bạn có thể:\n\n1. Thử tìm với tiêu chí khác\n2. Xem tất cả tour\n3. Liên hệ **0332510486** để đặt tour riêng! 📞"
        
        # Build response
        if filter_desc:
            filter_text = " và ".join(filter_desc)
            response = f"Tìm thấy **{len(filtered_tours)}** tour {filter_text}:\n\n"
        else:
            response = f"Tìm thấy **{len(filtered_tours)}** tour:\n\n"
        
        for idx, tour in enumerate(filtered_tours[:Config.MAX_TOURS_PER_RESPONSE], 1):
            response += f"{idx}. **{tour.get('tour_name', 'Tour')}**\n"
            
            if tour.get('duration'):
                response += f"   ⏱️ {tour['duration']}\n"
            
            if tour.get('location'):
                response += f"   📍 {tour['location']}\n"
            
            if tour.get('price'):
                price = tour['price']
                if len(price) > 50:
                    price = price[:50] + "..."
                response += f"   💰 {price}\n"
            
            if tour.get('summary'):
                summary = tour['summary']
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                response += f"   📝 {summary}\n"
            
            response += "\n"
        
        if len(filtered_tours) > Config.MAX_TOURS_PER_RESPONSE:
            response += f"... và {len(filtered_tours) - Config.MAX_TOURS_PER_RESPONSE} tour khác.\n\n"
        
        response += "Bạn muốn biết thêm chi tiết về tour nào? Hoặc liên hệ **0332510486** để đặt tour ngay! 📱"
        
        return response
    
    def _generate_tour_inquiry(self, search_results: List, metadata: Dict) -> str:
        """Generate response for general tour inquiry"""
        if not search_results:
            return self._generate_tour_list(metadata)
        
        response = "Dựa trên yêu cầu của bạn, tôi tìm thấy:\n\n"
        
        tours_mentioned = set()
        added_count = 0
        
        for score, entry in search_results[:Config.MAX_TOURS_PER_RESPONSE]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour:
                    tours_mentioned.add(tour_idx)
                    added_count += 1
                    
                    response += f"**{tour.get('tour_name', 'Tour')}**\n"
                    
                    if tour.get('location'):
                        response += f"📍 {tour['location']}\n"
                    if tour.get('duration'):
                        response += f"⏱️ {tour['duration']}\n"
                    if tour.get('price'):
                        price = tour['price']
                        if len(price) > 100:
                            price = price[:100] + "..."
                        response += f"💰 {price}\n"
                    if tour.get('summary'):
                        summary = tour['summary']
                        if len(summary) > 150:
                            summary = summary[:150] + "..."
                        response += f"📝 {summary}\n"
                    response += "\n"
        
        if added_count == 0:
            return self._generate_tour_list(metadata)
        
        response += "Bạn muốn biết thêm chi tiết gì? Hoặc liên hệ **0332510486** để đặt tour! 😊"
        
        return response
    
    def _generate_price_info(self, search_results: List, metadata: Dict) -> str:
        """Generate price information response"""
        if not search_results:
            return "Để biết giá cụ thể, vui lòng:\n\n1. Chọn tour bạn quan tâm\n2. Liên hệ **0332510486** để được báo giá chi tiết\n\n💰 Giá tour thường từ 890.000 VNĐ - 3.500.000 VNĐ tùy loại."
        
        response = "Thông tin giá các tour:\n\n"
        
        tours_mentioned = set()
        
        for score, entry in search_results[:Config.MAX_TOURS_PER_RESPONSE]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour and tour.get('price'):
                    tours_mentioned.add(tour_idx)
                    
                    response += f"**{tour.get('tour_name', 'Tour')}**\n"
                    response += f"💰 {tour['price']}\n\n"
        
        if not tours_mentioned:
            response = "Các tour Ruby Wings có giá từ **890.000 VNĐ** đến **3.500.000 VNĐ** tùy thời lượng và dịch vụ.\n\n"
            response += "Để biết giá chính xác, vui lòng:\n"
            response += "1. Chọn tour cụ thể\n"
            response += "2. Liên hệ **0332510486** để được báo giá chi tiết và ưu đãi! 📞"
        
        return response
    
    def _generate_booking_info(self, search_results: List, metadata: Dict) -> str:
        """Generate booking information response"""
        response = "🎯 **Đặt tour Ruby Wings**\n\n"
        response += "Để đặt tour, bạn có thể:\n\n"
        response += "1. **Chọn tour** từ danh sách trên\n"
        response += "2. **Cung cấp số điện thoại** để chúng tôi liên hệ tư vấn\n"
        response += "3. **Gọi trực tiếp 0332510486** để đặt ngay\n\n"
        response += "📋 **Thông tin cần chuẩn bị:**\n"
        response += "- Họ tên đầy đủ\n"
        response += "- Số điện thoại\n"
        response += "- Ngày dự kiến đi tour\n"
        response += "- Số lượng người tham gia\n\n"
        response += "Sau khi đặt, bạn sẽ nhận được xác nhận và hướng dẫn chi tiết qua SMS/Zalo. 📱"
        
        return response
    
    def _generate_contact_info(self) -> str:
        """Generate contact information response"""
        response = "📞 **Liên hệ Ruby Wings**\n\n"
        response += "**Hotline:** 0332510486\n"
        response += "**Zalo:** 0332510486\n"
        response += "**Email:** info@rubywings.vn\n\n"
        response += "⏰ **Thời gian làm việc:**\n"
        response += "- Thứ 2 - Thứ 6: 8:00 - 17:00\n"
        response += "- Thứ 7: 8:00 - 12:00\n"
        response += "- Chủ nhật: Nghỉ\n\n"
        response += "📍 **Địa chỉ:** Đà Nẵng, Việt Nam\n\n"
        response += "Chúng tôi sẵn sàng tư vấn và hỗ trợ bạn 24/7 qua hotline! 😊"
        
        return response
    
    def _generate_callback_request(self, metadata: Dict) -> str:
        """Generate callback request response"""
        response = "✅ **Yêu cầu gọi lại đã được ghi nhận!**\n\n"
        response += "Đội ngũ Ruby Wings sẽ liên hệ với bạn trong thời gian sớm nhất.\n\n"
        response += "📋 **Thông tin đã ghi nhận:**\n"
        
        if metadata.get('duration_days'):
            response += f"- Tour {metadata['duration_days']} ngày\n"
        
        if metadata.get('location'):
            response += f"- Địa điểm: {metadata['location']}\n"
        
        response += "\n**Hoặc bạn có thể gọi ngay:** 0332510486\n\n"
        response += "Cảm ơn bạn đã quan tâm đến Ruby Wings! 🌿"
        
        return response
    
    def _generate_lead_confirm(self, metadata: Dict) -> str:
        """Generate lead confirmation response"""
        phone = metadata.get('phone_number', '')
        masked_phone = phone[:3] + '***' + phone[-2:] if phone else '***'
        
        response = "✅ **Thông tin của bạn đã được lưu thành công!**\n\n"
        response += f"Số điện thoại: {masked_phone}\n"
        response += "Đội ngũ Ruby Wings sẽ liên hệ với bạn trong vòng 15 phút.\n\n"
        response += "📱 **Các bước tiếp theo:**\n"
        response += "1. Nhân viên sẽ gọi tư vấn chi tiết về tour\n"
        response += "2. Xác nhận thông tin và ngày đi\n"
        response += "3. Hướng dẫn thanh toán và chuẩn bị\n\n"
        response += "Cảm ơn bạn đã tin tưởng Ruby Wings! 🌈"
        
        return response
    
    def _generate_smalltalk(self, search_results: List, metadata: Dict) -> str:
        """Generate smalltalk response"""
        responses = [
            "Tôi có thể giúp gì cho bạn về các tour Ruby Wings? 😊",
            "Bạn muốn tìm hiểu về tour nào ạ? Tôi sẵn sàng hỗ trợ! 🌿",
            "Ruby Wings có nhiều tour trải nghiệm thú vị. Bạn quan tâm đến chủ đề nào? 🎒"
        ]
        
        return random.choice(responses)
    
    def _generate_fallback(self, search_results: List, metadata: Dict) -> str:
        """Generate fallback response"""
        return "Tôi có thể giúp gì cho bạn về các tour Ruby Wings? Bạn có thể:\n\n1. Hỏi về tour cụ thể\n2. Tìm tour theo thời gian/địa điểm\n3. Hỏi về giá cả\n4. Yêu cầu tư vấn đặt tour\n\nHoặc liên hệ **0332510486** để được hỗ trợ nhanh nhất! 📞"

# Initialize response generator
response_gen = ResponseGenerator()

# ==================== ENHANCED CHAT PROCESSOR ====================
class ChatProcessor:
    """Enhanced chat processor with intent-driven flow"""
    
    def __init__(self):
        self.response_generator = response_gen
        self.search_engine = search_engine
    
    def ensure_knowledge_loaded(self):
        """Ensure knowledge is loaded before processing"""
        if not state._knowledge_loaded:
            logger.warning("⚠️ Knowledge not initialized – initializing now")
            if not load_knowledge():
                logger.error("❌ Failed to load knowledge in chat processor")
                return False
            
            search_engine.load_index()
            logger.info("✅ Knowledge ready for chat")
            return True
        
        return True
    
    def process(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process user message with enhanced intent handling"""
        start_time = time.time()
        
        try:
            # CRITICAL: Ensure knowledge is loaded
            if not self.ensure_knowledge_loaded():
                logger.error("❌ Cannot process chat without knowledge")
                return {
                    'reply': "Xin lỗi, hệ thống đang khởi tạo dữ liệu. Vui lòng thử lại sau 5 giây hoặc liên hệ **0332510486**! 🙏",
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
            
            # Detect intent with metadata
            intent, confidence, metadata = detect_intent(user_message)
            context['intent'] = intent.name if hasattr(intent, 'name') else str(intent)
            context['intent_metadata'] = metadata
            
            # Update intent statistics
            state.stats['intent_counts'][context['intent']] += 1
            
            # Detect phone number
            phone = metadata.get('phone_number') or detect_phone_number(user_message)
            if phone:
                context['lead_phone'] = phone
                context['stage'] = ConversationStage.LEAD.value
                
                # Capture lead
                if Config.ENABLE_LEAD_CAPTURE:
                    self._capture_lead(phone, session_id, user_message, context)
            
            # Detect location
            location = metadata.get('detected_location') or extract_location_from_query(user_message)
            if location:
                context['location_filter'] = location
            
            # State machine transition
            if Config.STATE_MACHINE_ENABLED:
                context['stage'] = self._next_stage(context['stage'], intent)
            
            # Intent-aware search
            search_results = self.search_engine.search(
                user_message, 
                Config.TOP_K, 
                intent=context['intent'],
                metadata=metadata
            )
            
            # Extract mentioned tours
            mentioned_tours = []
            for score, entry in search_results:
                tour_idx = entry.get('tour_index')
                if tour_idx is not None and tour_idx not in mentioned_tours:
                    mentioned_tours.append(tour_idx)
            
            context['mentioned_tours'] = mentioned_tours
            
            # Generate intent-driven response
            response_text = self.response_generator.generate(
                user_message,
                search_results,
                context
            )
            
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
            
            # Keep only last N messages
            if len(context['conversation_history']) > Config.CONVERSATION_HISTORY_LIMIT * 2:
                context['conversation_history'] = context['conversation_history'][-Config.CONVERSATION_HISTORY_LIMIT * 2:]
            
            # Build enhanced result
            result = {
                'reply': response_text,
                'session_id': session_id,
                'session_state': {
                    'stage': context.get('stage').value if hasattr(context.get('stage'), 'value') else context.get('stage'),
                    'intent': context.get('intent'),
                    'intent_metadata': metadata,
                    'mentioned_tours': mentioned_tours,
                    'has_phone': bool(phone)
                },
                'intent': {
                    'name': intent.name if hasattr(intent, 'name') else str(intent),
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
                'response_type': self._get_response_type(context['intent'])
            }
            
            # Cache result
            state.cache_response(cache_key, result)
            
            # Update stats
            state.stats['requests'] += 1
            
            # Log with intent and processing time
            processing_time = result['processing_time_ms']
            intent_name = context['intent']
            
            logger.info(f"⏱️ Processed in {processing_time}ms | "
                       f"Intent: {intent_name} | "
                       f"Results: {len(search_results)} | "
                       f"Mapping: {len(state.mapping)} entries")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            traceback.print_exc()
            
            state.stats['errors'] += 1
            
            return {
                'reply': "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ **0332510486**! 🙏",
                'session_id': session_id,
                'error': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }
    
    def _next_stage(self, current_stage: str, intent) -> str:
        """Determine next conversation stage"""
        intent_name = intent.name if hasattr(intent, 'name') else str(intent)
        
        if isinstance(current_stage, str):
            try:
                current_stage = ConversationStage(current_stage)
            except (ValueError, AttributeError):
                current_stage = ConversationStage.EXPLORE
        
        transitions = {
            ConversationStage.EXPLORE: {
                'TOUR_INQUIRY': ConversationStage.SUGGEST,
                'TOUR_LIST': ConversationStage.SUGGEST,
                'TOUR_FILTER': ConversationStage.SUGGEST,
                'PRICE_ASK': ConversationStage.SUGGEST,
                'ABOUT_COMPANY': ConversationStage.EXPLORE,
                'PROVIDE_PHONE': ConversationStage.LEAD,
                'CALLBACK_REQUEST': ConversationStage.CALLBACK
            },
            ConversationStage.SUGGEST: {
                'BOOKING_REQUEST': ConversationStage.SELECT,
                'PROVIDE_PHONE': ConversationStage.LEAD,
                'TOUR_FILTER': ConversationStage.SUGGEST,
                'PRICE_ASK': ConversationStage.SUGGEST
            },
            ConversationStage.SELECT: {
                'BOOKING_REQUEST': ConversationStage.BOOK,
                'PROVIDE_PHONE': ConversationStage.BOOK
            },
            ConversationStage.BOOK: {
                'PROVIDE_PHONE': ConversationStage.LEAD
            }
        }
        
        next_stages = transitions.get(current_stage, {})
        next_stage = next_stages.get(intent_name, current_stage)
        
        return next_stage.value if hasattr(next_stage, 'value') else str(next_stage)
    
    def _get_response_type(self, intent: str) -> str:
        """Get response type based on intent"""
        response_types = {
            'GREETING': 'GREETING',
            'FAREWELL': 'FAREWELL',
            'ABOUT_COMPANY': 'COMPANY_INFO',
            'TOUR_LIST': 'TOUR_LIST',
            'TOUR_FILTER': 'TOUR_FILTER_RESULT',
            'TOUR_INQUIRY': 'TOUR_DETAILS',
            'PRICE_ASK': 'PRICE_INFO',
            'BOOKING_REQUEST': 'BOOKING_INFO',
            'CONTACT_INFO': 'CONTACT_INFO',
            'CALLBACK_REQUEST': 'CALLBACK_CONFIRM',
            'PROVIDE_PHONE': 'LEAD_CONFIRM',
            'LEAD_CAPTURED': 'LEAD_CONFIRM',
            'SMALLTALK': 'SMALLTALK',
            'UNKNOWN': 'GENERAL'
        }
        
        return response_types.get(intent, 'GENERAL')
    
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
            
            # Send to Meta CAPI
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
                    if Config.DEBUG_META_CAPI:
                        logger.debug(f"Meta CAPI result: {result}")
                except Exception as e:
                    state.stats['meta_capi_errors'] += 1
                    logger.error(f"Meta CAPI lead error: {e}")
            
            # Save to Google Sheets
            if Config.ENABLE_GOOGLE_SHEETS:
                self._save_to_sheets(lead_data)
            
            # Fallback storage (always save)
            if Config.ENABLE_FALLBACK_STORAGE:
                self._save_to_fallback(lead_data)
            
            # Update stats
            state.stats['leads'] += 1
            
            logger.info(f"📞 Lead captured: {phone_clean[:4]}***{phone_clean[-2:]}")
            
        except Exception as e:
            logger.error(f"Lead capture error: {e}")
            traceback.print_exc()
    
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
            logger.info(f"✅ Saved to Google Sheets (13 columns): {len(row)} values")
            
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

# Initialize chat processor
chat_processor = ChatProcessor()

# ==================== ROUTES (UNCHANGED - PRESERVE EXISTING FEATURES) ====================
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
        'version': '5.2.3-intent',
        'timestamp': datetime.now().isoformat(),
        'knowledge': {
            'loaded': state._knowledge_loaded,
            'tours': len(state.tours_db),
            'mapping_entries': len(state.mapping),
            'company_info_loaded': state._company_info_loaded,
            'tour_entities_loaded': state._tour_entities_loaded
        },
        'modules': {
            'openai': OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
            'entities': ENTITIES_AVAILABLE,
            'meta_capi': META_CAPI_AVAILABLE,
            'response_guard': RESPONSE_GUARD_AVAILABLE,
            'faiss': FAISS_AVAILABLE,
            'numpy': NUMPY_AVAILABLE
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Index route"""
    return jsonify({
        'service': 'Ruby Wings AI Chatbot',
        'version': '5.2.3 (Intent-Driven Fix)',
        'status': 'running',
        'knowledge_loaded': state._knowledge_loaded,
        'tours_available': len(state.tours_db),
        'endpoints': {
            'chat': '/api/chat',
            'save_lead': '/api/save-lead',
            'call_button': '/api/call-button',
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
                if Config.DEBUG_META_CAPI:
                    logger.debug(f"Meta CAPI result: {result}")
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
                    logger.info(f"✅ Form lead saved to Google Sheets (13 columns): {len(row)} values")
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
            'message': 'Lead đã được lưu! Đội ngũ Ruby Wings sẽ liên hệ sớm nhất. 📞',
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

@app.route('/api/call-button', methods=['POST', 'OPTIONS'])
def call_button():
    """Track call button click"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        
        page_url = data.get('page_url', '')
        call_type = data.get('call_type', 'regular')
        
        if Config.ENABLE_META_CAPI_CALL and META_CAPI_AVAILABLE:
            try:
                result = send_meta_call_button(
                    request,
                    page_url=page_url,
                    call_type=call_type,
                    button_location='fixed_bottom_left',
                    button_text='Gọi ngay'
                )
                state.stats['meta_capi_calls'] += 1
                logger.info(f"📞 Call button tracked: {call_type}")
                if Config.DEBUG_META_CAPI:
                    logger.debug(f"Meta CAPI result: {result}")
            except Exception as e:
                state.stats['meta_capi_errors'] += 1
                logger.error(f"Meta CAPI call error: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Call tracked',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Call button error: {e}")
        traceback.print_exc()
        state.stats['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Statistics endpoint"""
    return jsonify(state.get_stats())

@app.route('/sessions', methods=['GET'])
def sessions():
    """List sessions (admin only)"""
    secret = request.headers.get('X-Admin-Key', '')
    if secret != Config.SECRET_KEY:
        return jsonify({'error': 'Unauthorized'}), 403
    
    session_list = []
    with state._lock:
        for sid, ctx in state.session_contexts.items():
            session_list.append({
                'session_id': sid,
                'stage': ctx.get('stage'),
                'intent': ctx.get('intent'),
                'last_updated': ctx.get('last_updated', '').isoformat() if isinstance(ctx.get('last_updated'), datetime) else str(ctx.get('last_updated'))
            })
    
    return jsonify({
        'count': len(session_list),
        'sessions': session_list
    })

@app.route('/reindex', methods=['POST'])
def reindex():
    """Reload knowledge and index (admin only)"""
    secret = request.headers.get('X-Admin-Key', '')
    if secret != Config.SECRET_KEY:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        state._knowledge_loaded = False
        state._index_loaded = False
        
        load_knowledge()
        search_engine.load_index()
        
        return jsonify({
            'success': True,
            'message': 'Reindex complete',
            'tours': len(state.tours_db),
            'mappings': len(state.mapping)
        })
        
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/meta-health', methods=['GET'])
def meta_health():
    """Meta CAPI health check"""
    if META_CAPI_AVAILABLE:
        result = check_meta_capi_health()
        result['stats'] = {
            'calls': state.stats.get('meta_capi_calls', 0),
            'errors': state.stats.get('meta_capi_errors', 0)
        }
        return jsonify(result)
    else:
        return jsonify({
            'status': 'unavailable',
            'message': 'Meta CAPI module not loaded'
        })

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    """404 handler"""
    return jsonify({
        'error': 'Not found',
        'path': request.path
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500 handler"""
    logger.error(f"500 error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Vui lòng thử lại sau'
    }), 500

# ==================== INITIALIZATION ====================
def initialize_app():
    """Initialize application"""
    try:
        logger.info("🚀 Initializing Ruby Wings Chatbot v5.2.3 (Intent-Driven)...")
        logger.info(f"🚀 App startup – RAM profile: {Config.RAM_PROFILE}MB")
        
        Config.log_config()
        
        logger.info("🔍 Step 1: Loading knowledge base...")
        if not load_knowledge():
            logger.error("❌ Failed to load knowledge base on startup")
            logger.info("⚠️ Continuing anyway - features that don't need knowledge will work")
        else:
            logger.info("✅ Knowledge loaded successfully at startup")
        
        logger.info("🔍 Step 2: Loading search index...")
        if not search_engine.load_index():
            logger.warning("⚠️ Search index not loaded, using text search only")
        else:
            logger.info("✅ Search engine ready")
        
        if META_CAPI_AVAILABLE and Config.ENABLE_META_CAPI:
            logger.info("✅ Meta CAPI ready")
        else:
            logger.warning("⚠️ Meta CAPI not available")
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            logger.info("✅ OpenAI ready")
        else:
            logger.info("ℹ️ OpenAI not available, using intent-based responses")
        
        logger.info("=" * 60)
        logger.info("✅ RUBY WINGS CHATBOT READY!")
        logger.info(f"📊 Tours loaded: {len(state.tours_db)}")
        logger.info(f"🔍 Mapping entries: {len(state.mapping)}")
        logger.info(f"🏢 Company info: {'Yes' if state._company_info_loaded else 'No'}")
        logger.info(f"🌐 Server: {Config.HOST}:{Config.PORT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        logger.error("❌ App may not function correctly")

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
    initialize_app()