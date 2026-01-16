#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PHIÊN BẢN TỐI ƯU HOÀN THIỆN
Version: 5.2 Professional
Created: 2026-01-13
Author: Ruby Wings AI Team

TÍNH NĂNG CHÍNH:
1. State Machine nâng cao cho conversation flow
2. Location Filter thông minh với đề xuất khu vực
3. Intent Detection với 15+ loại intent
4. Phone Detection & Lead Capture tự động
5. Meta CAPI tracking đầy đủ
6. FAISS vector search với fallback numpy
7. Memory optimization cho Render 512MB
8. Session management với auto-cleanup
9. Enhanced response formatting với labels
10. Multi-tour comparison & recommendation

KIẾN TRÚC: Tương thích hoàn toàn với Render Docker + Gunicorn
"""

# ==================== IMPORTS & CONFIGURATION ====================
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Future
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


# ==================== PLATFORM DETECTION ====================
import platform
IS_WINDOWS = platform.system().lower().startswith("win")
IS_RENDER = "RENDER" in os.environ
IS_PRODUCTION = os.environ.get("FLASK_ENV") == "production"

# ==================== FLASK & WEB FRAMEWORK ====================
from flask import Flask, request, jsonify, g, session as flask_session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ruby_wings.log') if IS_PRODUCTION else logging.NullHandler()
    ]
)
logger = logging.getLogger("ruby-wings-pro")
# ==================== JSON LOG HELPER ====================
def log_event(event: str, level: str = "info", **data):
    """
    Structured JSON log for AI & production analysis
    """
    payload = {
        "ts": datetime.utcnow().isoformat(),
        "event": event,
        "service": "ruby-wings-chatbot",
        "env": "production" if IS_PRODUCTION else "development",
    }

    # Add runtime context if available
    try:
        from flask import g
        payload["request_id"] = getattr(g, "request_id", None)
    except Exception:
        payload["request_id"] = None

    payload.update(data)

    try:
        message = json.dumps(payload, ensure_ascii=False)
    except Exception:
        message = str(payload)

    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)


# ==================== ENVIRONMENT VARIABLES ====================
class Config:
    """Centralized configuration management"""
    
    # RAM & Performance
    RAM_PROFILE = os.environ.get("RAM_PROFILE", "512")
    IS_LOW_RAM = False  # FORCE ENABLE LLM ON 512MB (FAISS vẫn tắt)
    
    # Feature Toggles
    ENABLE_STATE_MACHINE = os.environ.get("ENABLE_STATE_MACHINE", "true").lower() == "true"
    ENABLE_LOCATION_FILTER = os.environ.get("ENABLE_LOCATION_FILTER", "true").lower() == "true"
    ENABLE_INTENT_DETECTION = os.environ.get("ENABLE_INTENT_DETECTION", "true").lower() == "true"
    ENABLE_PHONE_DETECTION = os.environ.get("ENABLE_PHONE_DETECTION", "true").lower() == "true"
    ENABLE_LEAD_CAPTURE = os.environ.get("ENABLE_LEAD_CAPTURE", "true").lower() == "true"
    ENABLE_LLM_FALLBACK = os.environ.get("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    ENABLE_CACHING = os.environ.get("ENABLE_CACHING", "true").lower() == "true" and not IS_LOW_RAM
    ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "false").lower() == "true"
    ENABLE_META_CAPI = os.environ.get("ENABLE_META_CAPI", "true").lower() == "true"
    FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "false").lower() == "true" and not IS_LOW_RAM
    
    # Core Config
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
    FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
    FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
    TOUR_ENTITIES_PATH = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")
    
    # Model Config
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    
    # Performance
    TOP_K = int(os.environ.get("TOP_K", "5"))
    MAX_TOURS_PER_RESPONSE = int(os.environ.get("MAX_TOURS_PER_RESPONSE", "3"))
    CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
    MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "100"))
    MAX_EMBEDDING_CACHE = int(os.environ.get("MAX_EMBEDDING_CACHE", "50"))
    CONVERSATION_HISTORY_LIMIT = int(os.environ.get("CONVERSATION_HISTORY_LIMIT", "10"))
    
    # Intent Detection
    INTENT_CONFIDENCE_THRESHOLD = float(os.environ.get("INTENT_CONFIDENCE_THRESHOLD", "0.6"))
    ENABLE_FUZZY_MATCHING = os.environ.get("ENABLE_FUZZY_MATCHING", "true").lower() == "true"
    ENABLE_PHONE_VALIDATION = os.environ.get("ENABLE_PHONE_VALIDATION", "true").lower() == "true"
    
    # Location Filter
    LOCATION_FILTER_STRICT = os.environ.get("LOCATION_FILTER_STRICT", "true").lower() == "true"
    ENABLE_REGION_FALLBACK = os.environ.get("ENABLE_REGION_FALLBACK", "true").lower() == "true"
    
    # Response Formatting
    ENABLE_TOUR_LABELS = os.environ.get("ENABLE_TOUR_LABELS", "true").lower() == "true"
    ENABLE_LOCATION_CONTEXT = os.environ.get("ENABLE_LOCATION_CONTEXT", "true").lower() == "true"
    
    # State Machine
    AUTO_STAGE_TRANSITION = os.environ.get("AUTO_STAGE_TRANSITION", "true").lower() == "true"
    STATE_MACHINE_ENABLED = os.environ.get("STATE_MACHINE_ENABLED", "true").lower() == "true"
    
    # Server
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "10000"))
    DEBUG = os.environ.get("FLASK_ENV") == "development"
    
    @classmethod
    def log_configuration(cls):
        """Log system configuration"""
        logger.info("=" * 60)
        logger.info("🚀 RUBY WINGS CHATBOT v5.2 PROFESSIONAL")
        logger.info("=" * 60)
        logger.info(f"🧠 RAM Profile: {cls.RAM_PROFILE}MB | Low RAM: {cls.IS_LOW_RAM}")
        logger.info(f"🌍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
        logger.info(f"🔧 Platform: {'Windows' if IS_WINDOWS else 'Linux/Render'}")
        logger.info(f"📊 OpenAI: {'Available' if cls.OPENAI_API_KEY else 'Not Available'}")
        
        # Feature summary
        features = [
            f"State Machine: {'✅' if cls.ENABLE_STATE_MACHINE else '❌'}",
            f"Location Filter: {'✅' if cls.ENABLE_LOCATION_FILTER else '❌'}",
            f"Intent Detection: {'✅' if cls.ENABLE_INTENT_DETECTION else '❌'}",
            f"Phone Detection: {'✅' if cls.ENABLE_PHONE_DETECTION else '❌'}",
            f"Lead Capture: {'✅' if cls.ENABLE_LEAD_CAPTURE else '❌'}",
            f"FAISS Search: {'✅' if cls.FAISS_ENABLED else '❌'}",
            f"Caching: {'✅' if cls.ENABLE_CACHING else '❌'}",
            f"Meta CAPI: {'✅' if cls.ENABLE_META_CAPI else '❌'}"
        ]
        
        for i in range(0, len(features), 2):
            if i+1 < len(features):
                logger.info(f"   {features[i]:<25} {features[i+1]}")
            else:
                logger.info(f"   {features[i]}")
        
        logger.info("=" * 60)

# ==================== LAZY IMPORTS ====================
# Dynamically import heavy dependencies only when needed

class LazyImporter:
    """Lazy import manager for memory optimization"""
    
    @staticmethod
    def import_numpy():
        try:
            import numpy as np
            return np, True
        except ImportError:
            return None, False
    
    @staticmethod
    def import_faiss():
        try:
            import faiss
            return faiss, True
        except ImportError:
            return None, False
    
    @staticmethod
    def import_openai():
        try:
            from openai import OpenAI
            return OpenAI, True
        except ImportError:
            return None, False
    
    @staticmethod
    def import_google_sheets():
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            return gspread, Credentials, True
        except ImportError:
            return None, None, False

# Initialize lazy imports
np, NUMPY_AVAILABLE = LazyImporter.import_numpy()
faiss, FAISS_AVAILABLE = LazyImporter.import_faiss()
OpenAI, OPENAI_AVAILABLE = LazyImporter.import_openai()

# ==================== META CAPI IMPORT ====================
try:
    from meta_capi import (
        send_meta_pageview, 
        send_meta_lead, 
        send_meta_call_button,
        check_meta_capi_health
    )
    META_CAPI_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import meta_capi: {e}")
    META_CAPI_AVAILABLE = False
    # Create dummy functions
    def send_meta_pageview(request): pass
    def send_meta_lead(*args, **kwargs): return None
    def send_meta_call_button(*args, **kwargs): return None
    def check_meta_capi_health(): return {"status": "unavailable"}

# ==================== DATA MODELS ====================
@dataclass
class Tour:
    """Tour data model with enhanced fields"""
    index: int
    name: str = ""
    duration: str = ""
    location: str = ""
    price: str = ""
    summary: str = ""
    includes: List[str] = field(default_factory=list)
    accommodation: str = ""
    meals: str = ""
    transport: str = ""
    notes: str = ""
    style: str = ""
    who_can_join: str = ""
    mission: str = ""
    event_support: str = ""
    cancellation_policy: str = ""
    booking_method: str = ""
    hotline: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_field(self, field_name: str) -> str:
        """Safely get field value"""
        return getattr(self, field_name, "")

@dataclass
class UserProfile:
    """User profile for personalization"""
    session_id: str
    phone: Optional[str] = None
    name: Optional[str] = None
    location_preference: Optional[str] = None
    budget_range: Optional[str] = None
    preferred_duration: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    viewed_tours: List[int] = field(default_factory=list)
    inquiry_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def update_activity(self):
        self.last_active = datetime.now()

@dataclass
class ConversationContext:
    """Enhanced conversation context with state machine"""
    session_id: str
    stage: str = "explore"  # explore, suggest, compare, select, book, lead, callback
    current_intent: str = "unknown"
    mentioned_tours: List[int] = field(default_factory=list)
    selected_tour_id: Optional[int] = None
    location_filter: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "stage": self.stage,
            "current_intent": self.current_intent,
            "mentioned_tours": self.mentioned_tours,
            "selected_tour_id": self.selected_tour_id,
            "location_filter": self.location_filter,
            "conversation_history": self.conversation_history[-5:],  # Last 5 messages
            "user_preferences": self.user_preferences,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    def update_stage(self, new_stage: str):
        """Update conversation stage with validation"""
        valid_stages = ["explore", "suggest", "compare", "select", "book", "lead", "callback"]
        if new_stage in valid_stages:
            self.stage = new_stage
            self.last_updated = datetime.now()
            return True
        return False
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content[:500],  # Limit length
            "timestamp": datetime.now().isoformat(),
            "stage": self.stage
        }
        if metadata:
            message.update(metadata)
        
        self.conversation_history.append(message)
        # Keep only last N messages
        if len(self.conversation_history) > Config.CONVERSATION_HISTORY_LIMIT:
            self.conversation_history = self.conversation_history[-Config.CONVERSATION_HISTORY_LIMIT:]
        
        self.last_updated = datetime.now()

@dataclass
class LeadData:
    """Lead data model for capture"""
    phone: str
    source_channel: str = "Chatbot"
    action_type: str = "Inquiry"
    contact_name: Optional[str] = None
    email: Optional[str] = None
    service_interest: Optional[str] = None
    preferred_date: Optional[str] = None
    participant_count: Optional[str] = None
    budget_range: Optional[str] = None
    note: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_row(self) -> List:
        """Convert to Google Sheets row"""
        return [
            self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            self.phone,
            self.contact_name or "",
            self.email or "",
            self.service_interest or "",
            self.preferred_date or "",
            self.participant_count or "",
            self.budget_range or "",
            self.source_channel,
            self.action_type,
            self.note or "",
            self.utm_source or "",
            self.utm_medium or "",
            self.utm_campaign or ""
        ]
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ==================== ENUMS ====================
class Intent(Enum):
    """Enhanced intent classification"""
    GREETING = auto()
    FAREWELL = auto()
    TOUR_INQUIRY = auto()
    PRICE_QUESTION = auto()
    LOCATION_QUESTION = auto()
    DURATION_QUESTION = auto()
    BOOKING_REQUEST = auto()
    CONTACT_REQUEST = auto()
    COMPARE_TOURS = auto()
    RECOMMENDATION = auto()
    DETAIL_REQUEST = auto()
    AVAILABILITY_CHECK = auto()
    CANCELLATION_POLICY = auto()
    PROVIDE_PHONE = auto()
    CALLBACK_REQUEST = auto()
    BOOKING_CONFIRM = auto()
    MODIFY_REQUEST = auto()
    COMPLAINT = auto()
    THANK_YOU = auto()
    SMALLTALK = auto()
    LEAD_CAPTURED = auto()
    UNKNOWN = auto()

class ConversationStage(Enum):
    """State machine stages"""
    EXPLORE = "explore"        # User exploring options
    SUGGEST = "suggest"        # AI suggesting tours
    COMPARE = "compare"        # Comparing multiple tours
    SELECT = "select"          # User selected a tour
    BOOK = "book"              # Booking process
    LEAD = "lead"              # Lead capture
    CALLBACK = "callback"      # Callback requested
    FOLLOW_UP = "follow_up"    # Follow up needed

# ==================== FLASK APPLICATION ====================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'ruby_wings_'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


# CORS configuration
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

# Proxy fix for Render
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# ==================== META CAPI MIDDLEWARE ====================
@app.before_request
def before_request():
    """Global before request handler"""
    g.start_time = time.time()
    g.request_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]
    
    # Meta CAPI pageview tracking
    if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
        try:
            send_meta_pageview(request)
        except Exception as e:
            log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

            logger.error(f"Meta CAPI pageview error: {e}")
            log_event(
    "request_received",
    method=request.method,
    path=request.path,
    ip=request.remote_addr,
)


@app.after_request
def after_request(response):
    """Global after request handler"""
    # Add processing time header
    if hasattr(g, 'start_time'):
        processing_time = time.time() - g.start_time
        response.headers['X-Processing-Time'] = f'{processing_time:.3f}s'
        response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    
    # CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

# ==================== LOCATION REGION MAPPING ====================
LOCATION_REGION_MAPPING = {
    # Miền Bắc
    "hà nội": "Miền Bắc", "hanoi": "Miền Bắc", "thủ đô": "Miền Bắc",
    "hạ long": "Miền Bắc", "halong": "Miền Bắc", "vịnh hạ long": "Miền Bắc",
    "sapa": "Miền Bắc", "sa pa": "Miền Bắc", "lào cai": "Miền Bắc",
    "ninh bình": "Miền Bắc", "tràng an": "Miền Bắc", "tam cốc": "Miền Bắc",
    "mai châu": "Miền Bắc", "maichau": "Miền Bắc",
    "mộc châu": "Miền Bắc", "mocchau": "Miền Bắc",
    "yên tử": "Miền Bắc", "yentu": "Miền Bắc",
    
    # Miền Trung
    "đà nẵng": "Miền Trung", "danang": "Miền Trung",
    "huế": "Miền Trung", "hue": "Miền Trung", "cố đô": "Miền Trung",
    "quảng trị": "Miền Trung", "quangtri": "Miền Trung",
    "bạch mã": "Miền Trung", "bachma": "Miền Trung",
    "hội an": "Miền Trung", "hoian": "Miền Trung",
    "quảng bình": "Miền Trung", "quangbinh": "Miền Trung",
    "động phong nha": "Miền Trung", "phongnha": "Miền Trung",
    
    # Miền Nam
    "hồ chí minh": "Miền Nam", "hcm": "Miền Nam", "sài gòn": "Miền Nam", "saigon": "Miền Nam",
    "cần thơ": "Miền Nam", "cantho": "Miền Nam",
    "phú quốc": "Miền Nam", "phuquoc": "Miền Nam", "đảo ngọc": "Miền Nam",
    "nha trang": "Miền Nam", "nhatrang": "Miền Nam",
    "đà lạt": "Miền Nam", "dalat": "Miền Nam", "thành phố ngàn hoa": "Miền Nam",
    "vũng tàu": "Miền Nam", "vungtau": "Miền Nam",
    "mũi né": "Miền Nam", "muine": "Miền Nam",
    
    # Khu vực đặc biệt
    "cát bà": "Miền Bắc", "catba": "Miền Bắc",
    "côn đảo": "Miền Nam", "condao": "Miền Nam",
    "hoàng liên sơn": "Miền Bắc",
}

REGION_TOURS = {
    "Miền Bắc": [
        "Hà Nội Văn Hiến", "Sapa Trekking", "Hạ Long Bay", "Ninh Bình Tràng An",
        "Mai Châu Homestay", "Mộc Châu", "Yên Tử Thiền", "Ba Vì Retreat"
    ],
    "Miền Trung": [
        "Huế Cố Đô", "Đà Nẵng Biển Xanh", "Bạch Mã", "Hội An Phố Cổ",
        "Quảng Trị Lịch Sử", "Phong Nha Kẻ Bàng", "Bà Nà Hills"
    ],
    "Miền Nam": [
        "Sài Gòn Văn Minh", "Cần Thơ Sông Nước", "Phú Quốc Đảo Ngọc",
        "Đà Lạt Ngàn Hoa", "Nha Trang Biển", "Vũng Tàu", "Mũi Né"
    ]
}

# ==================== KEYWORD FIELD MAPPING ====================
KEYWORD_FIELD_MAP: Dict[str, Dict] = {
    "tour_list": {
        "keywords": [
            "tên tour", "tour gì", "danh sách tour", "có những tour nào", "liệt kê tour",
            "show tour", "tour hiện có", "tour available", "liệt kê các tour đang có",
            "list tour", "tour đang bán", "tour hiện hành", "tour nào", "tours", "liệt kê các tour",
            "liệt kê các hành trình", "list tours", "show tours", "các tour hiện tại", "tour có sẵn"
        ],
        "field": "tour_name",
        "intent": Intent.TOUR_INQUIRY
    },
    "mission": {
        "keywords": ["tầm nhìn", "sứ mệnh", "giá trị cốt lõi", "triết lý", "vision", "mission"],
        "field": "mission",
        "intent": Intent.DETAIL_REQUEST
    },
    "summary": {
        "keywords": ["tóm tắt chương trình tour", "tóm tắt", "overview", "brief", "mô tả ngắn", "giới thiệu"],
        "field": "summary",
        "intent": Intent.DETAIL_REQUEST
    },
    "style": {
        "keywords": ["phong cách hành trình", "tính chất hành trình", "concept tour", "vibe tour", "style", "loại hình"],
        "field": "style",
        "intent": Intent.DETAIL_REQUEST
    },
    "transport": {
        "keywords": ["vận chuyển", "phương tiện", "di chuyển", "xe gì", "transportation", "máy bay", "ô tô"],
        "field": "transport",
        "intent": Intent.DETAIL_REQUEST
    },
    "includes": {
        "keywords": ["lịch trình chi tiết", "chương trình chi tiết", "chi tiết hành trình", "itinerary", "schedule", "includes", "hoạt động"],
        "field": "includes",
        "intent": Intent.DETAIL_REQUEST
    },
    "location": {
        "keywords": ["ở đâu", "đi đâu", "địa phương nào", "nơi nào", "điểm đến", "destination", "location", "địa điểm"],
        "field": "location",
        "intent": Intent.LOCATION_QUESTION
    },
    "duration": {
        "keywords": ["thời gian tour", "kéo dài", "mấy ngày", "bao lâu", "ngày đêm", "duration", "tour dài bao lâu", "tour bao nhiêu ngày", "2 ngày 1 đêm", "3 ngày 2 đêm"],
        "field": "duration",
        "intent": Intent.DURATION_QUESTION
    },
    "price": {
        "keywords": ["giá tour", "chi phí", "bao nhiêu tiền", "price", "cost", "giá cả", "phí", "bao nhiêu", "kinh phí"],
        "field": "price",
        "intent": Intent.PRICE_QUESTION
    },
    "notes": {
        "keywords": ["lưu ý", "ghi chú", "notes", "cần chú ý", "chú ý", "lưu ý gì"],
        "field": "notes",
        "intent": Intent.DETAIL_REQUEST
    },
    "accommodation": {
        "keywords": ["chỗ ở", "nơi lưu trú", "khách sạn", "homestay", "accommodation", "nhà nghỉ", "resort"],
        "field": "accommodation",
        "intent": Intent.DETAIL_REQUEST
    },
    "meals": {
        "keywords": ["ăn uống", "ẩm thực", "meals", "thực đơn", "bữa", "đồ ăn", "thức ăn"],
        "field": "meals",
        "intent": Intent.DETAIL_REQUEST
    },
    "event_support": {
        "keywords": ["hỗ trợ", "dịch vụ hỗ trợ", "event support", "dịch vụ tăng cường", "dịch vụ thêm"],
        "field": "event_support",
        "intent": Intent.DETAIL_REQUEST
    },
    "cancellation_policy": {
        "keywords": ["phí huỷ", "chính sách huỷ", "cancellation", "refund policy", "hoàn tiền", "hủy tour"],
        "field": "cancellation_policy",
        "intent": Intent.CANCELLATION_POLICY
    },
    "booking_method": {
        "keywords": ["đặt chỗ", "đặt tour", "booking", "cách đặt", "làm sao đặt", "đặt như thế nào", "đặt hàng"],
        "field": "booking_method",
        "intent": Intent.BOOKING_REQUEST
    },
    "who_can_join": {
        "keywords": ["phù hợp đối tượng", "ai tham gia", "who should join", "độ tuổi", "sức khỏe", "yêu cầu"],
        "field": "who_can_join",
        "intent": Intent.DETAIL_REQUEST
    },
    "hotline": {
        "keywords": ["hotline", "số điện thoại", "liên hệ", "contact number", "đường dây nóng", "tổng đài"],
        "field": "hotline",
        "intent": Intent.CONTACT_REQUEST
    },
}

# ==================== GLOBAL STATE MANAGEMENT ====================
class GlobalState:
    """Memory-optimized global state management"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize global state"""
        self.tours_db: Dict[int, Tour] = {}
        self.tour_name_to_index: Dict[str, int] = {}
        self.session_contexts: Dict[str, ConversationContext] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.response_cache: OrderedDict[str, Dict] = OrderedDict()
        self.embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        
        # Index management
        self.index = None
        self.mapping: List[Dict] = []
        self.flat_texts: List[str] = []
        
        # Performance tracking
        self.stats = {
            "requests_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "sessions_created": 0,
            "leads_captured": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        logger.info("🌐 Global state initialized")
    
    # === Tour Management ===
    def get_tour(self, index: int) -> Optional[Tour]:
        with self._lock:
            return self.tours_db.get(index)
    
    def get_tours_by_indices(self, indices: List[int]) -> List[Tour]:
        with self._lock:
            return [self.tours_db.get(idx) for idx in indices if idx in self.tours_db]
    
    def get_all_tour_indices(self) -> List[int]:
        with self._lock:
            return list(self.tours_db.keys())
    
    def add_tour(self, index: int, tour: Tour):
        with self._lock:
            self.tours_db[index] = tour
            if tour.name:
                self.tour_name_to_index[tour.name.lower()] = index
    
    def clear_tours(self):
        with self._lock:
            self.tours_db.clear()
            self.tour_name_to_index.clear()
    
    def find_tour_by_name(self, name: str) -> Optional[int]:
        with self._lock:
            return self.tour_name_to_index.get(name.lower().strip())
    
    # === Session Management ===
    def get_session_context(self, session_id: str) -> ConversationContext:
        with self._lock:
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = ConversationContext(session_id=session_id)
                self.stats["sessions_created"] += 1
                
                # Cleanup old sessions if needed
                if len(self.session_contexts) > Config.MAX_SESSIONS:
                    self._cleanup_old_sessions()
            
            return self.session_contexts[session_id]
    
    def update_session_context(self, session_id: str, updates: Dict[str, Any]):
        with self._lock:
            if session_id in self.session_contexts:
                ctx = self.session_contexts[session_id]
                for key, value in updates.items():
                    if hasattr(ctx, key):
                        setattr(ctx, key, value)
                ctx.last_updated = datetime.now()
    
    def get_user_profile(self, session_id: str) -> UserProfile:
        with self._lock:
            if session_id not in self.user_profiles:
                self.user_profiles[session_id] = UserProfile(session_id=session_id)
            return self.user_profiles[session_id]
    
    def _cleanup_old_sessions(self):
        """Cleanup old sessions based on last activity"""
        with self._lock:
            if len(self.session_contexts) <= Config.MAX_SESSIONS:
                return
            
            # Sort by last_updated
            sorted_sessions = sorted(
                self.session_contexts.items(),
                key=lambda x: x[1].last_updated
            )
            
            # Remove oldest 20%
            remove_count = max(1, len(sorted_sessions) // 5)
            for session_id, _ in sorted_sessions[:remove_count]:
                if session_id in self.session_contexts:
                    del self.session_contexts[session_id]
                if session_id in self.user_profiles:
                    del self.user_profiles[session_id]
            
            logger.info(f"🧹 Cleaned up {remove_count} old sessions")
    
    # === Cache Management ===
    def get_cached_response(self, key: str) -> Optional[Dict]:
        with self._lock:
            if key in self.response_cache:
                entry = self.response_cache[key]
                # Check TTL
                if time.time() - entry['timestamp'] < Config.CACHE_TTL_SECONDS:
                    self.response_cache.move_to_end(key)  # Mark as recently used
                    self.stats["cache_hits"] += 1
                    return entry['value']
                else:
                    del self.response_cache[key]
            
            self.stats["cache_misses"] += 1
            return None
    
    def cache_response(self, key: str, value: Dict):
        if not Config.ENABLE_CACHING:
            return
        
        with self._lock:
            self.response_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            
            # LRU eviction
            if len(self.response_cache) > Config.MAX_EMBEDDING_CACHE:
                self.response_cache.popitem(last=False)
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        with self._lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            return self.embedding_cache.get(text_hash)
    
    def cache_embedding(self, text: str, embedding: List[float]):
        with self._lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.embedding_cache[text_hash] = embedding
            
            # LRU eviction
            if len(self.embedding_cache) > Config.MAX_EMBEDDING_CACHE:
                self.embedding_cache.popitem(last=False)
    
    # === Statistics ===
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = datetime.now() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": uptime.total_seconds(),
                "uptime_human": str(uptime),
                "active_sessions": len(self.session_contexts),
                "tours_loaded": len(self.tours_db),
                "cache_size": len(self.response_cache),
                "embedding_cache_size": len(self.embedding_cache)
            }
    
    def increment_stat(self, stat_name: str, amount: int = 1):
        with self._lock:
            if stat_name in self.stats:
                self.stats[stat_name] += amount

# Initialize global state
state = GlobalState()

# ==================== UTILITY FUNCTIONS ====================
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

def extract_phone_number(text: str) -> Optional[str]:
    """Extract Vietnamese phone number from text"""
    if not text:
        return None
    
    # Vietnamese phone patterns
    patterns = [
        r'(?:84|0)?(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-9]|9[0-9])[0-9]{7}',
        r'\(\+84\)\s?\d{2,3}\s?\d{3}\s?\d{4}',
        r'\+\d{2}\s?\d{3}\s?\d{3}\s?\d{4}',
        r'0\d{9,10}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean and validate the phone number
            phone = matches[0].replace(' ', '').replace('(', '').replace(')', '').replace('+', '')
            if phone.startswith('84'):
                phone = '0' + phone[2:]
            elif not phone.startswith('0'):
                phone = '0' + phone
            
            # Validate length (10-11 digits for Vietnam)
            if 10 <= len(phone) <= 11 and phone.startswith('0'):
                return phone
    
    return None

def validate_phone_number(phone: str) -> bool:
    """Validate Vietnamese phone number"""
    if not phone:
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a valid Vietnamese mobile number
    viettel_prefixes = ['32', '33', '34', '35', '36', '37', '38', '39', '86', '96', '97', '98']
    mobifone_prefixes = ['70', '76', '77', '78', '79', '89', '90', '93']
    vinaphone_prefixes = ['81', '82', '83', '84', '85', '88', '91', '94']
    vietnamobile_prefixes = ['52', '56', '58', '92']
    gmobile_prefixes = ['59', '99']
    
    all_prefixes = viettel_prefixes + mobifone_prefixes + vinaphone_prefixes + vietnamobile_prefixes + gmobile_prefixes
    
    if len(digits) == 10 or len(digits) == 11:
        prefix = digits[1:3] if digits.startswith('0') else digits[0:2]
        return prefix in all_prefixes
    
    return False

def extract_location_from_query(query: str) -> Optional[str]:
    """Extract location from user query with enhanced matching"""
    if not query:
        return None
    
    query_lower = query.lower()
    
    # Check for exact location matches
    for location in LOCATION_REGION_MAPPING:
        if location in query_lower:
            return location
    
    # Check for patterns with location keywords
    location_patterns = [
        r'tại\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'ở\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'địa\s+điểm\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'destination\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'location\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'đi\s+([a-zA-ZÀ-ỹ\s]{2,})',
        r'đến\s+([a-zA-ZÀ-ỹ\s]{2,})'
    ]
    
    for pattern in location_patterns:
        matches = re.search(pattern, query_lower)
        if matches:
            location = matches.group(1).strip()
            # Validate location length and content
            if len(location) >= 2 and not any(word in location for word in ['bao nhiêu', 'thế nào', 'là gì']):
                return location
    
    return None

def find_region_for_location(location: str) -> Optional[str]:
    """Find region for a given location"""
    if not location:
        return None
    
    location_lower = location.lower()
    
    # Direct mapping
    for loc, region in LOCATION_REGION_MAPPING.items():
        if loc in location_lower:
            return region
    
    # Fuzzy matching
    location_words = set(location_lower.split())
    for loc, region in LOCATION_REGION_MAPPING.items():
        loc_words = set(loc.split())
        if location_words & loc_words:
            return region
    
    return None

# ==================== INTENT DETECTION ENGINE ====================
class IntentDetectionEngine:
    """Enhanced intent detection with confidence scoring"""
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.fuzzy_matcher = FuzzyMatcher() if Config.ENABLE_FUZZY_MATCHING else None
    
    def _build_intent_patterns(self) -> Dict[Intent, List[Tuple[str, float]]]:
        """Build intent patterns with confidence scores"""
        return {
            Intent.GREETING: [
                ("chào", 1.0), ("xin chào", 1.0), ("hello", 0.9), ("hi", 0.8),
                ("chào bạn", 1.0), ("chào anh", 1.0), ("chào chị", 1.0),
                ("good morning", 0.7), ("good afternoon", 0.7), ("good evening", 0.7)
            ],
            Intent.FAREWELL: [
                ("tạm biệt", 1.0), ("bye", 0.9), ("goodbye", 0.8),
                ("hẹn gặp lại", 0.9), ("cảm ơn", 0.7), ("thanks", 0.7),
                ("thank you", 0.7), ("cám ơn", 0.7)
            ],
            Intent.TOUR_INQUIRY: [
                ("tour", 0.8), ("hành trình", 0.8), ("chuyến đi", 0.7),
                ("có tour nào", 0.9), ("tour gì", 0.9), ("danh sách tour", 1.0),
                ("liệt kê tour", 0.9), ("tour hiện có", 0.9)
            ],
            Intent.PRICE_QUESTION: [
                ("giá", 0.9), ("bao nhiêu tiền", 1.0), ("chi phí", 0.9),
                ("price", 0.8), ("cost", 0.8), ("kinh phí", 0.9),
                ("tốn bao nhiêu", 0.9), ("phí", 0.8)
            ],
            Intent.LOCATION_QUESTION: [
                ("ở đâu", 0.9), ("đi đâu", 0.8), ("địa điểm", 0.9),
                ("location", 0.8), ("destination", 0.8), ("điểm đến", 0.9),
                ("nơi nào", 0.8)
            ],
            Intent.DURATION_QUESTION: [
                ("bao lâu", 0.9), ("mấy ngày", 1.0), ("thời gian", 0.8),
                ("duration", 0.7), ("kéo dài", 0.8), ("ngày đêm", 0.9)
            ],
            Intent.BOOKING_REQUEST: [
                ("đặt tour", 1.0), ("booking", 0.9), ("đặt chỗ", 0.9),
                ("reserve", 0.7), ("book", 0.8), ("muốn đặt", 0.9),
                ("cần đặt", 0.8), ("đặt hàng", 0.7)
            ],
            Intent.CONTACT_REQUEST: [
                ("liên hệ", 0.9), ("hotline", 1.0), ("số điện thoại", 1.0),
                ("contact", 0.8), ("phone", 0.7), ("đường dây nóng", 0.9),
                ("tổng đài", 0.9)
            ],
            Intent.COMPARE_TOURS: [
                ("so sánh", 0.9), ("compare", 0.8), ("khác nhau", 0.7),
                ("tour nào tốt hơn", 0.9), ("nên chọn tour nào", 0.9),
                ("tour nào phù hợp", 0.8)
            ],
            Intent.RECOMMENDATION: [
                ("gợi ý", 0.9), ("recommend", 0.8), ("suggest", 0.8),
                ("tour nào hay", 0.8), ("nên đi tour nào", 0.9),
                ("phù hợp với tôi", 0.8)
            ],
            Intent.DETAIL_REQUEST: [
                ("chi tiết", 0.8), ("details", 0.7), ("thông tin", 0.7),
                ("cụ thể", 0.7), ("nội dung", 0.7), ("mô tả", 0.7)
            ],
            Intent.AVAILABILITY_CHECK: [
                ("còn chỗ", 0.9), ("available", 0.8), ("trống", 0.7),
                ("có sẵn", 0.8), ("còn slot", 0.8), ("còn vé", 0.8)
            ],
            Intent.CANCELLATION_POLICY: [
                ("hủy tour", 0.9), ("cancellation", 0.8), ("hoàn tiền", 0.9),
                ("refund", 0.8), ("phí hủy", 1.0), ("chính sách hủy", 1.0)
            ],
            Intent.PROVIDE_PHONE: [
                ("số tôi là", 0.8), ("điện thoại", 0.7), ("phone của tôi", 0.8),
                ("liên hệ qua số", 0.9), ("gọi tôi số", 0.9)
            ],
            Intent.CALLBACK_REQUEST: [
                ("gọi lại", 1.0), ("callback", 0.9), ("liên hệ lại", 0.9),
                ("nhân viên gọi", 0.8), ("tư vấn qua điện thoại", 0.9)
            ],
            Intent.BOOKING_CONFIRM: [
                ("xác nhận đặt", 1.0), ("confirm booking", 0.9),
                ("đồng ý đặt", 0.9), ("ok book", 0.8), ("đặt đi", 0.8)
            ],
            Intent.MODIFY_REQUEST: [
                ("thay đổi", 0.9), ("modify", 0.8), ("chỉnh sửa", 0.8),
                ("đổi tour", 0.9), ("thay tour", 0.9), ("đổi ngày", 0.9)
            ],
            Intent.COMPLAINT: [
                ("khiếu nại", 1.0), ("complaint", 0.9), ("phàn nàn", 0.8),
                ("không hài lòng", 0.8), ("thất vọng", 0.7), ("tệ", 0.6)
            ],
            Intent.THANK_YOU: [
                ("cảm ơn", 1.0), ("thanks", 0.9), ("thank you", 0.9),
                ("cám ơn", 1.0), ("appreciate", 0.7), ("biết ơn", 0.8)
            ],
            Intent.SMALLTALK: [
                ("khỏe không", 0.9), ("how are you", 0.8), ("bạn khỏe không", 0.9),
                ("dạo này thế nào", 0.7), ("trò chuyện", 0.6), ("chat", 0.5)
            ]
        }
    
    def detect_intent(self, message: str) -> Tuple[Intent, float, Dict[str, Any]]:
        """Detect intent with confidence score and metadata"""
        if not message or not Config.ENABLE_INTENT_DETECTION:
            return Intent.UNKNOWN, 0.0, {}
        
        message_lower = message.lower().strip()
        metadata = {
            "original_message": message,
            "detected_phone": None,
            "detected_location": None,
            "keywords_found": []
        }
        
        # Extract phone number
        if Config.ENABLE_PHONE_DETECTION:
            phone = extract_phone_number(message)
            if phone and validate_phone_number(phone):
                metadata["detected_phone"] = phone
                # Phone detection strongly indicates PROVIDE_PHONE intent
                return Intent.PROVIDE_PHONE, 0.95, metadata
        
        # Extract location
        location = extract_location_from_query(message)
        if location:
            metadata["detected_location"] = location
        
        # Score each intent
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern, weight in patterns:
                if pattern in message_lower:
                    intent_scores[intent] += weight
                    metadata["keywords_found"].append(pattern)
        
        # Check keyword field mapping
        for field_config in KEYWORD_FIELD_MAP.values():
            for keyword in field_config["keywords"]:
                if keyword in message_lower:
                    intent = field_config.get("intent", Intent.UNKNOWN)
                    intent_scores[intent] += 0.5
                    metadata["keywords_found"].append(keyword)
        
        # Apply fuzzy matching if enabled
        if self.fuzzy_matcher:
            fuzzy_scores = self.fuzzy_matcher.match_intent(message_lower)
            for intent, score in fuzzy_scores.items():
                intent_scores[intent] += score * 0.3  # Lower weight for fuzzy matches
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / 3.0, 1.0)  # Normalize to 0-1
            
            # Apply threshold
            if confidence >= Config.INTENT_CONFIDENCE_THRESHOLD:
                log_event(
    "intent_detected",
    intent=best_intent[0].name,
    confidence=confidence,
    keywords=metadata.get("keywords_found", []),
)

                return best_intent[0], confidence, metadata
        
        # Fallback to keyword-based detection
        return self._fallback_intent_detection(message_lower, metadata)
    
    def _fallback_intent_detection(self, message: str, metadata: Dict) -> Tuple[Intent, float, Dict]:
        """Fallback intent detection logic"""
        # Check for question words
        question_words = ["ai", "cái gì", "gì", "nào", "đâu", "bao nhiêu", "tại sao", "thế nào"]
        if any(word in message for word in question_words):
            if "giá" in message or "bao nhiêu tiền" in message:
                return Intent.PRICE_QUESTION, 0.7, metadata
            elif "ở đâu" in message or "đi đâu" in message:
                return Intent.LOCATION_QUESTION, 0.7, metadata
            elif "bao lâu" in message or "mấy ngày" in message:
                return Intent.DURATION_QUESTION, 0.7, metadata
            else:
                return Intent.TOUR_INQUIRY, 0.6, metadata
        
        # Check for action words
        action_words = ["muốn", "cần", "tìm", "hỏi", "đặt", "book", "booking"]
        if any(word in message for word in action_words):
            if "đặt" in message or "book" in message:
                return Intent.BOOKING_REQUEST, 0.8, metadata
            else:
                return Intent.TOUR_INQUIRY, 0.6, metadata
        
        return Intent.UNKNOWN, 0.5, metadata

class FuzzyMatcher:
    """Fuzzy matching for intent detection"""
    
    def __init__(self):
        self.word_similarity_threshold = 0.7
    
    def match_intent(self, message: str) -> Dict[Intent, float]:
        """Fuzzy match message to intents"""
        # Simplified implementation
        # In production, use Levenshtein distance or ML model
        return {}

# Initialize intent detection
intent_detector = IntentDetectionEngine()

# ==================== LOCATION FILTER ENGINE ====================
class LocationFilterEngine:
    """Advanced location filtering with region fallback"""
    
    def __init__(self):
        self.location_cache = {}
    
    def filter_tours_by_location(self, tour_indices: List[int], location: str) -> Tuple[List[int], str, str]:
        """
        Filter tours by location with intelligent fallback
        Returns: (filtered_indices, message, region)
        """
        if not location or not Config.ENABLE_LOCATION_FILTER:
            return tour_indices, "", ""
        
        location_lower = location.lower()
        region = find_region_for_location(location)
        
        # Check cache
        cache_key = f"{location_lower}_{hash(str(tour_indices))}"
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
        
        # Get all tours
        all_tours = state.get_tours_by_indices(tour_indices)
        
        # Strict filtering first
        exact_matches = []
        for tour in all_tours:
            if tour and tour.location and location_lower in tour.location.lower():
                exact_matches.append(tour.index)
        
        if exact_matches:
            message = f"✅ Tìm thấy {len(exact_matches)} tour tại **{location.title()}**:"
            result = (exact_matches, message, region or "")
            self.location_cache[cache_key] = result
            return result
        
        # Region-based fallback
        if Config.ENABLE_REGION_FALLBACK and region:
            region_tours = []
            for tour in all_tours:
                if tour and tour.location:
                    # Check if tour location matches any keyword in region
                    tour_location_lower = tour.location.lower()
                    region_keywords = REGION_TOURS.get(region, [])
                    for keyword in region_keywords:
                        if keyword.lower() in tour_location_lower:
                            region_tours.append(tour.index)
                            break
            
            if region_tours:
                region_tours = list(set(region_tours))[:Config.MAX_TOURS_PER_RESPONSE]
                message = f"🔍 Không tìm thấy tour tại **{location.title()}**, nhưng có {len(region_tours)} tour tại **{region}**:"
                result = (region_tours, message, region)
                self.location_cache[cache_key] = result
                return result
        
        # General fallback - popular tours
        popular_tours = tour_indices[:min(Config.MAX_TOURS_PER_RESPONSE, len(tour_indices))]
        message = f"⚠️ Không tìm thấy tour tại **{location.title()}**. Dưới đây là các tour phổ biến:"
        
        result = (popular_tours, message, region or "")
        self.location_cache[cache_key] = result
        return result

# Initialize location filter
location_filter = LocationFilterEngine()

# ==================== STATE MACHINE ENGINE ====================
class StateMachineEngine:
    """Advanced state machine for conversation flow"""
    
    def __init__(self):
        self.state_transitions = self._build_state_transitions()
    
    def _build_state_transitions(self) -> Dict[str, Dict[str, str]]:
        """Define state transitions based on intent"""
        return {
            "explore": {
                "tour_inquiry": "suggest",
                "booking_request": "suggest",
                "recommendation": "suggest",
                "provide_phone": "lead",
                "callback_request": "callback"
            },
            "suggest": {
                "compare_tours": "compare",
                "tour_inquiry": "suggest",
                "select": "select",
                "provide_phone": "lead",
                "booking_request": "select"
            },
            "compare": {
                "select": "select",
                "tour_inquiry": "suggest",
                "provide_phone": "lead"
            },
            "select": {
                "booking_confirm": "book",
                "provide_phone": "book",
                "callback_request": "callback",
                "modify_request": "suggest"
            },
            "book": {
                "booking_confirm": "lead",
                "provide_phone": "lead",
                "callback_request": "callback"
            },
            "lead": {
                "thank_you": "explore",
                "tour_inquiry": "suggest"
            },
            "callback": {
                "thank_you": "explore",
                "tour_inquiry": "suggest"
            }
        }
    
    def get_next_state(self, current_state: str, intent: Intent) -> str:
        """Determine next state based on current state and intent"""
        if not Config.ENABLE_STATE_MACHINE:
            return current_state
        
        intent_key = self._intent_to_key(intent)
        
        # Get possible transitions from current state
        transitions = self.state_transitions.get(current_state, {})
        
        # Find matching transition
        for transition_intent, next_state in transitions.items():
            if transition_intent == intent_key:
                logger.info(f"🔄 State transition: {current_state} -> {next_state} (trigger: {intent_key})")
                log_event(
        "state_transition",
        from_state=current_state,
        to_state=next_state,
        intent=intent.name,
    )

                return next_state
        
        # Default: stay in current state
        return current_state
    
    def _intent_to_key(self, intent: Intent) -> str:
        """Convert Intent enum to string key"""
        intent_name = intent.name.lower()
        
        # Map to transition keys
        mapping = {
            "tour_inquiry": ["tour_inquiry", "price_question", "location_question", "duration_question"],
            "booking_request": ["booking_request"],
            "recommendation": ["recommendation"],
            "compare_tours": ["compare_tours"],
            "select": ["detail_request", "availability_check"],
            "booking_confirm": ["booking_confirm"],
            "provide_phone": ["provide_phone"],
            "callback_request": ["callback_request"],
            "modify_request": ["modify_request"],
            "thank_you": ["thank_you"],
            "greeting": ["greeting"],
            "farewell": ["farewell"]
        }
        
        for key, intent_list in mapping.items():
            if intent_name in intent_list:
                return key
        
        return "unknown"

# Initialize state machine
state_machine = StateMachineEngine()

# ==================== KNOWLEDGE BASE LOADER ====================
def load_knowledge_base():
    """Load and parse knowledge base with enhanced tour extraction"""
    if not os.path.exists(Config.KNOWLEDGE_PATH):
        logger.error(f"Knowledge file not found: {Config.KNOWLEDGE_PATH}")
        return False
    
    try:
        logger.info(f"📚 Loading knowledge base from {Config.KNOWLEDGE_PATH}")
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        # Clear existing data
        state.clear_tours()
        
        # Extract tours
        tours_data = knowledge.get('tours', [])
        
        for idx, tour_data in enumerate(tours_data):
            try:
                # Create Tour object with enhanced fields
                tour = Tour(
                    index=idx,
                    name=tour_data.get('tour_name', ''),
                    duration=tour_data.get('duration', ''),
                    location=tour_data.get('location', ''),
                    price=tour_data.get('price', ''),
                    summary=tour_data.get('summary', ''),
                    includes=tour_data.get('includes', []),
                    accommodation=tour_data.get('accommodation', ''),
                    meals=tour_data.get('meals', ''),
                    transport=tour_data.get('transport', ''),
                    notes=tour_data.get('notes', ''),
                    style=tour_data.get('style', ''),
                    who_can_join=tour_data.get('who_can_join', ''),
                    mission=tour_data.get('mission', ''),
                    event_support=tour_data.get('event_support', ''),
                    cancellation_policy=tour_data.get('cancellation_policy', ''),
                    booking_method=tour_data.get('booking_method', ''),
                    hotline=tour_data.get('hotline', '')
                )
                
                state.add_tour(idx, tour)
                
                # Index by name variations
                if tour.name:
                    # Index full name
                    state.tour_name_to_index[tour.name.lower()] = idx
                    
                    # Index individual words (for fuzzy matching)
                    name_words = tour.name.lower().split()
                    for word in name_words:
                        if len(word) > 2:  # Ignore short words
                            state.tour_name_to_index[word] = idx
                
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"❌ Error loading tour {idx}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(state.tours_db)} tours with enhanced metadata")
        
        # Load mapping for search
        load_search_mapping()
        
        return False
        
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        logger.error(f"❌ Failed to load knowledge base: {e}")
        traceback.print_exc()
        return False

def load_search_mapping():
    """Load or create search mapping"""
    try:
        if os.path.exists(Config.FAISS_MAPPING_PATH):
            with open(Config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                state.mapping = json.load(f)
            state.flat_texts = [m.get("text", "") for m in state.mapping]
            logger.info(f"✅ Loaded {len(state.mapping)} mapping entries")
        else:
            # Create mapping from tours
            state.mapping = []
            state.flat_texts = []
            
            for idx, tour in state.tours_db.items():
                # Add tour fields to mapping
                fields = [
                    ("tour_name", tour.name),
                    ("location", tour.location),
                    ("duration", tour.duration),
                    ("price", tour.price),
                    ("summary", tour.summary),
                    ("style", tour.style),
                    ("includes", "; ".join(tour.includes) if tour.includes else ""),
                    ("accommodation", tour.accommodation),
                    ("meals", tour.meals),
                    ("transport", tour.transport),
                    ("notes", tour.notes),
                    ("who_can_join", tour.who_can_join),
                    ("mission", tour.mission),
                    ("event_support", tour.event_support),
                    ("cancellation_policy", tour.cancellation_policy),
                    ("booking_method", tour.booking_method),
                    ("hotline", tour.hotline)
                ]
                
                for field_name, field_value in fields:
                    if field_value and str(field_value).strip():
                        state.mapping.append({
                            "path": f"tours[{idx}].{field_name}",
                            "text": str(field_value),
                            "tour_index": idx,
                            "field": field_name
                        })
                        state.flat_texts.append(str(field_value))
            
            logger.info(f"📝 Created {len(state.mapping)} mapping entries from tours")
            
            # Save mapping for future use
            save_mapping_to_disk()
        
        return True
        
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        logger.error(f"❌ Failed to load search mapping: {e}")
        return True

def save_mapping_to_disk():
    """Save mapping to disk"""
    try:
        with open(Config.FAISS_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(state.mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved mapping to {Config.FAISS_MAPPING_PATH}")
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        logger.error(f"❌ Failed to save mapping: {e}")

# ==================== EMBEDDING & SEARCH ENGINE ====================
class EmbeddingEngine:
    """Unified embedding engine with caching and fallback"""
    
    def __init__(self):
        self.client = None
        self.cache_hits = 0
        self.cache_misses = 0
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"❌ Failed to initialize OpenAI: {e}")
    
    @lru_cache(maxsize=1024)
    def get_embedding(self, text: str) -> Tuple[List[float], int]:
        """Get embedding with intelligent caching"""
        if not text:
            return [], 0
        
        # Check memory cache first
        cached = state.get_cached_embedding(text)
        if cached:
            self.cache_hits += 1
            return cached, len(cached)
        
        self.cache_misses += 1
        
        # Try OpenAI first
        if self.client:
            try:
                # Truncate if too long
                short_text = text[:2000] if len(text) > 2000 else text
                
                response = self.client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=short_text,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                dim = len(embedding)
                
                # Cache the result
                state.cache_embedding(text, embedding)
                
                return embedding, dim
                
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"OpenAI embedding error: {e}")
        
        # Fallback: deterministic embedding
        return self._get_fallback_embedding(text)
    
    def _get_fallback_embedding(self, text: str) -> Tuple[List[float], int]:
        """Generate deterministic fallback embedding"""
        # Simple hash-based embedding (deterministic)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        
        # Generate 1536-dim embedding (matches text-embedding-3-small)
        dim = 1536
        embedding = []
        
        for i in range(dim):
            # Deterministic "random" based on hash and position
            val = ((hash_int >> (i % 32)) & 0xFF) / 255.0
            # Add some position-based variation
            val = (val + (i % 7) / 7.0) % 1.0
            embedding.append(float(val))
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        # Cache
        state.cache_embedding(text, embedding)
        
        return embedding, dim
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding engine statistics"""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "openai_available": self.client is not None
        }

class SearchEngine:
    """Unified search engine with multiple backends"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.index = None
        self.index_loaded = False
        self._init_index()
    
    def _init_index(self):
        """Initialize search index"""
        try:
            # Try FAISS first
            if Config.FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info("📦 Loading FAISS index...")
                self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS index loaded: {self.index.ntotal} vectors")
                self.index_loaded = True
                return
            
            # Try numpy fallback
            if NUMPY_AVAILABLE and os.path.exists(Config.FALLBACK_VECTORS_PATH):
                logger.info("📦 Loading numpy vectors...")
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                
                if 'vectors' in data:
                    vectors = data['vectors']
                elif 'mat' in data:
                    vectors = data['mat']
                else:
                    # Get first array
                    first_key = list(data.keys())[0]
                    vectors = data[first_key]
                
                # Create simple numpy index
                self.index = NumpyIndex(vectors)
                logger.info(f"✅ Numpy index loaded: {vectors.shape[0]} vectors")
                self.index_loaded = True
                return
            
            logger.warning("⚠️ No index found, will use text-based search")
            self.index_loaded = False
            
        except Exception as e:
            log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

            logger.error(f"❌ Failed to load index: {e}")
            self.index_loaded = False
    
    def search(self, query: str, top_k: int = None, filters: Dict = None) -> List[Tuple[float, Dict]]:
        """Search for relevant passages"""
        if top_k is None:
            top_k = Config.TOP_K
        
        # Get query embedding
        embedding, dim = self.embedding_engine.get_embedding(query)
        if not embedding:
            return []
        
        # Text-based fallback if no index
        if not self.index_loaded:
            return self._text_search(query, top_k, filters)
        
        try:
            # Convert to numpy array
            query_vec = np.array([embedding], dtype='float32')
            
            # Search
            if isinstance(self.index, faiss.Index):
                # FAISS search
                scores, indices = self.index.search(query_vec, top_k)
                results = []
                
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(state.mapping):
                        # Apply filters if any
                        if filters and not self._passes_filters(state.mapping[idx], filters):
                            continue
                        results.append((float(score), state.mapping[idx]))
                
                return results
                
            elif isinstance(self.index, NumpyIndex):
                # Numpy index search
                scores, indices = self.index.search(query_vec, top_k)
                results = []
                
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(state.mapping):
                        # Apply filters if any
                        if filters and not self._passes_filters(state.mapping[idx], filters):
                            continue
                        results.append((float(score), state.mapping[idx]))
                
                return results
                
            else:
                return self._text_search(query, top_k, filters)
                
        except Exception as e:
            log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

            logger.error(f"❌ Search error: {e}")
            return self._text_search(query, top_k, filters)
    
    def _text_search(self, query: str, top_k: int, filters: Dict = None) -> List[Tuple[float, Dict]]:
        """Text-based search fallback"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for entry in state.mapping[:200]:  # Limit for performance
            text = entry.get('text', '').lower()
            
            # Simple keyword matching
            score = 0
            for word in query_words:
                if len(word) > 2 and word in text:
                    score += 1
            
            # Bonus for exact matches of important terms
            important_terms = ['tour', 'hành trình', 'chuyến đi', 'du lịch', 'travel']
            for term in important_terms:
                if term in query_lower and term in text:
                    score += 2
            
            # Apply filters
            if filters and not self._passes_filters(entry, filters):
                score = 0
            
            if score > 0:
                results.append((float(score), entry))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _passes_filters(self, entry: Dict, filters: Dict) -> bool:
        """Check if entry passes filters"""
        for key, value in filters.items():
            if key == "tour_index" and entry.get("tour_index") != value:
                return False
            if key == "field" and entry.get("field") != value:
                return False
            if key == "location" and value.lower() not in entry.get("text", "").lower():
                return False
        return True
    
    def search_by_field(self, field_name: str, limit: int = 10, tour_index: int = None) -> List[Dict]:
        """Search for entries by field name"""
        results = []
        count = 0
        
        for entry in state.mapping:
            if entry.get("field") == field_name:
                if tour_index is not None and entry.get("tour_index") != tour_index:
                    continue
                results.append(entry)
                count += 1
                if count >= limit:
                    break
        
        return results

class NumpyIndex:
    """Simple numpy-based index for cosine similarity"""
    
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype('float32')
        self.dim = vectors.shape[1]
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.normalized_vectors = self.vectors / (norms + 1e-12)
    
    def search(self, query_vec: np.ndarray, k: int):
        """Search for k nearest neighbors"""
        # Normalize query
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        
        # Cosine similarity
        similarities = np.dot(self.normalized_vectors, query_norm.T).flatten()
        
        # Get top k
        top_indices = np.argsort(-similarities)[:k]
        top_scores = similarities[top_indices]
        
        return top_scores.reshape(1, -1), top_indices.reshape(1, -1)

# Initialize search engine
search_engine = SearchEngine()

# ==================== RESPONSE GENERATOR ====================
class ResponseGenerator:
    """Advanced response generator with formatting and personalization"""
    
    def __init__(self):
        self.llm_client = None
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY and Config.ENABLE_LLM_FALLBACK:
            try:
                self.llm_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"❌ Failed to initialize LLM client: {e}")
    
    def generate_response(self, 
                         user_message: str, 
                         search_results: List[Tuple[float, Dict]], 
                         context: ConversationContext,
                         intent_data: Dict) -> str:
        """Generate response based on search results and context"""
        
        # Special handling for specific intents
        if intent_data.get("intent") == Intent.GREETING:
            return self._generate_greeting_response(context)
        
        if intent_data.get("intent") == Intent.FAREWELL:
            return self._generate_farewell_response(context)
        
        if intent_data.get("intent") == Intent.THANK_YOU:
            return self._generate_thank_you_response(context)
        
        if intent_data.get("intent") == Intent.PROVIDE_PHONE:
            return self._generate_phone_response(intent_data.get("metadata", {}))
        
        # Check if we have search results
        if not search_results:
            log_event(
    "no_search_result",
    user_message=user_message[:200],
    stage=context.stage,
)
            return self._generate_no_results_response(user_message, context)
        
        # Group results by tour
        tours_with_results = self._group_results_by_tour(search_results)
        
        # Generate response based on context stage
        if context.stage == "explore":
            return self._generate_explore_response(tours_with_results, context, intent_data)
        elif context.stage == "suggest":
            return self._generate_suggest_response(tours_with_results, context, intent_data)
        elif context.stage == "compare":
            return self._generate_compare_response(tours_with_results, context, intent_data)
        elif context.stage == "select":
            return self._generate_select_response(tours_with_results, context, intent_data)
        elif context.stage in ["book", "lead", "callback"]:
            return self._generate_booking_response(context, intent_data)
        else:
            return self._generate_default_response(tours_with_results, context, intent_data)
    
    def _generate_greeting_response(self, context: ConversationContext) -> str:
        """Generate greeting response"""
        greetings = [
            "Xin chào! Tôi là trợ lý AI của Ruby Wings. Rất vui được hỗ trợ bạn! 😊",
            "Chào bạn! Tôi có thể giúp gì cho bạn về các tour trải nghiệm của Ruby Wings? 🌿",
            "Xin chào! Hôm nay bạn cần tư vấn về tour nào? Tôi sẵn sàng hỗ trợ! ✨"
        ]
        
        # Check time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greetings.append("Chào buổi sáng! Một ngày mới tràn đầy năng lượng. Bạn muốn khám phá tour nào? 🌞")
        elif 12 <= hour < 18:
            greetings.append("Chào buổi chiều! Cần tư vấn tour cho chuyến đi sắp tới? 🌤️")
        else:
            greetings.append("Chào buổi tối! Đang lên kế hoạch cho chuyến đi tiếp theo? 🌙")
        
        return random.choice(greetings)
    
    def _generate_farewell_response(self, context: ConversationContext) -> str:
        """Generate farewell response"""
        farewells = [
            "Cảm ơn bạn đã trò chuyện! Hy vọng sớm được đồng hành cùng bạn. Chúc bạn một ngày tuyệt vời! ✨",
            "Rất vui được hỗ trợ bạn! Đừng ngần ngại quay lại nếu cần thêm thông tin. Tạm biệt! 👋",
            "Cảm ơn bạn! Chúc bạn có những trải nghiệm du lịch tuyệt vời. Hẹn gặp lại! 🌟"
        ]
        
        return random.choice(farewells)
    
    def _generate_thank_you_response(self, context: ConversationContext) -> str:
        """Generate thank you response"""
        responses = [
            "Rất vui được hỗ trợ bạn! Cần thêm thông tin gì nữa không? 😊",
            "Cảm ơn bạn! Tôi luôn sẵn sàng giúp đỡ. Bạn muốn biết thêm về tour nào? 🌿",
            "Không có gì! Đó là niềm vui của tôi. Còn điều gì tôi có thể giúp? ✨"
        ]
        
        return random.choice(responses)
    
    def _generate_phone_response(self, metadata: Dict) -> str:
        """Generate response for phone number"""
        phone = metadata.get("detected_phone")
        
        if phone:
            # Send lead to Meta CAPI
            if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
                try:
                    send_meta_lead(
                        request,
                        event_name="Lead",
                        phone=phone,
                        content_name="Chatbot Phone Capture"
                    )
                    logger.info(f"📞 Lead captured: {phone}")
                except Exception as e:
                    log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                    logger.error(f"Meta CAPI lead error: {e}")
            
            responses = [
                f"Cảm ơn bạn đã cung cấp số điện thoại **{phone}**. Đội ngũ Ruby Wings sẽ liên hệ với bạn sớm nhất! 📞",
                f"Đã ghi nhận số **{phone}**. Nhân viên tư vấn sẽ gọi lại cho bạn trong 5-10 phút! ⏱️",
                f"Tuyệt vời! Số **{phone}** đã được lưu lại. Bạn sẽ nhận được cuộc gọi tư vấn sớm! 📱"
            ]
            
            # Update lead count
            state.increment_stat("leads_captured")
            
            return random.choice(responses)
        else:
            return "Bạn vui lòng cung cấp số điện thoại của bạn nhé! 📞"
    
    def _generate_no_results_response(self, user_message: str, context: ConversationContext) -> str:
        """Generate response when no results found"""
        # Try to get some popular tours
        all_indices = state.get_all_tour_indices()
        popular_tours = all_indices[:min(3, len(all_indices))]
        tours = state.get_tours_by_indices(popular_tours)
        
        if tours:
            response = "Tôi chưa tìm thấy thông tin cụ thể cho câu hỏi của bạn. "
            response += "Dưới đây là một số tour phổ biến của Ruby Wings:\n\n"
            
            for i, tour in enumerate(tours, 1):
                if tour:
                    response += f"**{i}. {tour.name}**\n"
                    if tour.duration:
                        response += f"   ⏱️ {tour.duration}\n"
                    if tour.location:
                        response += f"   📍 {tour.location}\n"
                    response += "\n"
            
            response += "Bạn quan tâm tour nào? Tôi sẽ cung cấp thêm chi tiết! 😊"
        else:
            response = "Xin lỗi, tôi chưa thể tìm thấy thông tin phù hợp. "
            response += "Vui lòng liên hệ hotline **0332510486** để được tư vấn trực tiếp! 📞"
        
        return response
    
    def _generate_explore_response(self, tours_with_results: Dict, context: ConversationContext, intent_data: Dict) -> str:
        """Generate response for explore stage"""
        if not tours_with_results:
            return self._generate_no_results_response("", context)
        
        response = "Tôi tìm thấy một số tour phù hợp:\n\n"
        
        for idx, (tour_idx, entries) in enumerate(tours_with_results.items(), 1):
            if idx > Config.MAX_TOURS_PER_RESPONSE:
                break
            
            tour = state.get_tour(tour_idx)
            if not tour:
                continue
            
            # Add label based on ranking
            label = ""
            if Config.ENABLE_TOUR_LABELS:
                if idx == 1:
                    label = "🏆 **Phù hợp nhất:** "
                elif idx == 2:
                    label = "⭐ **Phổ biến:** "
                else:
                    label = "💰 **Giá tốt:** "
            
            response += f"{label}**{tour.name}**\n"
            
            # Add key information
            if tour.duration:
                response += f"   ⏱️ {tour.duration}\n"
            if tour.location:
                response += f"   📍 {tour.location}\n"
            if tour.price and len(tour.price) < 50:
                response += f"   💰 {tour.price}\n"
            
            # Add a brief summary if available
            if tour.summary and len(tour.summary) < 100:
                response += f"   📝 {tour.summary[:80]}...\n"
            
            response += "\n"
        
        response += "Bạn quan tâm tour nào nhất? Hoặc muốn so sánh các tour? 😊"
        
        return response
    
    def _generate_suggest_response(self, tours_with_results: Dict, context: ConversationContext, intent_data: Dict) -> str:
        """Generate response for suggest stage"""
        response = "Dựa trên yêu cầu của bạn, tôi đề xuất:\n\n"
        
        for idx, (tour_idx, entries) in enumerate(tours_with_results.items(), 1):
            if idx > Config.MAX_TOURS_PER_RESPONSE:
                break
            
            tour = state.get_tour(tour_idx)
            if not tour:
                continue
            
            response += f"**{idx}. {tour.name}**\n"
            
            # Highlight why this tour is suggested
            if intent_data.get("metadata", {}).get("detected_location"):
                location = intent_data["metadata"]["detected_location"]
                response += f"   ✅ Phù hợp với vị trí: {location.title()}\n"
            
            if intent_data.get("intent") == Intent.PRICE_QUESTION and tour.price:
                response += f"   💰 Giá: {tour.price}\n"
            
            if intent_data.get("intent") == Intent.DURATION_QUESTION and tour.duration:
                response += f"   ⏱️ Thời lượng: {tour.duration}\n"
            
            response += "\n"
        
        response += "Bạn muốn xem chi tiết tour nào? Hay muốn so sánh các tour? 🤔"
        
        return response
    
    def _group_results_by_tour(self, search_results: List[Tuple[float, Dict]]) -> Dict[int, List[Dict]]:
        """Group search results by tour index"""
        tours = defaultdict(list)
        
        for score, entry in search_results:
            tour_idx = entry.get("tour_index")
            if tour_idx is not None:
                tours[tour_idx].append({"score": score, "entry": entry})
        
        # Sort tours by total score
        sorted_tours = {}
        for tour_idx, entries in tours.items():
            total_score = sum(entry["score"] for entry in entries)
            sorted_tours[tour_idx] = {
                "total_score": total_score,
                "entries": entries
            }
        
        # Sort by total score descending
        sorted_tours = dict(sorted(
            sorted_tours.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        ))
        
        return sorted_tours
    
    def _generate_default_response(self, tours_with_results: Dict, context: ConversationContext, intent_data: Dict) -> str:
        """Generate default response"""
        response = "Dưới đây là thông tin tôi tìm được:\n\n"
        
        # Get the top result
        if tours_with_results:
            first_tour_idx = next(iter(tours_with_results))
            tour = state.get_tour(first_tour_idx)
            
            if tour:
                response += f"**{tour.name}**\n"
                
                # Add relevant fields based on intent
                intent = intent_data.get("intent")
                
                if intent == Intent.PRICE_QUESTION and tour.price:
                    response += f"💰 **Giá:** {tour.price}\n\n"
                elif intent == Intent.LOCATION_QUESTION and tour.location:
                    response += f"📍 **Địa điểm:** {tour.location}\n\n"
                elif intent == Intent.DURATION_QUESTION and tour.duration:
                    response += f"⏱️ **Thời lượng:** {tour.duration}\n\n"
                else:
                    # General info
                    if tour.summary:
                        response += f"📝 {tour.summary[:150]}...\n\n"
        
        response += "Bạn muốn biết thêm chi tiết gì về tour này? 😊"
        
        return response
    
    # Other response generation methods...
    def _generate_compare_response(self, tours_with_results: Dict, context: ConversationContext, intent_data: Dict) -> str:
        """Generate comparison response"""
        return "Tính năng so sánh tour đang được phát triển..."
    
    def _generate_select_response(self, tours_with_results: Dict, context: ConversationContext, intent_data: Dict) -> str:
        """Generate response for selected tour"""
        return "Bạn đã chọn một tour cụ thể..."
    
    def _generate_booking_response(self, context: ConversationContext, intent_data: Dict) -> str:
        """Generate booking response"""
        return "Hướng dẫn đặt tour..."

# Initialize response generator
response_generator = ResponseGenerator()

# ==================== CHAT PROCESSOR ====================
class ChatProcessor:
    """Main chat processing pipeline"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def process_message(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process chat message with full pipeline"""
        start_time = time.time()
        
        try:
            # Get or create session context
            context = state.get_session_context(session_id)
            user_profile = state.get_user_profile(session_id)
            
            # Update user activity
            user_profile.update_activity()
            
            # Generate cache key
            cache_key = None
            if Config.ENABLE_CACHING:
                cache_key = f"chat:{session_id}:{hashlib.md5(user_message.encode()).hexdigest()[:16]}"
                cached = state.get_cache(cached_key)
                if cached:
                    logger.info(f"💾 Cache hit for session {session_id}")
                    cached['processing_time_ms'] = int((time.time() - start_time) * 1000)
                    cached['from_cache'] = True
                    return cached
            
            # Detect intent
            intent, confidence, metadata = intent_detector.detect_intent(user_message)
            
            # Update context
            context.current_intent = intent.name
            context.last_updated = datetime.now()
            context.add_message("user", user_message, {
                "intent": intent.name,
                "confidence": confidence
            })
            
            # Update state machine
            if Config.ENABLE_STATE_MACHINE and Config.STATE_MACHINE_ENABLED:
                new_stage = state_machine.get_next_state(context.stage, intent)
                context.update_stage(new_stage)
            
            # Apply location filter if detected
            location = metadata.get("detected_location")
            if location and Config.ENABLE_LOCATION_FILTER:
                context.location_filter = location
            
            # Search for relevant information
            search_filters = {}
            if context.location_filter:
                search_filters["location"] = context.location_filter
            
            search_results = search_engine.search(user_message, filters=search_filters)
            
            # Generate response
            response_text = response_generator.generate_response(
                user_message, 
                search_results, 
                context,
                {"intent": intent, "metadata": metadata}
            )
            
            # Update context with bot response
            context.add_message("assistant", response_text)
            
            # Extract mentioned tours
            mentioned_tours = []
            for score, entry in search_results:
                tour_idx = entry.get("tour_index")
                if tour_idx is not None and tour_idx not in mentioned_tours:
                    mentioned_tours.append(tour_idx)
            
            context.mentioned_tours = mentioned_tours
            
            # Update user profile with viewed tours
            user_profile.viewed_tours.extend(mentioned_tours)
            user_profile.viewed_tours = list(set(user_profile.viewed_tours))[-20:]  # Keep last 20
            
            # Prepare response
            result = {
                "reply": response_text,
                "session_id": session_id,
                "session_state": context.to_dict(),
                "user_profile": {
                    "phone": user_profile.phone,
                    "viewed_tour_count": len(user_profile.viewed_tours),
                    "inquiry_count": len(user_profile.inquiry_history)
                },
                "intent": {
                    "name": intent.name,
                    "confidence": confidence,
                    "metadata": metadata
                },
                "search_info": {
                    "results_count": len(search_results),
                    "mentioned_tours": mentioned_tours
                },
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "from_cache": False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            if Config.ENABLE_CACHING and cache_key:
                state.cache_response(cache_key, result)
            
            # Update statistics
            state.increment_stat("requests_processed")
            
            logger.info(f"⏱️ Request processed in {result['processing_time_ms']}ms | "
                       f"Intent: {intent.name} | Stage: {context.stage} | "
                       f"Results: {len(search_results)}")
            
            return result
            
        except Exception as e:
            log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

            logger.error(f"❌ Chat processing error: {e}")
            traceback.print_exc()
            
            state.increment_stat("errors")
            
            return {
                "reply": "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. "
                        "Vui lòng thử lại sau hoặc liên hệ hotline **0332510486**.",
                "session_id": session_id,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown the processor"""
        self.executor.shutdown(wait=True)

# Initialize chat processor
chat_processor = ChatProcessor()

# ==================== API ENDPOINTS ====================
@app.route("/")
def home():
    """Home endpoint with system info"""
    return jsonify({
        "status": "ok",
        "version": "5.2 Professional",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "ram_profile": Config.RAM_PROFILE,
            "environment": "production" if IS_PRODUCTION else "development",
            "platform": platform.system(),
            "python_version": sys.version
        },
        "features": {
            "state_machine": Config.ENABLE_STATE_MACHINE,
            "location_filter": Config.ENABLE_LOCATION_FILTER,
            "intent_detection": Config.ENABLE_INTENT_DETECTION,
            "phone_detection": Config.ENABLE_PHONE_DETECTION,
            "lead_capture": Config.ENABLE_LEAD_CAPTURE,
            "llm_fallback": Config.ENABLE_LLM_FALLBACK,
            "caching": Config.ENABLE_CACHING,
            "faiss_enabled": Config.FAISS_ENABLED and FAISS_AVAILABLE,
            "meta_capi": Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE
        },
        "statistics": state.get_stats()
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "chatbot": "running",
            "knowledge_base": len(state.tours_db) > 0,
            "search_index": search_engine.index_loaded,
            "openai": OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
            "meta_capi": META_CAPI_AVAILABLE
        },
        "memory": {
            "sessions": len(state.session_contexts),
            "tours": len(state.tours_db),
            "cache_size": len(state.response_cache),
            "embedding_cache": len(state.embedding_cache)
        },
        "uptime": str(datetime.now() - state.stats["start_time"])
    }
    
    # Check if all critical components are healthy
    all_healthy = all(health_status["components"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    return jsonify(health_status)
logger.info("=== CHAT HIT ===")
logger.info(f"Headers: {dict(request.headers)}")
logger.info(f"JSON: {request.get_json(silent=True)}")

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_endpoint():
    """Main chat endpoint"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        
        if not user_message:
            return jsonify({
                "reply": "Xin chào! Tôi có thể giúp gì cho bạn về các tour Ruby Wings?",
                "session_id": "new",
                "error": "Empty message"
            })
        
        # Get or generate session ID
        session_id = data.get("session_id", "")
        if not session_id:
            # Generate from IP and timestamp
            ip = request.remote_addr or "0.0.0.0"
            timestamp = int(time.time() / 60)  # Change every minute
            session_id = hashlib.md5(f"{ip}_{timestamp}".encode()).hexdigest()[:12]
        
        # Process message
        result = chat_processor.process_message(user_message, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({
            "reply": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
    
logger.info("=== SAVE LEAD HIT ===")
logger.info(f"Headers: {dict(request.headers)}")
logger.info(f"JSON: {request.get_json(silent=True)}")
@app.route("/api/save-lead", methods=["POST", "OPTIONS"])
def save_lead():
    """Save lead data with Meta CAPI tracking"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        data = request.get_json() or {}
        
        # Extract lead data
        phone = data.get("phone", "").strip()
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        tour_interest = data.get("tour_interest", "").strip()
        note = data.get("note", "").strip()
        
        # Validate phone
        if not phone:
            return jsonify({"error": "Phone number is required"}), 400
        
        if not validate_phone_number(phone):
            return jsonify({"error": "Invalid phone number format"}), 400
        
        # Create lead data
        lead_data = LeadData(
            phone=phone,
            contact_name=name if name else None,
            email=email if email else None,
            service_interest=tour_interest if tour_interest else None,
            note=note if note else None,
            source_channel="Chatbot Lead Form",
            action_type="Form Submission"
        )
        
        # Send to Meta CAPI
        if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
            try:
                send_meta_lead(
                    request,
                    event_name="Lead",
                    phone=phone,
                    contact_name=name,
                    content_name=f"Tour Interest: {tour_interest}" if tour_interest else "General Inquiry",
                    value=0.0,
                    currency="VND"
                )
                logger.info(f"✅ Lead sent to Meta CAPI: {phone}")
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"Meta CAPI lead error: {e}")
        
        # Save to Google Sheets if enabled
        if Config.ENABLE_GOOGLE_SHEETS:
            try:
                gspread, Credentials, gs_available = LazyImporter.import_google_sheets()
                if gs_available:
                    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
                    if creds_json:
                        creds = Credentials.from_service_account_info(
                            json.loads(creds_json),
                            scopes=["https://www.googleapis.com/auth/spreadsheets"]
                        )
                        gc = gspread.authorize(creds)
                        
                        sheet_id = os.environ.get("GOOGLE_SHEET_ID")
                        sheet_name = os.environ.get("GOOGLE_SHEET_NAME", "Leads")
                        
                        if sheet_id:
                            sh = gc.open_by_key(sheet_id)
                            ws = sh.worksheet(sheet_name)
                            ws.append_row(lead_data.to_row())
                            logger.info(f"✅ Lead saved to Google Sheets: {phone}")
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"Google Sheets error: {e}")
        
        # Update statistics
        state.increment_stat("leads_captured")
        
        return jsonify({
            "success": True,
            "message": "Lead đã được lưu thành công! Đội ngũ Ruby Wings sẽ liên hệ với bạn sớm nhất.",
            "data": {
                "phone": phone[:3] + "****" + phone[-3:],
                "name": name,
                "tour_interest": tour_interest,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        logger.error(f"❌ Save lead error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    """Rebuild search index (admin only)"""
    # Check authorization
    secret = request.headers.get("X-Admin-Key", "")
    if secret != os.environ.get("ADMIN_SECRET", ""):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        # Reload knowledge base
        success = load_knowledge_base()
        
        if not success:
            return jsonify({"error": "Failed to load knowledge base"}), 500
        
        # Rebuild index in background
        def rebuild_task():
            try:
                # This would rebuild the FAISS index
                # For now, we just reload the mapping
                load_search_mapping()
                logger.info("✅ Index reloaded")
            except Exception as e:
                log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

                logger.error(f"❌ Index rebuild error: {e}")
        
        threading.Thread(target=rebuild_task, daemon=True).start()
        
        return jsonify({
            "ok": True,
            "message": "Index rebuild started",
            "tours": len(state.tours_db),
            "mappings": len(state.mapping)
        })
        
    except Exception as e:
        log_event(
    "system_error",
    level="error",
    error=str(e),
    traceback=traceback.format_exc()[:500],
)

        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics"""
    return jsonify(state.get_stats())

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List active sessions (admin only)"""
    secret = request.headers.get("X-Admin-Key", "")
    if secret != os.environ.get("ADMIN_SECRET", ""):
        return jsonify({"error": "Unauthorized"}), 403
    
    sessions = []
    with state._lock:
        for session_id, ctx in state.session_contexts.items():
            sessions.append({
                "session_id": session_id,
                "stage": ctx.stage,
                "last_updated": ctx.last_updated.isoformat(),
                "mentioned_tours": ctx.mentioned_tours,
                "conversation_length": len(ctx.conversation_history)
            })
    
    return jsonify({
        "count": len(sessions),
        "sessions": sessions
    })

# ==================== INITIALIZATION ====================
def initialize_application():
    """Initialize the application"""
    # Log configuration
    Config.log_configuration()
    
    # Load knowledge base
    logger.info("🚀 Initializing Ruby Wings Chatbot...")
    
    if not load_knowledge_base():
        logger.error("❌ Failed to load knowledge base")
        # Don't exit, allow the app to start but with limited functionality
    
    # Check Meta CAPI
    if Config.ENABLE_META_CAPI:
        if META_CAPI_AVAILABLE:
            logger.info("✅ Meta CAPI integration ready")
        else:
            logger.warning("⚠️ Meta CAPI not available")
    
    # Check OpenAI
    if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
        logger.info("✅ OpenAI integration ready")
    else:
        logger.warning("⚠️ OpenAI not available, using fallback mode")
    
    # Check search index
    if search_engine.index_loaded:
        logger.info("✅ Search index ready")
    else:
        logger.warning("⚠️ Using text-based search (no vector index)")
    
    logger.info("=" * 60)
    logger.info("✅ Ruby Wings Chatbot initialized successfully!")
    logger.info(f"🌐 Server will start on {Config.HOST}:{Config.PORT}")
    logger.info("=" * 60)

# ==================== APPLICATION START ====================
if __name__ == "__main__":
    # Initialize application
    initialize_application()
    
    # Run Flask development server
    # In production, Render will use Gunicorn via Dockerfile
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False
    )
else:
    # For Gunicorn
    initialize_application()