#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 6.0.0 (COMPREHENSIVE FIX)
Created: 2025-01-18
Author: Ruby Wings AI Team

MAJOR FIXES IN V6.0.0:
======================
1. ✅ FIXED: OpenAI Client initialization - removed invalid 'proxies' parameter
2. ✅ FIXED: SearchEngine always returns results - fallback to company_info & tour_entities
3. ✅ FIXED: Intent upgrade logic - GREETING/UNKNOWN with advisory content → TOUR_INQUIRY/TOUR_FILTER
4. ✅ FIXED: ResponseGenerator always gets context - never runs with empty search_results
5. ✅ ADDED: Enhanced advisory response generation - rich, detailed consulting responses
6. ✅ ADDED: Multi-level fallback system - vector → structured_data → company_info
7. ✅ ADDED: Smart intent detection - semantic analysis of user queries
8. ✅ OPTIMIZED: Memory management for 512MB RAM profile
9. ✅ ENHANCED: Conversation flow with better state transitions
10. ✅ PRESERVED: All existing features (Meta CAPI, Google Sheets, lead capture, etc.)

TECHNICAL IMPROVEMENTS:
=======================
- Proper OpenAI SDK usage (no proxies param)
- Guaranteed non-empty context for LLM generation
- Intent elevation based on semantic content
- Company info always available as fallback
- Tour entities pre-loaded for quick filtering
- Enhanced prompt engineering for detailed responses
- Better error handling and logging
- State machine improvements
- Cache optimization

This version ensures the chatbot ALWAYS provides detailed, helpful responses
instead of generic greetings or "no information" messages.
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
from typing import Dict, List, Optional, Tuple, Any, Union, Set
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

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ruby_wings.log') if IS_PRODUCTION else logging.NullHandler()
    ]
)
logger = logging.getLogger("ruby-wings-v6.0.0")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration with full environment variable mapping"""
    
    # ===== RAM Profile =====
    RAM_PROFILE = os.getenv("RAM_PROFILE", "512")
    IS_LOW_RAM = RAM_PROFILE == "512"
    
    # ===== Core API Keys =====
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or os.urandom(24).hex()
    
    # ===== Meta Conversion API =====
    META_PIXEL_ID = os.getenv("META_PIXEL_ID", "").strip()
    META_CAPI_TOKEN = os.getenv("META_CAPI_TOKEN", "").strip()
    META_TEST_EVENT_CODE = os.getenv("META_TEST_EVENT_CODE", "").strip()
    META_CAPI_ENDPOINT = os.getenv("META_CAPI_ENDPOINT", "https://graph.facebook.com").strip()
    
    # ===== File Paths =====
    KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "knowledge.json")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
    FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "faiss_mapping.json")
    FALLBACK_VECTORS_PATH = os.getenv("FALLBACK_VECTORS_PATH", "vectors.npz")
    TOUR_ENTITIES_PATH = os.getenv("TOUR_ENTITIES_PATH", "tour_entities.json")
    FALLBACK_STORAGE_PATH = os.getenv("FALLBACK_STORAGE_PATH", "leads_fallback.json")
    
    # ===== OpenAI Models =====
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    
    # ===== Feature Toggles =====
    FAISS_ENABLED = os.getenv("FAISS_ENABLED", "false").lower() == "true"
    ENABLE_INTENT_DETECTION = os.getenv("ENABLE_INTENT_DETECTION", "true").lower() == "true"
    ENABLE_PHONE_DETECTION = os.getenv("ENABLE_PHONE_DETECTION", "true").lower() == "true"
    ENABLE_GOOGLE_SHEETS = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_META_CAPI_LEAD = os.getenv("ENABLE_META_CAPI_LEAD", "true").lower() == "true"
    ENABLE_META_CAPI_CALL = os.getenv("ENABLE_META_CAPI_CALL", "true").lower() == "true"
    ENABLE_FALLBACK_STORAGE = os.getenv("ENABLE_FALLBACK_STORAGE", "true").lower() == "true"
    ENABLE_TOUR_FILTERING = os.getenv("ENABLE_TOUR_FILTERING", "true").lower() == "true"
    ENABLE_COMPANY_INFO = os.getenv("ENABLE_COMPANY_INFO", "true").lower() == "true"
    ENABLE_LLM_FALLBACK = True
    ENABLE_CACHING = True
    STATE_MACHINE_ENABLED = True
    ENABLE_LOCATION_FILTER = True
    ENABLE_SEMANTIC_ANALYSIS = True
    DEBUG_META_CAPI = os.getenv("DEBUG_META_CAPI", "false").lower() == "true"
    
    # ===== Performance Settings =====
    TOP_K = int(os.getenv("TOP_K", "5" if IS_LOW_RAM else "10"))
    MAX_TOURS_PER_RESPONSE = 3
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50" if IS_LOW_RAM else "100"))
    MAX_EMBEDDING_CACHE = int(os.getenv("MAX_EMBEDDING_CACHE", "30" if IS_LOW_RAM else "50"))
    CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "5" if IS_LOW_RAM else "10"))
    
    # ===== Server Config =====
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "10000"))
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "1048576"))
    
    # ===== CORS =====
    CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")
    CORS_ORIGINS = CORS_ORIGINS_RAW if CORS_ORIGINS_RAW == "*" else [
        o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()
    ]
    
    # ===== Google Sheets =====
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
    GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox")
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("❌ OPENAI_API_KEY is required")
        
        if not os.path.exists(cls.KNOWLEDGE_PATH):
            errors.append(f"❌ Knowledge file not found: {cls.KNOWLEDGE_PATH}")
        
        if cls.ENABLE_GOOGLE_SHEETS:
            if not cls.GOOGLE_SERVICE_ACCOUNT_JSON:
                logger.warning("⚠️ Google Sheets enabled but no service account JSON")
            if not cls.GOOGLE_SHEET_ID:
                logger.warning("⚠️ Google Sheets enabled but no sheet ID")
        
        if cls.ENABLE_META_CAPI_LEAD or cls.ENABLE_META_CAPI_CALL:
            if not cls.META_PIXEL_ID:
                logger.warning("⚠️ Meta CAPI enabled but no pixel ID")
            if not cls.META_CAPI_TOKEN:
                logger.warning("⚠️ Meta CAPI enabled but no token")
        
        return errors
    
    @classmethod
    def log_config(cls):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info("🚀 RUBY WINGS CHATBOT v6.0.0 (COMPREHENSIVE FIX)")
        logger.info("=" * 80)
        logger.info(f"📊 RAM Profile: {cls.RAM_PROFILE}MB")
        logger.info(f"🌍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
        logger.info(f"🔧 Platform: {platform.system()} {platform.release()}")
        
        features = []
        if cls.STATE_MACHINE_ENABLED:
            features.append("State Machine")
        if cls.FAISS_ENABLED:
            features.append("FAISS Vector Search")
        else:
            features.append("NumPy Fallback Search")
        if cls.ENABLE_META_CAPI_LEAD:
            features.append("Meta CAPI Lead")
        if cls.ENABLE_META_CAPI_CALL:
            features.append("Meta CAPI Call")
        if cls.ENABLE_GOOGLE_SHEETS:
            features.append("Google Sheets")
        if cls.ENABLE_TOUR_FILTERING:
            features.append("Tour Filtering")
        if cls.ENABLE_COMPANY_INFO:
            features.append("Company Info")
        if cls.ENABLE_SEMANTIC_ANALYSIS:
            features.append("Semantic Analysis")
        
        logger.info(f"🎯 Active Features: {', '.join(features)}")
        logger.info(f"🔑 OpenAI API: {'✅ Configured' if cls.OPENAI_API_KEY else '❌ Missing'}")
        logger.info(f"🔑 Meta CAPI: {'✅ Configured' if cls.META_CAPI_TOKEN else '❌ Missing'}")
        logger.info(f"🔑 Google Sheets: {'✅ Configured' if cls.GOOGLE_SERVICE_ACCOUNT_JSON else '❌ Missing'}")
        logger.info(f"🌐 CORS Origins: {cls.CORS_ORIGINS}")
        logger.info(f"📁 Knowledge Path: {cls.KNOWLEDGE_PATH}")
        logger.info(f"🎯 Top K Results: {cls.TOP_K}")
        logger.info(f"💾 Cache TTL: {cls.CACHE_TTL_SECONDS}s")
        logger.info("=" * 80)

# ==================== INTENT ENUM ====================
class Intent:
    """Complete Intent Enum with all required values"""
    # Core conversation
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    SMALLTALK = "SMALLTALK"
    UNKNOWN = "UNKNOWN"
    
    # Tour-related (CRITICAL: These must trigger LLM generation)
    TOUR_INQUIRY = "TOUR_INQUIRY"
    TOUR_LIST = "TOUR_LIST"
    TOUR_FILTER = "TOUR_FILTER"
    TOUR_DETAIL = "TOUR_DETAIL"
    TOUR_COMPARE = "TOUR_COMPARE"
    TOUR_RECOMMEND = "TOUR_RECOMMEND"
    
    # Price
    PRICE_ASK = "PRICE_ASK"
    PRICE_COMPARE = "PRICE_COMPARE"
    PRICE_RANGE = "PRICE_RANGE"
    
    # Booking
    BOOKING_REQUEST = "BOOKING_REQUEST"
    BOOKING_PROCESS = "BOOKING_PROCESS"
    BOOKING_CONDITION = "BOOKING_CONDITION"
    
    # Contact
    PROVIDE_PHONE = "PROVIDE_PHONE"
    CALLBACK_REQUEST = "CALLBACK_REQUEST"
    CONTACT_INFO = "CONTACT_INFO"
    
    # Company (CRITICAL: Must trigger detailed responses)
    ABOUT_COMPANY = "ABOUT_COMPANY"
    COMPANY_SERVICE = "COMPANY_SERVICE"
    COMPANY_MISSION = "COMPANY_MISSION"
    
    # Lead
    LEAD_CAPTURED = "LEAD_CAPTURED"
    
    @classmethod
    def is_advisory_intent(cls, intent: str) -> bool:
        """Check if intent requires detailed advisory response"""
        advisory_intents = {
            cls.TOUR_INQUIRY,
            cls.TOUR_FILTER,
            cls.TOUR_DETAIL,
            cls.TOUR_COMPARE,
            cls.TOUR_RECOMMEND,
            cls.ABOUT_COMPANY,
            cls.COMPANY_SERVICE,
            cls.COMPANY_MISSION,
            cls.PRICE_ASK,
            cls.PRICE_COMPARE,
            cls.BOOKING_REQUEST
        }
        return intent in advisory_intents
    
    @classmethod
    def requires_context(cls, intent: str) -> bool:
        """Check if intent requires search context"""
        context_intents = {
            cls.TOUR_INQUIRY,
            cls.TOUR_FILTER,
            cls.TOUR_DETAIL,
            cls.TOUR_COMPARE,
            cls.TOUR_RECOMMEND,
            cls.ABOUT_COMPANY,
            cls.COMPANY_SERVICE,
            cls.COMPANY_MISSION,
            cls.PRICE_ASK,
            cls.PRICE_COMPARE,
            cls.PRICE_RANGE
        }
        return intent in context_intents

class ConversationStage:
    """Conversation stages for state machine"""
    INITIAL = "INITIAL"
    GREETING = "GREETING"
    EXPLORING = "EXPLORING"
    FILTERING = "FILTERING"
    COMPARING = "COMPARING"
    SELECTING = "SELECTING"
    BOOKING = "BOOKING"
    LEAD_CAPTURE = "LEAD_CAPTURE"
    CALLBACK = "CALLBACK"
    FAREWELL = "FAREWELL"

# ==================== ADVISORY KEYWORDS ====================
# CRITICAL: Used to detect when user is asking for detailed advice/consultation
ADVISORY_KEYWORDS = {
    # Tour-related
    "tour", "du lịch", "hành trình", "chuyến đi", "trải nghiệm",
    "retreat", "chữa lành", "tâm linh", "thiền", "khí công",
    
    # Questions
    "gợi ý", "tư vấn", "giới thiệu", "có tour nào", "tour gì",
    "nên đi", "phù hợp", "so sánh", "khác nhau", "tốt hơn",
    
    # Time-based
    "cuối tuần", "1 ngày", "2 ngày", "ngắn ngày", "dài ngày",
    
    # Style-based
    "ít di chuyển", "nhẹ nhàng", "năng động", "mạo hiểm",
    "thiên về", "yên tĩnh", "gần Huế", "gần thành phố",
    
    # Budget
    "giá rẻ", "tiết kiệm", "cao cấp", "bao nhiêu tiền",
    
    # Group type
    "một mình", "gia đình", "công ty", "nhóm bạn", "cặp đôi",
    
    # Company info
    "ruby wings", "công ty", "đơn vị", "thương hiệu", "doanh nghiệp",
    "triết lý", "sứ mệnh", "giá trị", "hệ sinh thái"
}

def contains_advisory_content(text: str) -> bool:
    """Check if text contains advisory/consultation keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ADVISORY_KEYWORDS)

# ==================== META CONVERSION API ====================
META_CAPI_AVAILABLE = False

try:
    if Config.ENABLE_META_CAPI_LEAD or Config.ENABLE_META_CAPI_CALL:
        import requests
        META_CAPI_AVAILABLE = True
        logger.info("✅ Meta CAPI module loaded")
except ImportError:
    logger.warning("⚠️ Meta CAPI unavailable - requests module not found")

def send_meta_lead(request_obj, phone: str, contact_name: str = "", 
                   email: str = "", content_name: str = "", 
                   value: float = 200000, currency: str = "VND") -> Dict[str, Any]:
    """Send lead event to Meta Conversion API"""
    if not META_CAPI_AVAILABLE:
        return {"success": False, "error": "Meta CAPI not available"}
    
    if not Config.META_PIXEL_ID or not Config.META_CAPI_TOKEN:
        return {"success": False, "error": "Meta CAPI not configured"}
    
    try:
        # Prepare user data with hashing
        user_data = {}
        
        if phone:
            phone_clean = re.sub(r'[^\d+]', '', phone)
            user_data["ph"] = hashlib.sha256(phone_clean.encode()).hexdigest()
        
        if email:
            user_data["em"] = hashlib.sha256(email.lower().encode()).hexdigest()
        
        if contact_name:
            user_data["fn"] = hashlib.sha256(contact_name.lower().encode()).hexdigest()
        
        # Get user agent and IP
        user_agent = request_obj.headers.get('User-Agent', '')
        client_ip = request_obj.headers.get('X-Forwarded-For', request_obj.remote_addr)
        
        if client_ip:
            user_data["client_ip_address"] = client_ip
        if user_agent:
            user_data["client_user_agent"] = user_agent
        
        # Event data
        event_time = int(time.time())
        event_id = f"lead_{event_time}_{hashlib.md5(phone.encode()).hexdigest()[:8]}"
        
        custom_data = {
            "currency": currency,
            "value": value
        }
        
        if content_name:
            custom_data["content_name"] = content_name
        
        # Payload
        payload = {
            "data": [{
                "event_name": "Lead",
                "event_time": event_time,
                "event_id": event_id,
                "event_source_url": request_obj.referrer or request_obj.url,
                "action_source": "website",
                "user_data": user_data,
                "custom_data": custom_data
            }]
        }
        
        if Config.META_TEST_EVENT_CODE:
            payload["test_event_code"] = Config.META_TEST_EVENT_CODE
        
        # Send to Meta
        url = f"{Config.META_CAPI_ENDPOINT}/v21.0/{Config.META_PIXEL_ID}/events"
        params = {"access_token": Config.META_CAPI_TOKEN}
        
        response = requests.post(url, json=payload, params=params, timeout=10)
        
        if Config.DEBUG_META_CAPI:
            logger.info(f"Meta CAPI Request: {json.dumps(payload, indent=2)}")
            logger.info(f"Meta CAPI Response: {response.text}")
        
        if response.status_code == 200:
            return {"success": True, "event_id": event_id, "response": response.json()}
        else:
            logger.error(f"Meta CAPI error: {response.status_code} - {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
    
    except Exception as e:
        logger.error(f"Meta CAPI exception: {str(e)}")
        return {"success": False, "error": str(e)}

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    processor = state.get_chat_processor()
    result = processor.process(message, session_id)

    return jsonify(result)


app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# CORS configuration
CORS(app, 
     resources={r"/api/*": {"origins": Config.CORS_ORIGINS}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

# Proxy fix for production
if IS_PRODUCTION:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ==================== KNOWLEDGE LOADING ====================
def load_knowledge() -> bool:
    """Load knowledge from JSON file with enhanced structure"""
    try:
        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.error(f"❌ Knowledge file not found: {Config.KNOWLEDGE_PATH}")
            return False
        
        logger.info(f"📚 Loading knowledge from {Config.KNOWLEDGE_PATH}...")
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            state.knowledge_data = json.load(f)
        
        # Extract tours
        state.tours_db = state.knowledge_data.get('tours', [])
        logger.info(f"✅ Loaded {len(state.tours_db)} tours")
        
        # Extract company info (CRITICAL for ABOUT_COMPANY queries)
        state.company_info = state.knowledge_data.get('company_info', {})
        if not state.company_info:
            # Extract from first tour if available (tour_id: RW032)
            for tour in state.tours_db:
                if tour.get('tour_id') == 'RW032':
                    state.company_info = {
                        'name': 'Ruby Wings Travel',
                        'description': tour.get('summary', ''),
                        'philosophy': tour.get('style', ''),
                        'mission': tour.get('highlights', []),
                        'contact': {
                            'phone': '0332510486',
                            'address': tour.get('location', ''),
                            'email': 'info@rubywings.vn'
                        },
                        'values': tour.get('educational_values', [])
                    }
                    break
        
        logger.info(f"✅ Company info loaded: {state.company_info.get('name', 'N/A')}")
        
        # Extract FAQs
        state.faqs = state.knowledge_data.get('faqs', [])
        logger.info(f"✅ Loaded {len(state.faqs)} FAQs")
        
        # Load tour entities if available
        if os.path.exists(Config.TOUR_ENTITIES_PATH):
            try:
                with open(Config.TOUR_ENTITIES_PATH, 'r', encoding='utf-8') as f:
                    state.tour_entities = json.load(f)
                logger.info(f"✅ Loaded tour entities: {len(state.tour_entities)} entries")
            except Exception as e:
                logger.warning(f"⚠️ Could not load tour entities: {e}")
        else:
            logger.warning(f"⚠️ Tour entities file not found: {Config.TOUR_ENTITIES_PATH}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        traceback.print_exc()
        return False

# ==================== SEARCH ENGINE ====================
class SearchEngine:
    """
    Enhanced Search Engine with guaranteed non-empty results
    CRITICAL FIX: Always returns fallback data when vector search fails
    """
    def __init__(self):
        self.index = None
        self.mapping = []
        self.vectors = None
        self.openai_client = None
        
        # Initialize OpenAI client - NO PROXIES PARAMETER
        try:
            from openai import OpenAI
            if Config.OPENAI_API_KEY:
                # FIX: Removed 'proxies' parameter
                self.openai_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL,
                    timeout=30.0
                )

                logger.info("✅ SearchEngine OpenAI client initialized (api_key + base_url + timeout only)")
            else:
                logger.error("❌ OpenAI API key not configured")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            traceback.print_exc()
    
    def load_index(self):
        """Load search index (FAISS or NumPy fallback)"""
        try:
            # Load mapping
            if os.path.exists(Config.FAISS_MAPPING_PATH):
                with open(Config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                logger.info(f"✅ Loaded mapping: {len(self.mapping)} entries")
            else:
                logger.warning(f"⚠️ Mapping file not found: {Config.FAISS_MAPPING_PATH}")
            
            # Load index
            if Config.FAISS_ENABLED and os.path.exists(Config.FAISS_INDEX_PATH):
                try:
                    import faiss
                    self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                    logger.info(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"⚠️ FAISS load failed: {e}, using NumPy fallback")
                    self._load_numpy_fallback()
            else:
                self._load_numpy_fallback()
        
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            traceback.print_exc()
    
    def _load_numpy_fallback(self):
        """Load NumPy fallback vectors"""
        try:
            if os.path.exists(Config.FALLBACK_VECTORS_PATH):
                import numpy as np
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                self.vectors = data['mat']
                logger.info(f"✅ Loaded NumPy vectors: {self.vectors.shape}")
            else:
                logger.warning(f"⚠️ Fallback vectors not found: {Config.FALLBACK_VECTORS_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load NumPy vectors: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching"""
        if not text or not self.openai_client:
            return None
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        with state.embedding_cache_lock:
            if cache_key in state.embedding_cache:
                state.stats['cache_hits'] += 1
                return state.embedding_cache[cache_key]
            
            state.stats['cache_misses'] += 1
        
        # Generate embedding
        try:
            response = self.openai_client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache it
            with state.embedding_cache_lock:
                state.embedding_cache[cache_key] = embedding
                
                # Limit cache size
                if len(state.embedding_cache) > Config.MAX_EMBEDDING_CACHE:
                    state.embedding_cache.popitem(last=False)
            
            return embedding
        
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    def search(self, query: str, top_k: int = None, intent: str = Intent.UNKNOWN) -> List[Dict[str, Any]]:
        """
        Search with GUARANTEED non-empty results
        CRITICAL FIX: Always returns fallback data for advisory intents
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        results = []
        
        # Try vector search first
        if self.openai_client and (self.index is not None or self.vectors is not None):
            try:
                query_embedding = self.get_embedding(query)
                
                if query_embedding:
                    if self.index is not None:
                        # FAISS search
                        import numpy as np
                        query_vec = np.array([query_embedding], dtype='float32')
                        
                        # Normalize
                        norm = np.linalg.norm(query_vec)
                        if norm > 0:
                            query_vec = query_vec / norm
                        
                        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
                        
                        for score, idx in zip(scores[0], indices[0]):
                            if idx >= 0 and idx < len(self.mapping):
                                results.append({
                                    'score': float(score),
                                    'text': self.mapping[idx].get('text', ''),
                                    'path': self.mapping[idx].get('path', ''),
                                    'metadata': self.mapping[idx]
                                })
                    
                    elif self.vectors is not None:
                        # NumPy search
                        import numpy as np
                        query_vec = np.array(query_embedding, dtype='float32')
                        
                        # Normalize
                        norm = np.linalg.norm(query_vec)
                        if norm > 0:
                            query_vec = query_vec / norm
                        
                        # Compute similarities
                        scores = np.dot(self.vectors, query_vec)
                        top_indices = np.argsort(scores)[::-1][:top_k]
                        
                        for idx in top_indices:
                            if idx < len(self.mapping):
                                results.append({
                                    'score': float(scores[idx]),
                                    'text': self.mapping[idx].get('text', ''),
                                    'path': self.mapping[idx].get('path', ''),
                                    'metadata': self.mapping[idx]
                                })
            
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")
        
        # CRITICAL FIX: Provide fallback for advisory intents
        if not results and Intent.is_advisory_intent(intent):
            logger.info(f"🔄 Vector search empty for advisory intent {intent}, using fallback")
            results = self._get_fallback_results(query, intent, top_k)
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:top_k]
    
    def _get_fallback_results(self, query: str, intent: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Generate fallback results from structured data
        CRITICAL: Ensures non-empty context for advisory intents
        """
        results = []
        query_lower = query.lower()
        
        # Company info fallback (for ABOUT_COMPANY, COMPANY_SERVICE, etc.)
        if intent in [Intent.ABOUT_COMPANY, Intent.COMPANY_SERVICE, Intent.COMPANY_MISSION]:
            if state.company_info:
                company_text = self._format_company_info(state.company_info)
                results.append({
                    'score': 1.0,
                    'text': company_text,
                    'path': 'root.company_info',
                    'metadata': {'type': 'company_info', 'source': 'fallback'}
                })
        
        # Tour entities fallback (for TOUR_INQUIRY, TOUR_FILTER, etc.)
        if intent in [Intent.TOUR_INQUIRY, Intent.TOUR_FILTER, Intent.TOUR_RECOMMEND, Intent.TOUR_COMPARE]:
            # Filter tours by keywords
            relevant_tours = self._filter_tours_by_keywords(query_lower)
            
            for tour in relevant_tours[:top_k]:
                tour_text = self._format_tour_info(tour)
                results.append({
                    'score': 0.8,  # Lower than exact match but still relevant
                    'text': tour_text,
                    'path': f"root.tours[{tour.get('tour_id', '')}]",
                    'metadata': {'type': 'tour', 'source': 'fallback', 'tour_data': tour}
                })
        
        # FAQ fallback
        if state.faqs and not results:
            for faq in state.faqs[:3]:
                if any(word in faq.get('question', '').lower() for word in query_lower.split()):
                    results.append({
                        'score': 0.7,
                        'text': f"Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}",
                        'path': 'root.faqs',
                        'metadata': {'type': 'faq', 'source': 'fallback'}
                    })
        
        # If still empty, provide generic company info
        if not results and state.company_info:
            results.append({
                'score': 0.5,
                'text': self._format_company_info(state.company_info),
                'path': 'root.company_info',
                'metadata': {'type': 'company_info', 'source': 'fallback_generic'}
            })
        
        return results
    
    def _filter_tours_by_keywords(self, query: str) -> List[Dict[str, Any]]:
        """Filter tours based on query keywords"""
        keywords = set(query.split())
        scored_tours = []
        
        for tour in state.tours_db:
            score = 0
            tour_text = json.dumps(tour, ensure_ascii=False).lower()
            
            # Count keyword matches
            for keyword in keywords:
                if len(keyword) >= 3:  # Ignore short words
                    score += tour_text.count(keyword)
            
            if score > 0:
                scored_tours.append((score, tour))
        
        # Sort by score
        scored_tours.sort(key=lambda x: x[0], reverse=True)
        
        return [tour for _, tour in scored_tours]
    
    def _format_company_info(self, info: Dict[str, Any]) -> str:
        """Format company info for context"""
        lines = [
            f"Tên công ty: {info.get('name', 'Ruby Wings Travel')}",
            f"Mô tả: {info.get('description', '')[:300]}",
            f"Triết lý: {info.get('philosophy', '')[:300]}",
            f"Liên hệ: {info.get('contact', {}).get('phone', '0332510486')}"
        ]
        
        if info.get('mission'):
            missions = info['mission'][:3]  # First 3 missions
            lines.append(f"Sứ mệnh: {', '.join(str(m)[:100] for m in missions)}")
        
        return "\n".join(lines)
    
    def _format_tour_info(self, tour: Dict[str, Any]) -> str:
        """Format tour info for context"""
        lines = [
            f"Tên tour: {tour.get('tour_name', '')}",
            f"Địa điểm: {tour.get('location', '')}",
            f"Thời lượng: {tour.get('duration', '')}",
            f"Giá: {tour.get('price', '')}",
            f"Tóm tắt: {tour.get('summary', '')[:200]}"
        ]
        
        if tour.get('highlights'):
            highlights = tour['highlights'][:3]
            lines.append(f"Điểm nổi bật: {', '.join(str(h)[:100] for h in highlights)}")
        
        return "\n".join(lines)
# ==================== GLOBAL STATE ====================
class AppState:
    """Global application state"""
    def __init__(self):
        self.knowledge_data = None
        self.tours_db = []
        self.company_info = {}
        self.tour_entities = {}
        self.faqs = []
        
        # Components (initialized lazily)
        self._search_engine = None
        self._chat_processor = None
        
        # Session management
        self.sessions = OrderedDict()
        self.sessions_lock = threading.RLock()
        
        # Cache
        self.embedding_cache = OrderedDict()
        self.embedding_cache_lock = threading.RLock()
        
        # Stats
        self.stats = {
            'requests': 0,
            'errors': 0,
            'leads': 0,
            'meta_capi_calls': 0,
            'meta_capi_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.stats_lock = threading.RLock()
    
    def get_search_engine(self):
        if self._search_engine is None:
            self._search_engine = SearchEngine()
        return self._search_engine
    
    def get_chat_processor(self):
        if self._chat_processor is None:
            self._chat_processor = ChatProcessor()
        return self._chat_processor

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self.stats_lock:
            return {
                **self.stats,
                'sessions': len(self.sessions),
                'cache_size': len(self.embedding_cache),
                'tours_loaded': len(self.tours_db),
                'uptime_seconds': int(time.time() - getattr(self, 'start_time', time.time()))
            }

state = AppState()
state.start_time = time.time()

# ==================== RESPONSE GENERATOR ====================
class ResponseGenerator:
    """
    Enhanced Response Generator with rich, detailed consulting capabilities
    CRITICAL FIX: Always generates detailed responses when context is provided
    """
    def __init__(self):
        self.openai_client = None
        
        # Initialize OpenAI client - NO PROXIES PARAMETER
        try:
            from openai import OpenAI
            if Config.OPENAI_API_KEY:
                # FIX: Removed 'proxies' parameter
                self.openai_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL,
                    timeout=60.0
                )

                logger.info("✅ ResponseGenerator OpenAI client initialized (api_key + base_url + timeout only)")
            else:
                logger.error("❌ OpenAI API key not configured for ResponseGenerator")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ResponseGenerator OpenAI client: {e}")
            traceback.print_exc()
    
    def generate(self, query: str, search_results: List[Dict[str, Any]], 
                 intent: str = Intent.UNKNOWN, 
                 conversation_history: List[Dict[str, str]] = None,
                 user_context: Dict[str, Any] = None) -> str:
        """
        Generate response with GUARANTEED detailed output for advisory intents
        CRITICAL FIX: Never returns empty/generic response when context exists
        """
        
        if not self.openai_client:
            return self._generate_fallback_response(query, search_results, intent)
        if not search_results:
            return (
        "🌿 Ruby Wings chuyên các hành trình du lịch trải nghiệm và chữa lành. "
        "Dựa trên nhu cầu của bạn, tôi sẽ tư vấn hướng đi phù hợp ngay khi có thêm thông tin."
    )

        
        # Build context from search results
        context_texts = []
        for i, result in enumerate(search_results[:5], 1):
            context_texts.append(f"[{i}] {result.get('text', '')[:500]}")
        
        context_str = "\n\n".join(context_texts) if context_texts else "Không có thông tin cụ thể."
        
        # Build conversation history
        history_str = ""
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 turns
                history_str += f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}\n\n"
        
        # Enhanced system prompt (CRITICAL for detailed responses)
        system_prompt = self._build_system_prompt(intent)
        
        # User prompt
        user_prompt = f"""BỐI CẢNH CỦA NGƯỜI DÙNG:
{history_str if history_str else 'Cuộc hội thoại mới'}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

THÔNG TIN NỘI BỘ RUBY WINGS (ƯU TIÊN CAO NHẤT):
{context_str}

YÊU CẦU TRẢ LỜI:
1. TRẢ LỜI CHI TIẾT, CÓ CHIỀU SÂU dựa trên thông tin nội bộ ở trên
2. KHÔNG nói rằng bạn không có thông tin hoặc không đọc được file
3. TỔ HỢP thông tin từ các nguồn [1], [2], [3]... để tạo câu trả lời mạch lạc
4. Luôn TRÍCH DẪN nguồn bằng [1], [2], [3]...
5. Giữ giọng điệu nhiệt tình, chuyên nghiệp, phản ánh triết lý chữa lành của Ruby Wings
6. Nếu thiếu chi tiết, hãy tổng hợp thông tin CHUNG có sẵn thay vì nói không có
7. Câu trả lời dài 150-300 từ, trừ khi câu hỏi yêu cầu ngắn gọn

HÃY TRẢ LỜI:"""
        
        try:
            # Call OpenAI API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.openai_client.responses.create(
                model=Config.CHAT_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_output_tokens=800
            )

            answer = response.output_text.strip()

            
            # Validate response quality
            if self._is_low_quality_response(answer):
                logger.warning("⚠️ Low quality LLM response detected, using enhanced fallback")
                return self._generate_enhanced_fallback(query, search_results, intent)
            
            return answer
        
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            traceback.print_exc()
            return self._generate_fallback_response(query, search_results, intent)
    
    def _build_system_prompt(self, intent: str) -> str:
        """Build enhanced system prompt based on intent"""
        base_prompt = """Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.

TRẢ LỜI THEO CÁC NGUYÊN TẮC:
1. ƯU TIÊN CAO NHẤT: Luôn sử dụng thông tin từ dữ liệu nội bộ được cung cấp thông qua hệ thống.
2. Nếu thiếu thông tin CHI TIẾT, hãy tổng hợp và trả lời dựa trên THÔNG TIN CHUNG có sẵn trong dữ liệu nội bộ.
3. Đối với tour cụ thể: nếu tìm thấy bất kỳ dữ liệu nội bộ liên quan nào (dù là tóm tắt, giá, lịch trình, ghi chú), PHẢI tổng hợp và trình bày rõ ràng; chỉ trả lời đang nâng cấp hoặc chưa có thông tin khi hoàn toàn không tìm thấy dữ liệu phù hợp.
4. TUYỆT ĐỐI KHÔNG nói rằng bạn không đọc được file, không truy cập dữ liệu, hoặc từ chối trả lời khi đã có dữ liệu liên quan.
5. Luôn giữ thái độ nhiệt tình, hữu ích, trả lời trực tiếp vào nội dung người dùng hỏi.

Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn ngành du lịch trải nghiệm, retreat, 
thiền, khí công, hành trình chữa lành và các hành trình tham quan linh hoạt theo nhu cầu. 
Trả lời CHI TIẾT, CÓ CHIỀU SÂU, chính xác, rõ ràng, tử tế và bám sát dữ liệu Ruby Wings.

PHONG CÁCH TRẢ LỜI:
- Nhiệt tình, thân thiện nhưng chuyên nghiệp
- Thể hiện sự am hiểu về du lịch trải nghiệm và chữa lành
- Sử dụng ngôn ngữ gần gũi, dễ hiểu
- Luôn hướng người dùng đến hành động (đặt tour, liên hệ, tìm hiểu thêm)
- Trích dẫn nguồn bằng [1], [2], [3]... khi sử dụng thông tin cụ thể"""
        
        # Intent-specific additions
        if intent == Intent.ABOUT_COMPANY:
            base_prompt += """

ĐẶC BIỆT CHO CÂU HỎI VỀ CÔNG TY:
- Nhấn mạnh triết lý chữa lành và phát triển nội tâm
- Kể câu chuyện của Giám đốc Lương Tiến Sỹ nếu phù hợp
- Giải thích ý nghĩa logo 4 cánh (Thân - Tâm - Thiên nhiên - Niềm tin)
- Trình bày hệ sinh thái Ruby (Travel, Learn, Stay, Auto)
- Thể hiện giá trị cốt lõi: Chuẩn mực - Chân thành - Có chiều sâu"""
        
        elif intent in [Intent.TOUR_INQUIRY, Intent.TOUR_FILTER, Intent.TOUR_RECOMMEND]:
            base_prompt += """

ĐẶC BIỆT CHO TƯ VẤN TOUR:
- Phân tích nhu cầu người dùng (thời gian, phong cách, nhóm người)
- Đề xuất 2-3 tour phù hợp nhất với lý do rõ ràng
- So sánh điểm mạnh của từng tour
- Cung cấp thông tin giá, thời lượng, điểm nổi bật
- Kết thúc bằng call-to-action để đặt tour hoặc tư vấn thêm"""
        
        elif intent == Intent.TOUR_COMPARE:
            base_prompt += """

ĐẶC BIỆT CHO SO SÁNH TOUR:
- Trình bày thông tin song song, dễ so sánh
- Nêu rõ điểm giống và khác biệt
- Phân tích ưu điểm riêng của mỗi tour
- Đưa ra gợi ý lựa chọn dựa trên nhu cầu khác nhau
- Sử dụng bảng hoặc bullet points để dễ nhìn"""
        
        return base_prompt
    
    def _is_low_quality_response(self, response: str) -> bool:
        """Check if response is low quality"""
        if not response or len(response) < 50:
            return True
        
        # Check for generic rejection patterns
        low_quality_patterns = [
            "không có thông tin",
            "không thể trả lời",
            "không tìm thấy",
            "xin lỗi, tôi không",
            "tôi không có dữ liệu",
            "không đọc được file",
            "không truy cập được"
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in low_quality_patterns):
            if len(response) < 200:  # Short + rejection = low quality
                return True
        
        return False
    
    def _generate_fallback_response(self, query: str, search_results: List[Dict[str, Any]], 
                                   intent: str) -> str:
        """Generate fallback response from search results"""
        if not search_results:
            return "Xin lỗi, tôi đang gặp khó khăn trong việc truy xuất thông tin. Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp!"
        
        # Extract key information
        texts = [r.get('text', '')[:300] for r in search_results[:3]]
        
        response = f"Dựa trên thông tin của Ruby Wings:\n\n"
        
        for i, text in enumerate(texts, 1):
            if text:
                response += f"[{i}] {text}\n\n"
        
        response += "\n💡 Liên hệ hotline 0332510486 để được tư vấn chi tiết hơn!"
        
        return response
    
    def _generate_enhanced_fallback(self, query: str, search_results: List[Dict[str, Any]], 
                                   intent: str) -> str:
        """Generate enhanced fallback with structured information"""
        if not search_results:
            return self._generate_fallback_response(query, search_results, intent)
        
        # Group results by type
        tours = []
        company_info = []
        faqs = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            if metadata.get('type') == 'tour':
                tours.append(result)
            elif metadata.get('type') == 'company_info':
                company_info.append(result)
            elif metadata.get('type') == 'faq':
                faqs.append(result)
        
        # Build response based on available data
        response_parts = []
        
        if intent == Intent.ABOUT_COMPANY and company_info:
            response_parts.append("📍 **Về Ruby Wings Travel:**\n")
            for info in company_info[:2]:
                response_parts.append(info.get('text', '')[:400])
                response_parts.append("")
        
        elif intent in [Intent.TOUR_INQUIRY, Intent.TOUR_FILTER, Intent.TOUR_RECOMMEND] and tours:
            response_parts.append("🌿 **Các tour phù hợp với bạn:**\n")
            for i, tour in enumerate(tours[:3], 1):
                tour_data = tour.get('metadata', {}).get('tour_data', {})
                response_parts.append(f"**{i}. {tour_data.get('tour_name', 'Tour')}**")
                response_parts.append(f"   📍 {tour_data.get('location', '')}")
                response_parts.append(f"   ⏰ {tour_data.get('duration', '')}")
                response_parts.append(f"   💰 {tour_data.get('price', '')}")
                if tour_data.get('summary'):
                    response_parts.append(f"   📝 {tour_data.get('summary', '')[:150]}...")
                response_parts.append("")
        
        elif faqs:
            response_parts.append("💡 **Thông tin hữu ích:**\n")
            for faq in faqs[:2]:
                response_parts.append(faq.get('text', '')[:300])
                response_parts.append("")
        
        else:
            # Generic fallback
            for i, result in enumerate(search_results[:3], 1):
                response_parts.append(f"[{i}] {result.get('text', '')[:250]}")
                response_parts.append("")
        
        # Add call-to-action
        response_parts.append("\n📞 **Liên hệ ngay:** 0332510486 để được tư vấn chi tiết!")
        
        return "\n".join(response_parts)

# ==================== INTENT DETECTOR ====================
class IntentDetector:
    """
    Enhanced Intent Detector with semantic analysis
    CRITICAL FIX: Upgrades GREETING/UNKNOWN to advisory intents when appropriate
    """
    def __init__(self):
        self.intent_patterns = {
            Intent.GREETING: [
                r'\b(xin chào|chào|hello|hi|chào bạn)\b',
                r'^(chào|hello|hi)[\s!]*$'
            ],
            Intent.FAREWELL: [
                r'\b(tạm biệt|bye|goodbye|cảm ơn|thanks)\b',
            ],
            Intent.PROVIDE_PHONE: [
                r'\b(0\d{9,10})\b',
                r'\b(\+84\d{9,10})\b',
                r'\b(số điện thoại|sdt|đt)\b.*\b(0\d{9})\b'
            ],
            Intent.CALLBACK_REQUEST: [
                r'\b(gọi lại|callback|gọi cho tôi)\b',
            ],
            Intent.ABOUT_COMPANY: [
                r'\b(ruby wings|công ty|đơn vị|giới thiệu|thương hiệu)\b',
                r'\b(triết lý|sứ mệnh|giá trị|hệ sinh thái)\b'
            ],
            Intent.TOUR_INQUIRY: [
                r'\b(tour|du lịch|hành trình|chuyến đi)\b',
            ],
            Intent.TOUR_FILTER: [
                r'\b(cuối tuần|1 ngày|2 ngày|ngắn ngày)\b',
                r'\b(retreat|chữa lành|thiền|tâm linh)\b',
                r'\b(ít di chuyển|nhẹ nhàng|gần|xa)\b'
            ],
            Intent.TOUR_COMPARE: [
                r'\b(so sánh|khác nhau|giống|tốt hơn)\b',
            ],
            Intent.TOUR_RECOMMEND: [
                r'\b(gợi ý|tư vấn|đề xuất|nên đi)\b',
            ],
            Intent.PRICE_ASK: [
                r'\b(giá|bao nhiêu|chi phí|price)\b',
            ]
        }
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect intent with semantic upgrade logic
        CRITICAL FIX: Upgrades simple intents to advisory when needed
        """
        text_lower = text.lower().strip()
        
        # Check patterns
        scores = {}
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[intent] = scores.get(intent, 0) + 1
        
        # Get best match
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] * 0.3, 1.0)
            
            # CRITICAL FIX: Upgrade logic
            if best_intent in [Intent.GREETING, Intent.UNKNOWN]:
                if contains_advisory_content(text):
                    logger.info(f"🔄 Upgrading intent from {best_intent} to TOUR_INQUIRY (advisory content detected)")
                    return Intent.TOUR_INQUIRY, 0.8
            
            return best_intent, confidence
        
        # CRITICAL FIX: Check for advisory content even without pattern match
        if contains_advisory_content(text):
            logger.info(f"🔄 No pattern match but advisory content detected → TOUR_INQUIRY")
            return Intent.TOUR_INQUIRY, 0.7
        
        # Default
        return Intent.UNKNOWN, 0.5

# ==================== PHONE EXTRACTOR ====================
class PhoneExtractor:
    """Extract Vietnamese phone numbers"""
    def __init__(self):
        self.patterns = [
            r'(?:\+?84|0)(?:3[2-9]|5[2689]|7[06-9]|8[1-9]|9[0-9])\d{7}',
            r'(?:\+?84|0)\d{9,10}',
            r'\b\d{9,11}\b'
        ]
    
    def extract(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        text_clean = re.sub(r'[^\d\s\+]', '', text)
        
        for pattern in self.patterns:
            matches = re.findall(pattern, text_clean)
            if matches:
                for phone in matches:
                    phone_digits = re.sub(r'\D', '', phone)
                    
                    if 9 <= len(phone_digits) <= 11:
                        # Format to standard
                        if phone_digits.startswith('84'):
                            phone_digits = '0' + phone_digits[2:]
                        elif len(phone_digits) == 9:
                            phone_digits = '0' + phone_digits
                        
                        return phone_digits
        
        return None

# ==================== ADVANCED QUERY ANALYZER ====================
class AdvancedQueryAnalyzer:
    """
    Advanced semantic analysis of user queries
    Extracts: duration, budget, group_type, style_preferences, location
    """
    def __init__(self):
        self.duration_patterns = {
            '1_day': ['1 ngày', 'một ngày', 'ngắn ngày', 'trong ngày', 'cuối tuần'],
            '2_days': ['2 ngày', 'hai ngày', '1 đêm', 'qua đêm'],
            '3_days': ['3 ngày', 'ba ngày', '2 đêm'],
            'long': ['dài ngày', 'nhiều ngày', 'tuần', 'week']
        }
        
        self.budget_patterns = {
            'budget': ['giá rẻ', 'tiết kiệm', 'bình dân', 'phải chăng', 'dưới 1 triệu', 'dưới 2 triệu'],
            'midrange': ['trung bình', 'vừa phải', '1-2 triệu', '2-3 triệu'],
            'premium': ['cao cấp', 'sang trọng', 'luxury', 'premium', 'vip']
        }
        
        self.group_patterns = {
            'solo': ['một mình', 'cá nhân', 'tôi đi một mình', 'solo', 'độc hành'],
            'couple': ['cặp đôi', 'hai người', 'vợ chồng', 'bạn gái', 'bạn trai'],
            'family': ['gia đình', 'có trẻ em', 'có bé', 'cả nhà', 'cùng gia đình'],
            'group': ['nhóm bạn', 'bạn bè', 'team', 'đoàn', 'group'],
            'corporate': ['công ty', 'doanh nghiệp', 'team building', 'corporate']
        }
        
        self.style_patterns = {
            'retreat': ['retreat', 'nghỉ dưỡng', 'thư giãn', 'tĩnh tâm', 'chữa lành'],
            'spiritual': ['tâm linh', 'thiền', 'chánh niệm', 'nội tâm', 'khí công'],
            'adventure': ['mạo hiểm', 'thử thách', 'trekking', 'leo núi', 'khám phá'],
            'cultural': ['văn hóa', 'lịch sử', 'di tích', 'bản địa', 'truyền thống'],
            'nature': ['thiên nhiên', 'rừng', 'núi', 'biển', 'suối', 'sinh thái'],
            'gentle': ['nhẹ nhàng', 'ít di chuyển', 'dễ dàng', 'thoải mái', 'yên tĩnh']
        }
        
        self.location_patterns = {
            'Huế': ['huế', 'hue'],
            'Bạch Mã': ['bạch mã', 'bach ma', 'ngũ hồ'],
            'Quảng Trị': ['quảng trị', 'quang tri', 'hiền lương', 'cồn cỏ', 'cửa việt'],
            'Đà Nẵng': ['đà nẵng', 'da nang'],
            'Phong Nha': ['phong nha', 'quảng bình'],
            'Hội An': ['hội an', 'hoi an']
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Comprehensive query analysis"""
        query_lower = query.lower()
        
        analysis = {
            'duration': self._extract_duration(query_lower),
            'budget': self._extract_budget(query_lower),
            'group_type': self._extract_group_type(query_lower),
            'style_preferences': self._extract_style_preferences(query_lower),
            'locations': self._extract_locations(query_lower),
            'keywords': self._extract_keywords(query_lower),
            'is_comparison': self._is_comparison_query(query_lower),
            'is_recommendation': self._is_recommendation_query(query_lower)
        }
        
        return analysis
    
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration preference"""
        for duration, patterns in self.duration_patterns.items():
            if any(pattern in text for pattern in patterns):
                return duration
        return None
    
    def _extract_budget(self, text: str) -> Optional[str]:
        """Extract budget level"""
        for budget, patterns in self.budget_patterns.items():
            if any(pattern in text for pattern in patterns):
                return budget
        return None
    
    def _extract_group_type(self, text: str) -> Optional[str]:
        """Extract group type"""
        for group, patterns in self.group_patterns.items():
            if any(pattern in text for pattern in patterns):
                return group
        return None
    
    def _extract_style_preferences(self, text: str) -> List[str]:
        """Extract style preferences"""
        styles = []
        for style, patterns in self.style_patterns.items():
            if any(pattern in text for pattern in patterns):
                styles.append(style)
        return styles
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location preferences"""
        locations = []
        for location, patterns in self.location_patterns.items():
            if any(pattern in text for pattern in patterns):
                locations.append(location)
        return locations
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        # Remove common words
        stop_words = {'là', 'của', 'và', 'có', 'được', 'cho', 'với', 'tôi', 'bạn', 'anh', 'chị'}
        
        words = text.split()
        keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
        
        return keywords[:10]
    
    def _is_comparison_query(self, text: str) -> bool:
        """Check if query is asking for comparison"""
        comparison_words = ['so sánh', 'khác nhau', 'giống', 'tốt hơn', 'nên chọn', 'khác gì']
        return any(word in text for word in comparison_words)
    
    def _is_recommendation_query(self, text: str) -> bool:
        """Check if query is asking for recommendation"""
        recommendation_words = ['gợi ý', 'tư vấn', 'đề xuất', 'nên đi', 'phù hợp', 'chọn tour']
        return any(word in text for word in recommendation_words)

# ==================== TOUR FILTER ====================
class TourFilter:
    """
    Intelligent tour filtering based on user preferences
    Uses tour_entities.json structure for advanced filtering
    """
    def __init__(self):
        self.tour_entities = state.tour_entities
        self.tours_db = state.tours_db
    
    def filter(self, preferences: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Filter tours based on preferences
        Returns scored and ranked tours
        """
        scored_tours = []
        
        # Use tour_entities if available, otherwise use tours_db
        tours_to_filter = self.tour_entities if self.tour_entities else self._build_entities_from_tours()
        
        for tour_id, tour_data in tours_to_filter.items():
            score = self._calculate_tour_score(tour_data, preferences)
            
            if score > 0:
                scored_tours.append({
                    'tour_id': tour_id,
                    'tour_data': tour_data,
                    'score': score,
                    'match_reasons': self._get_match_reasons(tour_data, preferences)
                })
        
        # Sort by score
        scored_tours.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_tours[:max_results]
    
    def _calculate_tour_score(self, tour: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        """Calculate match score for a tour"""
        score = 0.0
        
        # Duration match
        if preferences.get('duration'):
            duration_days = tour.get('duration_days', 1)
            pref_duration = preferences['duration']
            
            if pref_duration == '1_day' and duration_days == 1:
                score += 3.0
            elif pref_duration == '2_days' and duration_days == 2:
                score += 3.0
            elif pref_duration == '3_days' and duration_days == 3:
                score += 3.0
            elif pref_duration == 'long' and duration_days >= 4:
                score += 3.0
        
        # Budget match
        if preferences.get('budget'):
            avg_price = tour.get('avg_price', 1500000)
            pref_budget = preferences['budget']
            
            if pref_budget == 'budget' and avg_price < 1000000:
                score += 2.5
            elif pref_budget == 'midrange' and 1000000 <= avg_price <= 2500000:
                score += 2.5
            elif pref_budget == 'premium' and avg_price > 2500000:
                score += 2.5
        
        # Group type match
        if preferences.get('group_type'):
            group_type = preferences['group_type']
            
            if group_type == 'family' and tour.get('family_friendly', False):
                score += 2.0
            elif group_type == 'corporate' and tour.get('corporate_friendly', False):
                score += 2.0
            elif group_type == 'solo' and tour.get('solo_friendly', False):
                score += 2.0
        
        # Style preferences match
        if preferences.get('style_preferences'):
            tour_tags = tour.get('tags', [])
            
            for style in preferences['style_preferences']:
                # Map style to tags
                style_tag_map = {
                    'retreat': 'retreat',
                    'spiritual': 'tâm_linh',
                    'adventure': 'mạo_hiểm',
                    'cultural': 'văn_hóa',
                    'nature': 'thiên_nhiên',
                    'gentle': 'ít_di_chuyển'
                }
                
                tag = style_tag_map.get(style, style)
                if tag in tour_tags:
                    score += 1.5
        
        # Location match
        if preferences.get('locations'):
            tour_location = tour.get('location', '')
            
            for location in preferences['locations']:
                if location.lower() in tour_location.lower():
                    score += 2.0
        
        # Keyword match
        if preferences.get('keywords'):
            tour_text = json.dumps(tour, ensure_ascii=False).lower()
            
            for keyword in preferences['keywords']:
                if keyword in tour_text:
                    score += 0.5
        
        # Popularity boost
        popularity_score = tour.get('popularity_score', 0)
        score += popularity_score * 0.1
        
        # Value boost
        value_score = tour.get('value_score', 0)
        score += value_score * 0.1
        
        return score
    
    def _get_match_reasons(self, tour: Dict[str, Any], preferences: Dict[str, Any]) -> List[str]:
        """Get reasons why tour matches preferences"""
        reasons = []
        
        if preferences.get('duration'):
            duration_days = tour.get('duration_days', 1)
            reasons.append(f"Phù hợp thời lượng {duration_days} ngày")
        
        if preferences.get('budget'):
            avg_price = tour.get('avg_price', 0)
            reasons.append(f"Phù hợp ngân sách (giá trung bình: {avg_price:,} VND)")
        
        if preferences.get('style_preferences'):
            matching_tags = []
            tour_tags = tour.get('tags', [])
            
            for style in preferences['style_preferences']:
                if style in tour_tags or style.replace('_', ' ') in ' '.join(tour_tags):
                    matching_tags.append(style)
            
            if matching_tags:
                reasons.append(f"Phong cách: {', '.join(matching_tags)}")
        
        if preferences.get('locations'):
            tour_location = tour.get('location', '')
            matching_locations = [loc for loc in preferences['locations'] if loc.lower() in tour_location.lower()]
            
            if matching_locations:
                reasons.append(f"Địa điểm: {', '.join(matching_locations)}")
        
        return reasons
    
    def _build_entities_from_tours(self) -> Dict[str, Any]:
        """Build tour entities from tours_db if tour_entities not available"""
        entities = {}
        
        for i, tour in enumerate(self.tours_db):
            tour_id = tour.get('tour_id', f'TOUR_{i}')
            
            entities[tour_id] = {
                'tour_name': tour.get('tour_name', ''),
                'location': tour.get('location', ''),
                'duration': tour.get('duration', ''),
                'duration_days': self._parse_duration(tour.get('duration', '')),
                'price': tour.get('price', ''),
                'avg_price': self._parse_avg_price(tour.get('price', '')),
                'summary': tour.get('summary', ''),
                'tags': self._extract_basic_tags(tour),
                'family_friendly': self._is_family_friendly(tour),
                'corporate_friendly': self._is_corporate_friendly(tour),
                'solo_friendly': True,  # Most tours are solo-friendly
                'popularity_score': 5,  # Default
                'value_score': 5  # Default
            }
        
        return entities
    
    def _parse_duration(self, duration_text: str) -> int:
        """Parse duration text to days"""
        if not duration_text:
            return 1
        
        match = re.search(r'(\d+)\s*ngày', duration_text.lower())
        if match:
            return int(match.group(1))
        
        return 1
    
    def _parse_avg_price(self, price_text: str) -> int:
        """Parse price text to average price"""
        if not price_text:
            return 1500000
        
        numbers = re.findall(r'[\d\.]+', price_text.replace(',', ''))
        
        prices = []
        for num in numbers:
            try:
                clean_num = int(num.replace('.', ''))
                if clean_num >= 1000:
                    prices.append(clean_num)
            except:
                pass
        
        if prices:
            return sum(prices) // len(prices)
        
        return 1500000
    
    def _extract_basic_tags(self, tour: Dict[str, Any]) -> List[str]:
        """Extract basic tags from tour"""
        tags = []
        
        tour_text = json.dumps(tour, ensure_ascii=False).lower()
        
        tag_keywords = {
            'retreat': ['retreat', 'nghỉ dưỡng', 'thư giãn'],
            'tâm_linh': ['tâm linh', 'thiền', 'chánh niệm'],
            'lịch_sử': ['lịch sử', 'di tích', 'chiến tranh'],
            'thiên_nhiên': ['thiên nhiên', 'rừng', 'núi', 'suối'],
            'văn_hóa': ['văn hóa', 'bản địa', 'dân tộc']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in tour_text for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _is_family_friendly(self, tour: Dict[str, Any]) -> bool:
        """Check if tour is family-friendly"""
        tour_text = json.dumps(tour, ensure_ascii=False).lower()
        family_keywords = ['gia đình', 'trẻ em', 'phù hợp gia đình', 'cả nhà']
        
        return any(keyword in tour_text for keyword in family_keywords)
    
    def _is_corporate_friendly(self, tour: Dict[str, Any]) -> bool:
        """Check if tour is corporate-friendly"""
        tour_text = json.dumps(tour, ensure_ascii=False).lower()
        corporate_keywords = ['team building', 'công ty', 'doanh nghiệp', 'corporate']
        
        return any(keyword in tour_text for keyword in corporate_keywords)

# ==================== CHAT PROCESSOR ====================
class ChatProcessor:
    """
    Main chat processing logic with enhanced flow
    CRITICAL FIX: Ensures all advisory intents get detailed responses
    """
    def __init__(self):
        self.search_engine = state.get_search_engine()
        self.response_generator = ResponseGenerator()
        self.intent_detector = IntentDetector()
        self.phone_extractor = PhoneExtractor()
        self.query_analyzer = AdvancedQueryAnalyzer()
        self.tour_filter = TourFilter()
    
    def process(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process user message and generate response"""
        try:
            # Update stats
            with state.stats_lock:
                state.stats['requests'] += 1
            
            # Get or create session
            session_data = self._get_session(session_id)
            
            # Detect intent
            intent, confidence = self.intent_detector.detect(user_message)
            logger.info(f"🎯 Detected intent: {intent} (confidence: {confidence:.2f})")
            
            # Advanced query analysis (for advisory intents)
            query_analysis = None
            if Intent.is_advisory_intent(intent):
                query_analysis = self.query_analyzer.analyze(user_message)
                logger.info(f"🔍 Query analysis: {json.dumps(query_analysis, ensure_ascii=False)}")
            
            # Extract phone if present
            phone = self.phone_extractor.extract(user_message)
            if phone:
                logger.info(f"📞 Phone detected: {phone[:4]}***")
                intent = Intent.PROVIDE_PHONE
            
            # Update session
            session_data['last_intent'] = intent
            session_data['last_query'] = user_message
            session_data['turn_count'] += 1
            
            if query_analysis:
                session_data['last_query_analysis'] = query_analysis
            
            # Process based on intent
            if intent == Intent.GREETING:
                response_text = self._handle_greeting(session_data)
            
            elif intent == Intent.FAREWELL:
                response_text = self._handle_farewell(session_data)
            
            elif intent == Intent.PROVIDE_PHONE:
                response_text = self._handle_phone_provision(phone, session_data)
            
            elif intent == Intent.CALLBACK_REQUEST:
                response_text = self._handle_callback_request(session_data)
            
            elif Intent.is_advisory_intent(intent):
                # CRITICAL PATH: Advisory intents MUST get detailed responses
                response_text = self._handle_advisory_query(
                    user_message, 
                    intent, 
                    session_data,
                    query_analysis
                )
            
            else:
                # Unknown/other intents: still try to be helpful
                response_text = self._handle_unknown(user_message, session_data)
            
            # Update conversation history
            session_data['history'].append({
                'user': user_message,
                'assistant': response_text,
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit history size
            if len(session_data['history']) > Config.CONVERSATION_HISTORY_LIMIT:
                session_data['history'] = session_data['history'][-Config.CONVERSATION_HISTORY_LIMIT:]
            
            # Update session
            self._update_session(session_id, session_data)
            
            return {
                'message': response_text,
                'intent': intent,
                'confidence': confidence,
                'session_id': session_id,
                'query_analysis': query_analysis
            }
        
        except Exception as e:
            logger.error(f"CHAT PROCESS ERROR: {str(e)}")
            traceback.print_exc()

            return {
                "message": "Xin lỗi, hệ thống đang xử lý chậm. Bạn vui lòng thử lại sau giây lát.",
                "intent": Intent.UNKNOWN,
                "confidence": 0.0,
                "session_id": session_id
            }

    
    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session data"""
        with state.sessions_lock:
            if session_id not in state.sessions:
                state.sessions[session_id] = {
                    'created_at': datetime.now(),
                    'last_active': datetime.now(),
                    'history': [],
                    'stage': ConversationStage.INITIAL,
                    'last_intent': Intent.UNKNOWN,
                    'last_query': '',
                    'last_query_analysis': None,
                    'turn_count': 0,
                    'leads_captured': [],
                    'selected_tours': [],
                    'preferences': {}
                }
                
                # Limit session count
                if len(state.sessions) > Config.MAX_SESSIONS:
                    state.sessions.popitem(last=False)
            
            session = state.sessions[session_id]
            session['last_active'] = datetime.now()
            
            return session
    
    def _update_session(self, session_id: str, session_data: Dict[str, Any]):
        """Update session data"""
        with state.sessions_lock:
            state.sessions[session_id] = session_data
    
    def _handle_greeting(self, session_data: Dict[str, Any]) -> str:
        """Handle greeting intent"""
        greetings = [
            "Xin chào! Tôi là trợ lý AI của Ruby Wings. Tôi có thể giúp bạn tìm hiểu về các tour trải nghiệm, retreat, và hành trình chữa lành. Bạn muốn biết thông tin gì? 🌿",
            "Chào bạn! Ruby Wings chuyên các hành trình du lịch trải nghiệm có chiều sâu. Bạn đang tìm kiếm tour nào? 😊",
            "Xin chào! Tôi có thể tư vấn cho bạn về các tour retreat, thiền, và du lịch chữa lành của Ruby Wings. Hãy cho tôi biết bạn quan tâm đến gì nhé! 🙏",
            "Chào bạn! Ruby Wings mang đến những hành trình có chiều sâu, kết hợp trải nghiệm và chữa lành nội tâm. Bạn muốn khám phá tour nào? 🌿"
        ]
        
        greeting = random.choice(greetings)
        
        # Add quick suggestions if first greeting
        if session_data['turn_count'] == 1:
            greeting += "\n\n💡 **Gợi ý:**\n"
            greeting += "• Tour retreat 1 ngày Bạch Mã\n"
            greeting += "• Hành trình Mưa Đỏ Trường Sơn 2N1Đ\n"
            greeting += "• Giới thiệu về Ruby Wings\n"
            greeting += "• So sánh các tour\n"
            greeting += "\nHoặc hãy cho tôi biết sở thích của bạn để tư vấn phù hợp nhất!"
        
        return greeting
    
    def _handle_farewell(self, session_data: Dict[str, Any]) -> str:
        """Handle farewell intent"""
        farewells = [
            "Cảm ơn bạn đã quan tâm đến Ruby Wings! Hẹn gặp lại. 🙏",
            "Rất vui được hỗ trợ bạn! Liên hệ 0332510486 nếu cần thêm thông tin. Chúc bạn một ngày tốt lành! 🌿",
            "Hẹn sớm gặp lại bạn trên những hành trình của Ruby Wings! 😊",
            "Cảm ơn bạn đã trò chuyện! Ruby Wings luôn sẵn sàng đồng hành cùng bạn. Hẹn gặp lại! 🌟"
        ]
        
        farewell = random.choice(farewells)
        
        # Add CTA if they showed interest but didn't book
        if session_data.get('selected_tours') and not session_data.get('leads_captured'):
            farewell += "\n\n📞 Nhớ liên hệ 0332510486 để đặt tour nhé!"
        
        return farewell
    
    def _handle_phone_provision(self, phone: str, session_data: Dict[str, Any]) -> str:
        """Handle phone number provision"""
        # Save lead
        lead_data = {
            'timestamp': datetime.now().isoformat(),
            'phone': phone,
            'source_channel': 'Chatbot',
            'action_type': 'Phone Provided',
            'session_id': str(id(session_data)),
            'intent': Intent.PROVIDE_PHONE,
            'last_query': session_data.get('last_query', ''),
            'preferences': session_data.get('preferences', {}),
            'selected_tours': session_data.get('selected_tours', [])
        }
        
        session_data['leads_captured'].append(lead_data)
        
        # Update stats
        with state.stats_lock:
            state.stats['leads'] += 1
        
        # TODO: Save to Google Sheets / Meta CAPI (implement in production)
        
        responses = [
            f"Cảm ơn bạn đã cung cấp số điện thoại {phone[:4]}***! Đội ngũ Ruby Wings sẽ liên hệ tư vấn trong vòng 30 phút. 📞",
            f"Đã nhận số điện thoại {phone[:4]}***! Chúng tôi sẽ gọi tư vấn chi tiết cho bạn trong thời gian sớm nhất. 🙏",
            f"Số {phone[:4]}*** đã được ghi nhận. Bộ phận tư vấn Ruby Wings sẽ liên hệ bạn ngay hôm nay! ✅"
        ]
        
        response = random.choice(responses)
        
        # Add context if they were looking at specific tours
        if session_data.get('selected_tours'):
            tour_names = [t.get('tour_name', '') for t in session_data['selected_tours'][:2]]
            response += f"\n\nChúng tôi sẽ tư vấn chi tiết về: {', '.join(tour_names)}"
        
        return response
    
    def _handle_callback_request(self, session_data: Dict[str, Any]) -> str:
        """Handle callback request"""
        return "Bạn muốn chúng tôi gọi lại khi nào? Vui lòng cung cấp:\n• Số điện thoại\n• Khung giờ thuận tiện (sáng/chiều/tối)\n\nVí dụ: \"0909123456, tôi rảnh chiều nay\"\n\n📞 Hoặc gọi ngay hotline 0332510486 để được hỗ trợ trực tiếp!"
    
    def _handle_advisory_query(self, query: str, intent: str, session_data: Dict[str, Any],
                               query_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle advisory queries with GUARANTEED detailed response
        CRITICAL FIX: Always provides rich context for LLM generation
        Uses intelligent filtering when query analysis is available
        """
        # Try intelligent filtering first if we have query analysis
        filtered_tours = []
        if query_analysis and (query_analysis.get('duration') or query_analysis.get('budget') or 
                              query_analysis.get('style_preferences') or query_analysis.get('locations')):
            logger.info("🎯 Using intelligent tour filtering")
            filtered_results = self.tour_filter.filter(query_analysis, max_results=5)
            
            if filtered_results:
                # Build context from filtered tours
                search_results = []
                for result in filtered_results:
                    tour_data = result['tour_data']
                    match_reasons = result.get('match_reasons', [])
                    
                    tour_text = f"""Tên tour: {tour_data.get('tour_name', '')}
Địa điểm: {tour_data.get('location', '')}
Thời lượng: {tour_data.get('duration', '')}
Giá: {tour_data.get('price', '')}
Tóm tắt: {tour_data.get('summary', '')[:300]}
Lý do phù hợp: {', '.join(match_reasons)}
Điểm phù hợp: {result.get('score', 0):.1f}/10"""
                    
                    search_results.append({
                        'score': result['score'],
                        'text': tour_text,
                        'path': f"root.tours[{result['tour_id']}]",
                        'metadata': {'type': 'tour', 'source': 'intelligent_filter', 'tour_data': tour_data}
                    })
                
                # Store selected tours in session
                session_data['selected_tours'] = [r['tour_data'] for r in filtered_results[:3]]
                
                logger.info(f"✅ Intelligent filtering returned {len(search_results)} tours")
        
        # Fallback to vector search if filtering didn't work
        if not filtered_tours:
            search_results = self.search_engine.search(query, top_k=Config.TOP_K, intent=intent)
            logger.info(f"🔍 Vector search returned {len(search_results)} results for intent {intent}")
        
        # CRITICAL FIX: Ensure non-empty search results
        if not search_results:
            logger.warning(f"⚠️ Empty search results for advisory intent {intent}, forcing fallback")
            search_results = self.search_engine._get_fallback_results(query, intent, Config.TOP_K)
        
        # Enrich context with company info for ABOUT_COMPANY queries
        if intent in [Intent.ABOUT_COMPANY, Intent.COMPANY_SERVICE, Intent.COMPANY_MISSION]:
            if state.company_info and len(search_results) < 3:
                company_text = self._format_company_context(state.company_info)
                search_results.insert(0, {
                    'score': 1.0,
                    'text': company_text,
                    'path': 'root.company_info',
                    'metadata': {'type': 'company_info', 'source': 'enrichment'}
                })
        
        # Build enhanced user context
        user_context = {
            'session_id': str(id(session_data)),
            'query_analysis': query_analysis,
            'selected_tours': session_data.get('selected_tours', []),
            'preferences': session_data.get('preferences', {}),
            'turn_count': session_data.get('turn_count', 0)
        }
        
        # Generate response
        response_text = self.response_generator.generate(
            query=query,
            search_results=search_results,
            intent=intent,
            conversation_history=session_data.get('history', []),
            user_context=user_context
        )
        
        # Validate response is not generic
        if self._is_generic_response(response_text):
            logger.warning("⚠️ Generic response detected, using enhanced fallback")
            response_text = self.response_generator._generate_enhanced_fallback(query, search_results, intent)
        
        # Add recommendations or next steps based on intent
        response_text = self._add_conversation_continuity(response_text, intent, session_data, query_analysis)
        
        return response_text
    
    def _handle_unknown(self, query: str, session_data: Dict[str, Any]) -> str:
        """Handle unknown queries - still try to be helpful"""
        # Try searching anyway
        search_results = self.search_engine.search(query, top_k=3, intent=Intent.UNKNOWN)
        
        if search_results:
            # Generate response from results
            return self.response_generator.generate(
                query=query,
                search_results=search_results,
                intent=Intent.UNKNOWN,
                conversation_history=session_data.get('history', [])
            )
        else:
            return """Tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể hỏi về:

🌿 **Tour du lịch trải nghiệm**
• Tour retreat 1 ngày
• Hành trình 2-3 ngày
• So sánh các tour
• Gợi ý tour phù hợp

🏢 **Về Ruby Wings**
• Triết lý và giá trị
• Hệ sinh thái dịch vụ
• Câu chuyện thương hiệu

💰 **Thông tin booking**
• Giá tour và chính sách
• Cách đặt tour
• Hỗ trợ tư vấn

📞 Hoặc liên hệ hotline **0332510486** để được hỗ trợ trực tiếp!"""
    
    def _format_company_context(self, company_info: Dict[str, Any]) -> str:
        """Format company info as rich context"""
        parts = [
            f"**{company_info.get('name', 'Ruby Wings Travel')}**",
            "",
            f"📍 Địa chỉ: {company_info.get('contact', {}).get('address', '148 Trương Gia Mô, Vỹ Dạ, TP. Huế')}",
            f"📞 Hotline: {company_info.get('contact', {}).get('phone', '0332510486')}",
            f"📧 Email: {company_info.get('contact', {}).get('email', 'info@rubywings.vn')}",
            "",
            "**Giới thiệu:**",
            company_info.get('description', '')[:400],
            "",
            "**Triết lý:**",
            company_info.get('philosophy', '')[:400]
        ]
        
        if company_info.get('mission'):
            parts.append("")
            parts.append("**Sứ mệnh:**")
            for i, mission in enumerate(company_info['mission'][:5], 1):
                parts.append(f"{i}. {mission}"[:150])
        
        if company_info.get('values'):
            parts.append("")
            parts.append("**Giá trị cốt lõi:**")
            for i, value in enumerate(company_info['values'][:5], 1):
                parts.append(f"{i}. {value}"[:150])
        
        return "\n".join(parts)
    
    def _add_conversation_continuity(self, response: str, intent: str, 
                                    session_data: Dict[str, Any],
                                    query_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Add conversation continuity suggestions"""
        # Don't add if response already has CTA
        if '0332510486' in response and len(response) > 500:
            return response
        
        suggestions = []
        
        if intent in [Intent.TOUR_INQUIRY, Intent.TOUR_FILTER, Intent.TOUR_RECOMMEND]:
            if session_data.get('selected_tours'):
                suggestions.append("💡 **Tiếp theo:**")
                suggestions.append("• So sánh các tour này")
                suggestions.append("• Đặt tour ngay")
                suggestions.append("• Cần tư vấn thêm về lịch trình")
                suggestions.append("")
                suggestions.append("📞 Liên hệ **0332510486** để đặt tour!")
        
        elif intent == Intent.ABOUT_COMPANY:
            suggestions.append("💡 **Tìm hiểu thêm:**")
            suggestions.append("• Xem các tour của Ruby Wings")
            suggestions.append("• Triết lý và giá trị cốt lõi")
            suggestions.append("• Hệ sinh thái dịch vụ")
        
        elif intent in [Intent.TOUR_COMPARE, Intent.PRICE_COMPARE]:
            suggestions.append("💡 **Quyết định tour phù hợp?**")
            suggestions.append("• Cung cấp số điện thoại để được tư vấn")
            suggestions.append("• Hỏi thêm về lịch trình chi tiết")
            suggestions.append("")
            suggestions.append("📞 Gọi ngay **0332510486** để được hỗ trợ!")
        
        if suggestions:
            if not response.endswith('\n\n'):
                response += "\n\n"
            response += "\n".join(suggestions)
        
        return response
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if response is too generic"""
        if len(response) < 100:
            return True
        
        generic_patterns = [
            "xin lỗi",
            "không có thông tin",
            "không thể",
            "không tìm thấy"
        ]
        
        response_lower = response.lower()
        generic_count = sum(1 for pattern in generic_patterns if pattern in response_lower)
        
        return generic_count >= 2

# ==================== ROUTES ====================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '6.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'], endpoint='api_chat')
def api_chat():
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
        logger.error(f"❌ /chat error: {e}")
        traceback.print_exc()
        
        with state.stats_lock:
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
        
        # Send to Meta CAPI
        if Config.ENABLE_META_CAPI_LEAD and META_CAPI_AVAILABLE:
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
                
                with state.stats_lock:
                    state.stats['meta_capi_calls'] += 1
                
                logger.info(f"✅ Form lead sent to Meta CAPI: {phone_clean[:4]}***")
            except Exception as e:
                with state.stats_lock:
                    state.stats['meta_capi_errors'] += 1
                logger.error(f"Meta CAPI error: {e}")
        
        # Save to Google Sheets
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
        
        # Fallback storage
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
        
        with state.stats_lock:
            state.stats['leads'] += 1
        
        return jsonify({
            'success': True,
            'message': 'Lead đã được lưu! Ruby Wings sẽ liên hệ sớm nhất. 📞',
            'data': {
                'phone': phone_clean[:3] + '***' + phone_clean[-2:],
                'timestamp': lead_data['timestamp']
            }
        })
    
    except Exception as e:
        logger.error(f"❌ Save lead error: {e}")
        traceback.print_exc()
        
        with state.stats_lock:
            state.stats['errors'] += 1
        
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Statistics endpoint"""
    return jsonify(state.get_stats())

# ==================== LEAD MANAGER ====================
class LeadManager:
    """
    Comprehensive lead management system
    Handles Google Sheets, Meta CAPI, and fallback storage
    """
    def __init__(self):
        self.google_sheets_enabled = Config.ENABLE_GOOGLE_SHEETS
        self.meta_capi_enabled = Config.ENABLE_META_CAPI_LEAD
        self.fallback_enabled = Config.ENABLE_FALLBACK_STORAGE
    
    def save_lead(self, lead_data: Dict[str, Any], request_obj=None) -> Dict[str, Any]:
        """Save lead to all configured destinations"""
        results = {
            'google_sheets': {'success': False, 'error': None},
            'meta_capi': {'success': False, 'error': None},
            'fallback': {'success': False, 'error': None}
        }
        
        # Google Sheets
        if self.google_sheets_enabled:
            try:
                self._save_to_google_sheets(lead_data)
                results['google_sheets']['success'] = True
                logger.info("✅ Lead saved to Google Sheets")
            except Exception as e:
                results['google_sheets']['error'] = str(e)
                logger.error(f"❌ Google Sheets save failed: {e}")
        
        # Meta CAPI
        if self.meta_capi_enabled and request_obj and META_CAPI_AVAILABLE:
            try:
                meta_result = send_meta_lead(
                    request_obj,
                    phone=lead_data.get('phone', ''),
                    contact_name=lead_data.get('contact_name', ''),
                    email=lead_data.get('email', ''),
                    content_name=lead_data.get('service_interest', 'General Inquiry'),
                    value=200000,
                    currency="VND"
                )
                
                results['meta_capi']['success'] = meta_result.get('success', False)
                
                with state.stats_lock:
                    if results['meta_capi']['success']:
                        state.stats['meta_capi_calls'] += 1
                    else:
                        state.stats['meta_capi_errors'] += 1
                
                logger.info(f"✅ Lead sent to Meta CAPI: {meta_result.get('event_id', 'N/A')}")
            except Exception as e:
                results['meta_capi']['error'] = str(e)
                logger.error(f"❌ Meta CAPI save failed: {e}")
        
        # Fallback storage
        if self.fallback_enabled:
            try:
                self._save_to_fallback(lead_data)
                results['fallback']['success'] = True
                logger.info("✅ Lead saved to fallback storage")
            except Exception as e:
                results['fallback']['error'] = str(e)
                logger.error(f"❌ Fallback save failed: {e}")
        
        # Update stats
        with state.stats_lock:
            state.stats['leads'] += 1
        
        return results
    
    def _save_to_google_sheets(self, lead_data: Dict[str, Any]):
        """Save lead to Google Sheets"""
        if not Config.GOOGLE_SERVICE_ACCOUNT_JSON or not Config.GOOGLE_SHEET_ID:
            raise ValueError("Google Sheets credentials not configured")
        
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
    
    def _save_to_fallback(self, lead_data: Dict[str, Any]):
        """Save lead to fallback JSON storage"""
        if os.path.exists(Config.FALLBACK_STORAGE_PATH):
            with open(Config.FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                leads = json.load(f)
        else:
            leads = []
        
        leads.append(lead_data)
        
        # Keep only last 1000 leads
        leads = leads[-1000:]
        
        with open(Config.FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
            json.dump(leads, f, ensure_ascii=False, indent=2)

# ==================== CONVERSATION STATE MANAGER ====================
class ConversationStateManager:
    """
    Manages conversation state transitions
    Implements a state machine for natural conversation flow
    """
    def __init__(self):
        self.state_transitions = {
            ConversationStage.INITIAL: [
                ConversationStage.GREETING,
                ConversationStage.EXPLORING
            ],
            ConversationStage.GREETING: [
                ConversationStage.EXPLORING,
                ConversationStage.FAREWELL
            ],
            ConversationStage.EXPLORING: [
                ConversationStage.FILTERING,
                ConversationStage.COMPARING,
                ConversationStage.SELECTING,
                ConversationStage.FAREWELL
            ],
            ConversationStage.FILTERING: [
                ConversationStage.COMPARING,
                ConversationStage.SELECTING,
                ConversationStage.EXPLORING
            ],
            ConversationStage.COMPARING: [
                ConversationStage.SELECTING,
                ConversationStage.BOOKING,
                ConversationStage.EXPLORING
            ],
            ConversationStage.SELECTING: [
                ConversationStage.BOOKING,
                ConversationStage.COMPARING,
                ConversationStage.EXPLORING
            ],
            ConversationStage.BOOKING: [
                ConversationStage.LEAD_CAPTURE,
                ConversationStage.CALLBACK,
                ConversationStage.FAREWELL
            ],
            ConversationStage.LEAD_CAPTURE: [
                ConversationStage.FAREWELL
            ],
            ConversationStage.CALLBACK: [
                ConversationStage.FAREWELL
            ],
            ConversationStage.FAREWELL: [
                ConversationStage.INITIAL
            ]
        }
    
    def get_next_stage(self, current_stage: str, intent: str, 
                       session_data: Dict[str, Any]) -> str:
        """Determine next conversation stage based on intent and context"""
        # Simple intent-to-stage mapping
        if intent == Intent.GREETING:
            return ConversationStage.GREETING
        
        elif intent == Intent.FAREWELL:
            return ConversationStage.FAREWELL
        
        elif intent in [Intent.TOUR_INQUIRY, Intent.ABOUT_COMPANY]:
            return ConversationStage.EXPLORING
        
        elif intent in [Intent.TOUR_FILTER, Intent.TOUR_RECOMMEND]:
            return ConversationStage.FILTERING
        
        elif intent == Intent.TOUR_COMPARE:
            return ConversationStage.COMPARING
        
        elif intent in [Intent.TOUR_DETAIL, Intent.PRICE_ASK]:
            if session_data.get('selected_tours'):
                return ConversationStage.SELECTING
            else:
                return ConversationStage.EXPLORING
        
        elif intent in [Intent.BOOKING_REQUEST, Intent.BOOKING_PROCESS]:
            return ConversationStage.BOOKING
        
        elif intent == Intent.PROVIDE_PHONE:
            return ConversationStage.LEAD_CAPTURE
        
        elif intent == Intent.CALLBACK_REQUEST:
            return ConversationStage.CALLBACK
        
        else:
            # Stay in current stage
            return current_stage
    
    def can_transition(self, from_stage: str, to_stage: str) -> bool:
        """Check if transition is valid"""
        allowed_transitions = self.state_transitions.get(from_stage, [])
        return to_stage in allowed_transitions
    
    def get_stage_prompt_hints(self, stage: str) -> List[str]:
        """Get conversation hints for current stage"""
        hints = {
            ConversationStage.INITIAL: [
                "Hỏi về tour Ruby Wings",
                "Tìm hiểu về công ty",
                "Gợi ý tour phù hợp"
            ],
            ConversationStage.GREETING: [
                "Tìm tour 1 ngày",
                "Tour retreat chữa lành",
                "Giới thiệu Ruby Wings"
            ],
            ConversationStage.EXPLORING: [
                "Xem chi tiết tour",
                "So sánh các tour",
                "Hỏi về giá"
            ],
            ConversationStage.FILTERING: [
                "Lọc theo ngân sách",
                "Chọn thời lượng",
                "Xem kết quả"
            ],
            ConversationStage.COMPARING: [
                "So sánh 2 tour",
                "Điểm khác biệt",
                "Tour nào tốt hơn"
            ],
            ConversationStage.SELECTING: [
                "Đặt tour này",
                "Hỏi thêm chi tiết",
                "Thay đổi tour"
            ],
            ConversationStage.BOOKING: [
                "Cung cấp số điện thoại",
                "Yêu cầu gọi lại",
                "Hỏi về thanh toán"
            ],
            ConversationStage.LEAD_CAPTURE: [
                "Chờ liên hệ",
                "Hỏi thêm thông tin"
            ],
            ConversationStage.CALLBACK: [
                "Chọn khung giờ",
                "Xác nhận yêu cầu"
            ],
            ConversationStage.FAREWELL: [
                "Bắt đầu lại",
                "Tìm tour khác"
            ]
        }
        
        return hints.get(stage, [])

# ==================== RESPONSE FORMATTER ====================
class ResponseFormatter:
    """
    Formats responses with rich text, emojis, and structured layout
    Ensures consistent, professional presentation
    """
    def __init__(self):
        self.tour_emojis = {
            'retreat': '🧘',
            'spiritual': '🙏',
            'nature': '🌿',
            'cultural': '🏛️',
            'adventure': '🏔️',
            'family': '👨‍👩‍👧‍👦',
            'corporate': '💼'
        }
    
    def format_tour_card(self, tour: Dict[str, Any], rank: Optional[int] = None) -> str:
        """Format tour as a rich card"""
        lines = []
        
        # Header with rank
        if rank:
            lines.append(f"**{rank}. {tour.get('tour_name', 'Tour')}** ⭐")
        else:
            lines.append(f"**{tour.get('tour_name', 'Tour')}**")
        
        lines.append("")
        
        # Key info
        if tour.get('location'):
            lines.append(f"📍 **Địa điểm:** {tour['location']}")
        
        if tour.get('duration'):
            lines.append(f"⏰ **Thời lượng:** {tour['duration']}")
        
        if tour.get('price'):
            lines.append(f"💰 **Giá:** {tour['price']}")
        
        lines.append("")
        
        # Summary
        if tour.get('summary'):
            summary = tour['summary'][:200]
            if len(tour['summary']) > 200:
                summary += "..."
            lines.append(f"📝 {summary}")
        
        # Tags/highlights
        if tour.get('tags'):
            tag_icons = []
            for tag in tour['tags'][:5]:
                icon = self.tour_emojis.get(tag, '🔹')
                tag_icons.append(f"{icon} {tag.replace('_', ' ').title()}")
            
            if tag_icons:
                lines.append("")
                lines.append(f"**Đặc điểm:** {' • '.join(tag_icons)}")
        
        return "\n".join(lines)
    
    def format_tour_list(self, tours: List[Dict[str, Any]], 
                        max_display: int = 3,
                        show_ranking: bool = True) -> str:
        """Format list of tours"""
        if not tours:
            return "Không tìm thấy tour phù hợp."
        
        lines = []
        
        for i, tour in enumerate(tours[:max_display], 1):
            rank = i if show_ranking else None
            card = self.format_tour_card(tour, rank=rank)
            lines.append(card)
            lines.append("\n" + "─" * 50 + "\n")
        
        # Add footer if more tours available
        if len(tours) > max_display:
            remaining = len(tours) - max_display
            lines.append(f"\n💡 *Còn {remaining} tour khác. Liên hệ để xem đầy đủ!*")
        
        return "\n".join(lines)
    
    def format_comparison_table(self, tours: List[Dict[str, Any]]) -> str:
        """Format tour comparison"""
        if not tours or len(tours) < 2:
            return "Cần ít nhất 2 tour để so sánh."
        
        lines = [
            "**SO SÁNH TOUR**",
            ""
        ]
        
        # Tour names
        tour_names = [t.get('tour_name', f'Tour {i+1}')[:30] for i, t in enumerate(tours[:3])]
        lines.append(f"**Tên tour:**")
        for i, name in enumerate(tour_names, 1):
            lines.append(f"{i}. {name}")
        lines.append("")
        
        # Duration
        lines.append(f"⏰ **Thời lượng:**")
        for i, tour in enumerate(tours[:3], 1):
            lines.append(f"{i}. {tour.get('duration', 'N/A')}")
        lines.append("")
        
        # Price
        lines.append(f"💰 **Giá:**")
        for i, tour in enumerate(tours[:3], 1):
            lines.append(f"{i}. {tour.get('price', 'N/A')}")
        lines.append("")
        
        # Location
        lines.append(f"📍 **Địa điểm:**")
        for i, tour in enumerate(tours[:3], 1):
            lines.append(f"{i}. {tour.get('location', 'N/A')}")
        lines.append("")
        
        # Highlights (first 2 for each)
        lines.append(f"🌟 **Điểm nổi bật:**")
        for i, tour in enumerate(tours[:3], 1):
            highlights = tour.get('highlights', [])[:2]
            if highlights:
                highlights_text = ", ".join(str(h)[:50] for h in highlights)
                lines.append(f"{i}. {highlights_text}")
            else:
                lines.append(f"{i}. Xem chi tiết tại website")
        
        return "\n".join(lines)
    
    def add_call_to_action(self, text: str, cta_type: str = 'default') -> str:
        """Add call-to-action to response"""
        ctas = {
            'default': "\n\n📞 **Liên hệ ngay:** 0332510486",
            'booking': "\n\n📞 **Đặt tour ngay:** 0332510486\n💬 Hoặc để lại số điện thoại, chúng tôi sẽ gọi lại!",
            'consultation': "\n\n💬 **Cần tư vấn thêm?** Gọi 0332510486\n📧 Email: info@rubywings.vn",
            'callback': "\n\n📞 **Để lại số điện thoại** và chúng tôi sẽ gọi lại trong 30 phút!",
            'website': "\n\n🌐 **Xem thêm:** https://www.rubywings.vn\n📞 **Hotline:** 0332510486"
        }
        
        cta = ctas.get(cta_type, ctas['default'])
        
        # Don't duplicate if CTA already present
        if '0332510486' in text:
            return text
        
        return text + cta
    
    def format_company_info(self, company_data: Dict[str, Any]) -> str:
        """Format company information"""
        lines = [
            f"🏢 **{company_data.get('name', 'Ruby Wings Travel')}**",
            "",
            "**Thông tin liên hệ:**"
        ]
        
        contact = company_data.get('contact', {})
        if contact.get('phone'):
            lines.append(f"📞 Hotline: {contact['phone']}")
        if contact.get('email'):
            lines.append(f"📧 Email: {contact['email']}")
        if contact.get('address'):
            lines.append(f"📍 Địa chỉ: {contact['address']}")
        
        lines.append("")
        
        if company_data.get('description'):
            lines.append("**Giới thiệu:**")
            lines.append(company_data['description'][:300])
            lines.append("")
        
        if company_data.get('philosophy'):
            lines.append("**Triết lý:**")
            lines.append(company_data['philosophy'][:300])
            lines.append("")
        
        if company_data.get('mission'):
            lines.append("**Sứ mệnh:**")
            for i, mission in enumerate(company_data['mission'][:3], 1):
                lines.append(f"{i}. {mission}"[:150])
            lines.append("")
        
        return "\n".join(lines)
    
    def format_faq_response(self, faq: Dict[str, Any]) -> str:
        """Format FAQ"""
        return f"**❓ {faq.get('question', '')}**\n\n{faq.get('answer', '')}"

# ==================== CACHE MANAGER ====================
class CacheManager:
    """
    Manages caching for embeddings and responses
    Optimized for low-RAM environments
    """
    def __init__(self):
        self.embedding_cache = OrderedDict()
        self.response_cache = OrderedDict()
        self.cache_lock = threading.RLock()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.embedding_cache:
                # Move to end (LRU)
                self.embedding_cache.move_to_end(cache_key)
                return self.embedding_cache[cache_key]
        
        return None
    
    def set_embedding(self, text: str, embedding: List[float]):
        """Cache embedding"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        with self.cache_lock:
            self.embedding_cache[cache_key] = embedding
            
            # Limit cache size
            if len(self.embedding_cache) > Config.MAX_EMBEDDING_CACHE:
                self.embedding_cache.popitem(last=False)
    
    def get_response(self, query: str, intent: str) -> Optional[str]:
        """Get cached response"""
        cache_key = hashlib.md5(f"{query}_{intent}".encode()).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.response_cache:
                entry = self.response_cache[cache_key]
                
                # Check if expired
                if (datetime.now() - entry['timestamp']).seconds < Config.CACHE_TTL_SECONDS:
                    self.response_cache.move_to_end(cache_key)
                    return entry['response']
                else:
                    # Remove expired
                    del self.response_cache[cache_key]
        
        return None
    
    def set_response(self, query: str, intent: str, response: str):
        """Cache response"""
        cache_key = hashlib.md5(f"{query}_{intent}".encode()).hexdigest()
        
        with self.cache_lock:
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': datetime.now()
            }
            
            # Limit cache size
            if len(self.response_cache) > 50:
                self.response_cache.popitem(last=False)
    
    def clear(self):
        """Clear all caches"""
        with self.cache_lock:
            self.embedding_cache.clear()
            self.response_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'embedding_cache_size': len(self.embedding_cache),
                'response_cache_size': len(self.response_cache),
                'embedding_cache_max': Config.MAX_EMBEDDING_CACHE,
                'response_cache_max': 50
            }

# ==================== INITIALIZATION ====================
def initialize_app():
    """Initialize application components"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Initializing Ruby Wings Chatbot v6.0.0 (COMPREHENSIVE FIX)")
        logger.info("=" * 80)
        
        # Validate config
        Config.log_config()
        errors = Config.validate()
        
        if errors:
            for error in errors:
                logger.error(error)
            logger.error("❌ Configuration validation failed")
            return False
        
        # Load knowledge
        logger.info("📚 Loading knowledge base...")
        if not load_knowledge():
            logger.error("❌ Failed to load knowledge")
            return False
        
        logger.info(f"✅ Knowledge loaded: {len(state.tours_db)} tours, {len(state.faqs)} FAQs")
        
        # Initialize search engine
        logger.info("🔍 Initializing search engine...")
        search_engine = state.get_search_engine()
        if search_engine:
            search_engine.load_index()
            logger.info("✅ Search engine initialized")
        
        # Initialize chat processor
        logger.info("💬 Initializing chat processor...")
        chat_processor = state.get_chat_processor()
        if chat_processor:
            logger.info("✅ Chat processor initialized")
        
        logger.info("=" * 80)
        logger.info("✅ RUBY WINGS CHATBOT READY!")
        logger.info(f"📊 Loaded: {len(state.tours_db)} tours")
        logger.info(f"🏢 Company: {state.company_info.get('name', 'N/A')}")
        logger.info(f"🌐 Server: {Config.HOST}:{Config.PORT}")
        logger.info(f"💾 RAM Profile: {Config.RAM_PROFILE}MB")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

# ==================== UTILITY FUNCTIONS ====================
def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for better matching"""
    if not text:
        return ""
    
    # Remove diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def extract_price_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract price range from text"""
    # Remove common separators
    text_clean = text.replace('.', '').replace(',', '').replace(' ', '')
    
    # Find all numbers
    numbers = re.findall(r'\d+', text_clean)
    
    prices = []
    for num_str in numbers:
        try:
            num = int(num_str)
            # Filter reasonable prices (> 1000 VND)
            if num >= 1000:
                prices.append(num)
        except:
            pass
    
    if len(prices) >= 2:
        return min(prices), max(prices)
    elif len(prices) == 1:
        return prices[0], prices[0]
    else:
        return None, None

def format_price_vietnamese(price: int) -> str:
    """Format price in Vietnamese style"""
    if price >= 1000000:
        # Millions
        millions = price / 1000000
        return f"{millions:.1f} triệu VNĐ"
    elif price >= 1000:
        # Thousands
        thousands = price / 1000
        return f"{thousands:.0f} nghìn VNĐ"
    else:
        return f"{price:,} VNĐ"

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts"""
    words1 = set(normalize_vietnamese(text1).split())
    words2 = set(normalize_vietnamese(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def extract_duration_days(duration_text: str) -> Optional[int]:
    """Extract number of days from duration text"""
    if not duration_text:
        return None
    
    # Pattern: "2 ngày 1 đêm" → 2
    match = re.search(r'(\d+)\s*ngày', duration_text.lower())
    if match:
        return int(match.group(1))
    
    # Check for keywords
    if 'dài ngày' in duration_text.lower() or 'nhiều ngày' in duration_text.lower():
        return 7  # Default for long trips
    
    if 'ngắn ngày' in duration_text.lower() or 'trong ngày' in duration_text.lower():
        return 1
    
    return None

def validate_phone_number(phone: str) -> bool:
    """Validate Vietnamese phone number"""
    if not phone:
        return False
    
    # Remove non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Check length (9-11 digits)
    if not (9 <= len(digits) <= 11):
        return False
    
    # Check Vietnamese prefixes
    if digits.startswith('84'):
        # International format
        digits = '0' + digits[2:]
    
    if not digits.startswith('0'):
        return False
    
    # Check valid prefixes (03x, 05x, 07x, 08x, 09x)
    second_digit = digits[1] if len(digits) > 1 else '0'
    valid_prefixes = ['3', '5', '7', '8', '9']
    
    return second_digit in valid_prefixes

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    import html as html_module
    text = html_module.unescape(text)
    
    return text

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length with suffix"""
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        truncated = truncated[:last_space]
    
    return truncated + suffix

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

def is_business_hours() -> bool:
    """Check if current time is within business hours"""
    now = datetime.now()
    
    # Business hours: 8 AM - 8 PM Vietnam time
    if 8 <= now.hour < 20:
        return True
    
    return False

def get_greeting_by_time() -> str:
    """Get appropriate greeting based on time of day"""
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "Chào buổi sáng"
    elif 12 <= hour < 18:
        return "Chào buổi chiều"
    elif 18 <= hour < 22:
        return "Chào buổi tối"
    else:
        return "Xin chào"

# ==================== ANALYTICS & LOGGING ====================
class AnalyticsTracker:
    """
    Track user interactions and generate analytics
    Helps improve chatbot performance
    """
    def __init__(self):
        self.interactions = []
        self.interactions_lock = threading.RLock()
    
    def track_interaction(self, session_id: str, event_type: str, 
                         data: Dict[str, Any]):
        """Track user interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'event_type': event_type,
            'data': data
        }
        
        with self.interactions_lock:
            self.interactions.append(interaction)
            
            # Keep only last 1000 interactions
            if len(self.interactions) > 1000:
                self.interactions = self.interactions[-1000:]
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular queries"""
        query_counts = defaultdict(int)
        
        with self.interactions_lock:
            for interaction in self.interactions:
                if interaction['event_type'] == 'query':
                    query = interaction['data'].get('query', '').lower()
                    if query:
                        query_counts[query] += 1
        
        # Sort by count
        popular = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'query': query, 'count': count}
            for query, count in popular[:limit]
        ]
    
    def get_popular_tours(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most viewed tours"""
        tour_counts = defaultdict(int)
        
        with self.interactions_lock:
            for interaction in self.interactions:
                if interaction['event_type'] == 'tour_view':
                    tour_id = interaction['data'].get('tour_id')
                    if tour_id:
                        tour_counts[tour_id] += 1
        
        # Sort by count
        popular = sorted(tour_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'tour_id': tour_id, 'views': count}
            for tour_id, count in popular[:limit]
        ]
    
    def get_conversion_rate(self) -> Dict[str, Any]:
        """Calculate conversion rate (leads / sessions)"""
        with self.interactions_lock:
            sessions = set()
            leads = 0
            
            for interaction in self.interactions:
                sessions.add(interaction['session_id'])
                if interaction['event_type'] == 'lead_captured':
                    leads += 1
            
            total_sessions = len(sessions)
            conversion_rate = (leads / total_sessions * 100) if total_sessions > 0 else 0
            
            return {
                'total_sessions': total_sessions,
                'total_leads': leads,
                'conversion_rate': round(conversion_rate, 2)
            }
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents"""
        intent_counts = defaultdict(int)
        
        with self.interactions_lock:
            for interaction in self.interactions:
                if interaction['event_type'] == 'query':
                    intent = interaction['data'].get('intent', Intent.UNKNOWN)
                    intent_counts[intent] += 1
        
        return dict(intent_counts)

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429

# ==================== HEALTH & MONITORING ====================
@app.route('/api/health/deep', methods=['GET'])
def deep_health_check():
    """Comprehensive health check"""
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '6.0.0',
        'components': {}
    }
    
    # Check OpenAI
    health['components']['openai'] = {
        'configured': bool(Config.OPENAI_API_KEY),
        'status': 'ok' if Config.OPENAI_API_KEY else 'not_configured'
    }
    
    # Check knowledge
    health['components']['knowledge'] = {
        'tours_count': len(state.tours_db),
        'faqs_count': len(state.faqs),
        'status': 'ok' if state.tours_db else 'no_data'
    }
    
    # Check search engine
    search_engine = state.get_search_engine()
    health['components']['search'] = {
        'initialized': search_engine is not None,
        'has_index': search_engine.index is not None if search_engine else False,
        'has_vectors': search_engine.vectors is not None if search_engine else False,
        'status': 'ok' if search_engine else 'not_initialized'
    }
    
    # Check integrations
    health['components']['integrations'] = {
        'google_sheets': Config.ENABLE_GOOGLE_SHEETS,
        'meta_capi': Config.ENABLE_META_CAPI_LEAD,
        'status': 'ok'
    }
    
    # Overall status
    component_statuses = [c['status'] for c in health['components'].values()]
    if all(s == 'ok' for s in component_statuses):
        health['status'] = 'healthy'
    elif any(s == 'not_configured' or s == 'not_initialized' for s in component_statuses):
        health['status'] = 'degraded'
    else:
        health['status'] = 'unhealthy'
    
    status_code = 200 if health['status'] in ['healthy', 'degraded'] else 503
    return jsonify(health), status_code

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Prometheus-style metrics endpoint"""
    stats = state.get_stats()
    
    metrics_text = f"""# HELP rubywings_requests_total Total number of requests
# TYPE rubywings_requests_total counter
rubywings_requests_total {stats['requests']}

# HELP rubywings_errors_total Total number of errors
# TYPE rubywings_errors_total counter
rubywings_errors_total {stats['errors']}

# HELP rubywings_leads_total Total number of leads captured
# TYPE rubywings_leads_total counter
rubywings_leads_total {stats['leads']}

# HELP rubywings_sessions_active Number of active sessions
# TYPE rubywings_sessions_active gauge
rubywings_sessions_active {stats['sessions']}

# HELP rubywings_cache_hits_total Total cache hits
# TYPE rubywings_cache_hits_total counter
rubywings_cache_hits_total {stats['cache_hits']}

# HELP rubywings_cache_misses_total Total cache misses
# TYPE rubywings_cache_misses_total counter
rubywings_cache_misses_total {stats['cache_misses']}

# HELP rubywings_uptime_seconds Uptime in seconds
# TYPE rubywings_uptime_seconds gauge
rubywings_uptime_seconds {stats['uptime_seconds']}
"""
    
    return metrics_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}

# ==================== ADMIN ROUTES ====================
@app.route('/api/admin/clear-cache', methods=['POST'])
def admin_clear_cache():
    """Clear all caches (admin only)"""
    # TODO: Add authentication
    
    with state.embedding_cache_lock:
        state.embedding_cache.clear()
    
    logger.info("🗑️ All caches cleared")
    
    return jsonify({
        'success': True,
        'message': 'Caches cleared successfully'
    })

@app.route('/api/admin/reload-knowledge', methods=['POST'])
def admin_reload_knowledge():
    """Reload knowledge base (admin only)"""
    # TODO: Add authentication
    
    try:
        if load_knowledge():
            logger.info("♻️ Knowledge reloaded successfully")
            return jsonify({
                'success': True,
                'message': 'Knowledge reloaded successfully',
                'tours_count': len(state.tours_db),
                'faqs_count': len(state.faqs)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to reload knowledge'
            }), 500
    
    except Exception as e:
        logger.error(f"❌ Knowledge reload failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/sessions', methods=['GET'])
def admin_sessions():
    """Get active sessions info (admin only)"""
    # TODO: Add authentication
    
    with state.sessions_lock:
        sessions_info = []
        
        for session_id, session_data in state.sessions.items():
            sessions_info.append({
                'session_id': session_id[:16] + '...',  # Truncate for privacy
                'created_at': session_data['created_at'].isoformat(),
                'last_active': session_data['last_active'].isoformat(),
                'turn_count': session_data['turn_count'],
                'last_intent': session_data.get('last_intent', 'N/A'),
                'leads_captured': len(session_data.get('leads_captured', []))
            })
    
    return jsonify({
        'total_sessions': len(sessions_info),
        'sessions': sessions_info
    })

# ==================== WEBHOOK HANDLERS ====================
@app.route('/api/webhook/facebook', methods=['GET', 'POST'])
def facebook_webhook():
    """Facebook Messenger webhook"""
    # TODO: Implement Facebook Messenger integration
    if request.method == 'GET':
        # Verification
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        # Verify token (use environment variable)
        if mode == 'subscribe' and token == os.getenv('FB_VERIFY_TOKEN', 'rubywings'):
            return challenge, 200
        else:
            return 'Verification failed', 403
    
    elif request.method == 'POST':
        # Handle message
        # TODO: Implement message handling
        return jsonify({'status': 'ok'}), 200

@app.route('/api/webhook/zalo', methods=['POST'])
def zalo_webhook():
    """Zalo webhook"""
    # TODO: Implement Zalo integration
    return jsonify({'status': 'ok'}), 200

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == '__main__':
    if initialize_app():
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False
        )
    else:
        logger.error("❌ Application failed to initialize")
        sys.exit(1)

# ==================== MODULE EXPORTS FOR GUNICORN ====================
# CRITICAL: Lazy proxy pattern for search_engine + availability flags

class _SearchEngineProxy:
    """Lazy proxy for search_engine - init on first access"""
    def __getattr__(self, name):
        return getattr(state.get_search_engine(), name)
    
    @property
    def _loaded(self):
        engine = state.get_search_engine()
        return getattr(engine, '_loaded', False)
    
    def load_index(self):
        engine = state.get_search_engine()
        if hasattr(engine, 'load_index'):
            return engine.load_index()

search_engine = _SearchEngineProxy()

# Availability flags
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = Config.FAISS_ENABLED
except:
    FAISS_AVAILABLE = False

try:
    import numpy
    NUMPY_AVAILABLE = True
except:
    NUMPY_AVAILABLE = False

__all__ = ["app", "search_engine", "OPENAI_AVAILABLE", "FAISS_AVAILABLE", "NUMPY_AVAILABLE", "state"]