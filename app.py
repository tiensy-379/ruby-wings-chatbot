#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 5.2.2 FIXED
Created: 2025-01-17
Author: Ruby Wings AI Team

ARCHITECTURE:
- Fully compatible with Render 512MB RAM
- Ready to scale to 2GB RAM with env variables only
- State Machine for conversation flow
- Location Filter with region fallback
- Intent Detection with phone capture
- Meta CAPI tracking (FIXED & ENHANCED)
- FAISS/Numpy hybrid search
- Session management with auto-cleanup
- Enhanced error handling
- Better lead capture integration

ĐỒNG BỘ: entities.py, meta_capi.py, response_guard.py, gunicorn.conf.py,
         build_index.py, knowledge.json, .env variables from Render

CHANGES IN v5.2.2:
- Fixed Meta CAPI integration
- Enhanced lead capture with fallback storage
- Better error handling for Google Sheets
- Improved session management
- Fixed CORS configuration
- Enhanced logging
- Better cache management
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
logger = logging.getLogger("ruby-wings-v5.2.2")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration - ĐỒNG BỘ với .env từ Render"""
    
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
    
    # Feature Toggles (QUAN TRỌNG: Đồng bộ với Render env vars)
    FAISS_ENABLED = os.getenv("FAISS_ENABLED", "false").lower() == "true"
    ENABLE_INTENT_DETECTION = os.getenv("ENABLE_INTENT_DETECTION", "true").lower() == "true"
    ENABLE_PHONE_DETECTION = os.getenv("ENABLE_PHONE_DETECTION", "true").lower() == "true"
    ENABLE_LEAD_CAPTURE = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_LLM_FALLBACK = True  # Always enabled
    ENABLE_CACHING = True  # Always enabled
    ENABLE_GOOGLE_SHEETS = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_META_CAPI = os.getenv("ENABLE_META_CAPI_LEAD", "true").lower() == "true"
    ENABLE_META_CAPI_CALL = os.getenv("ENABLE_META_CAPI_CALL", "true").lower() == "true"
    ENABLE_FALLBACK_STORAGE = os.getenv("ENABLE_FALLBACK_STORAGE", "true").lower() == "true"
    
    # State Machine
    STATE_MACHINE_ENABLED = True
    ENABLE_LOCATION_FILTER = True
    ENABLE_SEMANTIC_ANALYSIS = True
    
    # Performance Settings (tối ưu cho 512MB)
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
    
    # CORS (FIXED)
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
        logger.info("🚀 RUBY WINGS CHATBOT v5.2.2 PRODUCTION (FIXED)")
        logger.info("=" * 60)
        logger.info(f"📊 RAM Profile: {cls.RAM_PROFILE}MB")
        logger.info(f"🌍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
        logger.info(f"🔧 Platform: {platform.system()}")
        
        # Features
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
        
        logger.info(f"🎯 Features: {', '.join(features)}")
        logger.info(f"🔑 OpenAI: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        
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
    
    # Fallback definitions
    class ConversationStage:
        EXPLORE = "explore"
        SUGGEST = "suggest"
        COMPARE = "compare"
        SELECT = "select"
        BOOK = "book"
        LEAD = "lead"
        CALLBACK = "callback"
    
    class Intent:
        GREETING = "greeting"
        FAREWELL = "farewell"
        TOUR_INQUIRY = "tour_inquiry"
        PRICE_ASK = "price_ask"
        BOOKING_REQUEST = "booking_request"
        PROVIDE_PHONE = "provide_phone"
        CALLBACK_REQUEST = "callback_request"
        UNKNOWN = "unknown"
    
    def detect_intent(text): 
        text_lower = text.lower()
        if any(w in text_lower for w in ['xin chào', 'chào', 'hello', 'hi']):
            return Intent.GREETING, 0.9, {}
        if any(w in text_lower for w in ['tạm biệt', 'bye', 'cảm ơn']):
            return Intent.FAREWELL, 0.9, {}
        if any(w in text_lower for w in ['giá', 'bao nhiêu', 'price', 'cost']):
            return Intent.PRICE_ASK, 0.8, {}
        if any(w in text_lower for w in ['đặt', 'book', 'đăng ký']):
            return Intent.BOOKING_REQUEST, 0.8, {}
        return Intent.UNKNOWN, 0.5, {}
    
    def detect_phone_number(text):
        # Simple VN phone detection
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
        return None
    
    def get_region_from_location(location): 
        return None
    
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
    
    # Dummy functions
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
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_LENGTH", "1048576"))  # 1MB
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# Apply ProxyFix for Render
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# CORS (FIXED)
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
    """Memory-optimized global state"""
    
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
        # Core data
        self.tours_db: Dict[int, Dict] = {}
        self.tour_name_index: Dict[str, int] = {}
        
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
            "start_time": datetime.now()
        }
        
        self._knowledge_loaded = False
        self._index_loaded = False
        
        logger.info("🌐 Global state initialized")
    
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
                    "mentioned_tours": [],
                    "selected_tour_id": None,
                    "location_filter": None,
                    "lead_phone": None,
                    "conversation_history": [],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                self.stats["sessions"] += 1
                
                # Cleanup old sessions if needed
                if len(self.session_contexts) > Config.MAX_SESSIONS:
                    self._cleanup_sessions()
            
            return self.session_contexts[session_id]
    
    def _cleanup_sessions(self):
        """Remove old sessions"""
        with self._lock:
            # Sort by last_updated
            sorted_sessions = sorted(
                self.session_contexts.items(),
                key=lambda x: x[1].get("last_updated", datetime.min)
            )
            
            # Remove oldest 30%
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
                # Check TTL
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
            
            # LRU eviction
            if len(self.response_cache) > Config.MAX_EMBEDDING_CACHE:
                self.response_cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        with self._lock:
            uptime = datetime.now() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": int(uptime.total_seconds()),
                "active_sessions": len(self.session_contexts),
                "tours_loaded": len(self.tours_db),
                "cache_size": len(self.response_cache)
            }

# Initialize global state
state = GlobalState()

# ==================== KNOWLEDGE LOADER ====================
def load_knowledge():
    """Load knowledge base"""
    if state._knowledge_loaded:
        return True
    
    try:
        logger.info(f"📚 Loading knowledge from {Config.KNOWLEDGE_PATH}")
        
        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.error(f"❌ Knowledge file not found: {Config.KNOWLEDGE_PATH}")
            return False
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        # Load tours
        tours_data = knowledge.get('tours', [])
        
        for idx, tour_data in enumerate(tours_data):
            try:
                state.tours_db[idx] = tour_data
                
                # Index by name
                name = tour_data.get('tour_name', '')
                if name:
                    state.tour_name_index[name.lower()] = idx
                    
            except Exception as e:
                logger.error(f"❌ Error loading tour {idx}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(state.tours_db)} tours")
        
        # Load mapping
        if os.path.exists(Config.FAISS_MAPPING_PATH):
            with open(Config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                state.mapping = json.load(f)
            logger.info(f"✅ Loaded {len(state.mapping)} mapping entries")
        else:
            # Create simple mapping from tours
            state.mapping = []
            for idx, tour in state.tours_db.items():
                # Add key fields to mapping
                for field in ['tour_name', 'location', 'duration', 'price', 'summary', 'includes', 'style']:
                    value = tour.get(field, '')
                    if value:
                        if isinstance(value, list):
                            value = ' '.join(str(v) for v in value)
                        value_str = str(value).strip()
                        if value_str:
                            state.mapping.append({
                                "path": f"tours[{idx}].{field}",
                                "text": value_str,
                                "tour_index": idx
                            })
            logger.info(f"📝 Created {len(state.mapping)} mapping entries")
        
        state._knowledge_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        traceback.print_exc()
        return False

# ==================== SEARCH ENGINE ====================
class SearchEngine:
    """Unified search engine"""
    
    def __init__(self):
        self.openai_client = None
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(
                    api_key=Config.OPENAI_API_KEY
                )
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")
    
    def load_index(self):
        """Load search index"""
        if state._index_loaded:
            return True
        
        try:
            # Try FAISS first
            if Config.FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info("📦 Loading FAISS index...")
                state.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS loaded: {state.index.ntotal} vectors")
                state._index_loaded = True
                return True
            
            # Try numpy fallback
            if NUMPY_AVAILABLE and os.path.exists(Config.FALLBACK_VECTORS_PATH):
                logger.info("📦 Loading numpy vectors...")
                data = np.load(Config.FALLBACK_VECTORS_PATH)
                
                if 'mat' in data:
                    state.vectors = data['mat']
                elif 'vectors' in data:
                    state.vectors = data['vectors']
                else:
                    first_key = list(data.keys())[0]
                    state.vectors = data[first_key]
                
                # Normalize
                if state.vectors is not None:
                    norms = np.linalg.norm(state.vectors, axis=1, keepdims=True)
                    state.vectors = state.vectors / (norms + 1e-12)
                
                logger.info(f"✅ Numpy loaded: {state.vectors.shape[0]} vectors")
                state._index_loaded = True
                return True
            
            logger.warning("⚠️ No index found, using text search")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        if not text:
            return None
        
        # Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=text[:2000]  # Truncate
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
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[float, Dict]]:
        """Search for relevant passages"""
        if top_k is None:
            top_k = Config.TOP_K
        
        # Get query embedding
        embedding = self.get_embedding(query)
        if not embedding:
            return self._text_search(query, top_k)
        
        # FAISS search
        if state.index is not None and FAISS_AVAILABLE:
            try:
                query_vec = np.array([embedding], dtype='float32')
                scores, indices = state.index.search(query_vec, top_k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(state.mapping):
                        results.append((float(score), state.mapping[idx]))
                
                return results
            except Exception as e:
                logger.error(f"FAISS search error: {e}")
        
        # Numpy search
        if state.vectors is not None and NUMPY_AVAILABLE:
            try:
                query_vec = np.array([embedding], dtype='float32')
                query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
                
                similarities = np.dot(state.vectors, query_norm.T).flatten()
                top_indices = np.argsort(-similarities)[:top_k]
                
                results = []
                for idx in top_indices:
                    if 0 <= idx < len(state.mapping):
                        results.append((float(similarities[idx]), state.mapping[idx]))
                
                return results
            except Exception as e:
                logger.error(f"Numpy search error: {e}")
        
        # Text fallback
        return self._text_search(query, top_k)
    
    def _text_search(self, query: str, top_k: int) -> List[Tuple[float, Dict]]:
        """Simple text-based search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for entry in state.mapping[:200]:  # Limit for performance
            text = entry.get('text', '').lower()
            
            score = 0
            for word in query_words:
                if len(word) > 2 and word in text:
                    score += 1
            
            if score > 0:
                results.append((float(score), entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

# Initialize search engine
search_engine = SearchEngine()

# ==================== RESPONSE GENERATOR ====================
class ResponseGenerator:
    """Generate responses"""
    
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
        """Generate response"""
        
        # Handle special intents
        intent = context.get("intent", Intent.UNKNOWN)
        
        if intent == Intent.GREETING or intent == "GREETING":
            return random.choice([
                "Xin chào! Tôi là trợ lý AI của Ruby Wings. Rất vui được hỗ trợ bạn! 😊\n\nBạn muốn tìm hiểu về tour nào?",
                "Chào bạn! Tôi có thể giúp gì cho bạn về các tour Ruby Wings? 🌿"
            ])
        
        if intent == Intent.FAREWELL or intent == "FAREWELL":
            return random.choice([
                "Cảm ơn bạn! Chúc một ngày tuyệt vời! ✨",
                "Tạm biệt! Liên hệ **0332510486** nếu cần hỗ trợ nhé! 👋"
            ])
        
        # Check if we have results
        if not search_results:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Vui lòng liên hệ hotline **0332510486** để được tư vấn! 📞"
        
        # Build response from search results
        response = "Dựa trên yêu cầu của bạn, tôi tìm thấy:\n\n"
        
        # Group by tour
        tours_mentioned = set()
        for score, entry in search_results[:Config.MAX_TOURS_PER_RESPONSE]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour:
                    tours_mentioned.add(tour_idx)
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
        
        response += "Bạn muốn biết thêm chi tiết gì? Hoặc liên hệ **0332510486** để đặt tour! 😊"
        
        return response

# Initialize response generator
response_gen = ResponseGenerator()

# ==================== CHAT PROCESSOR ====================
class ChatProcessor:
    """Main chat processing engine"""
    
    def __init__(self):
        self.response_generator = response_gen
        self.search_engine = search_engine
    
    def process(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Process user message"""
        start_time = time.time()
        
        try:
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
            
            # Detect intent
            intent, confidence, metadata = detect_intent(user_message)
            context['intent'] = intent.name if hasattr(intent, 'name') else str(intent)
            
            # Detect phone number
            phone = metadata.get('phone_number') or detect_phone_number(user_message)
            if phone:
                context['lead_phone'] = phone
                context['stage'] = ConversationStage.LEAD
                
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
            
            # Search for relevant information
            search_results = self.search_engine.search(user_message, Config.TOP_K)
            
            # Extract mentioned tours
            mentioned_tours = []
            for score, entry in search_results:
                tour_idx = entry.get('tour_index')
                if tour_idx is not None and tour_idx not in mentioned_tours:
                    mentioned_tours.append(tour_idx)
            
            context['mentioned_tours'] = mentioned_tours
            
            # Generate response
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
            
            # Build result
            result = {
                'reply': response_text,
                'session_id': session_id,
                'session_state': {
                    'stage': context.get('stage'),
                    'intent': context.get('intent'),
                    'mentioned_tours': mentioned_tours,
                    'has_phone': bool(phone)
                },
                'intent': {
                    'name': intent.name if hasattr(intent, 'name') else str(intent),
                    'confidence': confidence
                },
                'search': {
                    'results_count': len(search_results),
                    'tours': mentioned_tours
                },
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'from_cache': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            state.cache_response(cache_key, result)
            
            # Update stats
            state.stats['requests'] += 1
            
            logger.info(f"⏱️ Processed in {result['processing_time_ms']}ms | "
                       f"Intent: {intent.name if hasattr(intent, 'name') else intent} | "
                       f"Stage: {context.get('stage')} | "
                       f"Results: {len(search_results)}")
            
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
        
        transitions = {
            ConversationStage.EXPLORE: {
                'TOUR_INQUIRY': ConversationStage.SUGGEST,
                'PRICE_ASK': ConversationStage.SUGGEST,
                'PROVIDE_PHONE': ConversationStage.LEAD,
                'CALLBACK_REQUEST': ConversationStage.CALLBACK
            },
            ConversationStage.SUGGEST: {
                'BOOKING_REQUEST': ConversationStage.SELECT,
                'PROVIDE_PHONE': ConversationStage.LEAD
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
        return next_stages.get(intent_name, current_stage)
    
    def _capture_lead(self, phone: str, session_id: str, message: str, context: Dict):
        """Capture lead data"""
        try:
            # Clean phone
            phone_clean = re.sub(r'[^\d+]', '', phone)
            
            # Create lead data following LeadData.to_row() (13 columns)
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
            
            # FIXED: 13 columns with str() cast
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
            # Don't raise - fallback storage will catch it
    
    def _save_to_fallback(self, lead_data: Dict):
        """Save to fallback JSON file"""
        try:
            # Load existing
            if os.path.exists(Config.FALLBACK_STORAGE_PATH):
                with open(Config.FALLBACK_STORAGE_PATH, 'r', encoding='utf-8') as f:
                    leads = json.load(f)
            else:
                leads = []
            
            # Add new lead
            leads.append(lead_data)
            
            # Keep only last 1000
            leads = leads[-1000:]
            
            # Save
            with open(Config.FALLBACK_STORAGE_PATH, 'w', encoding='utf-8') as f:
                json.dump(leads, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Saved to fallback storage")
            
        except Exception as e:
            logger.error(f"Fallback storage error: {e}")

# Initialize chat processor
chat_processor = ChatProcessor()

# ==================== ROUTES ====================
@app.before_request
def before_request():
    """Before request handler"""
    g.start_time = time.time()
    
    # Track pageview for Meta CAPI
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
    # Add processing time header
    if hasattr(g, 'start_time'):
        elapsed = (time.time() - g.start_time) * 1000
        response.headers['X-Processing-Time'] = f"{elapsed:.2f}ms"
    
    return response

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '5.2.2',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'openai': OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
            'entities': ENTITIES_AVAILABLE,
            'meta_capi': META_CAPI_AVAILABLE,
            'response_guard': RESPONSE_GUARD_AVAILABLE,
            'faiss': FAISS_AVAILABLE,
            'numpy': NUMPY_AVAILABLE
        },
        'knowledge': {
            'loaded': state._knowledge_loaded,
            'tours': len(state.tours_db)
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Index route"""
    return jsonify({
        'service': 'Ruby Wings AI Chatbot',
        'version': '5.2.2',
        'status': 'running',
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
        
        # Process message
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
    """Legacy /chat endpoint - backward compatible"""
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
        
        # Extract data
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
        
        # Clean phone
        phone_clean = re.sub(r'[^\d+]', '', phone)
        
        # Validate phone
        if not re.match(r'^(0|\+?84)\d{9,10}$', phone_clean):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Create lead data (13 columns)
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
                    
                    # FIXED: 13 columns
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
        
        # Update stats
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
        
        # Send to Meta CAPI
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
        # Reset flags
        state._knowledge_loaded = False
        state._index_loaded = False
        
        # Reload
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
        logger.info("🚀 Initializing Ruby Wings Chatbot v5.2.2...")
        
        # Log configuration
        Config.log_config()
        
        # Load knowledge
        if not load_knowledge():
            logger.warning("⚠️ Knowledge base not loaded, continuing anyway")
        
        # Load search index
        if not search_engine.load_index():
            logger.warning("⚠️ Search index not loaded, using text search")
        
        # Check integrations
        if META_CAPI_AVAILABLE and Config.ENABLE_META_CAPI:
            logger.info("✅ Meta CAPI ready")
        else:
            logger.warning("⚠️ Meta CAPI not available")
        
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            logger.info("✅ OpenAI ready")
        else:
            logger.warning("⚠️ OpenAI not available, using fallback")
        
        logger.info("=" * 60)
        logger.info("✅ RUBY WINGS CHATBOT READY!")
        logger.info(f"🌐 Server: {Config.HOST}:{Config.PORT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        traceback.print_exc()

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == '__main__':
    # Development mode
    initialize_app()
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False
    )
else:
    # Production mode (Gunicorn)
    initialize_app()