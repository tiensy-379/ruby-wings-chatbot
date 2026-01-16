#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUBY WINGS AI CHATBOT - PRODUCTION VERSION 5.2
Created: 2025-01-16
Author: Ruby Wings AI Team

ARCHITECTURE:
- Fully compatible with Render 512MB RAM
- Ready to scale to 2GB RAM with env variables only
- State Machine for conversation flow
- Location Filter with region fallback
- Intent Detection with phone capture
- Meta CAPI tracking
- FAISS/Numpy hybrid search
- Session management with auto-cleanup

ĐỒNG BỘ: entities.py, meta_capi.py, response_guard.py, gunicorn.conf.py,
         build_index.py, knowledge.json, .env.example.ini
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
from flask import Flask, request, jsonify, g
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
logger = logging.getLogger("ruby-wings-v5.2")

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
    SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
    
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
    ENABLE_INTENT_DETECTION = os.getenv("UPGRADE_7_STATE_MACHINE", "true").lower() == "true"
    ENABLE_PHONE_DETECTION = os.getenv("UPGRADE_7_STATE_MACHINE", "true").lower() == "true"
    ENABLE_LEAD_CAPTURE = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_LLM_FALLBACK = True  # Always enabled
    ENABLE_CACHING = os.getenv("CACHE_TTL_SECONDS", "300") != ""
    ENABLE_GOOGLE_SHEETS = os.getenv("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
    ENABLE_META_CAPI = os.getenv("ENABLE_META_CAPI_LEAD", "true").lower() == "true"
    ENABLE_META_CAPI_CALL = os.getenv("ENABLE_META_CAPI_CALL", "true").lower() == "true"
    ENABLE_FALLBACK_STORAGE = os.getenv("ENABLE_FALLBACK_STORAGE", "true").lower() == "true"
    
    # State Machine
    STATE_MACHINE_ENABLED = os.getenv("UPGRADE_7_STATE_MACHINE", "true").lower() == "true"
    ENABLE_LOCATION_FILTER = os.getenv("UPGRADE_1_MANDATORY_FILTER", "true").lower() == "true"
    ENABLE_SEMANTIC_ANALYSIS = os.getenv("UPGRADE_8_SEMANTIC_ANALYSIS", "true").lower() == "true"
    
    # Performance Settings (tối ưu cho 512MB)
    TOP_K = int(os.getenv("TOP_K", "5" if IS_LOW_RAM else "10"))
    MAX_TOURS_PER_RESPONSE = 3
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "180" if IS_LOW_RAM else "300"))
    MAX_SESSIONS = 50 if IS_LOW_RAM else 100
    MAX_EMBEDDING_CACHE = 30 if IS_LOW_RAM else 50
    CONVERSATION_HISTORY_LIMIT = 5 if IS_LOW_RAM else 10
    
    # Server Config
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "10000"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    
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
        logger.info("🚀 RUBY WINGS CHATBOT v5.2 PRODUCTION")
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
            features.append("Meta CAPI")
        if cls.ENABLE_GOOGLE_SHEETS:
            features.append("Google Sheets")
        
        logger.info(f"🎯 Features: {', '.join(features)}")
        logger.info(f"🔑 OpenAI: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        logger.info(f"📞 Meta Pixel: {cls.META_PIXEL_ID[:6]}...{cls.META_PIXEL_ID[-4:] if len(cls.META_PIXEL_ID) > 10 else ''}")
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
except ImportError as e:
    logger.error(f"❌ Failed to import entities.py: {e}")
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
    
    def detect_intent(text): return Intent.UNKNOWN, 0.5, {}
    def detect_phone_number(text): return None
    def extract_location_from_query(text): return None
    def get_region_from_location(location): return None

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
except ImportError as e:
    logger.warning(f"⚠️ meta_capi.py not available: {e}")
    META_CAPI_AVAILABLE = False
    
    # Dummy functions
    def send_meta_pageview(request): pass
    def send_meta_lead(*args, **kwargs): return None
    def send_meta_lead_from_entities(*args, **kwargs): return None
    def send_meta_call_button(*args, **kwargs): return None
    def check_meta_capi_health(): return {"status": "unavailable"}

try:
    from response_guard import validate_and_format_answer
    RESPONSE_GUARD_AVAILABLE = True
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
app.secret_key = Config.SECRET_KEY or os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_LENGTH", "16777216"))
app.config['JSON_AS_ASCII'] = False

# Apply ProxyFix for Render
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# CORS
if Config.CORS_ORIGINS == "*":
    CORS(app, origins="*")
else:
    CORS(app, origins=Config.CORS_ORIGINS.split(","))

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
                "uptime_seconds": uptime.total_seconds(),
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
                for field in ['tour_name', 'location', 'duration', 'price', 'summary']:
                    value = tour.get(field, '')
                    if value and str(value).strip():
                        state.mapping.append({
                            "path": f"tours[{idx}].{field}",
                            "text": str(value),
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
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL
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
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL
                )
            except Exception as e:
                logger.error(f"LLM client init failed: {e}")
    
    def generate(self, user_message: str, search_results: List, context: Dict) -> str:
        """Generate response"""
        
        # Handle special intents
        intent = context.get("intent", Intent.UNKNOWN)
        
        if intent == Intent.GREETING:
            return random.choice([
                "Xin chào! Tôi là trợ lý AI của Ruby Wings. Rất vui được hỗ trợ bạn! 😊",
                "Chào bạn! Tôi có thể giúp gì cho bạn về các tour Ruby Wings? 🌿"
            ])
        
        if intent == Intent.FAREWELL:
            return random.choice([
                "Cảm ơn bạn! Chúc một ngày tuyệt vời! ✨",
                "Tạm biệt! Liên hệ **0332510486** nếu cần hỗ trợ nhé! 👋"
            ])
        
        # Check if we have results
        if not search_results:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Vui lòng liên hệ hotline **0332510486**! 📞"
        
        # Build response from search results
        response = "Tôi tìm thấy thông tin sau:\n\n"
        
        # Group by tour
        tours_mentioned = set()
        for score, entry in search_results[:3]:
            tour_idx = entry.get('tour_index')
            if tour_idx is not None and tour_idx not in tours_mentioned:
                tour = state.get_tour(tour_idx)
                if tour:
                    tours_mentioned.add(tour_idx)
                    response += f"**{tour.get('tour_name', 'Tour')}**\n"
                    
                    if tour.get('location'):
                        response += f"   📍 {tour['location']}\n"
                    if tour.get('duration'):
                        response += f"   ⏱️ {tour['duration']}\n"
                    if tour.get('price'):
                        price = tour['price']
                        if len(price) > 80:
                            price = price[:80] + "..."
                        response += f"   💰 {price}\n"
                    response += "\n"
        
        response += "Bạn muốn biết thêm chi tiết gì? 😊"
        
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
                guarded = validate_and_format_answer(
                    response_text,
                    [(s, e) for s, e in search_results],
                    context=context
                )
                response_text = guarded.get('answer', response_text)
            
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
            # Create lead data
            lead_data = {
                'phone': phone,
                'session_id': session_id,
                'message': message[:200],
                'stage': context.get('stage'),
                'mentioned_tours': context.get('mentioned_tours', []),
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to Meta CAPI
            if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
                try:
                    send_meta_lead(
                        request,
                        phone=phone,
                        content_name="Chatbot Lead Capture",
                        value=200000,
                        currency="VND"
                    )
                    logger.info(f"✅ Lead sent to Meta: {phone[:4]}***")
                except Exception as e:
                    logger.error(f"Meta CAPI lead error: {e}")
            
            # Save to Google Sheets
            if Config.ENABLE_GOOGLE_SHEETS:
                self._save_to_sheets(lead_data)
            
            # Fallback storage
            if Config.ENABLE_FALLBACK_STORAGE:
                self._save_to_fallback(lead_data)
            
            # Update stats
            state.stats['leads'] += 1
            
            logger.info(f"📞 Lead captured: {phone[:4]}***{phone[-2:]}")
            
        except Exception as e:
            logger.error(f"Lead capture error: {e}")
    
    def _save_to_sheets(self, lead_data: Dict):
        """Save to Google Sheets"""
        try:
            if not Config.GOOGLE_SERVICE_ACCOUNT_JSON:
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
            
            # Prepare row
            row = [
                lead_data.get('timestamp', ''),
                lead_data.get('phone', ''),
                lead_data.get('session_id', ''),
                lead_data.get('message', ''),
                lead_data.get('stage', ''),
                ', '.join(map(str, lead_data.get('mentioned_tours', [])))
            ]
            
            ws.append_row(row)
            logger.info("✅ Saved to Google Sheets")
            
        except Exception as e:
            logger.error(f"Google Sheets error: {e}")
    
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

# ==================== FLASK MIDDLEWARE ====================
@app.before_request
def before_request():
    """Before request handler"""
    g.start_time = time.time()
    g.request_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]
    
    # Meta CAPI pageview
    if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
        try:
            send_meta_pageview(request)
        except Exception:
            pass

@app.after_request
def after_request(response):
    """After request handler"""
    # Add headers
    if hasattr(g, 'start_time'):
        processing_time = time.time() - g.start_time
        response.headers['X-Processing-Time'] = f'{processing_time:.3f}s'
    
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    # CORS
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

# ==================== API ENDPOINTS ====================
@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'ok',
        'version': '5.2',
        'service': 'Ruby Wings AI Chatbot',
        'timestamp': datetime.now().isoformat(),
        'system': {
            'ram_profile': Config.RAM_PROFILE,
            'environment': 'production' if IS_PRODUCTION else 'development',
            'platform': platform.system()
        },
        'features': {
            'state_machine': Config.STATE_MACHINE_ENABLED,
            'location_filter': Config.ENABLE_LOCATION_FILTER,
            'intent_detection': Config.ENABLE_INTENT_DETECTION,
            'phone_detection': Config.ENABLE_PHONE_DETECTION,
            'meta_capi': Config.ENABLE_META_CAPI,
            'google_sheets': Config.ENABLE_GOOGLE_SHEETS,
            'faiss': Config.FAISS_ENABLED and FAISS_AVAILABLE
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'knowledge_base': state._knowledge_loaded,
            'search_index': state._index_loaded,
            'openai': OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
            'meta_capi': META_CAPI_AVAILABLE
        },
        'stats': state.get_stats()
    }
    
    # Check if all critical components are ok
    critical_ok = health_data['components']['knowledge_base']
    health_data['status'] = 'healthy' if critical_ok else 'degraded'
    
    return jsonify(health_data)

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        
        if not user_message:
            return jsonify({
                'reply': 'Xin chào! Tôi có thể giúp gì cho bạn về các tour Ruby Wings? 🌿',
                'session_id': 'new',
                'error': 'Empty message'
            }), 400
        
        # Get or generate session ID
        session_id = data.get('session_id', '')
        if not session_id:
            ip = request.remote_addr or '0.0.0.0'
            timestamp = int(time.time() / 60)
            session_id = hashlib.md5(f"{ip}_{timestamp}".encode()).hexdigest()[:12]
        
        # Process message
        result = chat_processor.process(user_message, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({
            'reply': 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại! 🙏',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/save-lead', methods=['POST', 'OPTIONS'])
def save_lead():
    """Save lead endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json() or {}
        
        phone = data.get('phone', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        tour_interest = data.get('tour_interest', '').strip()
        
        if not phone:
            return jsonify({'error': 'Phone required'}), 400
        
        # Validate phone
        phone_clean = detect_phone_number(phone)
        if not phone_clean:
            return jsonify({'error': 'Invalid phone format'}), 400
        
        # Create lead
        lead_data = {
            'phone': phone_clean,
            'name': name,
            'email': email,
            'tour_interest': tour_interest,
            'source': 'Lead Form',
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to Meta CAPI
        if Config.ENABLE_META_CAPI and META_CAPI_AVAILABLE:
            try:
                send_meta_lead(
                    request,
                    phone=phone_clean,
                    contact_name=name,
                    content_name=f"Tour: {tour_interest}" if tour_interest else "General Inquiry",
                    value=200000,
                    currency="VND"
                )
            except Exception as e:
                logger.error(f"Meta CAPI error: {e}")
        
        # Save to Google Sheets
        if Config.ENABLE_GOOGLE_SHEETS:
            try:
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
                    lead_data['timestamp'],
                    phone_clean,
                    name,
                    email,
                    tour_interest,
                    'Lead Form'
                ]
                
                ws.append_row(row)
            except Exception as e:
                logger.error(f"Google Sheets error: {e}")
        
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
                send_meta_call_button(
                    request,
                    page_url=page_url,
                    call_type=call_type,
                    button_location='fixed_bottom_left',
                    button_text='Gọi ngay'
                )
                logger.info(f"📞 Call button tracked: {call_type}")
            except Exception as e:
                logger.error(f"Meta CAPI call error: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Call tracked',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Call button error: {e}")
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
        return jsonify({'error': str(e)}), 500

@app.route('/meta-health', methods=['GET'])
def meta_health():
    """Meta CAPI health check"""
    if META_CAPI_AVAILABLE:
        return jsonify(check_meta_capi_health())
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
        logger.info("🚀 Initializing Ruby Wings Chatbot v5.2...")
        
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