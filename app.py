#!/usr/bin/env python3
"""
Ruby Wings Chatbot v4.2 - Production-Grade Hybrid AI Assistant
=====================================================================
A sophisticated travel chatbot combining deterministic business logic with 
OpenAI-powered natural language understanding and generation.

Total Lines: 4,500+
Language: Python 3.8+
Dependencies: Flask, OpenAI, FAISS, NumPy, Gspread, Google Auth

Author: Ruby Wings Team
Last Updated: 2026-01-23
"""

# ============================================================================
# SECTION 1: IMPORTS & DEPENDENCIES (Lines 1-120)
# ============================================================================

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
import random
import pickle
import base64
from functools import lru_cache, wraps
from typing import (
    List, Tuple, Dict, Optional, Any, Set, Union, Callable, 
    Generator, DefaultDict, Deque
)
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher
from enum import Enum
from abc import ABC, abstractmethod
import inspect

# Flask & Web Framework
from flask import Flask, request, jsonify, g, Response
from flask_cors import CORS

# Data Science & Embeddings
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# OpenAI Integration
try:
    from openai import OpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AzureOpenAI = None

# Google Cloud Integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    from google.auth.exceptions import GoogleAuthError
    from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
    HAS_GOOGLE_SHEETS = True
except ImportError:
    HAS_GOOGLE_SHEETS = False

# HTTP Client
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None

# ============================================================================
# SECTION 2: CONFIGURATION & ENVIRONMENT (Lines 121-250)
# ============================================================================

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ruby_wings_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rbw_v4.2")

# Environment Variables
class Config:
    """Centralized configuration management"""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini").strip()
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    
    # Knowledge & Index Settings
    KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json").strip()
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin").strip()
    FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json").strip()
    FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz").strip()
    TOUR_ENTITIES_PATH = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json").strip()
    
    # Search Settings
    TOP_K = int(os.environ.get("TOP_K", "10"))
    SEMANTIC_THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD", "0.78"))
    
    # Memory & Performance
    RAM_PROFILE = os.environ.get("RAM_PROFILE", "2048").strip()
    IS_LOW_RAM = RAM_PROFILE == "512"
    IS_HIGH_RAM = RAM_PROFILE == "2048"
    MAX_CACHE_SIZE = 1000 if IS_HIGH_RAM else 500
    
    # Feature Flags
    FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")
    ENABLE_CACHING = os.environ.get("ENABLE_CACHING", "true").lower() in ("1", "true", "yes")
    ENABLE_SEMANTIC_SEARCH = os.environ.get("ENABLE_SEMANTIC_SEARCH", "true").lower() in ("1", "true", "yes")
    ENABLE_DEDUPLICATION = os.environ.get("ENABLE_DEDUPLICATION", "true").lower() in ("1", "true", "yes")
    
    # Google Sheets
    GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "1SdVbwkuxb8l1meEW--ddyfh4WmUvSXXMOPQ5bCyPkdk").strip()
    GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox").strip()
    GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "true").lower() in ("1", "true", "yes")
    
    # Meta CAPI
    META_CAPI_TOKEN = os.environ.get("META_CAPI_TOKEN", "").strip()
    META_PIXEL_ID = os.environ.get("META_PIXEL_ID", "").strip()
    ENABLE_META_CAPI = os.environ.get("ENABLE_META_CAPI", "true").lower() in ("1", "true", "yes")
    
    # Server Settings
    FLASK_ENV = os.environ.get("FLASK_ENV", "production").strip()
    DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
    SECRET_KEY = os.environ.get("SECRET_KEY", "ruby-wings-secret-2026").strip()
    HOST = os.environ.get("HOST", "0.0.0.0").strip()
    PORT = int(os.environ.get("PORT", "10000"))
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "https://www.rubywings.vn,http://localhost:3000").split(",")
    
    # Session Management
    SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "1800"))  # 30 minutes
    MAX_SESSION_SIZE = 100
    
    # Timeouts
    OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "30"))
    GOOGLE_SHEETS_TIMEOUT = int(os.environ.get("GOOGLE_SHEETS_TIMEOUT", "10"))
    
    # Validation
    MIN_MESSAGE_LENGTH = 2
    MAX_MESSAGE_LENGTH = 5000
    MAX_RESPONSE_LENGTH = 3000
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        if not cls.OPENAI_API_KEY:
            logger.warning("⚠️ OPENAI_API_KEY not configured - AI features disabled")
        if not cls.KNOWLEDGE_PATH or not os.path.exists(cls.KNOWLEDGE_PATH):
            logger.warning(f"⚠️ Knowledge base not found: {cls.KNOWLEDGE_PATH}")
        logger.info(f"✅ Configuration loaded | RAM: {cls.RAM_PROFILE}MB | Env: {cls.FLASK_ENV}")


# ============================================================================
# SECTION 3: DATA MODELS & DATACLASSES (Lines 251-600)
# ============================================================================

class ConversationState(Enum):
    """Conversation state enumeration"""
    INITIAL = "initial"
    GREETING = "greeting"
    ASKING_DETAILS = "asking_details"
    TOUR_SELECTED = "tour_selected"
    COMPARING = "comparing"
    RECOMMENDATION = "recommendation"
    BOOKING = "booking"
    FAREWELL = "farewell"


class QuestionType(Enum):
    """Question type classification"""
    GREETING = "greeting"
    INFORMATION = "information"
    LISTING = "listing"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    CALCULATION = "calculation"
    COMPLEX = "complex"
    FAREWELL = "farewell"
    SERVICE_INQUIRY = "service_inquiry"
    LOCATION_QUERY = "location_query"
    PRICE_INQUIRY = "price_inquiry"
    BOOKING_INFO = "booking_info"


class PriceLevel(Enum):
    """Price level categories"""
    BUDGET = "budget"          # < 1.5M VND
    MIDRANGE = "midrange"      # 1.5M - 3M VND
    PREMIUM = "premium"        # > 3M VND


class DurationType(Enum):
    """Tour duration categories"""
    ONE_DAY = "1_day"
    TWO_DAYS = "2_days"
    THREE_DAYS = "3_days"
    WEEK = "week"


@dataclass
class Tour:
    """Tour entity with comprehensive metadata"""
    index: int
    name: Optional[str] = None
    summary: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    price: Optional[str] = None
    includes: List[str] = field(default_factory=list)
    accommodation: Optional[str] = None
    meals: Optional[str] = None
    transport: Optional[str] = None
    notes: Optional[str] = None
    style: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    def is_complete(self):
        """Check if tour has essential information"""
        essential_fields = [self.name, self.duration, self.location, self.price]
        return all(essential_fields) and self.completeness_score >= 0.6


@dataclass
class UserProfile:
    """User profile for personalization"""
    age_group: Optional[str] = None
    group_type: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    budget_level: Optional[str] = None
    physical_level: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    
    def update(self, **kwargs):
        """Update profile fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class FilterSet:
    """Mandatory filter criteria"""
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    location: Optional[str] = None
    near_location: Optional[str] = None
    group_type: Optional[str] = None
    month: Optional[int] = None
    weekend: bool = False
    holiday: Optional[str] = None
    accessibility: Optional[str] = None
    
    def is_empty(self) -> bool:
        """Check if no filters are set"""
        return all(v is None or v is False for v in asdict(self).values())
    
    def to_dict(self):
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None and v is not False}
    
    def __str__(self):
        """String representation"""
        if self.is_empty():
            return "No filters"
        parts = []
        if self.price_min or self.price_max:
            parts.append(f"Price: {self.price_min or '0'} - {self.price_max or 'unlimited'}")
        if self.duration_min or self.duration_max:
            parts.append(f"Duration: {self.duration_min or '1'} - {self.duration_max or '∞'} days")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.group_type:
            parts.append(f"Group: {self.group_type}")
        return " | ".join(parts)


@dataclass
class ConversationContext:
    """Conversation session context"""
    session_id: str
    user_profile: UserProfile = field(default_factory=UserProfile)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_tours: List[int] = field(default_factory=list)
    last_successful_tours: List[int] = field(default_factory=list)
    mentioned_tours: Set[int] = field(default_factory=set)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    filters: FilterSet = field(default_factory=FilterSet)
    state: ConversationState = ConversationState.INITIAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    interaction_count: int = 0
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update context with new interaction"""
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if tour_indices:
            self.current_tours = tour_indices
            self.last_successful_tours = tour_indices
            self.mentioned_tours.update(tour_indices)
        
        self.last_updated = datetime.utcnow()
        self.interaction_count += 1
        
        # Keep history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-30:]
    
    def is_expired(self, timeout_seconds: int = 1800) -> bool:
        """Check if session expired"""
        age = (datetime.utcnow() - self.last_updated).total_seconds()
        return age > timeout_seconds


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int = 300
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if expired"""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


@dataclass
class ChatResponse:
    """Structured chat response"""
    reply: str
    sources: List[str] = field(default_factory=list)
    tour_indices: List[int] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    from_memory: bool = False
    confidence: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'reply': self.reply,
            'sources': self.sources,
            'tour_indices': self.tour_indices,
            'context': self.context,
            'processing_time_ms': self.processing_time_ms,
            'from_memory': self.from_memory,
            'confidence': self.confidence
        }


@dataclass
class SearchResult:
    """Search result with metadata"""
    tour_index: int
    relevance_score: float
    match_type: str
    reasoning: List[str] = field(default_factory=list)


# ============================================================================
# SECTION 4: JSON ENCODER & UTILITIES (Lines 601-750)
# ============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for custom types"""
    
    def default(self, obj):
        if isinstance(obj, (datetime, timedelta)):
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)


def safe_json_dumps(obj, **kwargs):
    """Safe JSON serialization"""
    try:
        return json.dumps(obj, cls=EnhancedJSONEncoder, **kwargs)
    except Exception as e:
        logger.error(f"JSON serialization error: {e}")
        return json.dumps({"error": str(e)})


def safe_json_loads(data):
    """Safe JSON deserialization"""
    try:
        return json.loads(data)
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return {}


def normalize_text_simple(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text"""
    if not text:
        return []
    numbers = re.findall(r'\d+', text)
    return [int(n) for n in numbers]


def sanitize_user_input(text: str) -> str:
    """Sanitize user input for security"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters except common punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\'"\']', '', text)
    
    # Limit length
    text = text[:Config.MAX_MESSAGE_LENGTH]
    
    return text.strip()


# ============================================================================
# SECTION 5: FLASK APP INITIALIZATION (Lines 751-850)
# ============================================================================

# Validate configuration
Config.validate()

# Create Flask app
app = Flask(__name__)
app.json_encoder = EnhancedJSONEncoder
app.config['JSON_ENCODER_CLASS'] = EnhancedJSONEncoder
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# CORS Configuration
CORS(
    app,
    origins=Config.CORS_ORIGINS,
    supports_credentials=True,
    allow_headers=['Content-Type', 'Authorization'],
    methods=['GET', 'POST', 'OPTIONS']
)

# Request context
@app.before_request
def before_request():
    """Setup request context"""
    g.start_time = time.time()
    g.request_id = hashlib.md5(
        f"{request.remote_addr}_{time.time()}".encode()
    ).hexdigest()[:12]


@app.after_request
def after_request(response):
    """Cleanup request context"""
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        response.headers['X-Process-Time'] = str(elapsed)
    return response


# ============================================================================
# SECTION 6: OPENAI CLIENT MANAGEMENT (Lines 851-950)
# ============================================================================

class OpenAIManager:
    """Singleton manager for OpenAI API"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.client = None
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        if Config.OPENAI_API_KEY and HAS_OPENAI:
            try:
                http_client = None
                if HAS_HTTPX:
                    http_client = httpx.Client(
                        timeout=Config.OPENAI_TIMEOUT,
                        follow_redirects=True
                    )
                
                self.client = OpenAI(
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL,
                    http_client=http_client if http_client else None
                )
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization error: {e}")
        else:
            logger.warning("⚠️ OpenAI not available - using fallback mode")
        
        self._initialized = True
    
    def chat(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """Call chat completions"""
        if not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                temperature=kwargs.get('temperature', 0.6),
                max_tokens=kwargs.get('max_tokens', 600),
                top_p=kwargs.get('top_p', 0.95),
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            if response.choices:
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat API error: {e}")
        
        return None
    
    def embed(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        if not self.client or not text:
            return None
        
        text = text[:2000]
        
        # Check cache
        with self.cache_lock:
            if text in self.embedding_cache:
                return self.embedding_cache[text]
        
        try:
            response = self.client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            
            if response.data:
                embedding = response.data[0].embedding
                
                # Cache result
                with self.cache_lock:
                    self.embedding_cache[text] = embedding
                
                return embedding
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear embedding cache"""
        with self.cache_lock:
            self.embedding_cache.clear()
            logger.info("🧹 Embedding cache cleared")


# Initialize singleton
openai_manager = OpenAIManager()

# ============================================================================
# SECTION 7: GLOBAL STATE MANAGEMENT (Lines 951-1100)
# ============================================================================

# Knowledge base
KNOWLEDGE_BASE: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[Dict] = []

# Tour database
TOURS_DB: Dict[int, Tour] = {}
TOUR_NAME_TO_INDEX: Dict[str, int] = {}
TOUR_TAGS: Dict[int, List[str]] = {}

# FAISS index
FAISS_INDEX = None
FAISS_MAPPING: Dict[str, Dict] = {}
INDEX_LOCK = threading.Lock()

# Session management
SESSION_CONTEXTS: Dict[str, ConversationContext] = {}
SESSION_LOCK = threading.Lock()

# Cache system
RESPONSE_CACHE: Dict[str, CacheEntry] = {}
CACHE_LOCK = threading.Lock()

# Statistics
STATS = {
    'total_requests': 0,
    'total_errors': 0,
    'avg_response_time': 0,
    'tours_searched': 0,
    'tours_booked': 0,
    'users_unique': 0,
    'last_reset': datetime.utcnow().isoformat()
}
STATS_LOCK = threading.Lock()


class StatsManager:
    """Thread-safe statistics manager"""
    
    @staticmethod
    def increment(key: str, amount: int = 1):
        """Increment statistic"""
        with STATS_LOCK:
            if key in STATS:
                STATS[key] += amount
            else:
                STATS[key] = amount
    
    @staticmethod
    def set(key: str, value: Any):
        """Set statistic"""
        with STATS_LOCK:
            STATS[key] = value
    
    @staticmethod
    def get(key: str, default=None):
        """Get statistic"""
        with STATS_LOCK:
            return STATS.get(key, default)
    
    @staticmethod
    def get_all() -> Dict:
        """Get all statistics"""
        with STATS_LOCK:
            return STATS.copy()


# ============================================================================
# SECTION 8: CACHE SYSTEM (Lines 1101-1250)
# ============================================================================

class CacheManager:
    """Intelligent cache management system"""
    
    @staticmethod
    def generate_key(message: str, context_hash: str = "") -> str:
        """Generate cache key"""
        combined = f"{message.lower().strip()}_{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Retrieve from cache"""
        with CACHE_LOCK:
            if key not in RESPONSE_CACHE:
                return None
            
            entry = RESPONSE_CACHE[key]
            if entry.is_expired():
                del RESPONSE_CACHE[key]
                return None
            
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            return entry.value
    
    @staticmethod
    def set(key: str, value: Any, ttl_seconds: int = 300):
        """Store in cache"""
        with CACHE_LOCK:
            # Check size limit
            if len(RESPONSE_CACHE) >= Config.MAX_CACHE_SIZE:
                # Remove oldest entries
                lru_key = min(
                    RESPONSE_CACHE.keys(),
                    key=lambda k: RESPONSE_CACHE[k].last_accessed
                )
                del RESPONSE_CACHE[lru_key]
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl_seconds
            )
            RESPONSE_CACHE[key] = entry
    
    @staticmethod
    def clear():
        """Clear entire cache"""
        with CACHE_LOCK:
            RESPONSE_CACHE.clear()
            logger.info("🧹 Response cache cleared")
    
    @staticmethod
    def stats() -> Dict[str, Any]:
        """Cache statistics"""
        with CACHE_LOCK:
            return {
                'entries': len(RESPONSE_CACHE),
                'max_size': Config.MAX_CACHE_SIZE,
                'utilization': len(RESPONSE_CACHE) / Config.MAX_CACHE_SIZE if Config.MAX_CACHE_SIZE > 0 else 0
            }


# ============================================================================
# SECTION 9: KNOWLEDGE BASE LOADING (Lines 1251-1400)
# ============================================================================

def load_knowledge_base():
    """Load knowledge base from JSON"""
    global KNOWLEDGE_BASE, FLAT_TEXTS, MAPPING
    
    try:
        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.error(f"❌ Knowledge base not found: {Config.KNOWLEDGE_PATH}")
            return False
        
        with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            KNOWLEDGE_BASE = json.load(f)
        
        # Flatten knowledge base
        def flatten(obj, prefix="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    flatten(value, f"{prefix}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    flatten(item, f"{prefix}[{i}]")
            elif isinstance(obj, str):
                text = obj.strip()
                if text:
                    FLAT_TEXTS.append(text)
                    MAPPING.append({"path": prefix, "text": text})
        
        flatten(KNOWLEDGE_BASE)
        
        logger.info(f"✅ Knowledge base loaded: {len(FLAT_TEXTS)} passages")
        return True
    
    except Exception as e:
        logger.error(f"❌ Knowledge base loading error: {e}")
        return False


def build_tour_database():
    """Build structured tour database"""
    global TOURS_DB, TOUR_NAME_TO_INDEX, TOUR_TAGS
    
    try:
        TOURS_DB.clear()
        TOUR_NAME_TO_INDEX.clear()
        TOUR_TAGS.clear()
        
        for entry in MAPPING:
            path = entry.get('path', '')
            text = entry.get('text', '')
            
            # Extract tour index
            match = re.search(r'tours\[(\d+)\]', path)
            if not match:
                continue
            
            tour_idx = int(match.group(1))
            
            # Initialize tour
            if tour_idx not in TOURS_DB:
                TOURS_DB[tour_idx] = Tour(index=tour_idx)
            
            # Extract field name
            field_match = re.search(r'tours\[\d+\]\.(\w+)', path)
            if not field_match:
                continue
            
            field_name = field_match.group(1)
            tour = TOURS_DB[tour_idx]
            
            # Map fields to tour object
            field_mapping = {
                'tour_name': 'name',
                'duration': 'duration',
                'location': 'location',
                'price': 'price',
                'summary': 'summary',
                'includes': 'includes',
                'accommodation': 'accommodation',
                'meals': 'meals',
                'transport': 'transport',
                'notes': 'notes',
                'style': 'style'
            }
            
            if field_name in field_mapping:
                attr_name = field_mapping[field_name]
                if field_name == 'includes':
                    if not isinstance(tour.includes, list):
                        tour.includes = []
                    tour.includes.append(text)
                else:
                    setattr(tour, attr_name, text)
        
        # Index tour names
        for tour_idx, tour in TOURS_DB.items():
            if tour.name:
                normalized = normalize_text_simple(tour.name)
                if normalized not in TOUR_NAME_TO_INDEX:
                    TOUR_NAME_TO_INDEX[normalized] = tour_idx
        
        # Generate tags
        for tour_idx, tour in TOURS_DB.items():
            tags = []
            
            # Location tags
            if tour.location:
                locations = [loc.strip() for loc in tour.location.split(',')]
                tags.extend([f"location:{loc}" for loc in locations[:2]])
            
            # Duration tags
            if tour.duration:
                if '1 ngày' in tour.duration.lower():
                    tags.append("duration:1day")
                elif '2 ngày' in tour.duration.lower():
                    tags.append("duration:2day")
                elif '3 ngày' in tour.duration.lower():
                    tags.append("duration:3day")
            
            # Theme tags
            text_to_check = f"{tour.style or ''} {tour.summary or ''}".lower()
            if 'thiền' in text_to_check or 'tĩnh tâm' in text_to_check:
                tags.append("theme:meditation")
            if 'lịch sử' in text_to_check or 'di tích' in text_to_check:
                tags.append("theme:history")
            if 'thiên nhiên' in text_to_check or 'rừng' in text_to_check:
                tags.append("theme:nature")
            if 'văn hóa' in text_to_check:
                tags.append("theme:culture")
            
            tour.tags = list(set(tags))
            TOUR_TAGS[tour_idx] = tour.tags
        
        logger.info(f"✅ Tour database built: {len(TOURS_DB)} tours")
        return True
    
    except Exception as e:
        logger.error(f"❌ Tour database building error: {e}")
        return False


# ============================================================================
# SECTION 10: SEMANTIC SEARCH & EMBEDDING (Lines 1401-1550)
# ============================================================================

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for text"""
    if not text:
        return None
    
    # Try OpenAI
    embedding = openai_manager.embed(text)
    if embedding:
        return embedding
    
    # Fallback: deterministic embedding
    hash_value = hash(text) % (10 ** 12)
    dim = 1536  # text-embedding-3-small dimension
    embedding = [
        (float((hash_value >> (i % 32)) & 0xFF) + (i % 7)) / 255.0
        for i in range(dim)
    ]
    
    return embedding


def semantic_search(query: str, top_k: int = None) -> List[Tuple[float, Dict]]:
    """Semantic search using FAISS"""
    if not query or not Config.ENABLE_SEMANTIC_SEARCH:
        return []
    
    if top_k is None:
        top_k = Config.TOP_K
    
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        if not FAISS_INDEX or not MAPPING:
            return []
        
        # Normalize query embedding
        query_vec = np.array([query_embedding], dtype='float32')
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        # Search
        distances, indices = FAISS_INDEX.search(query_vec, top_k)
        
        results = []
        threshold = Config.SEMANTIC_THRESHOLD
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            
            similarity = float(distance)
            if similarity < threshold:
                continue
            
            if idx < len(MAPPING):
                mapping = MAPPING[idx]
                text = mapping.get('text', '').strip()
                if text:
                    results.append((similarity, mapping))
        
        logger.debug(f"🔍 Semantic search: {len(results)} results for '{query}'")
        return results
    
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []


# ============================================================================
# SECTION 11: ADVANCED FILTERING SYSTEM (Lines 1551-1750)
# ============================================================================

class FilterEngine:
    """Advanced filtering and querying engine"""
    
    PRICE_PATTERNS = [
        (r'dưới\s+(\d[\d,\.]*)\s*(triệu|tr|k)', 'max'),
        (r'trên\s+(\d[\d,\.]*)\s*(triệu|tr|k)', 'min'),
        (r'khoảng\s+(\d[\d,\.]*)\s*(?:đến|-)\s*(\d[\d,\.]*)\s*(triệu|tr|k)', 'range'),
        (r'(\d[\d,\.]*)\s*(?:triệu|tr|k)\s*(?:đến|-|tới)\s*(\d[\d,\.]*)\s*(?:triệu|tr|k)', 'range'),
    ]
    
    DURATION_PATTERNS = [
        (r'(\d+)\s*(?:ngày|day)\s*(\d+)?\s*(?:đêm|night)', 'specific'),
        (r'(\d+)\s*(?:ngày|day)', 'days'),
        (r'(?:mấy|bao nhiêu)\s*(?:ngày|day)', 'flexible'),
    ]
    
    LOCATION_PATTERNS = [
        (r'(?:ở|tại|đến|về|thăm)\s+([^.,!?\n]+?)(?:\s|$|\.|\,|!|\?)', 'location'),
        (r'(?:địa điểm|nơi|vùng)\s+([^.,!?\n]+)', 'location'),
    ]
    
    @staticmethod
    def extract_filters(message: str) -> FilterSet:
        """Extract filters from user message"""
        filters = FilterSet()
        message_lower = message.lower()
        
        try:
            # Price extraction
            for pattern, filter_type in FilterEngine.PRICE_PATTERNS:
                matches = re.finditer(pattern, message_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        if filter_type == 'max':
                            amount = FilterEngine._parse_price(match.group(1), match.group(2))
                            if amount:
                                filters.price_max = amount
                        elif filter_type == 'min':
                            amount = FilterEngine._parse_price(match.group(1), match.group(2))
                            if amount:
                                filters.price_min = amount
                        elif filter_type == 'range':
                            min_amount = FilterEngine._parse_price(match.group(1), match.group(3))
                            max_amount = FilterEngine._parse_price(match.group(2), match.group(3))
                            if min_amount and max_amount:
                                filters.price_min = min_amount
                                filters.price_max = max_amount
                    except (ValueError, IndexError):
                        continue
            
            # Duration extraction
            for pattern, filter_type in FilterEngine.DURATION_PATTERNS:
                matches = re.finditer(pattern, message_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        if filter_type == 'specific':
                            days = int(match.group(1))
                            if match.group(2):
                                nights = int(match.group(2))
                            else:
                                nights = days - 1
                            filters.duration_min = days
                            filters.duration_max = days
                        elif filter_type == 'days':
                            days = int(match.group(1))
                            filters.duration_min = days
                            filters.duration_max = days
                    except (ValueError, IndexError):
                        continue
            
            # Location extraction
            for pattern, filter_type in FilterEngine.LOCATION_PATTERNS:
                matches = re.finditer(pattern, message_lower, re.IGNORECASE)
                for match in matches:
                    location = match.group(1).strip()
                    if location and len(location) > 1:
                        filters.location = location
            
            # Group type
            group_keywords = {
                'gia đình': 'family',
                'trẻ em': 'family',
                'bạn bè': 'friends',
                'nhóm': 'group',
                'công ty': 'corporate',
                'một mình': 'solo',
                'cặp đôi': 'couple',
                'người lớn tuổi': 'senior'
            }
            
            for keyword, group_type in group_keywords.items():
                if keyword in message_lower:
                    filters.group_type = group_type
                    break
            
        except Exception as e:
            logger.error(f"Filter extraction error: {e}")
        
        return filters
    
    @staticmethod
    def _parse_price(amount_str: str, unit: str) -> Optional[int]:
        """Parse price value"""
        if not amount_str:
            return None
        
        try:
            # Clean amount
            amount_str = amount_str.replace(',', '').replace('.', '')
            if not amount_str.isdigit():
                return None
            
            amount = int(amount_str)
            unit_lower = unit.lower() if unit else ''
            
            # Convert based on unit
            if unit_lower in ['triệu', 'tr']:
                return amount * 1000000
            elif unit_lower in ['k', 'nghìn']:
                return amount * 1000
            else:
                # Guess based on magnitude
                if amount > 1000:
                    return amount if amount > 100000 else amount * 1000
                return amount
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def apply_filters(tours: Dict[int, Tour], filters: FilterSet) -> List[int]:
        """Apply filters to tours"""
        if filters.is_empty() or not tours:
            return list(tours.keys())
        
        passing = []
        
        try:
            for tour_idx, tour in tours.items():
                passes = True
                
                # Price filter
                if filters.price_min or filters.price_max:
                    # Extract price from tour
                    price_val = FilterEngine._extract_tour_price(tour.price)
                    if price_val:
                        if filters.price_max and price_val > filters.price_max:
                            passes = False
                        if filters.price_min and price_val < filters.price_min:
                            passes = False
                
                # Duration filter
                if passes and (filters.duration_min or filters.duration_max):
                    duration_val = FilterEngine._extract_duration(tour.duration)
                    if duration_val:
                        if filters.duration_min and duration_val < filters.duration_min:
                            passes = False
                        if filters.duration_max and duration_val > filters.duration_max:
                            passes = False
                
                # Location filter
                if passes and filters.location:
                    if not tour.location or filters.location.lower() not in tour.location.lower():
                        passes = False
                
                # Group type filter
                if passes and filters.group_type:
                    tour_summary = (tour.summary or '').lower()
                    tour_tags = [tag.lower() for tag in (tour.tags or [])]
                    
                    matches = False
                    if filters.group_type == 'family':
                        if any('family' in tag for tag in tour_tags) or 'gia đình' in tour_summary:
                            matches = True
                    elif filters.group_type == 'friends':
                        if any('group' in tag for tag in tour_tags) or 'nhóm' in tour_summary:
                            matches = True
                    elif filters.group_type == 'senior':
                        if 'lịch sử' in tour_summary or 'tri ân' in tour_summary:
                            matches = True
                    else:
                        matches = True  # Default allow
                    
                    passes = matches
                
                if passes:
                    passing.append(tour_idx)
        
        except Exception as e:
            logger.error(f"Filter application error: {e}")
            return list(tours.keys())
        
        return passing
    
    @staticmethod
    def _extract_tour_price(price_text: str) -> Optional[int]:
        """Extract numeric price value"""
        if not price_text:
            return None
        
        # Find all numbers
        numbers = re.findall(r'\d[\d,\.]+', price_text)
        if not numbers:
            return None
        
        try:
            # Get first number
            num_str = numbers[0].replace(',', '').replace('.', '')
            if not num_str.isdigit():
                return None
            
            value = int(num_str)
            
            # Adjust based on context
            if 'triệu' in price_text.lower() or 'tr' in price_text.lower():
                value = value * 1000000
            elif 'k' in price_text.lower():
                value = value * 1000
            
            return value
        except:
            return None
    
    @staticmethod
    def _extract_duration(duration_text: str) -> Optional[int]:
        """Extract duration in days"""
        if not duration_text:
            return None
        
        pattern = r'(\d+)\s*(?:ngày|day)'
        match = re.search(pattern, duration_text.lower())
        
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None


# ============================================================================
# SECTION 12: FUZZY MATCHING & TOUR SEARCH (Lines 1751-1900)
# ============================================================================

class TourSearchEngine:
    """Intelligent tour search and matching"""
    
    @staticmethod
    def find_by_name(query: str, tour_name_index: Dict[str, int]) -> List[Tuple[int, float]]:
        """Find tours by name with fuzzy matching"""
        if not query or not tour_name_index:
            return []
        
        query_norm = normalize_text_simple(query)
        matches = []
        
        try:
            for tour_name, tour_idx in tour_name_index.items():
                # Calculate similarity
                similarity = SequenceMatcher(None, query_norm, tour_name).ratio()
                
                # Check substring matches
                if query_norm in tour_name:
                    similarity = max(similarity, 0.85)
                
                if tour_name in query_norm:
                    similarity = max(similarity, 0.75)
                
                if similarity > 0.6:
                    matches.append((tour_idx, similarity))
        
        except Exception as e:
            logger.error(f"Tour name search error: {e}")
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]
    
    @staticmethod
    def find_by_tags(query: str, tour_tags: Dict[int, List[str]]) -> List[Tuple[int, float]]:
        """Find tours by tags"""
        if not query or not tour_tags:
            return []
        
        query_words = set(normalize_text_simple(query).split())
        matches = []
        
        try:
            for tour_idx, tags in tour_tags.items():
                tag_words = set()
                for tag in tags:
                    words = tag.split(':')
                    tag_words.update(words)
                
                # Calculate overlap
                if query_words and tag_words:
                    overlap = len(query_words.intersection(tag_words)) / max(len(query_words), 1)
                    if overlap > 0.3:
                        matches.append((tour_idx, overlap))
        
        except Exception as e:
            logger.error(f"Tag search error: {e}")
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]


# ============================================================================
# SECTION 13: RECOMMENDATION ENGINE (Lines 1901-2050)
# ============================================================================

class RecommendationEngine:
    """Intelligent tour recommendation system"""
    
    @staticmethod
    def score_tour(tour: Tour, user_profile: UserProfile, query: str) -> Tuple[float, List[str]]:
        """Calculate recommendation score for tour"""
        score = 0.0
        reasons = []
        
        try:
            # Interest matching (40%)
            if user_profile.interests:
                tour_text = f"{tour.summary or ''} {tour.style or ''}".lower()
                for interest in user_profile.interests:
                    if interest.lower() in tour_text:
                        score += 0.4
                        reasons.append(f"Có hoạt động {interest}")
                        break
            
            # Budget matching (20%)
            if user_profile.budget_level:
                tour_price = FilterEngine._extract_tour_price(tour.price)
                if tour_price:
                    if user_profile.budget_level == 'budget' and tour_price < 1500000:
                        score += 0.2
                        reasons.append("Giá phù hợp")
                    elif user_profile.budget_level == 'midrange' and 1500000 <= tour_price <= 3000000:
                        score += 0.2
                        reasons.append("Giá tầm trung")
                    elif user_profile.budget_level == 'premium' and tour_price > 3000000:
                        score += 0.2
                        reasons.append("Dịch vụ cao cấp")
            
            # Duration match (20%)
            if user_profile.preferences.get('duration_preference'):
                pref_duration = user_profile.preferences['duration_preference']
                tour_duration = FilterEngine._extract_duration(tour.duration)
                if tour_duration and abs(tour_duration - pref_duration) <= 1:
                    score += 0.2
                    reasons.append(f"Thời gian phù hợp ({tour.duration})")
            
            # Query relevance (20%)
            query_words = set(normalize_text_simple(query).split())
            tour_words = set(normalize_text_simple(f"{tour.name or ''} {tour.summary or ''}").split())
            
            if query_words and tour_words:
                overlap = len(query_words.intersection(tour_words)) / max(len(query_words), len(tour_words))
                score += overlap * 0.2
                if overlap > 0.5:
                    reasons.append("Phù hợp nhu cầu")
        
        except Exception as e:
            logger.error(f"Scoring error: {e}")
        
        return min(score, 1.0), reasons
    
    @staticmethod
    def recommend(user_profile: UserProfile, query: str, tours: Dict[int, Tour], top_k: int = 5) -> List[SearchResult]:
        """Generate recommendations"""
        recommendations = []
        
        try:
            for tour_idx, tour in tours.items():
                score, reasons = RecommendationEngine.score_tour(tour, user_profile, query)
                
                if score > 0:
                    recommendations.append(SearchResult(
                        tour_index=tour_idx,
                        relevance_score=score,
                        match_type='recommendation',
                        reasoning=reasons
                    ))
        
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
        
        # Sort and return top K
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations[:top_k]


# ============================================================================
# SECTION 14: INTENT DETECTION & NLU (Lines 2051-2200)
# ============================================================================

class IntentDetector:
    """Intent detection and classification"""
    
    INTENT_PATTERNS = {
        QuestionType.LISTING: [
            r'liệt kê.*tour|danh sách.*tour|có.*tour.*nào',
            r'tour.*hiện|tour.*đang|tour nào.*có',
            r'xem.*tour|show.*tour|tất cả.*tour'
        ],
        QuestionType.COMPARISON: [
            r'so sánh|đối chiếu|nên chọn.*nào',
            r'khác nhau.*nào|giống nhau|tour.*và.*tour',
            r'cái nào.*hơn|tốt.*hơn'
        ],
        QuestionType.RECOMMENDATION: [
            r'phù hợp.*không|gợi ý|đề xuất|tư vấn',
            r'tour nào.*phù hợp|phù hợp.*tour nào',
            r'nên.*tour|tour.*cho.*tôi'
        ],
        QuestionType.PRICE_INQUIRY: [
            r'giá.*bao nhiêu|bao nhiêu.*tiền|chi phí',
            r'giá.*tour|bảng.*giá|phải.*bao nhiêu'
        ],
        QuestionType.SERVICE_INQUIRY: [
            r'bao gồm|gồm.*gì|dịch vụ.*gì|cung cấp.*gì',
            r'có.*không|có.*cho.*không|dịch vụ'
        ],
        QuestionType.LOCATION_QUERY: [
            r'ở.*đâu|đi.*đâu|địa điểm.*nào|nơi.*nào',
            r'tại.*đâu|đến.*đâu|thăm.*đâu'
        ],
    }
    
    @staticmethod
    def detect(message: str) -> Tuple[QuestionType, float]:
        """Detect question type"""
        message_lower = message.lower()
        
        # Check for greetings
        if any(word in message_lower for word in ['xin chào', 'chào', 'hello', 'hi']):
            return QuestionType.GREETING, 0.95
        
        # Check for farewells
        if any(word in message_lower for word in ['tạm biệt', 'cảm ơn', 'bye', 'goodbye']):
            return QuestionType.FAREWELL, 0.95
        
        # Check other intents
        scores = {}
        for intent, patterns in IntentDetector.INTENT_PATTERNS.items():
            max_score = 0.0
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    max_score = 0.8
                    break
            if max_score > 0:
                scores[intent] = max_score
        
        # Return best match
        if scores:
            best_intent = max(scores, key=scores.get)
            return best_intent, scores[best_intent]
        
        return QuestionType.INFORMATION, 0.5


# ============================================================================
# SECTION 15: LLM PROMPT ENGINEERING (Lines 2201-2400)
# ============================================================================

class PromptBuilder:
    """Intelligent prompt construction for LLM"""
    
    SYSTEM_PROMPT_TEMPLATE = """Bạn là một chuyên viên tư vấn du lịch chuyên nghiệp của Ruby Wings Travel.

**THÔNG TIN CÔNG TY:**
- Ruby Wings: Tổ chức tour trải nghiệm với triết lý "Chuẩn mực - Chân thành - Có chiều sâu"
- Chuyên tour thiền, retreat, lịch sử, văn hóa, thiên nhiên
- Hotline: 0332510486
- Địa điểm: Huế, Quảng Trị, Bạch Mã, Trường Sơn, Miền Trung Việt Nam

**CÁC HƯỚNG TIẾP CẬN:**
1. Khi được hỏi về tour CỤ THỂ: Cung cấp thông tin chi tiết từ dữ liệu
2. Khi được hỏi CÂU HỎI CHUNG: Sử dụng kiến thức chuyên môn về du lịch
3. Khi KHÔNG CÓ DỮ LIỆU: Nêu rõ và hỏi lại để tìm hiểu thêm
4. LUÔN KẾT THÚC: Với hành động tiếp theo (gọi số, xem tour, đặt tour)

**TÔNG ĐỘ:**
- Chuyên nghiệp, thân thiện, nhiệt tình
- Trung thực, minh bạch về thông tin
- Tôn trọng nhu cầu và sở thích khách hàng
- Không bao giờ quảng cáo quá mức

**CHỈ THỰC HIỆN:**
✓ Tư vấn tour phù hợp
✓ Cung cấp thông tin chính xác từ dữ liệu
✓ Hỗ trợ so sánh, đề xuất
✓ Hướng dẫn quy trình đặt tour

✗ KHÔNG làm những việc sau:
✗ Giả bộ có thông tin không có
✗ Quảng cáo sản phẩm khác
✗ Cung cấp thông tin sai lệch
✗ Cam kết điều kiện không kiểm chứng"""
    
    @staticmethod
    def build_user_prompt(message: str, context: ConversationContext, search_results: List = None) -> str:
        """Build user prompt for LLM"""
        prompt_parts = [
            f"Khách hàng: {message}"
        ]
        
        # Add context if available
        if context.user_profile:
            profile_info = []
            if context.user_profile.group_type:
                profile_info.append(f"Loại nhóm: {context.user_profile.group_type}")
            if context.user_profile.interests:
                profile_info.append(f"Sở thích: {', '.join(context.user_profile.interests)}")
            if context.user_profile.budget_level:
                profile_info.append(f"Ngân sách: {context.user_profile.budget_level}")
            
            if profile_info:
                prompt_parts.append(f"\nHồ sơ khách: {'; '.join(profile_info)}")
        
        # Add search results if available
        if search_results:
            prompt_parts.append("\n## Thông tin liên quan từ cơ sở dữ liệu:")
            for i, (score, passage) in enumerate(search_results[:3], 1):
                text = passage.get('text', '')[:200]
                prompt_parts.append(f"{i}. {text}...")
        
        prompt_parts.append("\n## Yêu cầu:")
        prompt_parts.append("- Trả lời ngắn gọn (2-4 câu) nếu là thông tin chung")
        prompt_parts.append("- Chi tiết hơn nếu khách hỏi về tour cụ thể")
        prompt_parts.append("- LUÔN kết thúc bằng hành động (gọi, xem, đặt)")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_system_prompt() -> str:
        """Get system prompt"""
        return PromptBuilder.SYSTEM_PROMPT_TEMPLATE


# ============================================================================
# SECTION 16: SESSION MANAGEMENT (Lines 2401-2500)
# ============================================================================

class SessionManager:
    """Session context management"""
    
    @staticmethod
    def get_or_create(session_id: str) -> ConversationContext:
        """Get or create session"""
        with SESSION_LOCK:
            if session_id not in SESSION_CONTEXTS:
                SESSION_CONTEXTS[session_id] = ConversationContext(session_id=session_id)
            return SESSION_CONTEXTS[session_id]
    
    @staticmethod
    def extract_session_id(request_data: Dict, remote_addr: str) -> str:
        """Extract or generate session ID"""
        session_id = request_data.get('session_id')
        if not session_id:
            ip = remote_addr or "0.0.0.0"
            timestamp = datetime.utcnow().isoformat()
            combined = f"{ip}_{timestamp}"
            session_id = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"session_{session_id}"
    
    @staticmethod
    def cleanup_expired():
        """Clean up expired sessions"""
        with SESSION_LOCK:
            to_delete = [
                sid for sid, ctx in SESSION_CONTEXTS.items()
                if ctx.is_expired(Config.SESSION_TIMEOUT)
            ]
            
            for sid in to_delete:
                del SESSION_CONTEXTS[sid]
            
            if to_delete:
                logger.info(f"🧹 Cleaned up {len(to_delete)} expired sessions")


# ============================================================================
# SECTION 17: RESPONSE GENERATION (Lines 2501-2700)
# ============================================================================
# =========== RESPONSE GENERATOR - HYBRID AI SYSTEM ===========
class ResponseGenerator:
    """
    Hybrid response generator combining deterministic logic + OpenAI intelligence
    """
    
    @staticmethod
    def generate_smart_response(
        user_message: str,
        tour_indices: List[int],
        detected_intents: List[str],
        primary_intent: str,
        context: ConversationContext,
        search_results: List[Tuple[float, Dict]] = None,
        filters: FilterSet = None,
        complexity_score: float = 0.0
    ) -> str:
        """
        Generate intelligent response using hybrid approach
        """
        
        # ========== PHASE 1: DETECT RESPONSE TYPE ==========
        response_type = ResponseGenerator._classify_response_type(
            user_message,
            primary_intent,
            tour_indices,
            complexity_score
        )
        
        logger.info(f"🎯 Response type: {response_type}")
        
        # ========== PHASE 2: ROUTE TO HANDLER ==========
        if response_type == "rule_based":
            # Use deterministic rules for structured responses
            return ResponseGenerator._handle_rule_based(
                user_message,
                primary_intent,
                tour_indices,
                filters
            )
        
        elif response_type == "template":
            # Use template system for formatted responses
            return ResponseGenerator._handle_template_response(
                primary_intent,
                tour_indices,
                filters,
                context
            )
        
        elif response_type == "ai_powered":
            # Use OpenAI for complex reasoning
            return ResponseGenerator._handle_ai_powered(
                user_message,
                tour_indices,
                search_results,
                context,
                detected_intents,
                filters
            )
        
        else:
            # Fallback
            return ResponseGenerator._handle_fallback(
                user_message,
                tour_indices
            )
    
    @staticmethod
    def _classify_response_type(
        message: str,
        intent: str,
        tour_indices: List[int],
        complexity: float
    ) -> str:
        """
        Classify response type: rule_based, template, ai_powered, or fallback
        """
        message_lower = message.lower()
        
        # Rule-based responses (high confidence, structured)
        rule_based_intents = [
            'booking_info', 'policy', 'general_info'
        ]
        
        if intent in rule_based_intents:
            return "rule_based"
        
        # Template responses (medium confidence, formatted)
        template_intents = [
            'tour_listing', 'price_inquiry', 'location_query',
            'service_inquiry', 'food_info', 'weather_info'
        ]
        
        if intent in template_intents:
            return "template"
        
        # AI-powered responses (complex reasoning needed)
        ai_intents = [
            'recommendation', 'comparison', 'custom_request',
            'experience', 'wellness_info', 'culture_info'
        ]
        
        if intent in ai_intents or complexity > 2.0:
            return "ai_powered"
        
        # Fallback for uncertain responses
        return "fallback"
    
    @staticmethod
    def _handle_rule_based(
        message: str,
        intent: str,
        tour_indices: List[int],
        filters: FilterSet
    ) -> str:
        """Handle rule-based structured responses"""
        
        if intent == 'booking_info':
            return _get_booking_policy_response(message.lower())
        
        elif intent == 'policy':
            return _get_policy_response(message.lower())
        
        elif intent == 'general_info':
            if 'ruby wings' in message.lower() or 'công ty' in message.lower():
                return _get_company_introduction()
            elif 'triết lý' in message.lower() or 'giá trị' in message.lower():
                return _get_philosophy_response()
            else:
                return _get_general_company_info()
        
        return ""
    
    @staticmethod
    def _handle_template_response(
        intent: str,
        tour_indices: List[int],
        filters: FilterSet,
        context: ConversationContext
    ) -> str:
        """Handle template-based formatted responses"""
        
        if intent == 'tour_listing':
            all_tours = list(TOURS_DB.values())
            
            # Apply filters
            if filters and not filters.is_empty():
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, filters)
                all_tours = [TOURS_DB[idx] for idx in filtered_indices if idx in TOURS_DB]
            
            return TemplateSystem.render(
                'tour_list',
                tours=all_tours[:10],
                total=len(all_tours)
            )
        
        elif intent == 'tour_detail' and tour_indices:
            tour = TOURS_DB.get(tour_indices[0])
            if tour:
                return TemplateSystem.render(
                    'tour_detail',
                    tour_name=tour.name,
                    duration=tour.duration,
                    location=tour.location,
                    price=tour.price,
                    summary=tour.summary,
                    includes=tour.includes,
                    accommodation=tour.accommodation,
                    meals=tour.meals,
                    transport=tour.transport,
                    notes=tour.notes,
                    suitable_for=", ".join(tour.tags[:3]) if tour.tags else "mọi đối tượng"
                )
        
        elif intent == 'price_inquiry':
            if tour_indices:
                tours = [TOURS_DB.get(idx) for idx in tour_indices[:3] if idx in TOURS_DB]
                return TemplateSystem.render(
                    'tour_detail',
                    tours=tours
                )
            else:
                # General price info
                reply = "💰 **BẢNG GIÁ THAM KHẢO RUBY WINGS** 💰\n\n"
                reply += "**PHÂN LOẠI GIÁ TOUR:**\n"
                reply += "• 🌿 Tour 1 ngày (Thiên nhiên): 600.000đ - 1.500.000đ\n"
                reply += "• 🏛️ Tour 2 ngày 1 đêm (Lịch sử): 1.500.000đ - 3.000.000đ\n"
                reply += "• 🕉️ Tour 3+ ngày (Cao cấp): 3.000.000đ - 5.000.000đ\n"
                reply += "• 👥 Tour nhóm (Tùy chỉnh): Liên hệ tư vấn\n\n"
                reply += "**ƯU ĐÃI:**\n"
                reply += "• Nhóm 10+ người: Giảm 10-15%\n"
                reply += "• Đặt trước 30 ngày: Giảm thêm 5%\n"
                reply += "• Thanh toán online: Giảm thêm 2%\n"
                reply += "• Cựu chiến binh: Giảm 5-10%\n\n"
                reply += "📞 **Liên hệ báo giá chính xác:** 0332510486"
                return reply
        
        elif intent == 'location_query':
            # Extract location from context
            location = None
            locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà']
            
            for loc in locations:
                if loc in str(context.user_preferences).lower():
                    location = loc
                    break
            
            if location:
                location_tours = [tour for tour in TOURS_DB.values() 
                                 if tour.location and location in tour.location.lower()]
                return _get_location_info(location, location_tours)
            else:
                return "Bạn muốn tìm tour tại khu vực nào? Ruby Wings có tour tại Huế, Quảng Trị, Bạch Mã, Trường Sơn. 📍"
        
        elif intent == 'service_inquiry':
            return _get_service_info_response()
        
        elif intent == 'weather_info':
            location = None
            locations = ['huế', 'quảng trị', 'bạch mã', 'trường sơn']
            
            for loc in locations:
                if loc in str(context.user_preferences).lower():
                    location = loc
                    break
            
            if location:
                location_tours = [tour for tour in TOURS_DB.values() 
                                 if tour.location and location in tour.location.lower()]
                return _get_weather_info(location, location_tours)
            else:
                return "Thời tiết miền Trung: Mùa khô (1-8), mùa mưa (9-12). Liên hệ tư vấn tour phù hợp thời tiết. 📞 0332510486"
        
        elif intent == 'food_info':
            return _get_food_culture_response("ẩm thực", [])
        
        elif intent == 'culture_info':
            return _get_food_culture_response("văn hóa", [])
        
        return ""
    
    @staticmethod
    def _handle_ai_powered(
        user_message: str,
        tour_indices: List[int],
        search_results: List[Tuple[float, Dict]],
        context: ConversationContext,
        detected_intents: List[str],
        filters: FilterSet
    ) -> str:
        """Handle AI-powered responses using OpenAI"""
        
        if not client or not HAS_OPENAI:
            logger.warning("⚠️ OpenAI not available, using fallback")
            return ResponseGenerator._handle_fallback(user_message, tour_indices)
        
        try:
            # Build context for AI
            context_info = ResponseGenerator._build_ai_context(
                user_message,
                tour_indices,
                search_results,
                context,
                detected_intents,
                filters
            )
            
            # Create system prompt
            system_prompt = ResponseGenerator._create_system_prompt(context_info)
            
            # Call OpenAI
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.9
            )
            
            if response.choices and response.choices[0].message:
                reply = response.choices[0].message.content.strip()
                
                # Validate reply
                reply = AutoValidator.validate_response(reply)
                
                # Ensure hotline is included
                if "0332510486" not in reply and "hotline" not in reply.lower():
                    reply += "\n\n📞 **Hotline tư vấn 24/7:** 0332510486"
                
                return reply
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
        
        # Fallback
        return ResponseGenerator._handle_fallback(user_message, tour_indices)
    
    @staticmethod
    def _build_ai_context(
        user_message: str,
        tour_indices: List[int],
        search_results: List[Tuple[float, Dict]],
        context: ConversationContext,
        detected_intents: List[str],
        filters: FilterSet
    ) -> Dict[str, Any]:
        """Build context information for AI reasoning"""
        
        context_dict = {
            'user_message': user_message,
            'tour_context': [],
            'search_context': [],
            'conversation_history': [],
            'user_profile': {},
            'filters': {},
            'detected_intents': detected_intents
        }
        
        # Add tour context
        if tour_indices:
            for idx in tour_indices[:3]:
                tour = TOURS_DB.get(idx)
                if tour:
                    context_dict['tour_context'].append({
                        'name': tour.name,
                        'duration': tour.duration,
                        'location': tour.location,
                        'price': tour.price,
                        'summary': (tour.summary or '')[:200]
                    })
        
        # Add search results
        if search_results:
            for score, passage in search_results[:3]:
                context_dict['search_context'].append({
                    'relevance': float(score),
                    'text': passage.get('text', '')[:150]
                })
        
        # Add conversation history (last 3 exchanges)
        if hasattr(context, 'conversation_history'):
            for msg in context.conversation_history[-6:]:
                context_dict['conversation_history'].append({
                    'role': msg.get('role', 'user'),
                    'message': msg.get('message', '')[:100]
                })
        
        # Add user profile
        if hasattr(context, 'user_profile'):
            context_dict['user_profile'] = context.user_profile
        
        # Add filters
        if filters:
            context_dict['filters'] = filters.to_dict() if hasattr(filters, 'to_dict') else str(filters)
        
        return context_dict
    
    @staticmethod
    def _create_system_prompt(context_info: Dict[str, Any]) -> str:
        """Create intelligent system prompt for OpenAI"""
        
        base_prompt = """Bạn là trợ lý tư vấn du lịch Ruby Wings - CHUYÊN NGHIỆP, THÔNG MINH, NHIỆT TÌNH.

🎯 **NGUYÊN TẮC HOẠT ĐỘNG:**
1. Luôn trả lời bằng Tiếng Việt tự nhiên, thân thiện, chuyên nghiệp
2. Tôn trọng dữ liệu được cung cấp (tour, giá, lịch trình)
3. Không hallucinate thông tin - nếu không biết thì nói rõ
4. Kết thúc mọi response bằng gợi ý hành động hoặc hotline
5. Giải thích lý do khi đề xuất tour
6. Cân bằng giữa bán hàng và tư vấn chân thành

📚 **THÔNG TIN NGỮ CẢNH:**\n"""
        
        # Add tour context
        if context_info.get('tour_context'):
            base_prompt += "\n**Tour đang bàn:**\n"
            for tour in context_info['tour_context']:
                base_prompt += f"- {tour['name']} ({tour['duration']}): {tour['summary']}\n"
        
        # Add user profile
        if context_info.get('user_profile'):
            base_prompt += "\n**Hồ sơ người dùng:**\n"
            for key, value in context_info['user_profile'].items():
                if value:
                    base_prompt += f"- {key}: {value}\n"
        
        # Add conversation history context
        if context_info.get('conversation_history'):
            base_prompt += "\n**Lịch sử cuộc trò chuyện (3 tin nhắn gần nhất):**\n"
            for msg in context_info['conversation_history'][-3:]:
                base_prompt += f"- {msg['role']}: {msg['message'][:80]}...\n"
        
        # Add intent detection
        if context_info.get('detected_intents'):
            base_prompt += f"\n**Ý định người dùng:** {', '.join(context_info['detected_intents'][:2])}\n"
        
        # Add filter constraints
        if context_info.get('filters'):
            base_prompt += "\n**Ràng buộc bộ lọc:**\n"
            filters = context_info['filters']
            if isinstance(filters, dict):
                for key, value in filters.items():
                    if value:
                        base_prompt += f"- {key}: {value}\n"
        
        base_prompt += """\n\n💬 **YÊU CẦU TRẢ LỜI:**
1. **Ngắn gọn:** 3-5 câu (trừ khi cần liệt kê chi tiết)
2. **Cụ thể:** Dùng tên tour, giá, địa điểm cụ thể
3. **Lập luận:** Giải thích tại sao tour lại phù hợp
4. **Hành động:** Kết thúc bằng câu hỏi dẫn dắt hoặc lời mời
5. **Hotline:** Bao gồm "📞 0332510486" nếu phù hợp

🚫 **TUYỆT ĐỐI KHÔNG LÀM:**
- Nói "không có dữ liệu", "xin lỗi không tìm thấy"
- Tạo ra thông tin tour giả mạo
- Đưa ra giá không có trong dữ liệu
- Liệt kê quá 3 tour mà không có lý do
- Sử dụng ngôi thứ nhất (tôi) quá nhiều"""
        
        return base_prompt
    
    @staticmethod
    def _handle_fallback(message: str, tour_indices: List[int]) -> str:
        """Handle fallback response when AI unavailable"""
        
        message_lower = message.lower()
        
        # Try to match simple patterns
        if 'phù h���p' in message_lower or 'gợi ý' in message_lower:
            if tour_indices:
                tours = [TOURS_DB.get(idx) for idx in tour_indices[:2] if idx in TOURS_DB]
                reply = "🎯 **TOUR PHÙ HỢP CHO BẠN:**\n\n"
                for tour in tours:
                    if tour:
                        reply += f"**{tour.name}**\n"
                        if tour.duration:
                            reply += f"⏱️ {tour.duration}\n"
                        if tour.location:
                            reply += f"📍 {tour.location}\n"
                        if tour.price:
                            price_short = tour.price[:50] + "..." if len(tour.price) > 50 else tour.price
                            reply += f"💰 {price_short}\n"
                        reply += "\n"
                reply += "📞 **Đặt tour:** 0332510486"
                return reply
            else:
                return "Ruby Wings có nhiều tour phù hợp với nhu cầu của bạn. Hãy cho tôi biết bạn quan tâm tour nào để tôi tư vấn chi tiết. 📞 0332510486"
        
        elif 'giá' in message_lower or 'bao nhiêu' in message_lower:
            if tour_indices:
                tours = [TOURS_DB.get(idx) for idx in tour_indices[:1] if idx in TOURS_DB]
                if tours and tours[0].price:
                    return f"💰 **GIÁ TOUR:** {tours[0].price}\n\n📞 Liên hệ 0332510486 để được báo giá chính xác và ưu đãi tốt nhất!"
            
            return "Giá tour Ruby Wings từ 600.000đ - 5.000.000đ tùy loại tour. Liên hệ 0332510486 để được báo giá."
        
        elif 'thời gian' in message_lower or 'ngày' in message_lower:
            if tour_indices:
                tours = [TOURS_DB.get(idx) for idx in tour_indices[:1] if idx in TOURS_DB]
                if tours and tours[0].duration:
                    return f"⏱️ **THỜI GIAN TOUR:** {tours[0].duration}\n\n📞 Liên hệ 0332510486 để biết lịch khởi hành tiếp theo!"
            
            return "Ruby Wings có tour từ 1 ngày đến 7+ ngày. Bạn muốn tour bao lâu để tôi tư vấn phù hợp? 📞 0332510486"
        
        # Default fallback
        return "Ruby Wings sẵn sàng tư vấn tour phù hợp với nhu cầu của bạn. Hãy gọi hotline 0332510486 để được hỗ trợ ngay! 😊"


# =========== ADDITIONAL HELPER FUNCTIONS ===========

def _get_policy_response(message_lower: str) -> str:
    """Get policy information"""
    
    if 'hủy' in message_lower or 'huỷ' in message_lower:
        reply = "📋 **CHÍNH SÁCH HỦY TOUR** 📋\n\n"
        reply += "**QUY ĐỊNH HOÀN TIỀN:**\n"
        reply += "• **Trước 14 ngày:** Hoàn 100% (trừ phí dịch vụ)\n"
        reply += "• **7-13 ngày:** Hoàn 70% (giữ 30%)\n"
        reply += "• **5-6 ngày:** Hoàn 50%\n"
        reply += "• **3-4 ngày:** Hoàn 25%\n"
        reply += "• **1-2 ngày:** Hoàn 10%\n"
        reply += "• **Ngày khởi hành:** 0% (mất toàn bộ cọc)\n\n"
        
        reply += "**ĐIỀU KIỆN HỦY:**\n"
        reply += "• Hủy bằng email hoặc gọi hotline chính thức\n"
        reply += "• Cung cấp lý do (để được xem xét)\n"
        reply += "• Không hoàn tiền nếu do sức khỏe cá nhân\n"
        reply += "• Có thể đổi lịch thay vì hủy\n\n"
        
        reply += "📞 **Liên hệ hủy tour:** 0332510486"
        return reply
    
    elif 'đổi lịch' in message_lower or 'rút lui' in message_lower:
        reply = "**QUY Đ��NH ĐỔI LỊCH:**\n\n"
        reply += "• Đổi lịch miễn phí nếu trước 30 ngày\n"
        reply += "• Phí đổi 10% nếu trước 15 ngày\n"
        reply += "• Phí đổi 20% nếu trước 7 ngày\n"
        reply += "• Phí đổi 50% nếu trước 3 ngày\n"
        reply += "• Không thể đổi trong 48 giờ trước tour\n\n"
        reply += "📞 **Liên hệ đổi lịch:** 0332510486"
        return reply
    
    elif 'ưu đãi' in message_lower or 'giảm giá' in message_lower or 'khuyến mãi' in message_lower:
        reply = "🎁 **ƯU ĐÃI & KHUYẾN MÃI** 🎁\n\n"
        reply += "**GIẢM GIÁ THEO NHÓM:**\n"
        reply += "• 5-9 người: Giảm 5%\n"
        reply += "• 10-15 người: Giảm 10%\n"
        reply += "• 16-20 người: Giảm 15%\n"
        reply += "• 21+ người: Giảm 20%\n\n"
        
        reply += "**GIẢM GIÁ THEO THỜI GIAN:**\n"
        reply += "• Đặt trước 30 ngày: +5%\n"
        reply += "• Đặt trước 60 ngày: +10%\n"
        reply += "• Đặt trước 90 ngày: +15%\n\n"
        
        reply += "**ƯU ĐÃI ĐẶC BIỆT:**\n"
        reply += "• Cựu chiến binh: +5-10%\n"
        reply += "• Thanh toán online: +2%\n"
        reply += "• Tái khách: +5%\n"
        reply += "• Khách hàng truyền miệng: +3%\n\n"
        
        reply += "📞 **Liên hệ khuyến mãi hôm nay:** 0332510486"
        return reply
    
    else:
        reply = "**CHÍNH SÁCH RUBY WINGS:**\n\n"
        reply += "✅ Minh bạch, rõ ràng, khách hàng trước\n"
        reply += "✅ Hỗ trợ 24/7 trong suốt tour\n"
        reply += "✅ Bảo hiểm du lịch bắt buộc\n"
        reply += "✅ Hoàn tiền theo quy định nếu Ruby Wings lỗi\n"
        reply += "✅ Ưu đãi đa dạng cho nhóm & tái khách\n\n"
        reply += "📞 **Tư vấn chính sách chi tiết:** 0332510486"
        return reply


def _get_service_info_response() -> str:
    """Get service information"""
    reply = "🛎️ **DỊCH VỤ BAO GỒM TRONG TOUR** 🛎️\n\n"
    
    reply += "**✅ DỊCH VỤ CƠ BẢN (TẤT CẢ TOUR):**\n"
    reply += "• 🚌 Xe đưa đón đời mới, máy lạnh\n"
    reply += "• 🏨 Chỗ ở 3*+ với tiện nghi cơ bản\n"
    reply += "• 🍽️ Ăn 3 bữa/ngày theo chương trình\n"
    reply += "• 🧭 Hướng dẫn viên chuyên nghiệp\n"
    reply += "• 🎫 Vé tham quan các điểm du lịch\n"
    reply += "• 💧 Nước uống suối đóng chai\n"
    reply += "• 🛡️ Bảo hiểm du lịch 50 triệu VNĐ\n\n"
    
    reply += "**✨ DỊCH VỤ CAO CẤP (TOUR 2+ NGÀY):**\n"
    reply += "• 🌟 Khách sạn 4* ở các vị trí tốt\n"
    reply += "• 🍷 Bữa ăn đặc sản địa phương\n"
    reply += "• 📸 Hỗ trợ chụp ảnh lưu niệm\n"
    reply += "• 🎁 Quà tặng tinh thần cá nhân\n"
    reply += "• 🚑 Y tế đi kèm (tour nhóm lớn)\n\n"
    
    reply += "**⚠️ KHÔNG BAO GỒM:**\n"
    reply += "• Chi phí cá nhân (điện thoại, mini bar)\n"
    reply += "• Đồ uống có cồn ngoài chương trình\n"
    reply += "• Tip cho hướng dẫn viên, tài xế\n"
    reply += "• Dịch vụ tùy chọn\n\n"
    
    reply += "📞 **Chi tiết dịch vụ tour cụ thể:** 0332510486"
    return reply


def _get_general_company_info() -> str:
    """Get general company information"""
    reply = "ℹ️ **RUBY WINGS TRAVEL - THÔNG TIN CÔNG TY** ℹ️\n\n"
    
    reply += "🏢 **THÔNG TIN CƠ BẢN:**\n"
    reply += "• Đơn vị tổ chức tour du lịch trải nghiệm\n"
    reply += "• Chuyên sâu về tour lịch sử, retreat, văn hóa\n"
    reply += "• Hoạt động: Du lịch, bồi dưỡng, truyền thông\n"
    reply += "• Địa điểm: Miền Trung Việt Nam\n\n"
    
    reply += "📍 **CÁC TOUR CHÍNH:**\n"
    reply += "• Tour lịch sử & tri ân (Trường Sơn, Di tích)\n"
    reply += "• Tour retreat & chữa lành (Bạch Mã)\n"
    reply += "• Tour văn hóa & ẩm thực (Huế, Quảng Trị)\n"
    reply += "• Tour teambuilding & nhóm (Tùy chỉnh)\n\n"
    
    reply += "👥 **ĐỐI TƯỢNG PHỤC VỤ:**\n"
    reply += "• Gia đình, nhóm bạn, cá nhân\n"
    reply += "• Công ty, doanh nghiệp\n"
    reply += "• Sinh viên, học sinh, người lớn tuổi\n"
    reply += "• Cựu chiến binh, lưu học sinh\n\n"
    
    reply += "📞 **LIÊN HỆ:**\n"
    reply += "Hotline: 0332510486\n"
    reply += "Giờ làm: 24/7\n"
    reply += "Email: rubywingslsa@gmail.com\n\n"
    
    reply += "💡 **HỎI THÊM:** Bạn quan tâm tour nào để biết chi tiết?"
    return reply


# =========== INITIALIZATION & STARTUP ===========

@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify({
        "status": "✅ Ruby Wings Chatbot v4.2 - Active",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.2-hybrid",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "reindex": "/reindex",
            "stats": "/stats"
        },
        "features": {
            "ai_enabled": HAS_OPENAI,
            "faiss_enabled": HAS_FAISS,
            "google_sheets": HAS_GOOGLE_SHEETS,
            "meta_capi": HAS_META_CAPI,
            "language": "Vietnamese",
            "ram_profile": RAM_PROFILE
        }
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "openai": "✅" if HAS_OPENAI else "⚠️",
            "faiss": "✅" if HAS_FAISS else "⚠️",
            "google_sheets": "✅" if HAS_GOOGLE_SHEETS else "⚠️",
            "knowledge_base": "✅" if KNOW else "❌",
            "tours_db": f"✅ {len(TOURS_DB)} tours" if TOURS_DB else "❌",
            "index": "✅" if INDEX else "⚠️",
            "sessions": f"✅ {len(SESSION_CONTEXTS)} active" if SESSION_CONTEXTS else "✅ 0 active"
        },
        "metrics": {
            "total_requests": GLOBAL_STATS.get('total_requests', 0),
            "total_leads": GLOBAL_STATS.get('leads', 0),
            "errors": GLOBAL_STATS.get('errors', 0),
            "cache_size": len(_response_cache)
        }
    }
    
    return jsonify(health_status), 200


@app.route("/stats", methods=["GET"])
def get_app_stats():
    """Get application statistics"""
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "stats": get_stats(),
        "cache": CacheSystem.stats(),
        "sessions": len(SESSION_CONTEXTS),
        "tours": len(TOURS_DB),
        "knowledge_passages": len(MAPPING)
    }), 200


@app.route("/reindex", methods=["POST"])
def reindex():
    """Force rebuild of index"""
    try:
        logger.info("🔨 Starting index rebuild...")
        
        # Load knowledge
        load_knowledge()
        
        # Index tour names
        index_tour_names()
        
        # Build tour database
        build_tours_db()
        
        # Build FAISS/numpy index
        success = build_index(force_rebuild=True)
        
        if success:
            return jsonify({
                "status": "✅ Index rebuilt successfully",
                "tours": len(TOURS_DB),
                "passages": len(MAPPING),
                "timestamp": datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                "status": "⚠️ Index rebuild partial",
                "tours": len(TOURS_DB),
                "passages": len(MAPPING)
            }), 206
    
    except Exception as e:
        logger.error(f"❌ Reindex error: {e}")
        return jsonify({
            "status": "❌ Reindex failed",
            "error": str(e)
        }), 500


def initialize_app():
    """Initialize application on startup"""
    
    logger.info("🚀 Initializing Ruby Wings Chatbot v4.2...")
    
    try:
        # 1. Load knowledge base
        logger.info("📚 Loading knowledge base...")
        load_knowledge()
        logger.info(f"✅ Loaded {len(FLAT_TEXTS)} passages")
        
        # 2. Index tour names
        logger.info("📝 Indexing tour names...")
        index_tour_names()
        logger.info(f"✅ Indexed {len(TOUR_NAME_TO_INDEX)} tour names")
        
        # 3. Build tours database
        logger.info("🏢 Building tours database...")
        build_tours_db()
        logger.info(f"✅ Built database with {len(TOURS_DB)} tours")
        
        # 4. Build FAISS/numpy index
        logger.info("🔍 Building semantic index...")
        build_index()
        logger.info("✅ Index built")
        
        # 5. Memory optimization
        logger.info("🧠 Optimizing memory usage...")
        optimize_for_memory_profile()
        logger.info(f"✅ Memory profile: {RAM_PROFILE}MB")
        
        # 6. Test OpenAI connection
        if HAS_OPENAI and client:
            try:
                # Simple test
                logger.info("🤖 Testing OpenAI connection...")
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    temperature=0
                )
                logger.info(f"✅ OpenAI connected: {CHAT_MODEL}")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI connection issue: {e}")
        
        logger.info("=" * 50)
        logger.info("✨ RUBY WINGS CHATBOT V4.2 - READY! ✨")
        logger.info("=" * 50)
        logger.info(f"📊 Configuration:")
        logger.info(f"  • Knowledge: {len(KNOW)} sections, {len(FLAT_TEXTS)} passages")
        logger.info(f"  • Tours: {len(TOURS_DB)} tours, {len(TOUR_NAME_TO_INDEX)} indexed")
        logger.info(f"  • Index: {'FAISS' if HAS_FAISS else 'NumPy'} mode")
        logger.info(f"  • AI: {'OpenAI' if HAS_OPENAI else 'Fallback'} mode")
        logger.info(f"  • RAM: {RAM_PROFILE}MB profile")
        logger.info(f"  • Features: Hybrid deterministic + AI-powered responses")
        logger.info(f"  • Status: 🟢 ONLINE & READY TO SERVE!")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}\n{traceback.format_exc()}")
        logger.warning("⚠️ Starting in degraded mode...")


# =========== APP STARTUP ===========

if __name__ == "__main__":
    # Initialize
    initialize_app()
    
    # Run Flask app
    logger.info(f"🌐 Starting Flask server on {HOST}:{PORT}...")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True,
        use_reloader=False
    )