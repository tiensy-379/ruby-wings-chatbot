#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ruby Wings Chatbot v4.2 - Production Grade Hybrid AI
Combines deterministic business logic with OpenAI intelligence
Repository: tiensy-379/ruby-wings-chatbot
Language: Python 98.9%
"""

import os
import sys
import json
import time
import threading
import logging
import hashlib
import re
import unicodedata
import traceback
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from functools import lru_cache, wraps
from difflib import SequenceMatcher

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ruby_wings_v4.2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ruby-wings-v4.2")

# ==================== IMPORTS WITH FALLBACKS ====================
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("✅ NumPy available")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️ NumPy not available - using fallback")
    np = None

try:
    import faiss
    HAS_FAISS = True
    logger.info("✅ FAISS available")
except ImportError:
    HAS_FAISS = False
    logger.warning("⚠️ FAISS not available - using numpy fallback")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("⚠️ OpenAI SDK not available")

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GOOGLE_SHEETS = True
except ImportError:
    HAS_GOOGLE_SHEETS = False

from flask import Flask, request, jsonify, g
from flask_cors import CORS

# ==================== DATACLASSES & ENUMS ====================

class QuestionType(Enum):
    """Question classification types"""
    LISTING = "listing"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    INFORMATION = "information"
    CALCULATION = "calculation"
    COMPLEX = "complex"


class ConversationState(Enum):
    """Conversation state tracking"""
    INITIAL = "initial"
    TOUR_SELECTED = "tour_selected"
    ASKING_DETAILS = "asking_details"
    COMPARING = "comparing"
    RECOMMENDATION = "recommendation"
    BOOKING = "booking"
    PAYMENT = "payment"
    COMPLETED = "completed"
    FAREWELL = "farewell"


class PriceLevel(Enum):
    """Price classification"""
    BUDGET = "budget"
    MIDRANGE = "midrange"
    PREMIUM = "premium"


class DurationType(Enum):
    """Duration classification"""
    ONE_DAY = "1_day"
    TWO_DAYS = "2_days"
    THREE_DAYS = "3_days"
    FOUR_PLUS = "4_plus"


@dataclass
class Tour:
    """Structured tour object"""
    index: int
    name: str = None
    duration: str = None
    location: str = None
    price: str = None
    summary: str = None
    includes: List[str] = field(default_factory=list)
    accommodation: str = None
    meals: str = None
    transport: str = None
    notes: str = None
    style: str = None
    tags: List[str] = field(default_factory=list)
    completeness_score: float = 0.0

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'index': self.index,
            'name': self.name,
            'duration': self.duration,
            'location': self.location,
            'price': self.price,
            'summary': self.summary,
            'includes': self.includes,
            'accommodation': self.accommodation,
            'meals': self.meals,
            'transport': self.transport,
            'notes': self.notes,
            'style': self.style,
            'tags': self.tags,
            'completeness_score': self.completeness_score
        }


@dataclass
class UserProfile:
    """User profile for personalized recommendations"""
    age_group: Optional[str] = None
    group_type: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    budget_level: Optional[str] = None
    physical_level: Optional[str] = None
    preferred_location: Optional[str] = None
    special_requirements: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0


@dataclass
class FilterSet:
    """Mandatory filters for tour search"""
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    location: Optional[str] = None
    near_location: Optional[str] = None
    month: Optional[int] = None
    weekend: bool = False
    holiday: Optional[str] = None
    group_type: Optional[str] = None

    def is_empty(self) -> bool:
        """Check if all filters are empty"""
        return all([
            self.price_min is None,
            self.price_max is None,
            self.duration_min is None,
            self.duration_max is None,
            self.location is None,
            self.near_location is None,
            self.month is None,
            not self.weekend,
            self.holiday is None,
            self.group_type is None
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'price_min': self.price_min,
            'price_max': self.price_max,
            'duration_min': self.duration_min,
            'duration_max': self.duration_max,
            'location': self.location,
            'near_location': self.near_location,
            'month': self.month,
            'weekend': self.weekend,
            'holiday': self.holiday,
            'group_type': self.group_type
        }

    def __str__(self) -> str:
        """String representation"""
        parts = []
        if self.price_min or self.price_max:
            parts.append(f"Giá: {self.price_min or 'min'}-{self.price_max or 'max'} VNĐ")
        if self.duration_min or self.duration_max:
            parts.append(f"Thời gian: {self.duration_min or 'min'}-{self.duration_max or 'max'} ngày")
        if self.location:
            parts.append(f"Địa điểm: {self.location}")
        if self.group_type:
            parts.append(f"Nhóm: {self.group_type}")
        return " | ".join(parts) if parts else "Không có bộ lọc"


@dataclass
class ConversationContext:
    """Session conversation context with full state tracking"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[Dict] = field(default_factory=list)
    current_tours: List[int] = field(default_factory=list)
    last_tour_name: Optional[str] = None
    last_successful_tours: List[int] = field(default_factory=list)
    mentioned_tours: Set[int] = field(default_factory=set)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    user_profile: Optional[UserProfile] = None
    filters_applied: FilterSet = field(default_factory=FilterSet)
    interaction_count: int = 0
    last_intent: Optional[str] = None

    def update(self, user_msg: str, bot_msg: str, tour_indices: List[int] = None):
        """Update context with new interaction"""
        self.last_updated = datetime.utcnow()
        self.interaction_count += 1
        
        self.conversation_history.append({
            'role': 'user',
            'message': user_msg,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self.conversation_history.append({
            'role': 'assistant',
            'message': bot_msg,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if tour_indices:
            self.last_successful_tours = tour_indices
            for idx in tour_indices:
                self.mentioned_tours.add(idx)

    def get_context_summary(self) -> str:
        """Get brief context summary"""
        summary = []
        if self.current_tours:
            summary.append(f"Current tours: {len(self.current_tours)}")
        if self.last_tour_name:
            summary.append(f"Last tour: {self.last_tour_name}")
        if self.user_profile and self.user_profile.group_type:
            summary.append(f"Group type: {self.user_profile.group_type}")
        return " | ".join(summary) if summary else "New session"


@dataclass
class CacheEntry:
    """Cache entry with TTL management"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int = 300
    access_count: int = 0
    last_accessed: datetime = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def age_seconds(self) -> float:
        """Get age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class ChatResponse:
    """Structured chat response"""
    reply: str
    sources: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    tour_indices: List[int] = field(default_factory=list)
    processing_time_ms: int = 0
    from_memory: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON"""
        return {
            'reply': self.reply,
            'sources': self.sources,
            'context': self.context,
            'tour_indices': self.tour_indices,
            'processing_time_ms': self.processing_time_ms,
            'from_memory': self.from_memory
        }


# ==================== ENVIRONMENT & CONFIGURATION ====================

# Memory profile
RAM_PROFILE = os.environ.get("RAM_PROFILE", "2048").strip()
IS_LOW_RAM = RAM_PROFILE == "512"
IS_HIGH_RAM = RAM_PROFILE == "2048"

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# Knowledge & Index paths
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

# Models
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "10"))

# FAISS
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# Server
FLASK_ENV = os.environ.get("FLASK_ENV", "production")
DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
SECRET_KEY = os.environ.get("SECRET_KEY", "ruby-wings-secret-2024")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "https://www.rubywings.vn,http://localhost:3000").split(",")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "10000"))

# Timeout
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))

# ==================== UPGRADE 1: MANDATORY FILTER SYSTEM ====================

class MandatoryFilterSystem:
    """UPGRADE 1: Extract and apply mandatory filters"""

    FILTER_PATTERNS = {
        'duration': [
            (r'(\d+)\s*ngày\s*(\d+)\s*đêm', 'days_nights'),
            (r'(\d+)\s*ngày', 'exact_duration'),
            (r'mấy\s*ngày|bao lâu', 'ask_duration'),
        ],
        'price': [
            (r'dưới\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'max_price'),
            (r'trên\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'min_price'),
            (r'khoảng\s*(\d[\d,\.]*)\s*(?:đến|-)\s*(\d[\d,\.]*)\s*(triệu|tr|k|nghìn)', 'price_range'),
            (r'giá.*bao nhiêu|bao nhiêu tiền', 'ask_price'),
        ],
        'location': [
            (r'(?:ở|tại|về|đến)\s+([^.,!?\n]+)', 'location'),
            (r'(?:địa điểm|nơi|vùng)\s+([^.,!?\n]+)', 'location'),
        ],
        'group_type': [
            (r'gia đình|family', 'family'),
            (r'cặp đôi|couple|đôi lứa', 'couple'),
            (r'nhóm bạn|bạn bè|friends', 'friends'),
            (r'công ty|doanh nghiệp|team building', 'corporate'),
            (r'một mình|solo', 'solo'),
            (r'người lớn tuổi|cao tuổi|cựu chiến binh', 'senior'),
        ]
    }

    @staticmethod
    def extract_filters(message: str) -> FilterSet:
        """Extract all mandatory filters from user message"""
        filters = FilterSet()
        
        if not message:
            return filters
        
        message_lower = message.lower()
        
        # Duration extraction
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['duration']:
            match = re.search(pattern, message_lower)
            if match:
                if filter_type == 'exact_duration':
                    try:
                        days = int(match.group(1))
                        filters.duration_min = days
                        filters.duration_max = days
                        logger.info(f"📅 Duration filter: {days} days")
                    except (ValueError, IndexError):
                        pass
                elif filter_type == 'days_nights':
                    try:
                        days = int(match.group(1))
                        filters.duration_min = days
                        filters.duration_max = days
                    except (ValueError, IndexError):
                        pass
        
        # Price extraction
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['price']:
            match = re.search(pattern, message_lower)
            if match:
                try:
                    if filter_type == 'max_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters.price_max = amount
                            logger.info(f"💰 Max price filter: {amount} VNĐ")
                    
                    elif filter_type == 'min_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters.price_min = amount
                            logger.info(f"💰 Min price filter: {amount} VNĐ")
                    
                    elif filter_type == 'price_range':
                        min_amt = MandatoryFilterSystem._parse_price(match.group(1), match.group(3))
                        max_amt = MandatoryFilterSystem._parse_price(match.group(2), match.group(3))
                        if min_amt and max_amt:
                            filters.price_min = min_amt
                            filters.price_max = max_amt
                            logger.info(f"💰 Price range: {min_amt}-{max_amt} VNĐ")
                except (ValueError, IndexError, AttributeError):
                    continue
        
        # Location extraction
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['location']:
            match = re.search(pattern, message_lower)
            if match:
                location = match.group(1).strip()
                if location and len(location) > 1:
                    filters.location = location
                    logger.info(f"📍 Location filter: {location}")
        
        # Group type extraction
        for pattern, group_type in MandatoryFilterSystem.FILTER_PATTERNS['group_type']:
            if re.search(pattern, message_lower):
                filters.group_type = group_type
                logger.info(f"👥 Group type: {group_type}")
                break
        
        return filters

    @staticmethod
    def _parse_price(amount_str: str, unit: str) -> Optional[int]:
        """Parse price string to integer"""
        if not amount_str:
            return None
        
        try:
            amount_str = amount_str.replace(',', '').replace('.', '')
            if not amount_str.isdigit():
                return None
            
            amount = int(amount_str)
            
            if unit in ['triệu', 'tr']:
                return amount * 1000000
            elif unit in ['k', 'nghìn']:
                return amount * 1000
            else:
                return amount if amount > 1000 else amount * 1000
        
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def apply_filters(tours_db: Dict[int, 'Tour'], filters: FilterSet) -> List[int]:
        """Apply filters to tour database"""
        if filters.is_empty() or not tours_db:
            return list(tours_db.keys())
        
        passing_tours = []
        
        for tour_idx, tour in tours_db.items():
            passes_all = True
            
            # Price filtering
            if passes_all and (filters.price_max is not None or filters.price_min is not None):
                tour_price_text = tour.price or ""
                if tour_price_text and tour_price_text.lower() != 'liên hệ':
                    tour_prices = MandatoryFilterSystem._extract_tour_prices(tour_price_text)
                    if tour_prices:
                        min_price = min(tour_prices)
                        max_price = max(tour_prices)
                        
                        if filters.price_max is not None and min_price > filters.price_max:
                            passes_all = False
                        if filters.price_min is not None and max_price < filters.price_min:
                            passes_all = False
            
            # Duration filtering
            if passes_all and (filters.duration_min is not None or filters.duration_max is not None):
                duration_text = (tour.duration or "").lower()
                tour_duration = MandatoryFilterSystem._extract_duration_days(duration_text)
                
                if tour_duration is not None:
                    if filters.duration_min is not None and tour_duration < filters.duration_min:
                        passes_all = False
                    if filters.duration_max is not None and tour_duration > filters.duration_max:
                        passes_all = False
            
            # Location filtering
            if passes_all and filters.location is not None:
                tour_location = (tour.location or "").lower()
                if filters.location.lower() not in tour_location:
                    passes_all = False
            
            # Group type filtering
            if passes_all and filters.group_type:
                tour_tags = [tag.lower() for tag in (tour.tags or [])]
                group_keywords = {
                    'family': ['family', 'gia đình'],
                    'senior': ['senior', 'lớn tuổi', 'cựu chiến binh'],
                    'friends': ['friends', 'nhóm', 'bạn'],
                    'corporate': ['corporate', 'team', 'công ty'],
                }
                
                if filters.group_type in group_keywords:
                    keywords = group_keywords[filters.group_type]
                    if not any(kw in ' '.join(tour_tags) for kw in keywords):
                        passes_all = False
            
            if passes_all:
                passing_tours.append(tour_idx)
        
        logger.info(f"✅ Filter applied: {len(passing_tours)}/{len(tours_db)} tours pass")
        return passing_tours

    @staticmethod
    def _extract_tour_prices(price_text: str) -> List[int]:
        """Extract prices from tour price text"""
        prices = []
        
        patterns = [
            r'(\d[\d,\.]+)\s*(?:triệu|tr)',
            r'(\d[\d,\.]+)\s*(?:k|nghìn)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, price_text, re.IGNORECASE)
            for match in matches:
                try:
                    num_str = match.group(1).replace(',', '').replace('.', '')
                    if num_str.isdigit():
                        num = int(num_str)
                        if 'triệu' in match.group(0).lower() or 'tr' in match.group(0).lower():
                            num = num * 1000000
                        elif 'k' in match.group(0).lower() or 'nghìn' in match.group(0).lower():
                            num = num * 1000
                        prices.append(num)
                except (ValueError, AttributeError):
                    continue
        
        return prices

    @staticmethod
    def _extract_duration_days(duration_text: str) -> Optional[int]:
        """Extract duration in days"""
        if not duration_text:
            return None
        
        patterns = [
            r'(\d+)\s*ngày',
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


# ==================== UPGRADE 2: DEDUPLICATION ENGINE ====================

class DeduplicationEngine:
    """UPGRADE 2: Remove duplicate and similar results"""

    SIMILARITY_THRESHOLD = 0.85

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between texts"""
        if not text1 or not text2:
            return 0.0
        
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def deduplicate_passages(passages: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        """Remove duplicate passages"""
        if len(passages) <= 1:
            return passages
        
        unique = []
        seen_texts = []
        
        for score, passage in sorted(passages, key=lambda x: x[0], reverse=True):
            text = passage.get('text', '').strip()
            
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = DeduplicationEngine.calculate_similarity(text, seen_text)
                if similarity > DeduplicationEngine.SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append((score, passage))
                seen_texts.append(text)
        
        logger.info(f"🔄 Deduplicated: {len(passages)} → {len(unique)} passages")
        return unique


# ==================== UPGRADE 3: ENHANCED FIELD DETECTOR ====================

class EnhancedFieldDetector:
    """UPGRADE 3: Better detection of user intent"""

    FIELD_PATTERNS = {
        'price': [r'giá.*bao nhiêu', r'bao nhiêu tiền', r'chi phí'],
        'duration': [r'bao lâu', r'mấy ngày', r'thời gian'],
        'location': [r'ở đâu', r'địa điểm', r'nơi nào'],
        'summary': [r'có gì', r'thế nào', r'giới thiệu'],
        'includes': [r'bao gồm', r'có gì', r'lịch trình'],
    }

    @staticmethod
    def detect_field(message: str) -> Tuple[Optional[str], float]:
        """Detect which field user is asking about"""
        if not message:
            return None, 0.0
        
        message_lower = message.lower()
        scores = {}
        
        for field, patterns in EnhancedFieldDetector.FIELD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    scores[field] = max(scores.get(field, 0.0), 0.9)
        
        if scores:
            best_field = max(scores.items(), key=lambda x: x[1])
            return best_field[0], best_field[1]
        
        return None, 0.0


# ==================== UPGRADE 4-10: CORE CHAT LOGIC ====================

class QuestionPipeline:
    """UPGRADE 4: Process different question types"""

    @staticmethod
    def classify_question(message: str) -> Tuple[QuestionType, float]:
        """Classify question type"""
        message_lower = message.lower()
        
        type_scores = defaultdict(float)
        
        # Comparison
        if any(kw in message_lower for kw in ['so sánh', 'khác nhau', 'tốt hơn']):
            type_scores[QuestionType.COMPARISON] = 0.9
        
        # Recommendation
        if any(kw in message_lower for kw in ['phù hợp', 'gợi ý', 'nên', 'tư vấn']):
            type_scores[QuestionType.RECOMMENDATION] = 0.9
        
        # Listing
        if any(kw in message_lower for kw in ['liệt kê', 'danh sách', 'có những']):
            type_scores[QuestionType.LISTING] = 0.9
        
        # Greeting
        if any(kw in message_lower for kw in ['xin chào', 'chào', 'hello', 'hi']):
            type_scores[QuestionType.GREETING] = 0.95
        
        # Farewell
        if any(kw in message_lower for kw in ['tạm biệt', 'goodbye', 'cảm ơn']):
            type_scores[QuestionType.FAREWELL] = 0.95
        
        # Default
        if not type_scores:
            type_scores[QuestionType.INFORMATION] = 0.6
        
        best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
        logger.info(f"🎯 Question classification: {best_type.value} ({best_score:.2f})")
        
        return best_type, best_score


class FuzzyMatcher:
    """UPGRADE 6: Handle tour name variations"""

    @staticmethod
    def normalize_vietnamese(text: str) -> str:
        """Normalize Vietnamese text"""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def find_similar_tours(query: str, tour_names: Dict[str, int]) -> List[Tuple[int, float]]:
        """Find similar tour names"""
        if not query or not tour_names:
            return []
        
        query_norm = FuzzyMatcher.normalize_vietnamese(query)
        matches = []
        
        for tour_name, tour_idx in tour_names.items():
            tour_norm = FuzzyMatcher.normalize_vietnamese(tour_name)
            
            # Similarity calculation
            similarity = SequenceMatcher(None, query_norm, tour_norm).ratio()
            
            # Word overlap
            query_words = set(query_norm.split())
            tour_words = set(tour_norm.split())
            if query_words and tour_words:
                overlap = len(query_words & tour_words) / max(len(query_words), len(tour_words))
                similarity = (similarity + overlap) / 2
            
            if similarity > 0.5:
                matches.append((tour_idx, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"🔍 Fuzzy matched: {len(matches)} tours for '{query}'")
        
        return matches[:5]


class ConversationStateMachine:
    """UPGRADE 7: Track conversation state"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = ConversationState.INITIAL
        self.context = ConversationContext(session_id=session_id)

    def update(self, user_msg: str, bot_msg: str, tour_indices: List[int] = None):
        """Update state based on interaction"""
        self.context.update(user_msg, bot_msg, tour_indices)
        
        message_lower = user_msg.lower()
        
        # State transitions
        if any(kw in message_lower for kw in ['tạm biệt', 'goodbye', 'cảm ơn']):
            self.state = ConversationState.FAREWELL
        elif 'so sánh' in message_lower:
            self.state = ConversationState.COMPARING
        elif 'phù hợp' in message_lower or 'gợi ý' in message_lower:
            self.state = ConversationState.RECOMMENDATION
        elif tour_indices:
            self.state = ConversationState.TOUR_SELECTED
        
        logger.info(f"🔄 State: {self.state.value}")


class SemanticAnalyzer:
    """UPGRADE 8: Deep semantic understanding"""

    @staticmethod
    def analyze_user_profile(message: str) -> UserProfile:
        """Build user profile from message"""
        profile = UserProfile()
        
        message_lower = message.lower()
        
        # Detect group type
        group_mapping = {
            'family': ['gia đình', 'con nhỏ', 'trẻ em'],
            'senior': ['người lớn tuổi', 'cao tuổi', 'cựu chiến binh'],
            'friends': ['nhóm bạn', 'bạn bè'],
            'corporate': ['công ty', 'team building'],
        }
        
        for group_type, keywords in group_mapping.items():
            if any(kw in message_lower for kw in keywords):
                profile.group_type = group_type
                profile.confidence_scores['group_type'] = 0.8
                break
        
        # Detect interests
        interest_mapping = {
            'history': ['lịch sử', 'di tích'],
            'nature': ['thiên nhiên', 'rừng'],
            'meditation': ['thiền', 'yoga'],
            'culture': ['văn hóa', 'ẩm thực'],
        }
        
        for interest, keywords in interest_mapping.items():
            if any(kw in message_lower for kw in keywords):
                profile.interests.append(interest)
                profile.confidence_scores[f'interest_{interest}'] = 0.8
        
        return profile


class AutoValidator:
    """UPGRADE 9: Validate information"""

    @staticmethod
    def validate_response(response: str) -> str:
        """Validate tour information in response"""
        if not response:
            return response
        
        # Check for unrealistic data
        if re.search(r'\d+\s*(?:triệu|tr)\s*(?:đêm|ngày)', response):
            logger.warning("⚠️ Unrealistic price-duration combo detected")
        
        return response


class TemplateSystem:
    """UPGRADE 10: Structured responses"""

    TEMPLATES = {
        'greeting': "👋 Xin chào! Tôi là trợ lý AI của Ruby Wings Travel.\n\n"
                   "Tôi có thể giúp bạn tìm tour phù hợp từ 32 hành trình đặc sắc. "
                   "Hãy cho biết nhu cầu của bạn! 😊",
        
        'farewell': "🙏 Cảm ơn bạn đã trao dồi cùng Ruby Wings!\n\n"
                   "Chúc bạn có một hành trình tuyệt vời. 📞 Hotline: 0332510486",
        
        'no_tours': "Hiện chưa có tour phù hợp với tiêu chí của bạn. "
                   "📞 Liên hệ 0332510486 để được thiết kế tour riêng!",
    }

    @staticmethod
    def render(template_name: str, **kwargs) -> str:
        """Render template"""
        return TemplateSystem.TEMPLATES.get(template_name, "")


# ==================== CACHE SYSTEM ====================

class CacheSystem:
    """Memory-optimized cache with TTL"""

    _cache: Dict[str, CacheEntry] = {}
    _lock = threading.Lock()

    @staticmethod
    def get_key(message: str, context_hash: str = "") -> str:
        """Generate cache key"""
        key_parts = [message]
        if context_hash:
            key_parts.append(context_hash)
        
        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get from cache"""
        with CacheSystem._lock:
            if key not in CacheSystem._cache:
                return None
            
            entry = CacheSystem._cache[key]
            if entry.is_expired():
                del CacheSystem._cache[key]
                return None
            
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            return entry.value

    @staticmethod
    def set(key: str, value: Any, ttl: int = 300):
        """Set in cache"""
        with CacheSystem._lock:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl
            )
            CacheSystem._cache[key] = entry
            
            # Clean expired entries
            expired = [k for k, v in CacheSystem._cache.items() if v.is_expired()]
            for k in expired:
                del CacheSystem._cache[k]
            
            logger.debug(f"💾 Cached: {key} (TTL: {ttl}s)")


# ==================== OPENAI INTEGRATION ====================

def create_openai_client():
    """Create OpenAI client"""
    if not HAS_OPENAI or not OPENAI_API_KEY:
        logger.warning("⚠️ OpenAI not available")
        return None
    
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        logger.info("✅ OpenAI client created")
        return client
    except Exception as e:
        logger.error(f"❌ OpenAI client error: {e}")
        return None


# Initialize OpenAI client
openai_client = create_openai_client()


def embed_text(text: str) -> Tuple[Optional[List[float]], int]:
    """Embed text using OpenAI"""
    if not openai_client:
        return None, 0
    
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:2000]
        )
        
        if response.data:
            embedding = response.data[0].embedding
            return embedding, len(embedding)
    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
    
    return None, 0


# ==================== KNOWLEDGE BASE LOADING ====================

KNOW = {}
FLAT_TEXTS = []
MAPPING = []
TOURS_DB = {}
TOUR_NAME_TO_INDEX = {}
SESSION_CONTEXTS = {}
SESSION_LOCK = threading.Lock()


def load_knowledge(path: str = KNOWLEDGE_PATH):
    """Load knowledge base from JSON"""
    global KNOW, FLAT_TEXTS, MAPPING
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
        logger.info(f"✅ Loaded knowledge from {path}")
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        KNOW = {}
        return
    
    # Flatten knowledge
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
            text = obj.strip()
            if text:
                FLAT_TEXTS.append(text)
                MAPPING.append({"path": prefix, "text": text})
    
    scan(KNOW)
    logger.info(f"📊 Knowledge flattened: {len(FLAT_TEXTS)} passages")


def build_tours_db():
    """Build structured tour database"""
    global TOURS_DB, TOUR_NAME_TO_INDEX
    
    TOURS_DB = {}
    TOUR_NAME_TO_INDEX = {}
    
    # Extract tours from MAPPING
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        
        if not path or not text:
            continue
        
        # Match tour index
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            continue
        
        tour_idx = int(tour_match.group(1))
        
        # Initialize tour
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = Tour(index=tour_idx)
        
        # Extract field
        field_match = re.search(r'tours\[\d+\]\.(\w+)', path)
        if field_match:
            field = field_match.group(1)
            tour = TOURS_DB[tour_idx]
            
            if field == 'tour_name':
                tour.name = text
                # Index tour name
                norm_name = FuzzyMatcher.normalize_vietnamese(text)
                if norm_name and norm_name not in TOUR_NAME_TO_INDEX:
                    TOUR_NAME_TO_INDEX[norm_name] = tour_idx
            
            elif field == 'duration':
                tour.duration = text
            elif field == 'location':
                tour.location = text
            elif field == 'price':
                tour.price = text
            elif field == 'summary':
                tour.summary = text
            elif field == 'includes':
                if isinstance(tour.includes, list):
                    tour.includes.append(text)
            elif field == 'accommodation':
                tour.accommodation = text
            elif field == 'meals':
                tour.meals = text
            elif field == 'transport':
                tour.transport = text
            elif field == 'notes':
                tour.notes = text
    
    logger.info(f"✅ Built tours database: {len(TOURS_DB)} tours")


def get_session_context(session_id: str) -> ConversationContext:
    """Get or create session context"""
    with SESSION_LOCK:
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = ConversationContext(session_id=session_id)
        return SESSION_CONTEXTS[session_id]


# ==================== FLASK APP ====================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, origins=CORS_ORIGINS)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tours_loaded": len(TOURS_DB),
        "passages_indexed": len(FLAT_TEXTS),
        "openai_available": HAS_OPENAI,
        "faiss_available": HAS_FAISS
    })


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Main chat endpoint - HYBRID AI + DETERMINISTIC LOGIC"""
    start_time = time.time()
    
    try:
        # ========== INPUT VALIDATION ==========
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        session_id = data.get("session_id", f"session_{hashlib.md5(request.remote_addr.encode()).hexdigest()[:12]}")
        
        if not user_message:
            return jsonify(TemplateSystem.render('greeting'))
        
        # ========== SESSION CONTEXT ==========
        context = get_session_context(session_id)
        context.last_updated = datetime.utcnow()
        context.interaction_count += 1
        
        # ========== PHASE 1: DETERMINISTIC LOGIC ==========
        
        # Extract mandatory filters
        filters = MandatoryFilterSystem.extract_filters(user_message)
        
        # Classify question
        question_type, confidence = QuestionPipeline.classify_question(user_message)
        
        # Detect field
        field, field_score = EnhancedFieldDetector.detect_field(user_message)
        
        # ========== PHASE 2: TOUR RESOLUTION ==========
        
        tour_indices = []
        
        # Strategy 1: Direct tour name matching
        for norm_name, idx in TOUR_NAME_TO_INDEX.items():
            if norm_name in user_message.lower():
                tour_indices.append(idx)
        
        # Strategy 2: Fuzzy matching
        if not tour_indices:
            fuzzy_matches = FuzzyMatcher.find_similar_tours(user_message, TOUR_NAME_TO_INDEX)
            tour_indices = [idx for idx, score in fuzzy_matches if score > 0.6]
        
        # Strategy 3: Filter-based search
        if not tour_indices and not filters.is_empty():
            tour_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, filters)
        
        # ========== PHASE 3: AI-POWERED RESPONSE ==========
        
        reply = ""
        
        # Handle greeting/farewell with templates
        if question_type == QuestionType.GREETING:
            reply = TemplateSystem.render('greeting')
        
        elif question_type == QuestionType.FAREWELL:
            reply = TemplateSystem.render('farewell')
        
        # Deterministic responses for booking/policy
        elif 'đặt' in user_message.lower() or 'booking' in user_message.lower():
            reply = _get_booking_response()
        
        # AI-powered for complex queries
        elif openai_client and (question_type in [QuestionType.COMPARISON, QuestionType.RECOMMENDATION]):
            reply = _get_ai_response(user_message, tour_indices, context, openai_client)
        
        # Rule-based for simple queries
        else:
            if tour_indices:
                reply = _format_tour_response(tour_indices)
            else:
                reply = _get_fallback_response(user_message, filters)
        
        # ========== RESPONSE FORMATTING ==========
        
        # Ensure response has hotline
        if "0332510486" not in reply:
            reply += "\n\n📞 **Hotline tư vấn 24/7:** 0332510486"
        
        # Update context
        context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        context.conversation_history.append({
            'role': 'assistant',
            'message': reply,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if tour_indices:
            context.last_successful_tours = tour_indices
        
        # ========== CACHE & RETURN ==========
        
        processing_time = time.time() - start_time
        
        response = ChatResponse(
            reply=reply,
            sources=[],
            context={
                'session_id': session_id,
                'question_type': question_type.value,
                'field': field,
                'tours_found': len(tour_indices),
                'filters_applied': not filters.is_empty()
            },
            tour_indices=tour_indices,
            processing_time_ms=int(processing_time * 1000)
        )
        
        return jsonify(response.to_dict())
    
    except Exception as e:
        logger.error(f"❌ Chat error: {e}\n{traceback.format_exc()}")
        
        error_response = ChatResponse(
            reply="⚡ Có chút trục trặc kỹ thuật. 📞 Hotline: 0332510486",
            context={'error': str(e)},
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        return jsonify(error_response.to_dict()), 500


def _get_ai_response(message: str, tour_indices: List[int], context: ConversationContext, client) -> str:
    """Use OpenAI for complex queries"""
    try:
        # Build context
        context_str = ""
        if tour_indices:
            for idx in tour_indices[:3]:
                tour = TOURS_DB.get(idx)
                if tour:
                    context_str += f"\nTour: {tour.name}\n"
                    if tour.price:
                        context_str += f"Giá: {tour.price}\n"
                    if tour.duration:
                        context_str += f"Thời gian: {tour.duration}\n"
        
        prompt = f"""Bạn là cố vấn du lịch Ruby Wings chuyên nghiệp.

Thông tin tour:
{context_str or 'Không có tour cụ thể'}

Yêu cầu: Trả lời câu hỏi một cách thân thiện, tự nhiên, chuyên nghiệp.
Luôn kết thúc bằng: "📞 Hotline: 0332510486"

Câu hỏi: {message}"""
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=600,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.choices:
            return response.choices[0].message.content or ""
    
    except Exception as e:
        logger.error(f"❌ AI error: {e}")
    
    return _get_fallback_response(message, FilterSet())


def _format_tour_response(tour_indices: List[int]) -> str:
    """Format tour information"""
    reply = "✨ **TOUR RUBY WINGS** ✨\n\n"
    
    for i, idx in enumerate(tour_indices[:3], 1):
        tour = TOURS_DB.get(idx)
        if not tour:
            continue
        
        reply += f"{i}. **{tour.name}**\n"
        if tour.duration:
            reply += f"   ⏱️ {tour.duration}\n"
        if tour.location:
            reply += f"   📍 {tour.location}\n"
        if tour.price:
            price_short = tour.price[:60] + "..." if len(tour.price) > 60 else tour.price
            reply += f"   💰 {price_short}\n"
        reply += "\n"
    
    return reply


def _get_booking_response() -> str:
    """Return booking information"""
    return """📝 **QUY TRÌNH ĐẶT TOUR**

1️⃣ **Tư vấn**: Liên hệ 0332510486 hoặc nhắn tin
2️⃣ **Đặt cọc**: 30% giá tour để xác nhận
3️⃣ **Thanh toán**: 70% còn lại trước 7 ngày khởi hành
4️⃣ **Khởi hành**: Hưởng thụ chuyến đi đặc sắc

📞 **Hotline: 0332510486**"""


def _get_fallback_response(message: str, filters: FilterSet) -> str:
    """Generate fallback response"""
    message_lower = message.lower()
    
    if 'giá' in message_lower:
        return "💰 **BẢNG GIÁ THAM KHẢO**\n\n" \
               "• Tour 1 ngày: 600k - 1.5 triệu\n" \
               "• Tour 2 ngày: 1.5 - 3 triệu\n" \
               "• Tour 3+ ngày: 3 - 5 triệu\n\n" \
               "📞 Hotline: 0332510486"
    
    elif 'phù hợp' in message_lower or 'nên' in message_lower:
        return "Ruby Wings có tour phù hợp với mọi nhu cầu:\n\n" \
               "👨‍👩‍👧‍👦 **Gia đình:** Tour nhẹ nhàng, an toàn\n" \
               "🏛️ **Lịch sử:** Tour lịch sử, di tích\n" \
               "🧘 **Retreat:** Tour thiền, chữa lành\n\n" \
               "📞 Liên hệ tư vấn: 0332510486"
    
    else:
        return "Ruby Wings có 32 tour du lịch trải nghiệm đặc sắc.\n\n" \
               "Bạn có thể hỏi:\n" \
               "• Tên tour cụ thể\n" \
               "• Giá tour\n" \
               "• Tour phù hợp gia đình\n\n" \
               "📞 Hotline: 0332510486"


# ==================== INITIALIZATION ====================

def initialize_app():
    """Initialize application on startup"""
    logger.info("🚀 Initializing Ruby Wings Chatbot v4.2...")
    
    # Load knowledge base
    load_knowledge()
    
    # Build tours database
    build_tours_db()
    
    logger.info("✅ Application initialized successfully")
    logger.info(f"📊 Stats: {len(TOURS_DB)} tours, {len(FLAT_TEXTS)} passages")


# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info(f"🌟 Ruby Wings Chatbot v4.2")
    logger.info(f"Config: CHAT_MODEL={CHAT_MODEL}, EMBEDDING_MODEL={EMBEDDING_MODEL}")
    logger.info(f"FAISS={FAISS_ENABLED}, OpenAI={HAS_OPENAI}, NumPy={NUMPY_AVAILABLE}")
    
    # Initialize
    initialize_app()
    
    # Run Flask
    logger.info(f"🚀 Starting Flask server on {HOST}:{PORT}")
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True,
        use_reloader=False
    )