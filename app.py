# app.py — RUBY WINGS CHATBOT v4.0 - COMPLETE UPGRADES
# =========== TÍCH HỢP ĐẦY ĐỦ 10 UPGRADES ===========
# =========== BẢO TOÀN TOÀN BỘ INTEGRATION HIỆN CÓ ===========

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
from typing import List, Tuple, Dict, Optional, Any, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from difflib import SequenceMatcher
from enum import Enum

import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS

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
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

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
        }
    
    @staticmethod
    def is_enabled(upgrade_name: str) -> bool:
        flags = UpgradeFlags.get_all_flags()
        return flags.get(f"UPGRADE_{upgrade_name}", False)

# =========== GLOBAL STATE ===========
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# Initialize OpenAI client
if HAS_OPENAI and OPENAI_API_KEY:
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=30.0,
            max_retries=3
        )
        logger.info("✅ OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ OpenAI client initialization failed: {e}")
        client = None
else:
    client = None
    logger.warning("⚠️ OpenAI client not available - using fallback responses")

# Knowledge base state
KNOW: Dict = {}                      # Raw knowledge.json data
FLAT_TEXTS: List[str] = []           # All text passages for indexing
MAPPING: List[Dict] = []             # Mapping from text to original path
INDEX = None                         # FAISS or numpy index
INDEX_LOCK = threading.Lock()        # Thread safety for index operations

# Tour databases
TOUR_NAME_TO_INDEX: Dict[str, int] = {}      # Normalized tour name → index
TOURS_DB: Dict[int, Dict[str, Any]] = {}     # Structured tour database
TOUR_TAGS: Dict[int, List[str]] = {}         # Auto-generated tags for filtering
TOUR_METADATA: Dict[int, Dict] = {}          # Additional metadata (popularity, etc.)

# Session management
SESSION_CONTEXTS: Dict[str, 'EnhancedContext'] = {}
SESSION_LOCK = threading.Lock()
SESSION_TIMEOUT = 1800  # 30 minutes

# Cache system
_response_cache: Dict[str, Tuple[datetime, Any]] = {}
_cache_lock = threading.Lock()

# =========== UPGRADE 1: MANDATORY FILTER SYSTEM ===========
class MandatoryFilterSystem:
    """
    UPGRADE 1: Extract and apply mandatory filters BEFORE semantic search
    Fixes: "tour dưới 1 triệu" should filter BEFORE searching
    """
    
    # Comprehensive filter patterns
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
    def extract_filters(message: str) -> Dict[str, Any]:
        """
        Extract ALL mandatory filters from user message
        Returns: Dict with filter type → value
        """
        if not message:
            return {}
        
        message_lower = message.lower()
        filters = {}
        
        # 1. DURATION FILTERS
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['duration']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                if filter_type == 'exact_duration':
                    try:
                        days = int(match.group(1))
                        filters['duration_exact'] = days
                        filters['duration_min'] = days
                        filters['duration_max'] = days
                    except (ValueError, IndexError):
                        pass
                elif filter_type == 'days_nights':
                    try:
                        days = int(match.group(1))
                        nights = int(match.group(2)) if match.group(2) else days
                        filters['duration_days'] = days
                        filters['duration_nights'] = nights
                    except (ValueError, IndexError):
                        pass
        
        # 2. PRICE FILTERS (MOST IMPORTANT - FIXES THE BUG)
        for pattern, filter_type in MandatoryFilterSystem.FILTER_PATTERNS['price']:
            matches = list(re.finditer(pattern, message_lower))
            for match in matches:
                try:
                    if filter_type == 'max_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters['price_max'] = amount
                            logger.info(f"💰 Extracted MAX price filter: {amount} VND")
                    
                    elif filter_type == 'min_price':
                        amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(2))
                        if amount:
                            filters['price_min'] = amount
                            logger.info(f"💰 Extracted MIN price filter: {amount} VND")
                    
                    elif filter_type == 'price_range':
                        min_amount = MandatoryFilterSystem._parse_price(match.group(1), match.group(3))
                        max_amount = MandatoryFilterSystem._parse_price(match.group(2), match.group(3))
                        if min_amount and max_amount:
                            filters['price_min'] = min_amount
                            filters['price_max'] = max_amount
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
                        filters['location'] = location
                    elif filter_type == 'near_location':
                        filters['near_location'] = location
        
        # 4. SPECIAL KEYWORDS that imply filters
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
                filters[filter_key] = value
        
        logger.info(f"🎯 Extracted filters: {filters}")
        return filters
    
    @staticmethod
    def _parse_price(amount_str: str, unit: str) -> Optional[int]:
        """Parse price string like '1.5 triệu' to integer VND"""
        if not amount_str:
            return None
        
        try:
            # Clean the amount string
            amount_str = amount_str.replace(',', '').replace('.', '')
            if not amount_str.isdigit():
                return None
            
            amount = int(amount_str)
            
            # Convert based on unit
            if unit in ['triệu', 'tr']:
                return amount * 1000000
            elif unit == 'k':
                return amount * 1000
            elif unit == 'nghìn':
                return amount * 1000
            else:
                # Assume it's already in VND
                return amount if amount > 1000 else amount * 1000
        
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def apply_filters(tours_db: Dict[int, Dict], filters: Dict) -> List[int]:
        """
        Apply mandatory filters to tour database
        Returns list of tour indices that pass ALL filters
        """
        if not filters or not tours_db:
            return list(tours_db.keys())
        
        passing_tours = []
        
        # FIX: Added try-catch for safe filtering
        try:
            for tour_idx, tour_data in tours_db.items():
                passes_all = True
                
                # PRICE FILTERING (CRITICAL FIX)
                if passes_all and ('price_max' in filters or 'price_min' in filters):
                    tour_price_text = tour_data.get('price', '')
                    if not tour_price_text:
                        # If tour has no price info and user wants price filter, exclude it
                        if 'price_max' in filters or 'price_min' in filters:
                            passes_all = False
                    else:
                        # Extract price range from tour
                        tour_prices = MandatoryFilterSystem._extract_tour_prices(tour_price_text)
                        if not tour_prices:
                            passes_all = False
                        else:
                            # Check against filters
                            min_tour_price = min(tour_prices)
                            max_tour_price = max(tour_prices)
                            
                            if 'price_max' in filters and min_tour_price > filters['price_max']:
                                passes_all = False
                            if 'price_min' in filters and max_tour_price < filters['price_min']:
                                passes_all = False
                
                # DURATION FILTERING
                if passes_all and ('duration_min' in filters or 'duration_max' in filters):
                    duration_text = tour_data.get('duration', '').lower()
                    tour_duration = MandatoryFilterSystem._extract_duration_days(duration_text)
                    
                    if tour_duration is not None:
                        if 'duration_min' in filters and tour_duration < filters['duration_min']:
                            passes_all = False
                        if 'duration_max' in filters and tour_duration > filters['duration_max']:
                            passes_all = False
                    else:
                        # If can't extract duration and user wants duration filter, be conservative
                        if 'duration_min' in filters or 'duration_max' in filters:
                            passes_all = False
                
                # LOCATION FILTERING
                if passes_all and ('location' in filters or 'near_location' in filters):
                    tour_location = tour_data.get('location', '').lower()
                    if 'location' in filters:
                        filter_location = filters['location'].lower()
                        if filter_location not in tour_location:
                            passes_all = False
                    if 'near_location' in filters:
                        near_location = filters['near_location'].lower()
                        if near_location not in tour_location:
                            passes_all = False
                
                if passes_all:
                    passing_tours.append(tour_idx)
            
            logger.info(f"🔍 After mandatory filtering: {len(passing_tours)}/{len(tours_db)} tours pass")
        except Exception as e:
            logger.error(f"❌ Error in apply_filters: {e}")
            # FALLBACK: Return all tours if filtering fails
            passing_tours = list(tours_db.keys())
        
        return passing_tours
    
    @staticmethod
    def _extract_tour_prices(price_text: str) -> List[int]:
        """Extract price numbers from tour price text"""
        prices = []
        
        # Find all number sequences
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
                                
                                # Convert based on context
                                if 'triệu' in match.group(0).lower() or 'tr' in match.group(0).lower():
                                    num = num * 1000000
                                elif 'k' in match.group(0).lower() or 'nghìn' in match.group(0).lower():
                                    num = num * 1000
                                
                                prices.append(num)
                except (ValueError, AttributeError):
                    continue
        
        # If no prices found with units, try raw numbers
        if not prices:
            raw_numbers = re.findall(r'\d[\d,\.]+', price_text)
            for num_str in raw_numbers[:2]:  # Take first 2 numbers
                try:
                    num_str = num_str.replace(',', '').replace('.', '')
                    if num_str.isdigit():
                        num = int(num_str)
                        # Assume it's in thousands if it's a reasonable tour price
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

# =========== UPGRADE 2: DEDUPLICATION ENGINE ===========
class DeduplicationEngine:
    """
    UPGRADE 2: Remove duplicate and highly similar results
    Fixes: Same tour appearing multiple times in results
    """
    
    SIMILARITY_THRESHOLD = 0.85  # 85% similarity = duplicate
    MIN_TEXT_LENGTH = 20  # Minimum text length to consider for deduplication
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = DeduplicationEngine._normalize_text(text1)
        text2_norm = DeduplicationEngine._normalize_text(text2)
        
        if len(text1_norm) < DeduplicationEngine.MIN_TEXT_LENGTH or len(text2_norm) < DeduplicationEngine.MIN_TEXT_LENGTH:
            return 0.0
        
        # Method 1: SequenceMatcher (good for reordered text)
        seq_ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Method 2: Jaccard similarity on words
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            jaccard = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard = len(intersection) / len(union)
        
        # Method 3: Prefix/suffix overlap (for similar beginnings/endings)
        prefix_len = min(50, min(len(text1_norm), len(text2_norm)))
        prefix1 = text1_norm[:prefix_len]
        prefix2 = text2_norm[:prefix_len]
        prefix_sim = SequenceMatcher(None, prefix1, prefix2).ratio()
        
        # Weighted combination
        similarity = (seq_ratio * 0.5) + (jaccard * 0.3) + (prefix_sim * 0.2)
        
        return similarity
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stopwords (Vietnamese)
        stopwords = {'và', 'của', 'cho', 'với', 'tại', 'ở', 'này', 'đó', 'kia', 'về', 'trong'}
        words = [word for word in text.split() if word not in stopwords]
        
        return ' '.join(words)
    
    @staticmethod
    def deduplicate_passages(passages: List[Tuple[float, Dict]], 
                            similarity_threshold: float = None) -> List[Tuple[float, Dict]]:
        """
        Remove duplicate passages from results
        Returns unique passages sorted by original score
        """
        if len(passages) <= 1:
            return passages
        
        threshold = similarity_threshold or DeduplicationEngine.SIMILARITY_THRESHOLD
        unique_passages = []
        seen_passages = []  # Store (normalized_text, path_pattern)
        
        # Sort by score descending
        sorted_passages = sorted(passages, key=lambda x: x[0], reverse=True)
        
        for score, passage in sorted_passages:
            text = passage.get('text', '').strip()
            path = passage.get('path', '')
            
            if not text or len(text) < DeduplicationEngine.MIN_TEXT_LENGTH:
                # Keep very short passages (might be important)
                unique_passages.append((score, passage))
                continue
            
            # Check if this passage is similar to any we've already kept
            is_duplicate = False
            for seen_text, seen_path in seen_passages:
                # First check if from same tour (same path pattern)
                tour_match1 = re.search(r'tours\[(\d+)\]', path)
                tour_match2 = re.search(r'tours\[(\d+)\]', seen_path)
                
                if tour_match1 and tour_match2:
                    if tour_match1.group(1) == tour_match2.group(1):
                        # Same tour, check field type
                        field1 = path.split('.')[-1] if '.' in path else ''
                        field2 = seen_path.split('.')[-1] if '.' in seen_path else ''
                        if field1 == field2:
                            # Same tour and field, definitely duplicate
                            is_duplicate = True
                            break
                
                # Check text similarity
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
    def merge_similar_tours(tour_indices: List[int], tours_db: Dict[int, Dict]) -> List[int]:
        """Merge tours that are essentially the same"""
        if len(tour_indices) <= 1:
            return tour_indices
        
        # Group tours by name similarity
        tour_groups = []
        processed = set()
        
        for i, idx1 in enumerate(tour_indices):
            if idx1 in processed:
                continue
            
            group = [idx1]
            tour1 = tours_db.get(idx1, {})
            name1 = tour1.get('tour_name', '').strip()
            
            if not name1:
                processed.add(idx1)
                tour_groups.append(group)
                continue
            
            for j, idx2 in enumerate(tour_indices[i+1:], i+1):
                if idx2 in processed:
                    continue
                
                tour2 = tours_db.get(idx2, {})
                name2 = tour2.get('tour_name', '').strip()
                
                if not name2:
                    continue
                
                # Check if tours are similar
                similarity = DeduplicationEngine.calculate_similarity(name1, name2)
                if similarity > 0.9:  # Very high similarity
                    group.append(idx2)
                    processed.add(idx2)
            
            processed.add(idx1)
            tour_groups.append(group)
        
        # For each group, keep the tour with most complete information
        best_tours = []
        for group in tour_groups:
            if not group:
                continue
            
            if len(group) == 1:
                best_tours.append(group[0])
                continue
            
            # Score tours by completeness
            best_score = -1
            best_idx = group[0]
            
            for idx in group:
                tour = tours_db.get(idx, {})
                score = 0
                
                # Check for important fields
                if tour.get('tour_name'):
                    score += 2
                if tour.get('duration'):
                    score += 2
                if tour.get('location'):
                    score += 2
                if tour.get('price'):
                    score += 3
                if tour.get('includes'):
                    score += 2
                if tour.get('summary'):
                    score += 1
                
                # Longer text fields are better
                for field in ['includes', 'summary', 'notes']:
                    value = tour.get(field, '')
                    if isinstance(value, str) and len(value) > 50:
                        score += 1
                    elif isinstance(value, list) and value:
                        score += len(value)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            best_tours.append(best_idx)
        
        logger.info(f"🔄 Tour merging: {len(tour_indices)} → {len(best_tours)} unique tours")
        return best_tours

# =========== UPGRADE 3: ENHANCED FIELD DETECTION ===========
class EnhancedFieldDetector:
    """
    UPGRADE 3: Better detection of what user is asking for
    Fixes: "tour có gì hay?" should detect as asking for summary
    """
    
    # Extended field mapping with confidence weights
    FIELD_DETECTION_RULES = [
        # TOUR LIST - Highest priority
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
        
        # PRICE - Very important
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
        
        # SUMMARY / WHAT'S INCLUDED (Often confused)
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
        
        # INCLUDES / ITINERARY
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
        
        # ACCOMMODATION
        {
            "field": "accommodation",
            "patterns": [
                (r'chỗ ở.*thế nào|nơi ở.*ra sao|khách sạn.*nào', 0.9),
                (r'ở đâu.*đêm|nghỉ ở đâu|ngủ ở đâu', 0.8),
                (r'có ở lại.*đêm|qua đêm.*ở đâu|đêm.*ở đâu', 0.8),
                (r'homestay|khách sạn|resort|nhà nghỉ', 0.7),
            ],
            "keywords": [
                ("chỗ ở", 0.8), ("nơi ở", 0.7), ("khách sạn", 0.7),
                ("homestay", 0.7), ("resort", 0.6), ("nhà nghỉ", 0.6),
                ("nghỉ đêm", 0.7), ("qua đêm", 0.7),
            ]
        },
        
        # MEALS
        {
            "field": "meals",
            "patterns": [
                (r'ăn uống.*thế nào|đồ ăn.*ra sao|thức ăn.*gì', 0.9),
                (r'có ăn.*gì|được ăn.*gì|bao gồm.*ăn', 0.8),
                (r'bữa ăn.*nào|ăn.*ở đâu|thực đơn.*gì', 0.7),
                (r'ẩm thực.*gì|đặc sản.*gì|món ăn.*gì', 0.7),
            ],
            "keywords": [
                ("ăn uống", 0.8), ("đồ ăn", 0.7), ("thức ăn", 0.7),
                ("bữa ăn", 0.7), ("ẩm thực", 0.7), ("đặc sản", 0.6),
                ("món ăn", 0.6), ("thực đơn", 0.7),
            ]
        },
        
        # TRANSPORT
        {
            "field": "transport",
            "patterns": [
                (r'di chuyển.*bằng gì|đi lại.*thế nào|xe cộ.*gì', 0.9),
                (r'phương tiện.*gì|xe.*gì|di chuyển.*ra sao', 0.8),
                (r'có xe.*đưa đón|xe đưa đón|vận chuyển.*gì', 0.7),
            ],
            "keywords": [
                ("di chuyển", 0.8), ("phương tiện", 0.8), ("xe", 0.7),
                ("đi lại", 0.7), ("vận chuyển", 0.7), ("đưa đón", 0.7),
            ]
        },
    ]
    
    @staticmethod
    def detect_field_with_confidence(message: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Detect which field user is asking about with confidence scores
        Returns: (best_field, confidence, all_scores)
        """
        if not message:
            return None, 0.0, {}
        
        message_lower = message.lower()
        scores = {}
        
        for rule in EnhancedFieldDetector.FIELD_DETECTION_RULES:
            field = rule["field"]
            field_score = 0.0
            
            # Check patterns (higher weight)
            for pattern, weight in rule["patterns"]:
                if re.search(pattern, message_lower):
                    field_score = max(field_score, weight)
            
            # Check keywords (medium weight)
            for keyword, weight in rule["keywords"]:
                if keyword in message_lower:
                    # Keyword position matters - earlier is more important
                    position = message_lower.find(keyword)
                    position_factor = 1.0 - (position / max(len(message_lower), 1))
                    adjusted_weight = weight * (0.7 + 0.3 * position_factor)
                    field_score = max(field_score, adjusted_weight)
            
            # Additional factors
            if field_score > 0:
                # Length of match matters
                field_score = min(field_score * 1.1, 1.0)
            
            scores[field] = field_score
        
        # Find best field
        best_field = None
        best_score = 0.0
        
        for field, score in scores.items():
            if score > best_score:
                best_score = score
                best_field = field
        
        # Special case: if asking "tour có gì" without specifying, default to summary
        if (best_score < 0.3 and 
            ("có gì" in message_lower or "thế nào" in message_lower) and
            "tour" in message_lower):
            best_field = "summary"
            best_score = 0.6
        
        logger.info(f"🔍 Field detection: '{message}' → {best_field} (confidence: {best_score:.2f})")
        return best_field, best_score, scores

# =========== UPGRADE 4: QUESTION PIPELINE ===========
class QuestionPipeline:
    """
    UPGRADE 4: Process different types of questions differently
    Fixes: "so sánh tour A và B" needs special handling
    """
    
    class QuestionType(Enum):
        INFORMATION = "information"        # Basic info request
        COMPARISON = "comparison"         # Compare 2+ items
        RECOMMENDATION = "recommendation" # Ask for suggestions
        LISTING = "listing"               # List items
        CALCULATION = "calculation"       # Calculate something
        CONFIRMATION = "confirmation"     # Confirm something
        GREETING = "greeting"             # Hello, hi, etc.
        FAREWELL = "farewell"             # Goodbye, thanks
        COMPLEX = "complex"               # Multi-part question
    
    @staticmethod
    def classify_question(message: str) -> Tuple[QuestionType, float, Dict]:
        """
        Classify question type with confidence and metadata
        """
        message_lower = message.lower()
        type_scores = defaultdict(float)
        metadata = {}
        
        # FIX: Improved listing detection - only trigger on clear listing requests
        # Not on "các tour còn lại" which might be comparison
        listing_patterns = [
            (r'^liệt kê.*tour|^danh sách.*tour|^có những tour nào$', 0.95),
            (r'^tất cả.*tour$|^mọi.*tour$|^mấy.*tour có$', 0.9),
            (r'^tour nào.*có\?*$|^hiện có.*tour\?*$', 0.85),
            (r'^kể tên.*tour|^nêu tên.*tour$', 0.9),
        ]
        
        for pattern, weight in listing_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionPipeline.QuestionType.LISTING] = max(
                    type_scores[QuestionPipeline.QuestionType.LISTING], weight
                )
        
        # FIX: Improved comparison detection - capture "điểm gì khác biệt"
        comparison_patterns = [
            (r'so sánh.*và|đối chiếu.*và', 0.95),
            (r'khác nhau.*nào|giống nhau.*nào', 0.9),
            (r'điểm.*khác biệt|điểm.*khác', 0.9),
            (r'nên chọn.*nào|tốt hơn.*nào|hơn kém.*nào', 0.85),
            (r'tour.*và.*tour', 0.8),
            (r'sánh.*với|đối chiếu.*với', 0.8),
        ]
        
        for pattern, weight in comparison_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionPipeline.QuestionType.COMPARISON] = max(
                    type_scores[QuestionPipeline.QuestionType.COMPARISON], weight
                )
                # Extract entities being compared
                metadata['comparison_type'] = 'direct'
        
        # FIX: Reduce false positive for "các tour còn lại" - NOT listing
        if 'còn lại' in message_lower and 'tour' in message_lower:
            # This is more likely to be comparison or information, not listing
            if QuestionPipeline.QuestionType.LISTING in type_scores:
                type_scores[QuestionPipeline.QuestionType.LISTING] *= 0.3  # Reduce confidence
        
        # RECOMMENDATION detection - HIGH PRIORITY
        recommendation_patterns = [
            (r'phù hợp.*với|nên đi.*nào|gợi ý.*tour', 0.9),
            (r'tour nào.*tốt|hành trình nào.*hay', 0.85),
            (r'đề xuất.*tour|tư vấn.*tour|chọn.*nào', 0.8),
            (r'cho.*tôi|dành cho.*tôi|hợp với.*tôi', 0.7),
            (r'nếu.*thì.*nên.*tour|nên chọn.*tour', 0.8),
        ]
        
        for pattern, weight in recommendation_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionPipeline.QuestionType.RECOMMENDATION] = max(
                    type_scores[QuestionPipeline.QuestionType.RECOMMENDATION], weight
                )
        
        # GREETING detection - LOWER PRIORITY (ONLY if no other intent)
        greeting_words = ['xin chào', 'chào', 'hello', 'hi', 'helo', 'chao']
        greeting_score = 0.0
        for word in greeting_words:
            if word in message_lower:
                # Only count as greeting if it's at the beginning or standalone
                if message_lower.startswith(word) or f" {word} " in message_lower or message_lower.endswith(f" {word}"):
                    greeting_score += 0.3
        
        # FIX 2: Greeting only triggers if no other intent with confidence > 0.3
        other_intent_score = max([score for qtype, score in type_scores.items() 
                                 if qtype != QuestionPipeline.QuestionType.GREETING], default=0.0)
        
        if greeting_score > 0.8 and other_intent_score < 0.3:
            type_scores[QuestionPipeline.QuestionType.GREETING] = min(greeting_score, 1.0)
        
        # FAREWELL detection
        farewell_words = ['tạm biệt', 'cảm ơn', 'thanks', 'thank you', 'bye', 'goodbye']
        if any(word in message_lower for word in farewell_words):
            type_scores[QuestionPipeline.QuestionType.FAREWELL] = 0.95
        
        # CALCULATION detection
        calculation_patterns = [
            (r'tính toán|tính.*bao nhiêu|tổng.*bao nhiêu', 0.9),
            (r'cộng.*lại|nhân.*lên|chia.*ra', 0.8),
            (r'bao nhiêu.*người|mấy.*người|số lượng.*người', 0.7),
        ]
        
        for pattern, weight in calculation_patterns:
            if re.search(pattern, message_lower):
                type_scores[QuestionPipeline.QuestionType.CALCULATION] = max(
                    type_scores[QuestionPipeline.QuestionType.CALCULATION], weight
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
            type_scores[QuestionPipeline.QuestionType.COMPLEX] = min(complex_score / 2, 1.0)
            metadata['complex_parts'] = QuestionPipeline._split_complex_question(message)
        
        # DEFAULT: INFORMATION request
        if not type_scores:
            type_scores[QuestionPipeline.QuestionType.INFORMATION] = 0.6
        else:
            # Still might be information request even if other types detected
            info_keywords = ['là gì', 'bao nhiêu', 'ở đâu', 'khi nào', 'thế nào', 'ai', 'tại sao']
            if any(keyword in message_lower for keyword in info_keywords):
                type_scores[QuestionPipeline.QuestionType.INFORMATION] = max(
                    type_scores.get(QuestionPipeline.QuestionType.INFORMATION, 0),
                    0.5
                )
        
        # Determine best type
        best_type = QuestionPipeline.QuestionType.INFORMATION
        best_score = 0.0
        
        for qtype, score in type_scores.items():
            if score > best_score:
                best_score = score
                best_type = qtype
        
        # Special handling for ambiguous cases
        if best_score < 0.5:
            # Default to INFORMATION with medium confidence
            best_type = QuestionPipeline.QuestionType.INFORMATION
            best_score = 0.5
        
        logger.info(f"🎯 Question classification: '{message}' → {best_type.value} (score: {best_score:.2f})")
        return best_type, best_score, metadata
    
    @staticmethod
    def _split_complex_question(message: str) -> List[str]:
        """Split complex multi-part question into simpler parts"""
        # Common Vietnamese conjunctions for splitting
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
    def process_comparison_question(tour_indices: List[int], tours_db: Dict[int, Dict], 
                                  aspect: str = "", context: Dict = None) -> str:
        """
        Process comparison question between tours
        """
        if len(tour_indices) < 2:
            return "Cần ít nhất 2 tour để so sánh."
        
        # Get tour data
        tours_to_compare = []
        for idx in tour_indices[:3]:  # Max 3 tours for comparison
            if idx in tours_db:
                tours_to_compare.append((idx, tours_db[idx]))
        
        if len(tours_to_compare) < 2:
            return "Không tìm thấy đủ thông tin tour để so sánh."
        
        # Determine aspect to compare
        if not aspect:
            # Auto-detect aspect from common fields
            common_fields = ['price', 'duration', 'location', 'accommodation', 'meals']
            aspect = common_fields[0]
        
        # Build comparison table
        result_lines = []
        
        # Header
        headers = ["TIÊU CHÍ"]
        for idx, tour in tours_to_compare:
            tour_name = tour.get('tour_name', f'Tour #{idx}')
            headers.append(tour_name[:25])  # Limit name length
        
        result_lines.append(" | ".join(headers))
        result_lines.append("-" * (len(headers) * 30))
        
        # Comparison fields (priority order)
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
                value = tour.get(field, 'N/A')
                if isinstance(value, list):
                    value = ', '.join(value[:2])
                row.append(str(value)[:30])  # Limit value length
                all_values.append(str(value).lower())
            
            # Only add row if there are differences
            if len(set(all_values)) > 1 or aspect == field:
                result_lines.append(" | ".join(row))
        
        # Add summary/recommendation
        result_lines.append("\n" + "="*50)
        result_lines.append("**ĐÁNH GIÁ & GỢI Ý:**")
        
        # Duration-based recommendation - FIX: Changed 'd' to 'x' to avoid NameError
        durations = [tour.get('duration', '') for _, tour in tours_to_compare]
        if durations and any('1 ngày' in x for x in durations) and any('2 ngày' in x for x in durations):
            result_lines.append("• Nếu bạn có ít thời gian: Chọn tour 1 ngày")
            result_lines.append("• Nếu muốn trải nghiệm sâu: Chọn tour 2 ngày")
        
        # Price-based recommendation
        prices = []
        for _, tour in tours_to_compare:
            price_text = tour.get('price', '')
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

# =========== UPGRADE 5: COMPLEX QUERY SPLITTER ===========
class ComplexQueryProcessor:
    """
    UPGRADE 5: Handle complex multi-condition queries
    Fixes: "tour 1 ngày giá rẻ ở Huế" has multiple conditions
    """
    
    @staticmethod
    def split_query(query: str) -> List[Dict[str, Any]]:
        """
        Split complex query into sub-queries with priorities
        """
        sub_queries = []
        
        # First, check if it's actually a complex query
        complexity_score = ComplexQueryProcessor._calculate_complexity(query)
        if complexity_score < 1.5:
            # Simple query, no splitting needed
            return [{
                'query': query,
                'priority': 1.0,
                'filters': {},
                'focus': 'general'
            }]
        
        # Extract different types of conditions
        conditions = ComplexQueryProcessor._extract_conditions(query)
        
        if len(conditions) <= 1:
            return [{
                'query': query,
                'priority': 1.0,
                'filters': conditions[0] if conditions else {},
                'focus': 'general'
            }]
        
        # Create sub-queries based on condition combinations
        # Priority 1: All conditions together (most specific)
        sub_queries.append({
            'query': query,
            'priority': 1.0,
            'filters': ComplexQueryProcessor._merge_conditions(conditions),
            'focus': 'specific'
        })
        
        # Priority 2: Location + one other condition
        location_conds = [c for c in conditions if 'location' in c]
        other_conds = [c for c in conditions if 'location' not in c]
        
        if location_conds and other_conds:
            for other_cond in other_conds[:2]:  # Top 2 other conditions
                merged = ComplexQueryProcessor._merge_conditions(location_conds + [other_cond])
                sub_queries.append({
                    'query': f"{query} (focus on location + {list(other_cond.keys())[0]})",
                    'priority': 0.8,
                    'filters': merged,
                    'focus': list(other_cond.keys())[0]
                })
        
        # Priority 3: Individual important conditions
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
        
        # Sort by priority
        sub_queries.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"🔀 Split query into {len(sub_queries)} sub-queries")
        return sub_queries[:3]  # Return top 3
    
    @staticmethod
    def _calculate_complexity(query: str) -> float:
        """Calculate how complex a query is"""
        complexity = 0.0
        
        # Number of keywords indicating different aspects
        aspects = {
            'price': ['giá', 'tiền', 'chi phí', 'đắt', 'rẻ'],
            'duration': ['ngày', 'đêm', 'bao lâu', 'thời gian'],
            'location': ['ở', 'tại', 'đến', 'về', 'địa điểm'],
            'quality': ['tốt', 'hay', 'đẹp', 'hấp dẫn', 'thú vị'],
            'type': ['thiền', 'khí công', 'retreat', 'chữa lành'],
        }
        
        query_lower = query.lower()
        
        # Count distinct aspects mentioned
        distinct_aspects = 0
        for aspect, keywords in aspects.items():
            if any(keyword in query_lower for keyword in keywords):
                distinct_aspects += 1
        
        complexity += distinct_aspects * 0.5
        
        # Length of query (normalized)
        complexity += min(len(query.split()) / 10, 1.0)
        
        # Presence of conjunctions
        conjunctions = ['và', 'với', 'có', 'cho', 'mà', 'nhưng']
        for conj in conjunctions:
            if conj in query_lower:
                complexity += 0.3
        
        return complexity
    
    @staticmethod
    def _extract_conditions(query: str) -> List[Dict[str, Any]]:
        """Extract individual conditions from query"""
        conditions = []
        
        # Use MandatoryFilterSystem to extract filters
        filters = MandatoryFilterSystem.extract_filters(query)
        
        # Convert filters to individual conditions
        for filter_type, value in filters.items():
            if filter_type.startswith('price_'):
                conditions.append({'price': {filter_type: value}})
            elif filter_type.startswith('duration_'):
                conditions.append({'duration': {filter_type: value}})
            elif filter_type == 'location':
                conditions.append({'location': value})
            elif filter_type == 'near_location':
                conditions.append({'near_location': value})
        
        # Extract keyword-based conditions
        query_lower = query.lower()
        
        # Quality conditions
        if any(word in query_lower for word in ['rẻ', 'giá rẻ', 'tiết kiệm']):
            conditions.append({'price_quality': 'budget'})
        if any(word in query_lower for word in ['cao cấp', 'sang', 'premium']):
            conditions.append({'price_quality': 'premium'})
        
        # Activity type conditions
        if 'thiền' in query_lower:
            conditions.append({'activity_type': 'meditation'})
        if 'khí công' in query_lower:
            conditions.append({'activity_type': 'qigong'})
        if 'retreat' in query_lower:
            conditions.append({'activity_type': 'retreat'})
        if 'chữa lành' in query_lower:
            conditions.append({'activity_type': 'healing'})
        
        # FIX 3: Extract tour names from complex queries (e.g., "so sánh tour A và tour B")
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
                        # Skip short/meaningless names
                        if len(tour_name) < 3 or tour_name in ['này', 'đó', 'kia', 'còn']:
                            continue
                        # Try to find tour index by name
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
                        # Prefer more specific values
                        if isinstance(value, dict) or (isinstance(value, str) and len(value) > len(str(merged[key]))):
                            merged[key] = value
                else:
                    merged[key] = value
        
        return merged

# =========== UPGRADE 6: FUZZY MATCHING ===========
class FuzzyMatcher:
    """
    UPGRADE 6: Handle misspellings and variations in tour names
    Fixes: "Bach Ma" should match "Bạch Mã"
    """
    
    SIMILARITY_THRESHOLD = 0.75
    
    @staticmethod
    def normalize_vietnamese(text: str) -> str:
        """
        Normalize Vietnamese text for fuzzy matching
        - Remove diacritics
        - Standardize common variations
        """
        if not text:
            return ""
        
        text = text.lower()
        
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Common Vietnamese variations
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
        
        # Remove extra spaces and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def find_similar_tours(query: str, tour_names: Dict[str, int]) -> List[Tuple[int, float]]:
        """
        Find tours with names similar to query
        Returns list of (tour_index, similarity_score)
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
            
            # Calculate similarity
            similarity = SequenceMatcher(None, query_norm, tour_norm).ratio()
            
            # Boost similarity if query is substring of tour name or vice versa
            if query_norm in tour_norm or tour_norm in query_norm:
                similarity = min(similarity + 0.2, 1.0)
            
            # Check for common words
            query_words = set(query_norm.split())
            tour_words = set(tour_norm.split())
            common_words = query_words.intersection(tour_words)
            
            if common_words:
                word_boost = len(common_words) * 0.1
                similarity = min(similarity + word_boost, 1.0)
            
            if similarity >= FuzzyMatcher.SIMILARITY_THRESHOLD:
                matches.append((tour_idx, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"🔍 Fuzzy matching: '{query}' → {len(matches)} matches")
        return matches
    
    @staticmethod
    def find_tour_by_partial_name(partial_name: str, tours_db: Dict[int, Dict]) -> List[int]:
        """
        Find tours by partial name match
        """
        if not partial_name or not tours_db:
            return []
        
        partial_norm = FuzzyMatcher.normalize_vietnamese(partial_name)
        matches = []
        
        for tour_idx, tour_data in tours_db.items():
            tour_name = tour_data.get('tour_name', '')
            if not tour_name:
                continue
            
            tour_norm = FuzzyMatcher.normalize_vietnamese(tour_name)
            
            # Check if partial name appears in tour name
            if partial_norm in tour_norm:
                # Calculate how much of the partial name matches
                match_ratio = len(partial_norm) / len(tour_norm) if tour_norm else 0
                matches.append((tour_idx, match_ratio))
        
        # Sort by match ratio
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in matches[:3]]  # Return top 3

# =========== UPGRADE 7: STATE MACHINE ===========
class ConversationStateMachine:
    """
    UPGRADE 7: Track conversation state for better context
    Fixes: "tour đó giá bao nhiêu?" needs to know what "tour đó" refers to
    """
    
    class State(Enum):
        INITIAL = "initial"              # Starting state
        TOUR_SELECTED = "tour_selected"  # User has selected a tour
        COMPARING = "comparing"          # User is comparing tours
        ASKING_DETAILS = "asking_details" # Asking about specific details
        RECOMMENDATION = "recommendation" # In recommendation flow
        BOOKING = "booking"              # Talking about booking
        FAREWELL = "farewell"            # Ending conversation
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = self.State.INITIAL
        self.context = {
            'current_tours': [],      # List of tour indices currently being discussed
            'last_question': None,    # Last question asked
            'last_response': None,    # Last response given
            'preferences': {},        # User preferences gathered
            'conversation_history': [],  # Last N messages
            'mentioned_tours': set(), # All tours mentioned in conversation
            'current_focus': None,    # What the user is currently focusing on
            'last_successful_tours': [], # Last tours that were successfully found
        }
        self.transitions = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update state based on new interaction"""
        self.last_updated = datetime.utcnow()
        
        # Update conversation history
        self.context['conversation_history'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'user': user_message,
            'bot': bot_response,
            'tours': tour_indices or []
        })
        
        # Keep only last 10 messages
        if len(self.context['conversation_history']) > 10:
            self.context['conversation_history'] = self.context['conversation_history'][-10:]
        
        # Update mentioned tours
        if tour_indices:
            self.context['mentioned_tours'].update(tour_indices)
            self.context['current_tours'] = tour_indices
            # Keep track of last successful tours
            self.context['last_successful_tours'] = tour_indices
        
        self.context['last_question'] = user_message
        self.context['last_response'] = bot_response
        
        # Determine new state based on message
        new_state = self._determine_state(user_message, bot_response)
        
        # Record transition
        self.transitions.append({
            'timestamp': datetime.utcnow().isoformat(),
            'from': self.state.value,
            'to': new_state.value,
            'message': user_message[:100]
        })
        
        self.state = new_state
        
        logger.info(f"🔄 State update: {self.state.value} for session {self.session_id}")
    
    def _determine_state(self, user_message: str, bot_response: str) -> State:
        """Determine new state based on current interaction"""
        message_lower = user_message.lower()
        
        # Check for farewell
        farewell_words = ['tạm biệt', 'cảm ơn', 'thanks', 'bye', 'goodbye']
        if any(word in message_lower for word in farewell_words):
            return self.State.FAREWELL
        
        # Check if user is asking about a specific tour
        tour_ref_patterns = [
            r'tour này', r'tour đó', r'tour đang nói', r'cái tour',
            r'nó', r'cái đó', r'cái này', r'đấy'
        ]
        
        if any(re.search(pattern, message_lower) for pattern in tour_ref_patterns):
            if self.context['current_tours']:
                return self.State.TOUR_SELECTED
            elif self.context['last_successful_tours']:
                # Use last successful tours if current_tours is empty
                self.context['current_tours'] = self.context['last_successful_tours']
                return self.State.TOUR_SELECTED
        
        # Check for comparison
        if 'so sánh' in message_lower or 'sánh' in message_lower or 'khác biệt' in message_lower:
            return self.State.COMPARING
        
        # Check for recommendation request
        if any(word in message_lower for word in ['phù hợp', 'gợi ý', 'đề xuất', 'tư vấn', 'nên chọn']):
            return self.State.RECOMMENDATION
        
        # Check for booking intent
        if any(word in message_lower for word in ['đặt', 'booking', 'đăng ký', 'giữ chỗ']):
            return self.State.BOOKING
        
        # Default: asking for details about current focus
        if self.context['current_tours']:
            return self.State.ASKING_DETAILS
        
        return self.State.INITIAL
    
    def get_context_hint(self) -> str:
        """Get hint about current context for LLM prompt"""
        hints = []
        
        if self.state == self.State.TOUR_SELECTED and self.context['current_tours']:
            tour_indices = self.context['current_tours']
            if len(tour_indices) == 1:
                hints.append(f"User is asking about tour index {tour_indices[0]}")
            else:
                hints.append(f"User is asking about tours {tour_indices}")
        
        if self.context['preferences']:
            prefs = []
            for key, value in self.context['preferences'].items():
                prefs.append(f"{key}: {value}")
            if prefs:
                hints.append(f"User preferences: {', '.join(prefs)}")
        
        return "; ".join(hints) if hints else "No specific context"
    
    def extract_reference(self, message: str) -> List[int]:
        """Extract tour reference from message using conversation context"""
        message_lower = message.lower()
        
        # FIX 2: Check context BEFORE trying to extract new references
        if self.context['current_tours']:
            # Check if message contains references to current tours
            for tour_idx in self.context['current_tours']:
                tour_name = TOURS_DB.get(tour_idx, {}).get('tour_name', '').lower()
                if tour_name:
                    # Simple word matching
                    tour_words = set(tour_name.split())
                    msg_words = set(message_lower.split())
                    common = tour_words.intersection(msg_words)
                    if common and len(common) >= 1:
                        logger.info(f"🔄 State machine: Using current tour {tour_idx}")
                        return self.context['current_tours']
        
        # Direct references to "tour này", "tour đó", etc.
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
                if self.context['current_tours']:
                    logger.info(f"🔄 State machine: Resolved reference to {self.context['current_tours']}")
                    return self.context['current_tours']
                elif self.context['last_successful_tours']:
                    # Use last successful tours
                    logger.info(f"🔄 State machine: Using last successful tours {self.context['last_successful_tours']}")
                    return self.context['last_successful_tours']
        
        # Try to match with recently mentioned tours
        if self.context['mentioned_tours']:
            # Check if message contains words from recently mentioned tour names
            recent_tours = list(self.context['mentioned_tours'])
            for tour_idx in recent_tours[-3:]:  # Check last 3 mentioned tours
                # Get tour name from global TOURS_DB
                tour_name = TOURS_DB.get(tour_idx, {}).get('tour_name', '').lower()
                if tour_name:
                    # Simple word overlap check
                    tour_words = set(tour_name.split())
                    msg_words = set(message_lower.split())
                    common = tour_words.intersection(msg_words)
                    if common and len(common) >= 1:
                        logger.info(f"🔄 State machine: Matched to recently mentioned tour {tour_idx}")
                        return [tour_idx]
        
        return []

# =========== UPGRADE 8: DEEP SEMANTIC ANALYSIS ===========
class SemanticAnalyzer:
    """
    UPGRADE 8: Deep understanding of user intent beyond keywords
    Fixes: "tour nào hợp với người già?" should understand user needs
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
    def analyze_user_profile(message: str, current_context: Dict = None) -> Dict[str, Any]:
        """
        Analyze message to build user profile
        """
        profile = current_context or {
            'age_group': None,
            'group_type': None,
            'interests': [],
            'budget_level': None,
            'physical_level': None,
            'special_needs': [],
            'confidence_scores': {}
        }
        
        message_lower = message.lower()
        
        # Detect profile attributes
        for category, patterns in SemanticAnalyzer.USER_PROFILE_PATTERNS.items():
            for pattern, value in patterns:
                if re.search(pattern, message_lower):
                    if category == 'interests':
                        if value not in profile['interests']:
                            profile['interests'].append(value)
                            profile['confidence_scores'][f'interest_{value}'] = 0.8
                    else:
                        profile[category] = value
                        profile['confidence_scores'][category] = 0.8
        
        # Infer additional attributes from context
        SemanticAnalyzer._infer_attributes(profile, message_lower)
        
        # Calculate overall confidence
        profile['overall_confidence'] = SemanticAnalyzer._calculate_confidence(profile)
        
        logger.info(f"👤 User profile analysis: {profile}")
        return profile
    
    @staticmethod
    def _infer_attributes(profile: Dict, message_lower: str):
        """Infer additional attributes from context"""
        # Infer age group from other attributes
        if not profile['age_group']:
            # Kiểm tra an toàn cho group_type
            group_type_value = profile.get('group_type') if profile else None
            if group_type_value and isinstance(group_type_value, str) and 'family_with_kids' in group_type_value:
                profile['age_group'] = 'middle_aged'
                profile['confidence_scores']['age_group'] = 0.6
            elif 'senior' in message_lower or 'già' in message_lower:
                profile['age_group'] = 'senior'
                profile['confidence_scores']['age_group'] = 0.7
        
        # Infer physical level from interests
        if not profile['physical_level']:
            if 'adventure' in profile['interests']:
                profile['physical_level'] = 'challenging'
                profile['confidence_scores']['physical_level'] = 0.6
            elif 'spiritual' in profile['interests'] or 'wellness' in profile['interests']:
                profile['physical_level'] = 'easy'
                profile['confidence_scores']['physical_level'] = 0.6
        
        # Infer budget from keywords
        if not profile['budget_level']:
            budget_keywords = {
                'budget': ['rẻ', 'tiết kiệm', 'ít tiền', 'kinh tế'],
                'premium': ['cao cấp', 'sang', 'đắt', 'premium']
            }
            
            for level, keywords in budget_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    profile['budget_level'] = level
                    profile['confidence_scores']['budget_level'] = 0.7
                    break
    
    @staticmethod
    def _calculate_confidence(profile: Dict) -> float:
        """Calculate overall confidence in user profile"""
        if not profile['confidence_scores']:
            return 0.0
        
        total = 0.0
        count = 0
        
        for key, score in profile['confidence_scores'].items():
            total += score
            count += 1
        
        return total / max(count, 1)
    
    @staticmethod
    def match_tours_to_profile(profile: Dict, tours_db: Dict[int, Dict], 
                              max_results: int = 5) -> List[Tuple[int, float, List[str]]]:
        """
        Match tours to user profile with explanation
        Returns: [(tour_index, match_score, match_reasons)]
        """
        matches = []
        
        for tour_idx, tour_data in tours_db.items():
            score = 0.0
            reasons = []
            
            # Get tour tags (pre-computed)
            tour_tags = TOUR_TAGS.get(tour_idx, [])
            
            # Match age group
            if profile['age_group']:
                if profile['age_group'] == 'senior':
                    # Senior-friendly tours
                    if any('easy' in tag for tag in tour_tags):
                        score += 0.3
                        reasons.append("phù hợp người lớn tuổi")
                    if any('nature' in tag for tag in tour_tags):
                        score += 0.2
                        reasons.append("thiên nhiên nhẹ nhàng")
            
            # Match interests
            if profile['interests']:
                for interest in profile['interests']:
                    # Check if interest matches tour tags or summary
                    tour_summary = tour_data.get('summary', '').lower()
                    if (interest in tour_summary or 
                        any(interest in tag for tag in tour_tags)):
                        score += 0.4
                        reasons.append(f"có yếu tố {interest}")
            
            # Match budget
            if profile['budget_level']:
                tour_price = tour_data.get('price', '')
                price_nums = re.findall(r'\d[\d,\.]+', tour_price)
                
                if price_nums:
                    try:
                        # Simple price extraction
                        first_price = int(price_nums[0].replace(',', '').replace('.', ''))
                        
                        if profile['budget_level'] == 'budget' and first_price < 2000000:
                            score += 0.3
                            reasons.append("giá hợp lý")
                        elif profile['budget_level'] == 'premium' and first_price > 2500000:
                            score += 0.3
                            reasons.append("cao cấp")
                        elif profile['budget_level'] == 'midrange' and 1500000 <= first_price <= 3000000:
                            score += 0.3
                            reasons.append("giá vừa phải")
                    except:
                        pass
            
            # Match physical level
            if profile['physical_level']:
                if profile['physical_level'] == 'easy':
                    if any('easy' in tag or 'meditation' in tag for tag in tour_tags):
                        score += 0.2
                        reasons.append("hoạt động nhẹ nhàng")
            
            if score > 0:
                matches.append((tour_idx, score, reasons))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:max_results]

# =========== UPGRADE 9: AUTO-VALIDATION SYSTEM ===========
class AutoValidator:
    """
    UPGRADE 9: Validate and correct information before returning
    Fixes: Prevent returning "tour 5 ngày 4 đêm" (unrealistic)
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
                'min_tour_price': 500000,    # 500k VND
                'max_tour_price': 10000000,  # 10 million VND
                'common_ranges': [
                    (800000, 1500000),   # 1-day tours
                    (1500000, 2500000),  # 2-day tours
                    (2500000, 4000000),  # 3-day tours
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
        
        'dates': {
            'patterns': [
                r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
                r'ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})',
            ],
            'constraints': {
                'min_date': '2024-01-01',
                'max_date': '2025-12-31'
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
        
        # Validate duration - FIX: Use try-catch to prevent crashes
        try:
            validated = AutoValidator._validate_duration(validated)
        except Exception as e:
            logger.error(f"❌ Duration validation error: {e}")
        
        # Validate price
        try:
            validated = AutoValidator._validate_price(validated)
        except Exception as e:
            logger.error(f"❌ Price validation error: {e}")
        
        # Validate location names
        try:
            validated = AutoValidator._validate_locations(validated)
        except Exception as e:
            logger.error(f"❌ Location validation error: {e}")
        
        # Validate dates
        try:
            validated = AutoValidator._validate_dates(validated)
        except Exception as e:
            logger.error(f"❌ Date validation error: {e}")
        
        # Check for unrealistic information
        try:
            validated = AutoValidator._check_unrealistic_info(validated)
        except Exception as e:
            logger.error(f"❌ Unrealistic info check error: {e}")
        
        # Add disclaimer if needed
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
                    if match.lastindex == 2:  # "X ngày Y đêm"
                        days = int(match.group(1))
                        nights = int(match.group(2))
                        
                        constraints = AutoValidator.VALIDATION_RULES['duration']['constraints']
                        
                        # Check if combination is valid
                        valid_combos = constraints['valid_day_night_combos']
                        # FIX: Changed 'd' and 'n' to 'days' and 'nights'
                        is_valid_combo = any(days == d2 and nights == n2 for d2, n2 in valid_combos)
                        
                        # Check max limits
                        if days > constraints['max_days'] or nights > constraints['max_nights']:
                            # Replace with common duration
                            replacement = random.choice(constraints['common_durations'])
                            text = text.replace(match.group(0), replacement)
                            logger.warning(f"⚠️ Corrected unrealistic duration: {days} ngày {nights} đêm → {replacement}")
                        
                        elif not is_valid_combo:
                            # Fix to nearest valid combo
                            valid_days = min(days, constraints['max_days'])
                            valid_nights = min(nights, constraints['max_nights'])
                            # Make nights = days or days-1 (common pattern)
                            if abs(valid_days - valid_nights) > 1:
                                valid_nights = valid_days
                            
                            replacement = f"{valid_days} ngày {valid_nights} đêm"
                            text = text.replace(match.group(0), replacement)
                            logger.info(f"🔄 Fixed duration combo: {replacement}")
                    
                    elif match.lastindex == 1:  # "X ngày" or "Y đêm"
                        num = int(match.group(1))
                        constraints = AutoValidator.VALIDATION_RULES['duration']['constraints']
                        
                        if num > constraints['max_days']:
                            # Cap at max
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
                    # Try to extract price amount
                    amount_str = match.group(1).replace(',', '').replace('.', '')
                    if not amount_str.isdigit():
                        continue
                    
                    amount = int(amount_str)
                    
                    # Get unit if present
                    unit = match.group(2).lower() if match.lastindex >= 2 else ''
                    
                    # Convert to VND
                    if unit in ['triệu', 'tr']:
                        amount = amount * 1000000
                    elif unit in ['k', 'nghìn']:
                        amount = amount * 1000
                    
                    constraints = AutoValidator.VALIDATION_RULES['price']['constraints']
                    
                    # Check price range
                    if amount < constraints['min_tour_price']:
                        # Too cheap for a tour
                        replacement = "giá hợp lý"
                        text = text.replace(match.group(0), replacement)
                        logger.warning(f"⚠️ Corrected too-low price: {amount} → {replacement}")
                    
                    elif amount > constraints['max_tour_price']:
                        # Too expensive for a tour
                        replacement = "giá cao cấp"
                        text = text.replace(match.group(0), replacement)
                        logger.warning(f"⚠️ Corrected too-high price: {amount} → {replacement}")
                
                except (ValueError, IndexError, AttributeError):
                    continue
        
        return text
    
    @staticmethod
    def _validate_locations(text: str) -> str:
        """Validate location names"""
        # Simple check for obviously wrong locations
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
    def _validate_dates(text: str) -> str:
        """Validate date information"""
        # Currently just logs, doesn't modify
        date_patterns = AutoValidator.VALIDATION_RULES['dates']['patterns']
        
        for pattern in date_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                logger.info(f"📅 Found dates in response: {[m.group(0) for m in matches]}")
        
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

# =========== UPGRADE 10: TEMPLATE SYSTEM ===========
class TemplateSystem:
    """
    UPGRADE 10: Beautiful, structured responses for different question types
    Fixes: Ugly, unstructured responses
    """
    
    TEMPLATES = {
        # TOUR LIST TEMPLATE
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
        
        # TOUR DETAIL TEMPLATE
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
            'footer': "📞 **ĐẶT TOUR & TƯ VẤN:** 0332510486\n"
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
        
        # COMPARISON TEMPLATE
        'comparison': {
            'header': "📊 **SO SÁNH TOUR**\n\n",
            'table_header': "| Tiêu chí | {tour1} | {tour2} |\n|----------|----------|----------|\n",
            'table_row': "| {criterion} | {value1} | {value2} |\n",
            'recommendation': "\n💡 **GỢI Ý LỰA CHỌN:**\n{recommendations}\n",
            'footer': "\n📞 **Tư vấn chi tiết:** 0332510486\n"
                     "🤔 *Cần so sánh thêm tiêu chí nào?*",
        },
        # RECOMMENDATION TEMPLATE
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
        
        # INFORMATION RESPONSE TEMPLATE
        'information': {
            'header': "ℹ️ **THÔNG TIN:**\n\n",
            'content': "{content}\n",
            'sources': "\n📚 *Nguồn thông tin từ dữ liệu Ruby Wings*",
            'footer': "\n📞 **Hotline hỗ trợ:** 0332510486",
        },
        
        # GREETING TEMPLATE
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
        
        # FAREWELL TEMPLATE
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
        
        # Build response based on template structure
        response_parts = []
        
        # Add header
        if 'header' in template_data:
            header = template_data['header']
            for key, value in kwargs.items():
                header = header.replace(f'{{{key}}}', str(value))
            response_parts.append(header)
        
        # Handle different template types
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
        
        # Add footer
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
        for i, tour in enumerate(tours[:10], 1):  # Limit to 10 tours
            # Get emoji based on duration
            duration = tour.get('duration', '')
            emoji = template_data['emoji_map'].get('default')
            for dur_pattern, dur_emoji in template_data['emoji_map'].items():
                if dur_pattern in duration.lower():
                    emoji = dur_emoji
                    break
            
            item_template = template_data['item']
            item = item_template.format(
                index=i,
                tour_name=tour.get('tour_name', f'Tour #{i}'),
                emoji=emoji or '✨',
                duration=tour.get('duration', 'Đang cập nhật'),
                location=tour.get('location', 'Đang cập nhật'),
                price=tour.get('price', 'Liên hệ để biết giá'),
                summary=tour.get('summary', 'Tour trải nghiệm đặc sắc')[:100] + '...'
            )
            items.append(item)
        
        return '\n'.join(items)
    
    @staticmethod
    def _render_tour_detail(template_data: Dict, kwargs: Dict) -> str:
        """Render tour detail template"""
        sections = []
        
        for section_name, section_template in template_data['sections'].items():
            # Get value from kwargs or use default
            value = kwargs.get(section_name, template_data['default_values'].get(section_name, ''))
            
            if value and value != template_data['default_values'].get(section_name):
                # Format list values
                if isinstance(value, list):
                    if section_name == 'includes':
                        value = '\n'.join([f'   • {item}' for item in value[:5]])  # Limit to 5 items
                    else:
                        value = ', '.join(value[:3])  # Limit to 3 items
                
                section = section_template.format(**{section_name: value})
                sections.append(section)
        
        return '\n'.join(sections)
    
    @staticmethod
    def _render_comparison(template_data: Dict, kwargs: Dict) -> str:
        """Render comparison template"""
        comparison_table = []
        
        # Add table header
        tour1_name = kwargs.get('tour1_name', 'Tour 1')[:20]
        tour2_name = kwargs.get('tour2_name', 'Tour 2')[:20]
        table_header = template_data['table_header'].format(tour1=tour1_name, tour2=tour2_name)
        comparison_table.append(table_header)
        
        # Add comparison rows
        criteria = kwargs.get('criteria', [])
        for criterion in criteria[:8]:  # Limit to 8 criteria
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
        
        # Top recommendation
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
        
        # Other recommendations
        other_tours = kwargs.get('other_tours', [])
        if other_tours:
            recommendation_text.append(template_data['other_recommendations'])
            
            for tour in other_tours[:2]:  # Limit to 2 other tours
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
        
        # Add sources if available
        if kwargs.get('has_sources'):
            info_text += template_data['sources']
        
        return info_text

# =========== GLOBAL HELPER FUNCTIONS ===========
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

def get_session_context(session_id: str) -> 'EnhancedContext':
    """Get or create context for session"""
    with SESSION_LOCK:
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = EnhancedContext()
        
        # Clean old sessions
        now = datetime.utcnow()
        to_delete = []
        for sid, ctx in SESSION_CONTEXTS.items():
            if (now - ctx.timestamp).total_seconds() > SESSION_TIMEOUT:
                to_delete.append(sid)
        
        for sid in to_delete:
            del SESSION_CONTEXTS[sid]
        
        return SESSION_CONTEXTS[session_id]

def extract_session_id(request_data: Dict, remote_addr: str) -> str:
    """Extract or create session ID"""
    session_id = request_data.get("session_id")
    if not session_id:
        ip = remote_addr or "0.0.0.0"
        current_hour = datetime.utcnow().strftime("%Y%m%d%H")
        unique_str = f"{ip}_{current_hour}"
        session_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    return f"session_{session_id}"

# =========== ENHANCED CONTEXT CLASS ===========
class EnhancedContext:
    """Enhanced context for conversation"""
    
    def __init__(self):
        self.tour_mentions = []  # (tour_idx, confidence, timestamp)
        self.user_preferences = {
            "duration_pref": None,
            "price_range": None,
            "interests": [],
            "location_pref": None,
            "group_type": None,
            "physical_level": None,
        }
        self.conversation_history = []
        self.last_tour_indices = []
        self.last_tour_name = None
        self.last_action = None
        self.timestamp = datetime.utcnow()
        self.state_machine = None
        self.user_profile = {}
        # FIX 5: Add memory check for recent questions
        self.recent_questions = deque(maxlen=5)
        self.recent_responses = deque(maxlen=5)
    
    def update_from_message(self, message: str):
        """Update context from user message"""
        text_l = message.lower()
        
        # Extract interests
        interests_map = {
            'nature': ['thiên nhiên', 'rừng', 'cây', 'núi'],
            'history': ['lịch sử', 'di tích', 'chiến tranh', 'tri ân'],
            'culture': ['văn hóa', 'cộng đồng', 'dân tộc', 'truyền thống'],
            'spiritual': ['thiền', 'tâm linh', 'tĩnh tâm', 'yoga'],
            'wellness': ['khí công', 'sức khỏe', 'chữa lành'],
            'adventure': ['phiêu lưu', 'mạo hiểm', 'khám phá'],
        }
        
        for interest, keywords in interests_map.items():
            if any(keyword in text_l for keyword in keywords):
                if interest not in self.user_preferences["interests"]:
                    self.user_preferences["interests"].append(interest)
        
        # Extract duration preference
        if "1 ngày" in text_l or "1ngày" in text_l:
            self.user_preferences["duration_pref"] = "1day"
        elif "2 ngày" in text_l or "2ngày" in text_l:
            self.user_preferences["duration_pref"] = "2day"
        elif "3 ngày" in text_l or "3ngày" in text_l:
            self.user_preferences["duration_pref"] = "3day"
        
        # Extract price preference
        if any(word in text_l for word in ["rẻ", "giá rẻ", "tiết kiệm"]):
            self.user_preferences["price_range"] = "budget"
        elif any(word in text_l for word in ["vừa phải", "trung bình"]):
            self.user_preferences["price_range"] = "midrange"
        elif any(word in text_l for word in ["cao cấp", "sang", "premium"]):
            self.user_preferences["price_range"] = "premium"
        
        # Update timestamp
        self.timestamp = datetime.utcnow()
    
    def get_preferences_summary(self) -> str:
        """Get summary of user preferences"""
        prefs = []
        
        if self.user_preferences["duration_pref"]:
            prefs.append(f"Thời gian: {self.user_preferences['duration_pref']}")
        
        if self.user_preferences["price_range"]:
            prefs.append(f"Ngân sách: {self.user_preferences['price_range']}")
        
        if self.user_preferences["interests"]:
            prefs.append(f"Sở thích: {', '.join(self.user_preferences['interests'])}")
        
        return "; ".join(prefs) if prefs else "Chưa có thông tin sở thích"
    
    # FIX 5: Add memory check methods
    def check_recent_question(self, current_question: str) -> bool:
        """Check if similar question was asked recently"""
        if not self.recent_questions:
            return False
        
        current_lower = current_question.lower()
        for past_question in self.recent_questions:
            past_lower = past_question.lower()
            # Simple similarity check
            if (current_lower in past_lower or past_lower in current_lower or
                SequenceMatcher(None, current_lower, past_lower).ratio() > 0.7):
                return True
        return False
    
    def add_to_history(self, question: str, response: str):
        """Add question and response to history"""
        self.recent_questions.append(question)
        self.recent_responses.append(response)
    
    def get_recent_response(self, question: str) -> Optional[str]:
        """Get recent response for similar question"""
        if not self.recent_questions or not self.recent_responses:
            return None
        
        current_lower = question.lower()
        for i, past_question in enumerate(self.recent_questions):
            past_lower = past_question.lower()
            if (current_lower in past_lower or past_lower in current_lower or
                SequenceMatcher(None, current_lower, past_lower).ratio() > 0.7):
                return self.recent_responses[i]
        return None

# =========== KNOWLEDGE BASE FUNCTIONS ===========
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
                    # Keep the longer name if there's a conflict
                    existing_txt = MAPPING[next(
                        i for i, m2 in enumerate(MAPPING) 
                        if re.search(rf"\[{prev}\]", m2.get('path','')) and ".tour_name" in m2.get('path','')
                    )].get("text","")
                    if len(txt) > len(existing_txt):
                        TOUR_NAME_TO_INDEX[norm] = idx
    
    logger.info(f"📝 Indexed {len(TOUR_NAME_TO_INDEX)} tour names")

def build_tours_db():
    """Build structured tour database from MAPPING"""
    global TOURS_DB, TOUR_TAGS, TOUR_METADATA
    
    TOURS_DB.clear()
    TOUR_TAGS.clear()
    TOUR_METADATA.clear()
    
    # First pass: collect all fields for each tour
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        
        if not path or not text:
            continue
        
        # Extract tour index
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            continue
        
        tour_idx = int(tour_match.group(1))
        
        # Extract field name
        field_match = re.search(r'tours\[\d+\]\.(\w+)(?:\[\d+\])?', path)
        if not field_match:
            continue
        
        field_name = field_match.group(1)
        
        # Initialize tour entry
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = {}
            TOUR_METADATA[tour_idx] = {
                'completeness_score': 0,
                'popularity_score': 0.5,
                'last_mentioned': None,
            }
        
        # Handle field values
        current_value = TOURS_DB[tour_idx].get(field_name)
        if current_value is None:
            TOURS_DB[tour_idx][field_name] = text
        elif isinstance(current_value, list):
            current_value.append(text)
        elif isinstance(current_value, str):
            TOURS_DB[tour_idx][field_name] = [current_value, text]
    
    # Second pass: generate tags and metadata
    for tour_idx, tour_data in TOURS_DB.items():
        tags = []
        
        # Location tags
        location = tour_data.get("location", "")
        if location:
            locations = [loc.strip() for loc in location.split(",") if loc.strip()]
            tags.extend([f"location:{loc}" for loc in locations[:2]])
        
        # Duration tags
        duration = tour_data.get("duration", "")
        if duration:
            duration_lower = duration.lower()
            if "1 ngày" in duration_lower:
                tags.append("duration:1day")
            elif "2 ngày" in duration_lower:
                tags.append("duration:2day")
            elif "3 ngày" in duration_lower:
                tags.append("duration:3day")
            else:
                # Extract number of days
                day_match = re.search(r'(\d+)\s*ngày', duration_lower)
                if day_match:
                    days = int(day_match.group(1))
                    tags.append(f"duration:{days}day")
        
        # Price tags
        price = tour_data.get("price", "")
        if price:
            # Extract numeric values
            price_nums = re.findall(r'[\d,\.]+', price)
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
        style = tour_data.get("style", "")
        summary = tour_data.get("summary", "")
        text_to_check = (style + " " + summary).lower()
        
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
        tour_name = tour_data.get('tour_name', '')
        if tour_name:
            name_lower = tour_name.lower()
            if "bạch mã" in name_lower:
                tags.append("destination:bachma")
            if "trường sơn" in name_lower:
                tags.append("destination:truongson")
            if "quảng trị" in name_lower:
                tags.append("destination:quangtri")
            if "huế" in name_lower:
                tags.append("destination:hue")
                # Destination tags from tour name
        tour_name = tour_data.get('tour_name', '')
        if tour_name:
            name_lower = tour_name.lower()
            if "bạch mã" in name_lower:
                tags.append("destination:bachma")
            if "trường sơn" in name_lower:
                tags.append("destination:truongson")
            if "quảng trị" in name_lower:
                tags.append("destination:quangtri")
            if "huế" in name_lower:
                tags.append("destination:hue")
        
        TOUR_TAGS[tour_idx] = list(set(tags))
        
        # Calculate completeness score
        completeness = 0
        important_fields = ['tour_name', 'duration', 'location', 'price', 'summary']
        for field in important_fields:
            if field in tour_data and tour_data[field]:
                completeness += 1
        
        TOUR_METADATA[tour_idx]['completeness_score'] = completeness / len(important_fields)
    
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

# =========== CACHE SYSTEM ===========
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
                cached_time, value = _response_cache[key]
                if (datetime.utcnow() - cached_time).total_seconds() < ttl_seconds:
                    logger.info(f"💾 Cache hit for key: {key[:20]}...")
                    return value
                else:
                    # Expired
                    del _response_cache[key]
            return None
    
    @staticmethod
    def set(key: str, value: Any):
        """Set item in cache"""
        with _cache_lock:
            _response_cache[key] = (datetime.utcnow(), value)
            
            # Clean old entries (keep max 1000)
            if len(_response_cache) > 1000:
                # Remove oldest 200 entries
                sorted_items = sorted(_response_cache.items(), 
                                     key=lambda x: x[1][0])
                for old_key in [k for k, _ in sorted_items[:200]]:
                    if old_key in _response_cache:
                        del _response_cache[old_key]

# =========== MAIN CHAT ENDPOINT WITH ALL UPGRADES ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Main chat endpoint with all 10 upgrades integrated
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        
        if not user_message:
            return jsonify({
                "reply": "Xin chào! Tôi có thể giúp gì cho bạn về các tour của Ruby Wings?",
                "sources": [],
                "context": {},
                "processing_time": 0
            })
        
        # Extract session ID
        session_id = extract_session_id(data, request.remote_addr)
        
        # Get or create session context
        context = get_session_context(session_id)
        
        # FIX 5: Check memory before processing
        recent_response = context.get_recent_response(user_message)
        if recent_response and context.check_recent_question(user_message):
            logger.info("💭 Using cached response from recent conversation")
            processing_time = time.time() - start_time
            return jsonify({
                "reply": recent_response,
                "sources": [],
                "context": {
                    "session_id": session_id,
                    "from_memory": True,
                    "processing_time_ms": int(processing_time * 1000)
                }
            })
        
        context.update_from_message(user_message)
        
        # Initialize state machine if not exists
        if UpgradeFlags.is_enabled("7_STATE_MACHINE"):
            if not context.state_machine:
                context.state_machine = ConversationStateMachine(session_id)
        
        # FIX 2: Inject context BEFORE any processing
        # Always check state machine for references first
        state_tour_indices = []
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            state_tour_indices = context.state_machine.extract_reference(user_message)
            if state_tour_indices:
                logger.info(f"🔄 State machine injected tours: {state_tour_indices}")
                context.last_tour_indices = state_tour_indices
        
        # =========== UPGRADE 5: COMPLEX QUERY SPLITTER ===========
        sub_queries = []
        if UpgradeFlags.is_enabled("5_QUERY_SPLITTER"):
            sub_queries = ComplexQueryProcessor.split_query(user_message)
        
        # =========== UPGRADE 1: MANDATORY FILTER EXTRACTION ===========
        mandatory_filters = {}
        if UpgradeFlags.is_enabled("1_MANDATORY_FILTER"):
            mandatory_filters = MandatoryFilterSystem.extract_filters(user_message)
            
            # Apply filters to available tours
            if mandatory_filters and TOURS_DB:
                filtered_indices = MandatoryFilterSystem.apply_filters(TOURS_DB, mandatory_filters)
                if filtered_indices:
                    # Store filtered indices for later use
                    # FIX 2: Combine with state machine tours if available
                    if state_tour_indices:
                        # Intersection of state tours and filtered tours
                        combined = [idx for idx in state_tour_indices if idx in filtered_indices]
                        context.last_tour_indices = combined if combined else filtered_indices
                    else:
                        context.last_tour_indices = filtered_indices
                    logger.info(f"🔍 Applied mandatory filters: {mandatory_filters}")
        
        # =========== UPGRADE 6: FUZZY MATCHING ===========
        fuzzy_matches = []
        if UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            fuzzy_matches = FuzzyMatcher.find_similar_tours(user_message, TOUR_NAME_TO_INDEX)
            if fuzzy_matches:
                fuzzy_indices = [idx for idx, _ in fuzzy_matches]
                logger.info(f"🔍 Fuzzy matches found: {fuzzy_indices}")
                
                # Combine with context tours
                if context.last_tour_indices:
                    context.last_tour_indices = list(set(context.last_tour_indices + fuzzy_indices))
                else:
                    context.last_tour_indices = fuzzy_indices
        
        # =========== UPGRADE 3: ENHANCED FIELD DETECTION ===========
        requested_field = None
        field_confidence = 0.0
        if UpgradeFlags.is_enabled("3_ENHANCED_FIELDS"):
            requested_field, field_confidence, field_scores = EnhancedFieldDetector.detect_field_with_confidence(user_message)
        
        # =========== UPGRADE 4: QUESTION CLASSIFICATION ===========
        question_type = QuestionPipeline.QuestionType.INFORMATION
        question_confidence = 0.0
        question_metadata = {}
        
        if UpgradeFlags.is_enabled("4_QUESTION_PIPELINE"):
            question_type, question_confidence, question_metadata = QuestionPipeline.classify_question(user_message)
        
        # =========== UPGRADE 8: SEMANTIC ANALYSIS ===========
        user_profile = {}
        if UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            user_profile = SemanticAnalyzer.analyze_user_profile(user_message, context.user_profile)
            context.user_profile = user_profile
        
        # =========== UPGRADE 7: STATE MACHINE PROCESSING ===========
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            # Update state machine with placeholder
            placeholder_response = "Processing your request..."
            context.state_machine.update(user_message, placeholder_response, context.last_tour_indices)
        
        # =========== TOUR RESOLUTION ===========
        tour_indices = context.last_tour_indices or []
        
        # FIX 3: Extract tour names from comparison questions
        if question_type == QuestionPipeline.QuestionType.COMPARISON and not tour_indices:
            # Try to extract tour names from the message
            comparison_tour_names = []
            # Look for patterns like "tour A và tour B"
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
                            # Find tour by name
                            for norm_name, idx in TOUR_NAME_TO_INDEX.items():
                                if tour_name in norm_name or FuzzyMatcher.normalize_vietnamese(tour_name) in norm_name:
                                    comparison_tour_names.append(idx)
                                    break
            
            if len(comparison_tour_names) >= 2:
                tour_indices = comparison_tour_names[:2]
                context.last_tour_indices = tour_indices
                logger.info(f"🔍 Extracted tours for comparison: {tour_indices}")
        
        # =========== PROCESS BY QUESTION TYPE ===========
        reply = ""
        sources = []
        
        # Check cache first
        cache_key = None
        if UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            context_hash = hashlib.md5(json.dumps({
                'tour_indices': tour_indices,
                'field': requested_field,
                'question_type': question_type.value,
                'filters': mandatory_filters
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = CacheSystem.get_cache_key(user_message, context_hash)
            cached_response = CacheSystem.get(cache_key)
            
            if cached_response:
                logger.info("💾 Using cached response")
                return jsonify(cached_response)
        
        # GREETING
        if question_type == QuestionPipeline.QuestionType.GREETING:
            reply = TemplateSystem.render('greeting')
        
        # FAREWELL
        elif question_type == QuestionPipeline.QuestionType.FAREWELL:
            reply = TemplateSystem.render('farewell')
        
        # COMPARISON
        elif question_type == QuestionPipeline.QuestionType.COMPARISON:
            if len(tour_indices) >= 2:
                # Use UPGRADE 4 comparison processing
                comparison_result = QuestionPipeline.process_comparison_question(
                    tour_indices, TOURS_DB, "", question_metadata
                )
                reply = comparison_result
            else:
                # FIX 3: Try to suggest tours for comparison
                if TOURS_DB:
                    # Get 2 most popular tours
                    all_tours = list(TOURS_DB.items())
                    if len(all_tours) >= 2:
                        tour1_idx, tour1_data = all_tours[0]
                        tour2_idx, tour2_data = all_tours[1]
                        reply = f"Bạn có thể so sánh:\n1. {tour1_data.get('tour_name', f'Tour #{tour1_idx}')}\n2. {tour2_data.get('tour_name', f'Tour #{tour2_idx}')}\n\nHãy cho tôi biết bạn muốn so sánh tour nào cụ thể."
                    else:
                        reply = "Hiện chỉ có 1 tour trong hệ thống, không thể so sánh."
                else:
                    reply = "Bạn muốn so sánh tour nào với nhau? Vui lòng nêu tên 2 tour trở lên."
        
        # RECOMMENDATION
        elif question_type == QuestionPipeline.QuestionType.RECOMMENDATION:
            # FIX 5: Check if this is actually a comparison misclassified
            if 'so sánh' in user_message.lower() or 'với' in user_message.lower():
                # Reclassify as comparison
                question_type = QuestionPipeline.QuestionType.COMPARISON
                # Try to extract tours
                if not tour_indices and TOURS_DB:
                    tour_indices = list(TOURS_DB.keys())[:2]
                    reply = f"Tôi thấy bạn muốn so sánh. Bạn có thể so sánh:\n1. {TOURS_DB[tour_indices[0]].get('tour_name', f'Tour #{tour_indices[0]}')}\n2. {TOURS_DB[tour_indices[1]].get('tour_name', f'Tour #{tour_indices[1]}')}"
                else:
                    reply = "Bạn muốn so sánh tour nào với nhau?"
            elif UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
                # Use semantic analysis for recommendations
                profile_matches = SemanticAnalyzer.match_tours_to_profile(
                    user_profile, TOURS_DB, max_results=3
                )
                
                if profile_matches:
                    # Format recommendations
                    recommendations = []
                    for idx, score, reasons in profile_matches:
                        tour_data = TOURS_DB.get(idx, {})
                        recommendations.append({
                            'name': tour_data.get('tour_name', f'Tour #{idx}'),
                            'score': score,
                            'reasons': reasons,
                            'duration': tour_data.get('duration', ''),
                            'location': tour_data.get('location', ''),
                            'price': tour_data.get('price', ''),
                        })
                    
                    if recommendations:
                        reply = TemplateSystem.render('recommendation',
                            top_tour=recommendations[0] if recommendations else None,
                            other_tours=recommendations[1:] if len(recommendations) > 1 else [],
                            criteria=context.get_preferences_summary()
                        )
                    else:
                        reply = "Hiện chưa tìm thấy tour phù hợp với yêu cầu của bạn."
                else:
                    reply = "Xin lỗi, tôi chưa hiểu rõ bạn cần tour như thế nào. " \
                           "Bạn có thể nói cụ thể hơn về sở thích và yêu cầu của mình không?"
            else:
                # Simple recommendation fallback
                if TOURS_DB:
                    top_tours = list(TOURS_DB.items())[:2]
                    reply = "Dựa trên thông tin hiện có, tôi đề xuất bạn tham khảo:\n"
                    for idx, tour_data in top_tours:
                        reply += f"• {tour_data.get('tour_name', f'Tour #{idx}')}\n"
                    reply += "\n💡 Bạn có thể hỏi chi tiết về từng tour cụ thể."
                else:
                    reply = "Hiện chưa có thông tin tour để đề xuất."
        
        # LISTING
        elif question_type == QuestionPipeline.QuestionType.LISTING or requested_field == "tour_name":
            # Get all tours
            all_tours = []
            for idx, tour_data in TOURS_DB.items():
                all_tours.append({
                    'tour_name': tour_data.get('tour_name', f'Tour #{idx}'),
                    'duration': tour_data.get('duration', ''),
                    'location': tour_data.get('location', ''),
                    'price': tour_data.get('price', ''),
                    'summary': tour_data.get('summary', ''),
                })
            
            # Apply UPGRADE 2: DEDUPLICATION
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and all_tours:
                # Simple deduplication by tour name
                seen_names = set()
                unique_tours = []
                for tour in all_tours:
                    name = tour['tour_name']
                    if name not in seen_names:
                        seen_names.add(name)
                        unique_tours.append(tour)
                all_tours = unique_tours
            
            # Limit number of tours
            all_tours = all_tours[:15]
            
            # Use UPGRADE 10: TEMPLATE SYSTEM
            if UpgradeFlags.is_enabled("10_TEMPLATE_SYSTEM"):
                reply = TemplateSystem.render('tour_list', tours=all_tours)
            else:
                # Fallback format
                if all_tours:
                    reply = "✨ **Danh sách tour Ruby Wings:** ✨\n\n"
                    for i, tour in enumerate(all_tours[:10], 1):
                        reply += f"{i}. **{tour['tour_name']}**\n"
                        if tour['duration']:
                            reply += f"   ⏱️ {tour['duration']}\n"
                        if tour['location']:
                            reply += f"   📍 {tour['location']}\n"
                        reply += "\n"
                    reply += "💡 *Hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour*"
                else:
                    reply = "Hiện chưa có thông tin tour trong hệ thống."
        
        # FIELD-SPECIFIC QUERY
        elif requested_field and field_confidence > 0.3:
            # Get field information
            if tour_indices:
                # Get info for specific tours
                field_info = []
                for idx in tour_indices:
                    if idx in TOURS_DB:
                        tour_data = TOURS_DB[idx]
                        field_value = tour_data.get(requested_field)
                        if field_value:
                            if isinstance(field_value, list):
                                field_text = "\n".join([f"• {item}" for item in field_value])
                            else:
                                field_text = field_value
                            
                            tour_name = tour_data.get('tour_name', f'Tour #{idx}')
                            field_info.append(f"**{tour_name}**:\n{field_text}")
                
                if field_info:
                    reply = "\n\n".join(field_info)
                    # Add sources
                    field_passages = get_passages_by_field(requested_field, tour_indices=tour_indices)
                    sources = [m for _, m in field_passages]
                else:
                    reply = f"Không tìm thấy thông tin về {requested_field} cho tour đã chọn."
            else:
                # General field information
                field_passages = get_passages_by_field(requested_field, limit=5)
                if field_passages:
                    field_texts = [m.get('text', '') for _, m in field_passages]
                    reply = "**Thông tin chung:**\n" + "\n".join([f"• {text}" for text in field_texts[:3]])
                    sources = [m for _, m in field_passages]
                else:
                    reply = f"Hiện không có thông tin về {requested_field} trong dữ liệu."
        
        # DEFAULT: SEMANTIC SEARCH + LLM
        else:
            # Perform semantic search
            search_results = query_index(user_message, TOP_K)
            
            # Apply UPGRADE 2: DEDUPLICATION
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and search_results:
                search_results = DeduplicationEngine.deduplicate_passages(search_results)
            
            # Prepare context for LLM
            llm_context = {
                'user_message': user_message,
                'tour_indices': tour_indices,
                'question_type': question_type.value,
                'requested_field': requested_field,
                'user_preferences': context.user_preferences,
                'current_tours': [],
                'filters': mandatory_filters
            }
            
            # Add tour information if available
            if tour_indices:
                for idx in tour_indices[:2]:  # Limit to 2 tours
                    if idx in TOURS_DB:
                        llm_context['current_tours'].append({
                            'index': idx,
                            'name': TOURS_DB[idx].get('tour_name', f'Tour #{idx}'),
                            'duration': TOURS_DB[idx].get('duration', ''),
                            'location': TOURS_DB[idx].get('location', ''),
                            'price': TOURS_DB[idx].get('price', ''),
                        })
            
            # Prepare prompt
            prompt = _prepare_llm_prompt(user_message, search_results, llm_context)
            
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
                # Fallback response
                reply = _generate_fallback_response(user_message, search_results, tour_indices)
            
            # Update sources
            sources = [m for _, m in search_results]
        
        # =========== UPGRADE 9: AUTO-VALIDATION ===========
        if UpgradeFlags.is_enabled("9_AUTO_VALIDATION"):
            reply = AutoValidator.validate_response(reply)
        
        # =========== UPDATE CONTEXT ===========
        context.last_action = "chat_response"
        context.timestamp = datetime.utcnow()
        
        if tour_indices:
            # Find tour name for the first tour
            if tour_indices[0] in TOURS_DB:
                context.last_tour_name = TOURS_DB[tour_indices[0]].get('tour_name')
        
        # Update state machine with actual response
        if UpgradeFlags.is_enabled("7_STATE_MACHINE") and context.state_machine:
            context.state_machine.update(user_message, reply, tour_indices)
        
        # FIX 5: Add to memory
        context.add_to_history(user_message, reply)
        
        # =========== PREPARE RESPONSE ===========
        processing_time = time.time() - start_time
        
        response_data = {
            "reply": reply,
            "sources": sources,
            "context": {
                "session_id": session_id,
                "tour_indices": tour_indices,
                "last_tour_name": context.last_tour_name,
                "user_preferences": context.user_preferences,
                "question_type": question_type.value,
                "requested_field": requested_field,
                "processing_time_ms": int(processing_time * 1000),
                "from_memory": False
            }
        }
        
        # Cache the response
        if cache_key and UpgradeFlags.get_all_flags().get("ENABLE_CACHING", True):
            CacheSystem.set(cache_key, response_data)
        
        # Log processing info
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
    
    # Add context info
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
        if 'price_max' in filters:
            filter_strs.append(f"giá dưới {filters['price_max']:,} VND")
        if 'price_min' in filters:
            filter_strs.append(f"giá trên {filters['price_min']:,} VND")
        if 'location' in filters:
            filter_strs.append(f"địa điểm: {filters['location']}")
        if filter_strs:
            prompt_parts.append(f"- Bộ lọc: {', '.join(filter_strs)}")
    
    prompt_parts.append("")
    prompt_parts.append("DỮ LIỆU NỘI BỘ RUBY WINGS:")
    
    if search_results:
        for i, (score, passage) in enumerate(search_results[:5], 1):
            text = passage.get('text', '')[:300]  # Limit text length
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
    # FIX 4: Handle price filter queries specifically
    message_lower = user_message.lower()
    
    if 'dưới' in message_lower and ('triệu' in message_lower or 'tiền' in message_lower):
        # Price filter query
        if not tour_indices and TOURS_DB:
            # Try to get some tours anyway
            all_tours = list(TOURS_DB.items())[:3]
            response = "Dựa trên yêu cầu của bạn, tôi đề xuất các tour có giá hợp lý:\n"
            for idx, tour_data in all_tours:
                tour_name = tour_data.get('tour_name', f'Tour #{idx}')
                price = tour_data.get('price', 'Liên hệ để biết giá')
                response += f"• **{tour_name}**: {price}\n"
            response += "\n💡 *Liên hệ hotline 0332510486 để biết giá chính xác và ưu đãi*"
            return response
    
    if not search_results:
        if tour_indices and TOURS_DB:
            # We have tour indices but no search results
            response = "Thông tin về tour bạn quan tâm:\n"
            for idx in tour_indices[:2]:
                if idx in TOURS_DB:
                    tour_data = TOURS_DB[idx]
                    response += f"\n**{tour_data.get('tour_name', f'Tour #{idx}')}**\n"
                    if tour_data.get('duration'):
                        response += f"⏱️ {tour_data['duration']}\n"
                    if tour_data.get('location'):
                        response += f"📍 {tour_data['location']}\n"
                    if tour_data.get('price'):
                        response += f"💰 {tour_data['price']}\n"
            response += "\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết*"
            return response
        else:
            return "Xin lỗi, hiện không tìm thấy thông tin liên quan trong dữ liệu. " \
                   "Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp."
    
    # Use top 3 results
    top_results = search_results[:3]
    response_parts = ["Tôi tìm thấy một số thông tin liên quan:"]
    
    for i, (score, passage) in enumerate(top_results, 1):
        text = passage.get('text', '')[:150]
        if text:
            response_parts.append(f"\n{i}. {text}")
    
    response_parts.append("\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết*")
    
    return "".join(response_parts)

# =========== OTHER ENDPOINTS (UNCHANGED) ===========
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
    """Save lead to Google Sheets"""
    try:
        if not request.is_json:
            return jsonify({"error": "JSON required", "success": False}), 400
        
        data = request.get_json() or {}
        phone = (data.get("phone") or "").strip()
        
        if not phone:
            return jsonify({"error": "Phone required", "success": False}), 400
        
        lead_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "source_channel": data.get("source_channel", "Website"),
            "action_type": data.get("action_type", "Click Call"),
            "page_url": data.get("page_url", ""),
            "contact_name": data.get("contact_name", ""),
            "phone": phone,
            "service_interest": data.get("service_interest", ""),
            "note": data.get("note", ""),
            "status": "New",
        }
        
        logger.info(f"💾 Processing lead: {phone}")
        
        # Try Google Sheets
        sheets_success = False
        if ENABLE_GOOGLE_SHEETS and HAS_GOOGLE_SHEETS:
            try:
                gc = get_gspread_client()
                if gc:
                    sh = gc.open_by_key(GOOGLE_SHEET_ID)
                    ws = sh.worksheet(GOOGLE_SHEET_NAME)
                    
                    row = [
                        lead_data["timestamp"],
                        lead_data["source_channel"],
                        lead_data["action_type"],
                        lead_data["page_url"],
                        lead_data["contact_name"],
                        lead_data["phone"],
                        lead_data["service_interest"],
                        lead_data["note"],
                        lead_data["status"]
                    ]
                    
                    ws.append_row(row)
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
                
                leads.append(lead_data)
                
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
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# =========== INDEX MANAGEMENT ===========
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

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, Dict]]:
    """Query the index"""
    global INDEX
    
    if not query or INDEX is None:
        return []
    
    # Get query embedding
    emb, _ = embed_text(query)
    if not emb:
        return []
    
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    
    # Search
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

@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """Embed text using OpenAI or fallback"""
    if not text:
        return [], 0
    
    text = text[:2000]  # Limit length
    
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

class NumpyIndex:
    """Simple numpy-based index"""
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if mat is not None else np.empty((0, 0), dtype="float32")
        self.dim = self.mat.shape[1] if self.mat.size > 0 else None
    
    def search(self, qvec, k):
        if self.mat.size == 0:
            return np.empty((1, 0)), np.empty((1, 0), dtype=int)
        
        q = qvec.astype("float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        
        return scores.astype("float32"), idx.astype("int64")
    
    def save(self, path):
        np.savez_compressed(path, mat=self.mat)
    
    @classmethod
    def load(cls, path):
        arr = np.load(path)
        return cls(arr['mat'])

# =========== INITIALIZATION ===========
def initialize_app():
    """Initialize the application"""
    logger.info("🚀 Starting Ruby Wings Chatbot v4.0...")
    
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
        time.sleep(2)  # Wait a bit for app to start
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
    
    logger.info("✅ Application initialized successfully")

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
        