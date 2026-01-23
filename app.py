"""
RUBY WINGS CHATBOT v5.0 - CẬP NHẬT KNOWLEDGE.JSON
Cấu trúc hoàn chỉnh theo đề cương chuẩn hóa
"""

# ==================== PHẦN 1: IMPORTS & CONFIG ====================
import json
import logging
import re
import hashlib
import time
import os
import threading
import traceback
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
from collections import defaultdict, deque
from difflib import SequenceMatcher
import requests
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
app = Flask(__name__)

from flask_cors import CORS

# Try to import FAISS with fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("⚠️ FAISS not available, using fallback")

# Các biến config
LLM_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_PATH = "knowledge.json"
CAPI_ENABLED = True
CAPI_URL = "https://graph.facebook.com/v18.0/me/messages"
SESSION_TIMEOUT = 1800
CACHE_TTL = 300
MAX_TOURS_RETURN = 10
SEMANTIC_MIN_SCORE = 0.75
TOP_K = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ruby_wings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== PHẦN 2: DATACLASSES & ENUMS ====================

class QuestionType(Enum):
    """Các loại câu hỏi - GIỮ NGUYÊN từ app gốc"""
    LIST_TOURS = "list_tours"
    TOUR_DETAIL = "tour_detail"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    GENERAL_INFO = "general_info"
    UNKNOWN = "unknown"

class ConversationState(Enum):
    """Trạng thái hội thoại - GIỮ NGUYÊN"""
    INITIAL = "initial"
    FILTERING = "filtering"
    DETAIL_VIEW = "detail_view"
    COMPARISON = "comparison"
    RECOMMENDING = "recommending"
    CLOSING = "closing"

@dataclass
class Tour:
    """
    TOUR OBJECT MỚI - TƯƠNG THÍCH KNOWLEDGE.JSON
    Kế thừa tất cả field từ knowledge.json + thêm computed fields
    """
    # Primary fields từ knowledge.json
    id: int
    tour_name: str
    summary: str
    location: str
    duration: str  # Giữ nguyên string format
    price: str     # Giữ nguyên string format
    includes: List[str]
    notes: str
    style: str
    transport: str
    accommodation: str
    meals: str
    event_support: str
    
    # Computed fields để hỗ trợ filter/search
    price_numeric: Optional[float] = None    # Giá đã parse sang số
    duration_numeric: Optional[int] = None   # Thời gian đã parse sang số ngày
    category: Optional[str] = None          # Loại tour (auto-categorized)
    rating: Optional[float] = 4.5           # Rating mặc định
    
    # Backward compatibility fields
    description: str = ""                   # Map từ summary
    highlights: List[str] = field(default_factory=list)  # Map từ includes
    name: str = ""                          # Alias cho tour_name
    tags: List[str] = field(default_factory=list)       # Auto-generated tags

@dataclass
class FilterSet:
    """Bộ lọc - THÊM field style để filter theo knowledge.json"""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    location: Optional[str] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    style: Optional[str] = None            # NEW: Filter theo field style
    category: Optional[str] = None
    include_keywords: Optional[List[str]] = None  # Tìm trong includes
    group_type: Optional[str] = None
    
    def is_empty(self) -> bool:
        """Check if filter set is empty"""
        return all(
            value is None or (isinstance(value, list) and not value)
            for value in [
                self.min_price, self.max_price, self.location,
                self.duration_min, self.duration_max, self.style,
                self.category, self.include_keywords, self.group_type
            ]
        )

@dataclass
class ConversationContext:
    """Context hội thoại - GIỮ NGUYÊN từ app gốc"""
    session_id: str
    last_tours_mentioned: List[int] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    last_question_type: Optional[QuestionType] = None
    current_state: ConversationState = ConversationState.INITIAL
    active_filters: Optional[FilterSet] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Additional fields for backward compatibility
    current_tours: List[int] = field(default_factory=list)
    last_successful_tours: List[int] = field(default_factory=list)
    mentioned_tours: List[int] = field(default_factory=list)
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update conversation context"""
        self.last_activity = time.time()
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': user_message,
            'bot': bot_response,
            'tours': tour_indices
        })
        
        if tour_indices:
            self.last_successful_tours = tour_indices
            self.current_tours = tour_indices
            self.mentioned_tours.extend(tour_indices)
            
            # Keep only recent tours
            if len(self.mentioned_tours) > 20:
                self.mentioned_tours = self.mentioned_tours[-20:]

@dataclass
class ChatResponse:
    """Response format - GIỮ NGUYÊN từ app gốc"""
    reply: str
    tour_name: Optional[str] = None
    tour_indices: Optional[List[int]] = None
    action: str = "continue"
    context: Optional[Dict] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict] = None

@dataclass  
class CacheEntry:
    """Cache entry - GIỮ NGUYÊN từ app gốc"""
    value: Any
    expiry: float
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > self.expiry

@dataclass
class LLMRequest:
    """LLM request - GIỮ NGUYÊN từ app gốc"""
    prompt: str
    model: str = "llama2"
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 500

# ==================== PHẦN 3: KNOWLEDGE PROCESSING ====================

class KnowledgeLoader:
    """Hệ thống load và parse knowledge.json"""
    
    @staticmethod
    def load_knowledge_file(file_path: str = KNOWLEDGE_PATH) -> Dict:
        """Load raw JSON data từ knowledge.json"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load knowledge file: {e}")
            return {"tours": []}
    
    @staticmethod
    def parse_tour_data(raw_tour: Dict, tour_id: int) -> Tour:
        """Parse một tour từ raw JSON data sang Tour object"""
        
        # Parse numeric values
        price_numeric = KnowledgeParser.parse_price_string(raw_tour.get('price', ''))
        duration_numeric = KnowledgeParser.parse_duration_string(raw_tour.get('duration', ''))
        
        # Auto-categorize
        category = KnowledgeParser.categorize_tour(raw_tour)
        
        # Create Tour object
        tour = Tour(
            id=tour_id,
            tour_name=raw_tour.get('tour_name', ''),
            summary=raw_tour.get('summary', ''),
            location=raw_tour.get('location', ''),
            duration=raw_tour.get('duration', ''),
            price=raw_tour.get('price', ''),
            includes=raw_tour.get('includes', []),
            notes=raw_tour.get('notes', ''),
            style=raw_tour.get('style', ''),
            transport=raw_tour.get('transport', ''),
            accommodation=raw_tour.get('accommodation', ''),
            meals=raw_tour.get('meals', ''),
            event_support=raw_tour.get('event_support', ''),
            price_numeric=price_numeric,
            duration_numeric=duration_numeric,
            category=category,
            description=raw_tour.get('summary', ''),
            highlights=raw_tour.get('includes', [])[:3],
            name=raw_tour.get('tour_name', ''),
            tags=KnowledgeParser.generate_tags(raw_tour)
        )
        
        return tour
    
    @classmethod
    def build_tours_database(cls) -> Dict[int, Tour]:
        """Xây dựng database tours từ knowledge.json"""
        knowledge_data = cls.load_knowledge_file()
        tours_db = {}
        
        for idx, tour_data in enumerate(knowledge_data.get('tours', [])):
            tour = cls.parse_tour_data(tour_data, idx)
            tours_db[idx] = tour
        
        logger.info(f"Loaded {len(tours_db)} tours from knowledge.json")
        return tours_db

class KnowledgeParser:
    """Parser utilities cho knowledge.json fields"""
    
    @staticmethod
    def parse_price_string(price_str: str) -> Optional[float]:
        """Parse chuỗi giá sang số"""
        if not price_str or not isinstance(price_str, str):
            return None
        
        # Tìm tất cả số trong chuỗi (hỗ trợ cả dấu phẩy và chấm)
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', price_str.replace(',', '.'))
        if numbers:
            try:
                value = float(numbers[0])
                # Giả định giá tính theo triệu VND
                return value * 1000000
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def parse_duration_string(duration_str: str) -> Optional[int]:
        """Parse chuỗi thời gian sang số ngày"""
        if not duration_str or not isinstance(duration_str, str):
            return None
        
        numbers = re.findall(r'\d+', duration_str)
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def categorize_tour(tour_data: Dict) -> str:
        """Phân loại tour tự động dựa trên style và location"""
        style = (tour_data.get('style') or '').lower()
        location = (tour_data.get('location') or '').lower()
        
        category_keywords = {
            'adventure': ['mạo hiểm', 'khám phá', 'trekking', 'leo núi', 'phượt'],
            'relaxation': ['nghỉ dưỡng', 'thư giãn', 'biển', 'spa', 'resort'],
            'cultural': ['văn hóa', 'lịch sử', 'di sản', 'di tích', 'truyền thống'],
            'culinary': ['ẩm thực', 'ăn uống', 'đặc sản', 'food tour'],
            'event': ['sự kiện', 'team building', 'hội nghị', 'tổ chức'],
            'family': ['gia đình', 'trẻ em', 'trải nghiệm gia đình'],
            'luxury': ['cao cấp', 'sang trọng', '5 sao', 'VIP']
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in style or keyword in location:
                    return category
        
        return 'general'
    
    @staticmethod
    def generate_tags(tour_data: Dict) -> List[str]:
        """Tạo tags cho tour dựa trên dữ liệu"""
        tags = []
        
        # Location tags
        location = (tour_data.get('location') or '').lower()
        if location:
            for loc in ['huế', 'quảng trị', 'bạch mã', 'trường sơn']:
                if loc in location:
                    tags.append(f"location:{loc}")
        
        # Style tags
        style = (tour_data.get('style') or '').lower()
        if style:
            for st in ['thiền', 'khí công', 'retreat', 'lịch sử', 'văn hóa']:
                if st in style:
                    tags.append(f"style:{st}")
        
        # Duration tags
        duration = (tour_data.get('duration') or '').lower()
        if '1 ngày' in duration:
            tags.append("duration:1day")
        elif '2 ngày' in duration:
            tags.append("duration:2day")
        elif '3 ngày' in duration:
            tags.append("duration:3day")
        
        # Price tags (if price available)
        price = tour_data.get('price', '')
        price_numeric = KnowledgeParser.parse_price_string(price)
        if price_numeric:
            if price_numeric < 1000000:
                tags.append("price:budget")
            elif price_numeric < 3000000:
                tags.append("price:midrange")
            else:
                tags.append("price:premium")
        
        return list(set(tags))

# ==================== PHẦN 4: 10 UPGRADES SYSTEMS ====================

class MandatoryFilterSystemV2:
    """
    Upgrade 1: Mandatory Filter System
    CẬP NHẬT: Hỗ trợ các field mới từ knowledge.json
    """
    
    @staticmethod
    def extract_filters(message: str) -> FilterSet:
        """Trích xuất filter từ message với knowledge.json fields"""
        filters = FilterSet()
        msg_lower = message.lower()
        
        # 1. Price filter
        price_patterns = [
            (r'giá\s*(?:dưới|dưới\s*)?\s*(\d+(?:[.,]\d+)?)\s*tr?i?ệ?u?', 'max'),
            (r'giá\s*(?:trên|trên\s*)?\s*(\d+(?:[.,]\d+)?)\s*tr?i?ệ?u?', 'min'),
            (r'(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*tr?i?ệ?u?', 'range'),
            (r'khoảng\s*(\d+(?:[.,]\d+)?)\s*tr?i?ệ?u?', 'approx')
        ]
        
        for pattern, ptype in price_patterns:
            matches = re.findall(pattern, msg_lower)
            if matches:
                if ptype == 'max':
                    filters.max_price = float(matches[0].replace(',', '.')) * 1000000
                elif ptype == 'min':
                    filters.min_price = float(matches[0].replace(',', '.')) * 1000000
                elif ptype == 'range':
                    filters.min_price = float(matches[0][0].replace(',', '.')) * 1000000
                    filters.max_price = float(matches[0][1].replace(',', '.')) * 1000000
                elif ptype == 'approx':
                    price = float(matches[0].replace(',', '.')) * 1000000
                    filters.min_price = price * 0.8
                    filters.max_price = price * 1.2
                break
        
        # 2. Location filter
        common_locations = [
            'hà nội', 'hanoi', 'sapa', 'hạ long', 'halong', 'nha trang',
            'đà nẵng', 'danang', 'hội an', 'hoian', 'phú quốc', 'phuquoc',
            'cần thơ', 'cantho', 'miền bắc', 'miền nam', 'miền trung',
            'huế', 'quảng trị', 'bạch mã', 'trường sơn', 'đông hà'
        ]
        for loc in common_locations:
            if loc in msg_lower:
                filters.location = loc
                break
        
        # 3. Duration filter
        duration_patterns = [
            r'(\d+)\s*ngày',
            r'(\d+)\s*-\s*(\d+)\s*ngày',
            r'khoảng\s*(\d+)\s*ngày'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, msg_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    filters.duration_min = int(matches[0][0])
                    filters.duration_max = int(matches[0][1])
                else:
                    dur = int(matches[0])
                    filters.duration_min = dur
                    filters.duration_max = dur
                break
        
        # 4. NEW: Style filter (từ knowledge.json field "style")
        style_keywords = [
            'văn hóa', 'ẩm thực', 'nghỉ dưỡng', 'mạo hiểm', 'khám phá',
            'gia đình', 'cá nhân', 'nhóm', 'team building', 'sự kiện',
            'thiền', 'khí công', 'retreat', 'chữa lành'
        ]
        for style in style_keywords:
            if style in msg_lower:
                filters.style = style
                break
        
        # 5. Include keywords filter (tìm trong field "includes")
        include_keywords = ['ăn sáng', 'vé máy bay', 'khách sạn', 'hướng dẫn viên', 'bảo hiểm']
        found_includes = []
        for keyword in include_keywords:
            if keyword in msg_lower:
                found_includes.append(keyword)
        if found_includes:
            filters.include_keywords = found_includes
        
        # 6. Group type filter
        group_keywords = {
            'family': ['gia đình', 'trẻ em', 'con nít', 'bố mẹ'],
            'friends': ['nhóm bạn', 'bạn bè', 'bạn trẻ'],
            'corporate': ['công ty', 'team building', 'doanh nghiệp'],
            'solo': ['một mình', 'đi lẻ', 'solo'],
            'couple': ['cặp đôi', 'đôi lứa', 'người yêu']
        }
        
        for group_type, keywords in group_keywords.items():
            for keyword in keywords:
                if keyword in msg_lower:
                    filters.group_type = group_type
                    break
            if filters.group_type:
                break
        
        return filters
    
    @staticmethod
    def apply_filters(tours_db: Dict[int, Tour], filters: FilterSet) -> List[int]:
        """Áp dụng filter lên tours database (hỗ trợ knowledge.json fields)"""
        if filters.is_empty():
            return list(tours_db.keys())
        
        filtered_tours = []
        
        for tour_id, tour in tours_db.items():
            # 1. Price filter (sử dụng price_numeric đã parse)
            if filters.min_price is not None and tour.price_numeric is not None:
                if tour.price_numeric < filters.min_price:
                    continue
            
            if filters.max_price is not None and tour.price_numeric is not None:
                if tour.price_numeric > filters.max_price:
                    continue
            
            # 2. Location filter
            if filters.location:
                if filters.location.lower() not in tour.location.lower():
                    continue
            
            # 3. Duration filter (sử dụng duration_numeric đã parse)
            if filters.duration_min is not None and tour.duration_numeric is not None:
                if tour.duration_numeric < filters.duration_min:
                    continue
            
            if filters.duration_max is not None and tour.duration_numeric is not None:
                if tour.duration_numeric > filters.duration_max:
                    continue
            
            # 4. NEW: Style filter
            if filters.style and tour.style:
                if filters.style.lower() not in tour.style.lower():
                    continue
            
            # 5. NEW: Include keywords filter
            if filters.include_keywords:
                includes_lower = [inc.lower() for inc in tour.includes]
                found_all = all(
                    any(keyword in inc for inc in includes_lower)
                    for keyword in filters.include_keywords
                )
                if not found_all:
                    continue
            
            # 6. Category filter
            if filters.category and tour.category:
                if filters.category.lower() != tour.category.lower():
                    continue
            
            # 7. Group type filter
            if filters.group_type:
                if filters.group_type == 'family':
                    if not any(tag.startswith('style:') for tag in tour.tags):
                        continue
                elif filters.group_type == 'solo':
                    # Solo travelers might prefer certain styles
                    if tour.style and 'nhóm' in tour.style.lower():
                        continue
            
            filtered_tours.append(tour_id)
        
        return filtered_tours

class DeduplicationEngine:
    """
    Upgrade 2: Deduplication Engine - GIỮ NGUYÊN
    """
    
    SIMILARITY_THRESHOLD = 0.85
    MIN_TEXT_LENGTH = 20
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        if not text1 or not text2:
            return 0.0
        
        text1_norm = DeduplicationEngine._normalize_text(text1)
        text2_norm = DeduplicationEngine._normalize_text(text2)
        
        if len(text1_norm) < DeduplicationEngine.MIN_TEXT_LENGTH or len(text2_norm) < DeduplicationEngine.MIN_TEXT_LENGTH:
            return 0.0
        
        seq_ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            jaccard = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard = len(intersection) / len(union)
        
        prefix_len = min(50, min(len(text1_norm), len(text2_norm)))
        prefix1 = text1_norm[:prefix_len]
        prefix2 = text2_norm[:prefix_len]
        prefix_sim = SequenceMatcher(None, prefix1, prefix2).ratio()
        
        similarity = (seq_ratio * 0.5) + (jaccard * 0.3) + (prefix_sim * 0.2)
        
        return similarity
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        stopwords = {'và', 'của', 'cho', 'với', 'tại', 'ở', 'này', 'đó', 'kia', 'về', 'trong'}
        words = [word for word in text.split() if word not in stopwords]
        
        return ' '.join(words)
    
    @staticmethod
    def deduplicate_passages(passages: List[Tuple[float, Dict]], 
                            similarity_threshold: float = None) -> List[Tuple[float, Dict]]:
        """
        Remove duplicate passages from results
        """
        if len(passages) <= 1:
            return passages
        
        threshold = similarity_threshold or DeduplicationEngine.SIMILARITY_THRESHOLD
        unique_passages = []
        seen_passages = []
        
        sorted_passages = sorted(passages, key=lambda x: x[0], reverse=True)
        
        for score, passage in sorted_passages:
            text = passage.get('text', '').strip()
            path = passage.get('path', '')
            
            if not text or len(text) < DeduplicationEngine.MIN_TEXT_LENGTH:
                unique_passages.append((score, passage))
                continue
            
            is_duplicate = False
            for seen_text, seen_path in seen_passages:
                tour_match1 = re.search(r'tours\[(\d+)\]', path)
                tour_match2 = re.search(r'tours\[(\d+)\]', seen_path)
                
                if tour_match1 and tour_match2:
                    if tour_match1.group(1) == tour_match2.group(1):
                        field1 = path.split('.')[-1] if '.' in path else ''
                        field2 = seen_path.split('.')[-1] if '.' in seen_path else ''
                        if field1 == field2:
                            is_duplicate = True
                            break
                
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
    def merge_similar_tours(tour_indices: List[int], tours_db: Dict[int, Tour]) -> List[int]:
        """Merge tours that are essentially the same"""
        if len(tour_indices) <= 1:
            return tour_indices
        
        tour_groups = []
        processed = set()
        
        for i, idx1 in enumerate(tour_indices):
            if idx1 in processed:
                continue
            
            group = [idx1]
            tour1 = tours_db.get(idx1)
            name1 = (tour1.tour_name if tour1 else "").strip()
            
            if not name1:
                processed.add(idx1)
                tour_groups.append(group)
                continue
            
            for j, idx2 in enumerate(tour_indices[i+1:], i+1):
                if idx2 in processed:
                    continue
                
                tour2 = tours_db.get(idx2)
                name2 = (tour2.tour_name if tour2 else "").strip()
                
                if not name2:
                    continue
                
                similarity = DeduplicationEngine.calculate_similarity(name1, name2)
                if similarity > 0.9:
                    group.append(idx2)
                    processed.add(idx2)
            
            processed.add(idx1)
            tour_groups.append(group)
        
        best_tours = []
        for group in tour_groups:
            if not group:
                continue
            
            if len(group) == 1:
                best_tours.append(group[0])
                continue
            
            best_score = -1
            best_idx = group[0]
            
            for idx in group:
                tour = tours_db.get(idx)
                if not tour:
                    continue
                
                score = 0
                
                if tour.tour_name:
                    score += 2
                if tour.duration:
                    score += 2
                if tour.location:
                    score += 2
                if tour.price:
                    score += 3
                if tour.includes:
                    score += 2
                if tour.summary:
                    score += 1
                
                for field in [tour.includes, tour.summary, tour.notes]:
                    if isinstance(field, str) and len(field) > 50:
                        score += 1
                    elif isinstance(field, list) and field:
                        score += len(field)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            best_tours.append(best_idx)
        
        logger.info(f"🔄 Tour merging: {len(tour_indices)} → {len(best_tours)} unique tours")
        return best_tours

class EnhancedFieldDetectorV2:
    """
    Upgrade 3: Enhanced Field Detector
    CẬP NHẬT: Detect các field mới từ knowledge.json
    """
    
    @staticmethod
    def detect_field_with_confidence(message: str) -> Tuple[Optional[str], float, Dict]:
        """Phát hiện field được hỏi với knowledge.json structure"""
        msg_lower = message.lower()
        
        # Field mapping: từ khóa -> field trong knowledge.json
        field_mappings = {
            'tour_name': {
                'keywords': ['tên tour', 'tour nào', 'tour gì', 'tour tên là'],
                'weight': 1.0
            },
            'price': {
                'keywords': ['giá', 'giá cả', 'chi phí', 'bao nhiêu tiền', 'giá tour'],
                'weight': 1.0
            },
            'duration': {
                'keywords': ['thời gian', 'bao lâu', 'mấy ngày', 'kéo dài', 'duration'],
                'weight': 0.9
            },
            'location': {
                'keywords': ['địa điểm', 'ở đâu', 'nơi nào', 'điểm đến', 'location'],
                'weight': 0.9
            },
            'includes': {
                'keywords': ['bao gồm', 'có gì', 'dịch vụ', 'tiện ích', 'included'],
                'weight': 0.8
            },
            'style': {
                'keywords': ['phong cách', 'loại hình', 'dạng tour', 'kiểu tour', 'style'],
                'weight': 0.7
            },
            'transport': {
                'keywords': ['phương tiện', 'di chuyển', 'xe cộ', 'vận chuyển', 'transport'],
                'weight': 0.6
            },
            'accommodation': {
                'keywords': ['chỗ ở', 'khách sạn', 'nơi ở', 'lưu trú', 'accommodation'],
                'weight': 0.6
            },
            'meals': {
                'keywords': ['ăn uống', 'bữa ăn', 'ẩm thực', 'đồ ăn', 'meals'],
                'weight': 0.6
            },
            'event_support': {
                'keywords': ['hỗ trợ sự kiện', 'tổ chức event', 'sự kiện', 'event support'],
                'weight': 0.5
            },
            'summary': {
                'keywords': ['tóm tắt', 'mô tả', 'giới thiệu', 'summary', 'overview'],
                'weight': 0.7
            },
            'notes': {
                'keywords': ['lưu ý', 'chú ý', 'cần biết', 'notes', 'ghi chú'],
                'weight': 0.5
            }
        }
        
        field_scores = {}
        for field, config in field_mappings.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in msg_lower:
                    score += 1
            
            if score > 0:
                # Tính confidence dựa trên số keyword match và weight
                base_confidence = min(0.3 + (score * 0.15), 0.9)
                weighted_confidence = base_confidence * config['weight']
                field_scores[field] = weighted_confidence
        
        if not field_scores:
            return None, 0.3, {}
        
        # Tìm field có confidence cao nhất
        best_field = max(field_scores.items(), key=lambda x: x[1])
        return best_field[0], best_field[1], field_scores
    
    @staticmethod
    def get_field_value(tour: Tour, field_name: str) -> Any:
        """Lấy giá trị field từ Tour object"""
        if field_name == 'tour_name':
            return tour.tour_name
        elif field_name == 'price':
            return tour.price
        elif field_name == 'duration':
            return tour.duration
        elif field_name == 'location':
            return tour.location
        elif field_name == 'includes':
            return tour.includes
        elif field_name == 'style':
            return tour.style
        elif field_name == 'transport':
            return tour.transport
        elif field_name == 'accommodation':
            return tour.accommodation
        elif field_name == 'meals':
            return tour.meals
        elif field_name == 'event_support':
            return tour.event_support
        elif field_name == 'summary':
            return tour.summary
        elif field_name == 'notes':
            return tour.notes
        else:
            return None

class KnowledgeAwareQuestionPipeline:
    """
    Upgrade 4: Question Pipeline
    CẬP NHẬT: Hiểu các câu hỏi liên quan đến knowledge.json fields
    """
    
    @staticmethod
    def classify_question(message: str) -> Tuple[QuestionType, float, Dict]:
        """Phân loại câu hỏi với knowledge.json context"""
        msg_lower = message.lower()
        
        # Kiểm tra greeting
        greetings = ['xin chào', 'hello', 'hi', 'chào bạn', 'chào']
        if any(g in msg_lower for g in greetings):
            return QuestionType.GREETING, 0.95, {'greeting_type': 'standard'}
        
        # Kiểm tra farewell
        farewells = ['tạm biệt', 'bye', 'cảm ơn', 'thanks', 'kết thúc']
        if any(f in msg_lower for f in farewells):
            return QuestionType.FAREWELL, 0.95, {'farewell_type': 'standard'}
        
        # Câu hỏi liệt kê tour
        list_keywords = ['danh sách', 'liệt kê', 'có những tour nào', 'tour nào có', 'các tour']
        list_count = sum(1 for kw in list_keywords if kw in msg_lower)
        if list_count > 0:
            confidence = min(0.7 + (list_count * 0.1), 0.95)
            return QuestionType.LIST_TOURS, confidence, {'list_type': 'general'}
        
        # Câu hỏi chi tiết tour
        detail_keywords = ['chi tiết', 'thông tin', 'giới thiệu', 'mô tả', 'tour này']
        detail_count = sum(1 for kw in detail_keywords if kw in msg_lower)
        if detail_count > 0:
            confidence = min(0.65 + (detail_count * 0.1), 0.9)
            return QuestionType.TOUR_DETAIL, confidence, {'detail_type': 'general'}
        
        # Câu hỏi so sánh
        compare_keywords = ['so sánh', 'khác nhau', 'nên chọn', 'cái nào tốt', 'cái nào hay']
        compare_count = sum(1 for kw in compare_keywords if kw in msg_lower)
        if compare_count > 0:
            confidence = min(0.6 + (compare_count * 0.15), 0.9)
            return QuestionType.COMPARISON, confidence, {'compare_type': 'general'}
        
        # Câu hỏi đề xuất
        recommend_keywords = ['đề xuất', 'gợi ý', 'nên đi', 'phù hợp', 'tư vấn']
        recommend_count = sum(1 for kw in recommend_keywords if kw in msg_lower)
        if recommend_count > 0:
            confidence = min(0.7 + (recommend_count * 0.1), 0.95)
            return QuestionType.RECOMMENDATION, confidence, {'recommend_type': 'general'}
        
        # Câu hỏi về field cụ thể trong knowledge.json
        field_detector = EnhancedFieldDetectorV2()
        field_name, field_confidence, _ = field_detector.detect_field_with_confidence(message)
        if field_confidence > 0.6:
            return QuestionType.GENERAL_INFO, field_confidence, {'field_name': field_name}
        
        return QuestionType.UNKNOWN, 0.5, {'reason': 'no_keywords_matched'}

class ComplexQueryProcessor:
    """Upgrade 5: Complex Query Processor - GIỮ NGUYÊN"""
    
    @staticmethod
    def split_query(query: str) -> List[Dict[str, Any]]:
        """Split complex query into sub-queries"""
        sub_queries = []
        
        # Simple implementation - can be enhanced
        if ' và ' in query or ',' in query:
            parts = re.split(r' và |,', query)
            for part in parts:
                if part.strip():
                    sub_queries.append({
                        'query': part.strip(),
                        'priority': 0.8,
                        'filters': {},
                        'focus': 'general'
                    })
        else:
            sub_queries.append({
                'query': query,
                'priority': 1.0,
                'filters': {},
                'focus': 'general'
            })
        
        return sub_queries

class FuzzyMatcher:
    """Upgrade 6: Fuzzy Matcher - GIỮ NGUYÊN"""
    
    def __init__(self, tours_db: Dict[int, Tour]):
        self.tours_db = tours_db
    
    def find_similar_tours(self, query: str, tour_names: Dict[str, int]) -> List[Tuple[int, float]]:
        """Find tours with similar names"""
        matches = []
        query_norm = self.normalize_text(query)
        
        for name, idx in tour_names.items():
            name_norm = self.normalize_text(name)
            similarity = SequenceMatcher(None, query_norm, name_norm).ratio()
            
            if similarity > 0.6:
                matches.append((idx, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]
    
    def find_tour_by_partial_name(self, partial_name: str) -> List[int]:
        """Find tours by partial name match"""
        partial_norm = self.normalize_text(partial_name)
        matches = []
        
        for idx, tour in self.tours_db.items():
            tour_name_norm = self.normalize_text(tour.tour_name)
            if partial_norm in tour_name_norm:
                matches.append(idx)
        
        return matches
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fuzzy matching"""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class ConversationStateMachine:
    """Upgrade 7: Conversation State Machine - GIỮ NGUYÊN"""
    
    def __init__(self, initial_state: ConversationState = ConversationState.INITIAL):
        self.current_state = initial_state
        self.state_history = []
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update state based on interaction"""
        # Simple state transitions based on message content
        msg_lower = user_message.lower()
        
        if 'so sánh' in msg_lower:
            self.current_state = ConversationState.COMPARISON
        elif 'chi tiết' in msg_lower or 'thông tin' in msg_lower:
            self.current_state = ConversationState.DETAIL_VIEW
        elif 'đề xuất' in msg_lower or 'gợi ý' in msg_lower:
            self.current_state = ConversationState.RECOMMENDING
        elif 'tạm biệt' in msg_lower or 'bye' in msg_lower:
            self.current_state = ConversationState.CLOSING
        
        self.state_history.append({
            'timestamp': time.time(),
            'state': self.current_state.value,
            'message': user_message[:100]
        })
    
    def extract_reference(self, message: str) -> List[int]:
        """Extract tour reference from message"""
        # Simple implementation - look for tour names
        msg_lower = message.lower()
        references = []
        
        for idx, tour in tours_db.items():
            tour_name_lower = tour.tour_name.lower()
            if tour_name_lower in msg_lower:
                references.append(idx)
        
        return references

class SemanticAnalyzer:
    """Upgrade 8: Semantic Analyzer - GIỮ NGUYÊN"""
    
    @staticmethod
    def analyze_user_profile(message: str, current_context: ConversationContext) -> Dict:
        """Analyze user profile from message"""
        profile = {
            'interests': [],
            'budget': None,
            'group_type': None,
            'preferred_duration': None
        }
        
        msg_lower = message.lower()
        
        # Detect interests
        interest_keywords = {
            'history': ['lịch sử', 'chiến tranh', 'di tích'],
            'nature': ['thiên nhiên', 'rừng', 'núi'],
            'wellness': ['thiền', 'khí công', 'chữa lành'],
            'culture': ['văn hóa', 'ẩm thực', 'truyền thống']
        }
        
        for interest, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in msg_lower:
                    profile['interests'].append(interest)
                    break
        
        return profile
    
    @staticmethod
    def match_tours_to_profile(profile: Dict, tours_db: Dict[int, Tour]) -> List[Tuple]:
        """Match tours to user profile"""
        matches = []
        
        for idx, tour in tours_db.items():
            score = 0
            
            # Match interests with tour style
            if profile.get('interests') and tour.style:
                tour_style_lower = tour.style.lower()
                for interest in profile['interests']:
                    interest_keywords = {
                        'history': ['lịch sử', 'chiến tranh'],
                        'nature': ['thiên nhiên', 'rừng'],
                        'wellness': ['thiền', 'khí công'],
                        'culture': ['văn hóa', 'ẩm thực']
                    }
                    
                    if any(keyword in tour_style_lower for keyword in interest_keywords.get(interest, [])):
                        score += 1
            
            # Match budget
            if profile.get('budget') and tour.price_numeric:
                if tour.price_numeric <= profile['budget']:
                    score += 1
            
            if score > 0:
                matches.append((idx, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

class AutoValidator:
    """Upgrade 9: Auto Validator - GIỮ NGUYÊN"""
    
    @staticmethod
    def validate_response(response: str) -> str:
        """Validate and correct response"""
        # Simple validation - ensure hotline is included
        if '0332510486' not in response:
            response += "\n\n📞 Liên hệ hotline 0332510486 để được tư vấn chi tiết!"
        
        return response
    
    @staticmethod
    def safe_validate(reply: dict) -> dict:
        """Safe validation wrapper"""
        try:
            if not isinstance(reply, dict):
                return reply
            
            if 'reply' in reply:
                reply['reply'] = AutoValidator.validate_response(reply['reply'])
            
            return reply
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return reply

class KnowledgeTemplateSystem:
    """
    Upgrade 10: Template System
    CẬP NHẬT: Templates cho knowledge.json fields
    """
    
    TEMPLATES = {
        # General templates
        'greeting': """Xin chào! 👋 Tôi là trợ lý du lịch của Ruby Wings.

Tôi có thể giúp bạn:
• Tìm kiếm tour theo yêu cầu
• Cung cấp thông tin chi tiết về tour
• So sánh các tour với nhau
• Đề xuất tour phù hợp với nhu cầu

Bạn đang tìm kiếm tour như thế nào?""",
        
        'farewell': """Cảm ơn bạn đã sử dụng dịch vụ của Ruby Wings! 🌟

Nếu bạn cần thêm thông tin về bất kỳ tour nào, đừng ngần ngại quay lại.

Chúc bạn có một chuyến đi tuyệt vời! ✈️""",
        
        # Tour list template
        'tour_list': """🎯 **Tôi tìm thấy {count} tour phù hợp với yêu cầu của bạn:**

{tour_items}

💡 **Gợi ý:**
• Gõ số thứ tự để xem chi tiết tour
• Hoặc hỏi thêm về tiêu chí cụ thể (giá, thời gian, địa điểm)""",
        
        'tour_item': """{idx}. **{tour_name}**
   📍 {location} | ⏱ {duration} | 💰 {price}
   🎯 {summary}""",
        
        # Tour detail template với knowledge.json fields
        'tour_detail_full': """🌟 **{tour_name}**

📋 **Tóm tắt:** {summary}
📍 **Địa điểm:** {location}
⏱ **Thời gian:** {duration}
💰 **Giá:** {price}
🎨 **Phong cách:** {style}

🚌 **Phương tiện di chuyển:** {transport}
🏨 **Chỗ ở:** {accommodation}
🍽 **Ăn uống:** {meals}

✅ **Dịch vụ bao gồm:**
{includes_formatted}

📝 **Lưu ý quan trọng:** {notes}

🎪 **Hỗ trợ sự kiện:** {event_support}

💎 **Loại tour:** {category} | ⭐ **Đánh giá:** {rating}/5""",
        
        # Field-specific templates
        'field_price': """💰 **Giá tour {tour_name}:**
{price}

💡 *Giá đã bao gồm thuế và phí dịch vụ*""",
        
        'field_includes': """✅ **Tour {tour_name} bao gồm:**

{includes_formatted}

💡 *Tất cả dịch vụ đã được kiểm duyệt và đảm bảo chất lượng*""",
        
        'field_duration': """⏱ **Thời gian tour {tour_name}:**
{duration}

📅 *Lịch trình chi tiết có thể điều chỉnh theo yêu cầu*""",
        
        'field_location': """📍 **Địa điểm tour {tour_name}:**
{location}

🗺️ *Bản đồ và hướng dẫn di chuyển sẽ được cung cấp đầy đủ*""",
        
        # Comparison template
        'comparison': """🔄 **So sánh {count} tour:**

{comparison_table}

📊 **Tóm tắt:**
{summary}

💡 **Gợi ý:** {suggestion}""",
        
        # Recommendation template
        'recommendation': """🎯 **Đề xuất phù hợp với bạn:**

{recommended_tour}

📈 **Lý do đề xuất:**
{reasons}

🤔 **Tour khác có thể xem xét:**
{alternatives}""",
        
        # Error/fallback templates
        'no_results': """😕 **Không tìm thấy tour phù hợp**

Tôi không tìm thấy tour nào đáp ứng yêu cầu của bạn. Bạn có thể:

1. **Mở rộng tiêu chí tìm kiếm**
2. **Thay đổi ngân sách hoặc thời gian**
3. **Xem danh sách tất cả tour có sẵn**

Bạn muốn thử cách nào?""",
        
        'general_fallback': """🤔 **Tôi hiểu bạn đang hỏi về:**
_{user_message}_

Hiện tôi có thể giúp bạn với:
• Thông tin về {available_fields}
• So sánh các tour
• Đề xuất tour phù hợp

Bạn muốn tìm hiểu cụ thể về điều gì?"""
    }
    
    @classmethod
    def render(cls, template_name: str, **kwargs) -> str:
        """Render template với data"""
        template = cls.TEMPLATES.get(template_name)
        if not template:
            return f"Template '{template_name}' not found"
        
        try:
            # Xử lý đặc biệt cho includes (chuyển list -> string)
            if 'includes' in kwargs and isinstance(kwargs['includes'], list):
                includes_items = [f"• {item}" for item in kwargs['includes']]
                kwargs['includes_formatted'] = "\n".join(includes_items)
            
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Template rendering error: {e}")
            return template

# ==================== PHẦN 5: SUPPORT FUNCTIONS ====================

class CacheSystem:
    """Cache System"""
    
    def __init__(self):
        self.cache = {}
    
    def get_cache_key(self, query: str, context_hash: str = "") -> str:
        """Generate cache key"""
        key_parts = [query]
        if context_hash:
            key_parts.append(context_hash)
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                logger.debug(f"💾 Cache hit for key: {key[:20]}...")
                return entry.value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, expiry: int = None):
        """Set item in cache"""
        ttl = expiry or CACHE_TTL
        cache_entry = CacheEntry(
            value=value,
            expiry=time.time() + ttl
        )
        self.cache[key] = cache_entry
        
        # Clean up expired entries occasionally
        if len(self.cache) > 100:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up expired cache entries"""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

def get_session_context(session_id: str) -> ConversationContext:
    """Lấy context từ session"""
    with SESSION_LOCK:
        if session_id in sessions:
            context = sessions[session_id]
            # Check if session has expired
            if time.time() - context.last_activity > SESSION_TIMEOUT:
                logger.info(f"Session {session_id} expired, creating new")
                context = ConversationContext(session_id=session_id)
                sessions[session_id] = context
            return context
        else:
            context = ConversationContext(session_id=session_id)
            sessions[session_id] = context
            return context

def save_session_context(session_id: str, context: ConversationContext):
    """Lưu context vào session"""
    with SESSION_LOCK:
        sessions[session_id] = context

def extract_session_id(request_data: Dict, remote_addr: str) -> str:
    """Extract session ID"""
    session_id = request_data.get("session_id")
    if not session_id:
        ip = remote_addr or "0.0.0.0"
        current_hour = datetime.now().strftime("%Y%m%d%H")
        unique_str = f"{ip}_{current_hour}"
        session_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    return f"session_{session_id}"

def llm_request(request_data: LLMRequest) -> str:
    """Gửi request đến LLM"""
    try:
        import requests
        response = requests.post(
            LLM_URL,
            json={
                "model": request_data.model,
                "prompt": request_data.prompt,
                "stream": request_data.stream,
                "temperature": request_data.temperature,
                "max_tokens": request_data.max_tokens
            },
            timeout=30
        )
        
        if response.status_code == 200:
            if request_data.stream:
                return response.text
            else:
                data = response.json()
                return data.get("response", "")
        else:
            logger.error(f"LLM request failed: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"LLM request error: {e}")
        return ""

def parse_llm_response(llm_response: str) -> Dict:
    """Parse LLM response"""
    # Simple parsing - can be enhanced
    return {"reply": llm_response}

@lru_cache(maxsize=100)
def embed_text(text: str) -> Tuple[List[float], int]:
    """Tạo embedding cho text"""
    # Fallback embedding - can be replaced with actual embedding model
    if not text:
        return [], 0
    
    # Simple hash-based embedding for fallback
    h = hash(text) % (10 ** 12)
    dim = 1536  # OpenAI embedding dimension
    embedding = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 
                 for i in range(dim)]
    
    return embedding, dim

class NumpyIndex:
    """Simple numpy-based index"""
    
    def __init__(self, mat=None):
        if mat is None:
            self.mat = np.empty((0, 0), dtype="float32")
        else:
            self.mat = np.asarray(mat, dtype="float32")
            if self.mat.ndim == 1:
                self.mat = self.mat.reshape(1, -1)
        
        if self.mat.shape[0] > 0 and self.mat.ndim == 2:
            self.dim = int(self.mat.shape[1])
        else:
            self.dim = 0
        self.size = int(self.mat.shape[0])
    
    def is_empty(self):
        return self.mat.shape[0] == 0
    
    def search(self, query_vec, k=5):
        if self.is_empty():
            return [], []
        
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        # Normalize for cosine similarity
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        mat_norm = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        
        sims = np.dot(mat_norm, q_norm.T).reshape(-1)
        topk = np.argsort(-sims)[:k]
        
        return sims[topk].tolist(), topk.tolist()
    
    def save(self, path):
        np.savez_compressed(path, mat=self.mat)
    
    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            return cls(arr['mat'])
        except Exception as e:
            logger.error(f"Failed to load numpy index: {e}")
            return cls()

def query_index(query: str, top_k: int = 5, min_score: float = SEMANTIC_MIN_SCORE) -> List[Tuple[float, Dict]]:
    """Semantic search"""
    if search_index is None:
        return []
    
    try:
        embedding, _ = embed_text(query)
        if not embedding:
            return []
        
        scores, indices = search_index.search(embedding, k=top_k)
        
        results = []
        for score, idx in zip(scores, indices):
            if score < min_score:
                continue
            
            if idx < len(MAPPING):
                passage = MAPPING[idx]
                results.append((float(score), passage))
        
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def build_index(force_rebuild: bool = False) -> bool:
    """Build search index"""
    global search_index
    
    try:
        if not FLAT_TEXTS:
            logger.warning("No texts to index")
            return False
        
        logger.info(f"🔨 Building index for {len(FLAT_TEXTS)} passages...")
        
        # Generate embeddings
        vectors = []
        for text in FLAT_TEXTS:
            emb, _ = embed_text(text)
            if emb:
                vectors.append(np.array(emb, dtype="float32"))
        
        if not vectors:
            logger.error("No embeddings generated")
            return False
        
        # Create index
        mat = np.vstack(vectors)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        
        if HAS_FAISS:
            import faiss
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            search_index = index
            logger.info("✅ Built FAISS index")
        else:
            search_index = NumpyIndex(mat)
            logger.info("✅ Built numpy index")
        
        return True
    except Exception as e:
        logger.error(f"Index building error: {e}")
        return False

def send_capi_event(session_id: str, user_message: str, bot_response: str):
    """Gửi event đến CAPI"""
    if not CAPI_ENABLED:
        return
    
    try:
        # Implement CAPI sending logic here
        pass
    except Exception as e:
        logger.error(f"CAPI error: {e}")

def generate_session_id() -> str:
    """Generate session ID"""
    import uuid
    return f"session_{uuid.uuid4().hex[:12]}"

def cleanup_expired_sessions():
    """Dọn dẹp session hết hạn"""
    with SESSION_LOCK:
        expired_keys = []
        current_time = time.time()
        
        for session_id, context in sessions.items():
            if current_time - context.last_activity > SESSION_TIMEOUT:
                expired_keys.append(session_id)
        
        for key in expired_keys:
            del sessions[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired sessions")

def _prepare_llm_prompt(user_message: str, search_results: List, context: Dict) -> str:
    """Chuẩn bị prompt cho LLM"""
    prompt_parts = [
        "Bạn là trợ lý tư vấn du lịch Ruby Wings - CHUYÊN NGHIỆP, THÔNG MINH, NHIỆT TÌNH.",
        "",
        "⚠️ QUY TẮC NGHIÊM NGẶT:",
        "1. LUÔN trả lời bằng tiếng Việt",
        "2. Giữ thái độ nhiệt tình, thân thiện",
        "3. KHÔNG bịa thông tin nếu không biết",
        "4. LUÔN đề cập hotline 0332510486 khi kết thúc",
        "",
        "📚 THÔNG TIN NGỮ CẢNH:",
    ]
    
    # Add context info
    if context.get('user_preferences'):
        prefs = context['user_preferences']
        prompt_parts.append(f"- Sở thích người dùng: {prefs}")
    
    if context.get('current_tours'):
        tours_info = context['current_tours']
        prompt_parts.append(f"- Tour đang nói đến: {tours_info}")
    
    # Add search results
    prompt_parts.append("")
    prompt_parts.append("📝 DỮ LIỆU TÌM THẤY:")
    
    if search_results:
        for i, (score, passage) in enumerate(search_results[:5], 1):
            text = passage.get('text', '')[:200]
            prompt_parts.append(f"[{i}] {text}")
    else:
        prompt_parts.append("(Không có dữ liệu cụ thể)")
    
    # Add user message
    prompt_parts.append("")
    prompt_parts.append("💬 CÂU HỎI CỦA KHÁCH:")
    prompt_parts.append(user_message)
    
    # Add instructions
    prompt_parts.append("")
    prompt_parts.append("🎯 HÃY TRẢ LỜI:")
    prompt_parts.append("1. Dựa trên dữ liệu có sẵn")
    prompt_parts.append("2. Ngắn gọn, rõ ràng")
    prompt_parts.append("3. Kết thúc bằng hotline 0332510486")
    
    return "\n".join(prompt_parts)

def _generate_fallback_response(user_message: str, search_results: List, tour_indices: List[int] = None) -> str:
    """Generate fallback response"""
    # Use template system for fallback
    if tour_indices and tours_db:
        tour_list = []
        for idx in tour_indices[:3]:
            tour = tours_db.get(idx)
            if tour:
                tour_list.append(tour)
        
        if tour_list:
            tour_items = []
            for i, tour in enumerate(tour_list, 1):
                tour_items.append(
                    KnowledgeTemplateSystem.TEMPLATES['tour_item'].format(
                        idx=i,
                        tour_name=tour.tour_name,
                        location=tour.location,
                        duration=tour.duration,
                        price=tour.price,
                        summary=tour.summary[:100] + "..."
                    )
                )
            
            return KnowledgeTemplateSystem.TEMPLATES['tour_list'].format(
                count=len(tour_list),
                tour_items="\n\n".join(tour_items)
            )
    
    return KnowledgeTemplateSystem.TEMPLATES['general_fallback'].format(
        user_message=user_message,
        available_fields="giá cả, thời gian, địa điểm, dịch vụ bao gồm"
    )

# ==================== PHẦN 6: GLOBAL VARIABLES & INITIALIZATION ====================

# Global variables
tours_db: Dict[int, Tour] = {}
tour_name_index: Dict[str, int] = {}
search_index = None
sessions: Dict[str, ConversationContext] = {}
SESSION_LOCK = threading.Lock()
cache_system = CacheSystem()

# Knowledge base state
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[Dict] = []

def initialize_system():
    """Khởi tạo hệ thống - CẬP NHẬT với knowledge.json"""
    global tours_db, tour_name_index, KNOW, FLAT_TEXTS, MAPPING
    
    # Load tours từ knowledge.json
    tours_db = KnowledgeLoader.build_tours_database()
    
    # Build tour name index
    tour_name_index = {}
    for tid, tour in tours_db.items():
        normalized_name = tour.tour_name.lower().strip()
        if normalized_name:
            tour_name_index[normalized_name] = tid
    
    # Load knowledge for indexing
    KNOW = KnowledgeLoader.load_knowledge_file()
    
    # Flatten knowledge for indexing
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
    
    # Build search index
    build_index(force_rebuild=False)
    
    logger.info(f"✅ System initialized with {len(tours_db)} tours, {len(FLAT_TEXTS)} passages")

# Gọi khởi tạo
initialize_system()

# ==================== PHẦN 7: FLASK APP ====================

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chatbot": "running",
            "tours_db": len(tours_db),
            "knowledge_base": len(KNOW.get('tours', [])),
            "sessions": len(sessions)
        }
    })

@app.route('/api/tours', methods=['GET'])
def get_tours():
    """Get all tours"""
    try:
        tours_list = []
        for idx, tour in tours_db.items():
            tours_list.append({
                "id": idx,
                "tour_name": tour.tour_name,
                "summary": tour.summary,
                "location": tour.location,
                "duration": tour.duration,
                "price": tour.price,
                "category": tour.category,
                "style": tour.style
            })
        
        return jsonify({
            "success": True,
            "count": len(tours_list),
            "tours": tours_list
        })
    except Exception as e:
        logger.error(f"Error getting tours: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tours/<int:tour_id>', methods=['GET'])
def get_tour_detail(tour_id: int):
    """Get tour details by ID"""
    try:
        tour = tours_db.get(tour_id)
        if not tour:
            return jsonify({"error": "Tour not found"}), 404
        
        return jsonify({
            "success": True,
            "tour": asdict(tour)
        })
    except Exception as e:
        logger.error(f"Error getting tour {tour_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_tours():
    """Search tours by query"""
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Apply filters if provided
        filters_data = data.get('filters', {})
        filters = FilterSet(
            min_price=filters_data.get('min_price'),
            max_price=filters_data.get('max_price'),
            location=filters_data.get('location'),
            duration_min=filters_data.get('duration_min'),
            duration_max=filters_data.get('duration_max'),
            style=filters_data.get('style'),
            category=filters_data.get('category')
        )
        
        # First apply mandatory filters
        filtered_indices = MandatoryFilterSystemV2.apply_filters(tours_db, filters)
        
        # Then apply fuzzy matching if needed
        if query and filtered_indices:
            fuzzy_matcher = FuzzyMatcher(tours_db)
            filtered_tours_db = {idx: tours_db[idx] for idx in filtered_indices}
            
            # Create tour name index for filtered tours
            filtered_tour_names = {}
            for idx in filtered_indices:
                tour = tours_db[idx]
                normalized_name = tour.tour_name.lower().strip()
                if normalized_name:
                    filtered_tour_names[normalized_name] = idx
            
            # Find similar tours
            fuzzy_matches = fuzzy_matcher.find_similar_tours(query, filtered_tour_names)
            if fuzzy_matches:
                filtered_indices = [idx for idx, _ in fuzzy_matches]
        
        # Prepare results
        results = []
        for idx in filtered_indices[:MAX_TOURS_RETURN]:
            tour = tours_db[idx]
            results.append({
                "id": idx,
                "tour_name": tour.tour_name,
                "summary": tour.summary,
                "location": tour.location,
                "duration": tour.duration,
                "price": tour.price,
                "category": tour.category,
                "style": tour.style,
                "rating": tour.rating
            })
        
        return jsonify({
            "success": True,
            "count": len(results),
            "tours": results,
            "total_matches": len(filtered_indices)
        })
    except Exception as e:
        logger.error(f"Search error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==================== PHẦN 8: CHAT ENDPOINT (ĐỂ CHÈN THỦ CÔNG) ====================

# ==================== PHẦN 8: CHAT ENDPOINT - KNOWLEDGE.JSON INTEGRATION ====================

@app.route("/chat", methods=["POST"])
def chat_endpoint_knowledge():
    """
    Main chat endpoint với full knowledge.json integration
    Version: Knowledge-Aware Chatbot V1.0
    """
    start_time = time.time()
    
    try:
        # ================== 1. NHẬN REQUEST & PARSE ==================
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        session_id = extract_session_id(data, request.remote_addr)
        
        if not user_message:
            return jsonify(asdict(ChatResponse(
                reply=KnowledgeTemplateSystem.render('greeting'),
                tour_indices=[],
                action="continue",
                context={"session_id": session_id},
                metadata={"processing_time_ms": 0}
            )))
        
        logger.info(f"📩 Received message from {session_id}: {user_message[:100]}...")
        
        # ================== 2. KHỞI TẠO & LOAD DỮ LIỆU ==================
        # 2.1 Lấy context từ session
        context = get_session_context(session_id)
        context.last_activity = time.time()
        
        # 2.2 Thêm user message vào history
        context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # 2.3 Check cache
        cache_key = None
        if UpgradeFlags.is_enabled("CACHE_SYSTEM"):
            context_hash = hashlib.md5(json.dumps({
                'last_tours': context.last_tours_mentioned[-3:] if context.last_tours_mentioned else [],
                'state': context.current_state.value,
                'filters': context.active_filters.to_dict() if context.active_filters else {}
            }, sort_keys=True).encode()).hexdigest()
            
            cache_key = cache_system.get_cache_key(user_message, context_hash)
            cached_response = cache_system.get(cache_key)
            
            if cached_response:
                logger.info(f"💾 Cache hit for key: {cache_key[:50]}...")
                cached_response['metadata']['from_cache'] = True
                cached_response['metadata']['processing_time_ms'] = int((time.time() - start_time) * 1000)
                return jsonify(cached_response)
        
        # ================== 3. PHÂN TÍCH CÂU HỎI VỚI KNOWLEDGE.JSON ==================
        # 3.1 Phân loại câu hỏi với knowledge context
        question_type, confidence, type_details = KnowledgeAwareQuestionPipeline.classify_question(user_message)
        context.last_question_type = question_type
        
        logger.info(f"🎯 Question type: {question_type.value} (confidence: {confidence:.2f})")
        
        # 3.2 Trích xuất filter với knowledge.json fields
        mandatory_filters = MandatoryFilterSystemV2.extract_filters(user_message)
        if not mandatory_filters.is_empty():
            context.active_filters = mandatory_filters
            logger.info(f"🔍 Filters extracted: {mandatory_filters}")
        
        # 3.3 Phát hiện field được hỏi từ knowledge.json
        field_name, field_confidence, field_scores = EnhancedFieldDetectorV2.detect_field_with_confidence(user_message)
        if field_name:
            logger.info(f"📊 Field detected: {field_name} (confidence: {field_confidence:.2f})")
        
        # 3.4 Phân tích semantic user profile
        user_profile = {}
        if UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            user_profile = SemanticAnalyzer.analyze_user_profile(user_message, context)
            if user_profile:
                context.user_preferences.update(user_profile)
                logger.info(f"👤 User profile updated: {user_profile}")
        
        # ================== 4. TÌM KIẾM TOUR TỪ KNOWLEDGE.JSON ==================
        tour_indices = []
        resolved_tours = []
        
        # 4.1 DIRECT TOUR NAME MATCHING với knowledge.json
        message_lower = user_message.lower()
        for tour_id, tour in tours_db.items():
            # Kiểm tra tên tour
            if tour.tour_name and tour.tour_name.lower() in message_lower:
                if tour_id not in tour_indices:
                    tour_indices.append(tour_id)
                    resolved_tours.append(tour)
            
            # Kiểm tra trong các field khác của knowledge.json
            search_text = f"{tour.summary} {tour.location} {tour.style} {' '.join(tour.includes)}".lower()
            important_keywords = ['bạch mã', 'trường sơn', 'huế', 'quảng trị', 'thiền', 'retreat']
            
            for keyword in important_keywords:
                if keyword in message_lower and keyword in search_text:
                    if tour_id not in tour_indices:
                        tour_indices.append(tour_id)
                        resolved_tours.append(tour)
                    break
        
        # 4.2 Áp dụng MANDATORY FILTERS với knowledge.json fields
        if not mandatory_filters.is_empty():
            filtered_indices = MandatoryFilterSystemV2.apply_filters(tours_db, mandatory_filters)
            
            if filtered_indices:
                if tour_indices:
                    # Kết hợp với logic AND: tour phải thỏa cả tìm kiếm và filter
                    combined = list(set(tour_indices) & set(filtered_indices))
                    if combined:
                        tour_indices = combined
                        logger.info(f"✅ Combined search and filter results: {len(tour_indices)} tours")
                    else:
                        # Nếu không có tour nào thỏa cả hai, ưu tiên filter
                        tour_indices = filtered_indices
                        logger.info(f"⚠️ No tours match both search and filter, using filter results: {len(tour_indices)} tours")
                else:
                    tour_indices = filtered_indices
                    logger.info(f"🔍 Using filter-only results: {len(tour_indices)} tours")
        
        # 4.3 FUZZY MATCHING nếu chưa đủ kết quả
        if len(tour_indices) < 3 and UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            try:
                fuzzy_matcher = FuzzyMatcher(tours_db)
                fuzzy_results = fuzzy_matcher.find_tour_by_partial_name(user_message)
                
                for tour_id in fuzzy_results:
                    if tour_id not in tour_indices:
                        tour_indices.append(tour_id)
                        tour = tours_db.get(tour_id)
                        if tour:
                            resolved_tours.append(tour)
                
                if fuzzy_results:
                    logger.info(f"🔍 Added {len(fuzzy_results)} tours from fuzzy matching")
            except Exception as e:
                logger.error(f"Fuzzy matching error: {e}")
        
        # 4.4 SEMANTIC SEARCH với FAISS index
        if len(tour_indices) < 5 and UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS") and search_index is not None:
            try:
                semantic_results = query_index(user_message, top_k=7, min_score=SEMANTIC_MIN_SCORE)
                
                for score, passage in semantic_results:
                    if 'tour_id' in passage:
                        tour_id = passage['tour_id']
                        if tour_id not in tour_indices:
                            tour_indices.append(tour_id)
                            tour = tours_db.get(tour_id)
                            if tour:
                                resolved_tours.append(tour)
                
                if semantic_results:
                    logger.info(f"🧠 Added {len(semantic_results)} tours from semantic search")
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
        
        # 4.5 PROFILE-BASED RECOMMENDATION
        if len(tour_indices) < 3 and user_profile and UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            try:
                profile_matches = SemanticAnalyzer.match_tours_to_profile(user_profile, tours_db)
                for tour_id, score in profile_matches:
                    if score > 0.7 and tour_id not in tour_indices:
                        tour_indices.append(tour_id)
                        tour = tours_db.get(tour_id)
                        if tour:
                            resolved_tours.append(tour)
                
                if profile_matches:
                    logger.info(f"👤 Added {len(profile_matches)} tours from profile matching")
            except Exception as e:
                logger.error(f"Profile matching error: {e}")
        
        # 4.6 DEDUPLICATION
        if UpgradeFlags.is_enabled("2_DEDUPLICATION"):
            try:
                original_count = len(tour_indices)
                tour_indices = DeduplicationEngine.merge_similar_tours(tour_indices, tours_db)
                if original_count != len(tour_indices):
                    logger.info(f"🔄 Deduplication: {original_count} -> {len(tour_indices)} tours")
            except Exception as e:
                logger.error(f"Deduplication error: {e}")
        
        # 4.7 Sắp xếp theo relevance với knowledge.json fields
        def calculate_relevance_score(tour_id: int) -> float:
            """Tính điểm relevance dựa trên knowledge.json fields"""
            tour = tours_db.get(tour_id)
            if not tour:
                return 0
            
            score = 0
            
            # 1. Direct name match (cao nhất)
            if tour.tour_name and tour.tour_name.lower() in message_lower:
                score += 100
            
            # 2. Field match
            if field_name:
                field_value = EnhancedFieldDetectorV2.get_field_value(tour, field_name)
                if field_value:
                    if isinstance(field_value, str) and field_value.lower() in message_lower:
                        score += 50
                    elif isinstance(field_value, list):
                        for item in field_value:
                            if item.lower() in message_lower:
                                score += 20
                                break
            
            # 3. Location match
            if tour.location and any(loc in message_lower for loc in ['huế', 'quảng trị', 'bạch mã', 'trường sơn']):
                for loc in ['huế', 'quảng trị', 'bạch mã', 'trường sơn']:
                    if loc in message_lower and loc in tour.location.lower():
                        score += 30
                        break
            
            # 4. Style match
            if tour.style and 'style' in type_details.get('field_name', ''):
                score += 15
            
            # 5. Price range match (nếu có filter)
            if mandatory_filters and (mandatory_filters.min_price or mandatory_filters.max_price):
                if tour.price_numeric:
                    if mandatory_filters.min_price and tour.price_numeric >= mandatory_filters.min_price:
                        score += 10
                    if mandatory_filters.max_price and tour.price_numeric <= mandatory_filters.max_price:
                        score += 10
            
            # 6. Rating bonus
            if tour.rating:
                score += tour.rating * 5
            
            return score
        
        # Sắp xếp theo relevance score
        tour_indices.sort(key=lambda x: calculate_relevance_score(x), reverse=True)
        
        # 4.8 Giới hạn số lượng
        tour_indices = tour_indices[:MAX_TOURS_RETURN]
        
        # Cập nhật resolved_tours
        resolved_tours = [tours_db.get(idx) for idx in tour_indices if tours_db.get(idx)]
        
        logger.info(f"✅ Found {len(tour_indices)} tours: {tour_indices}")
        
        # ================== 5. XÂY DỰNG RESPONSE VỚI KNOWLEDGE.JSON TEMPLATES ==================
        reply = ""
        warnings = []
        metadata = {
            "tour_count": len(tour_indices),
            "question_type": question_type.value,
            "confidence": confidence,
            "field_detected": field_name,
            "filters_applied": not mandatory_filters.is_empty() if mandatory_filters else False
        }
        
        # 5.1 XỬ LÝ THEO QUESTION TYPE VỚI KNOWLEDGE.JSON TEMPLATES
        if question_type == QuestionType.GREETING:
            reply = KnowledgeTemplateSystem.render('greeting')
            context.current_state = ConversationState.INITIAL
            
        elif question_type == QuestionType.FAREWELL:
            reply = KnowledgeTemplateSystem.render('farewell')
            context.current_state = ConversationState.CLOSING
            
        elif question_type == QuestionType.LIST_TOURS:
            if tour_indices:
                # Nhóm tour theo category từ knowledge.json
                tours_by_category = {}
                for tour in resolved_tours:
                    category = tour.category or 'general'
                    if category not in tours_by_category:
                        tours_by_category[category] = []
                    tours_by_category[category].append(tour)
                
                # Tạo danh sách tour có nhóm
                tour_items_by_category = []
                for category, tours in tours_by_category.items():
                    category_tours = []
                    for idx, tour in enumerate(tours[:4], 1):
                        tour_item = KnowledgeTemplateSystem.render('tour_item',
                            idx=idx,
                            tour_name=tour.tour_name,
                            location=tour.location,
                            duration=tour.duration,
                            price=tour.price,
                            summary=(tour.summary[:120] + '...') if tour.summary and len(tour.summary) > 120 else (tour.summary or "Không có mô tả")
                        )
                        category_tours.append(tour_item)
                    
                    if category_tours:
                        category_name = {
                            'adventure': '🏔️ MẠO HIỂM & KHÁM PHÁ',
                            'relaxation': '🌿 NGHỈ DƯỠNG & THƯ GIÃN',
                            'cultural': '🏛️ VĂN HÓA & LỊCH SỬ',
                            'culinary': '🍜 ẨM THỰC & ĐẶC SẢN',
                            'event': '🎪 SỰ KIỆN & TEAM BUILDING',
                            'family': '👨‍👩‍👧‍👦 GIA ĐÌNH & NHÓM',
                            'luxury': '💎 CAO CẤP & SANG TRỌNG',
                            'general': '✨ TỔNG HỢP'
                        }.get(category, category.upper())
                        
                        tour_items_by_category.append(f"**{category_name}**\n" + "\n".join(category_tours))
                
                tour_items_str = "\n\n".join(tour_items_by_category)
                
                reply = KnowledgeTemplateSystem.render('tour_list',
                    count=len(tour_indices),
                    tour_items=tour_items_str
                )
                
                # Thêm filter info nếu có
                if mandatory_filters and not mandatory_filters.is_empty():
                    filter_info = []
                    if mandatory_filters.location:
                        filter_info.append(f"📍 Địa điểm: {mandatory_filters.location}")
                    if mandatory_filters.style:
                        filter_info.append(f"🎨 Phong cách: {mandatory_filters.style}")
                    if mandatory_filters.min_price or mandatory_filters.max_price:
                        price_range = []
                        if mandatory_filters.min_price:
                            price_range.append(f"từ {mandatory_filters.min_price:,.0f} VNĐ")
                        if mandatory_filters.max_price:
                            price_range.append(f"đến {mandatory_filters.max_price:,.0f} VNĐ")
                        filter_info.append(f"💰 Giá: {' '.join(price_range)}")
                    
                    if filter_info:
                        reply += f"\n\n🔍 **Đang áp dụng bộ lọc:**\n" + "\n".join([f"• {info}" for info in filter_info])
            else:
                reply = KnowledgeTemplateSystem.render('no_results')
                warnings.append("Không tìm thấy tour nào phù hợp với yêu cầu")
            
            context.current_state = ConversationState.FILTERING
            
        elif question_type == QuestionType.TOUR_DETAIL:
            if tour_indices:
                # Hiển thị chi tiết đầy đủ từ knowledge.json
                tour = resolved_tours[0] if resolved_tours else None
                if tour:
                    reply = KnowledgeTemplateSystem.render('tour_detail_full',
                        tour_name=tour.tour_name,
                        summary=tour.summary,
                        location=tour.location,
                        duration=tour.duration,
                        price=tour.price,
                        style=tour.style,
                        transport=tour.transport,
                        accommodation=tour.accommodation,
                        meals=tour.meals,
                        includes=tour.includes,
                        notes=tour.notes,
                        event_support=tour.event_support,
                        category=tour.category or 'general',
                        rating=tour.rating or 4.5
                    )
                    
                    # Gợi ý các tour tương tự dựa trên style và category
                    similar_tours = []
                    for other_tour in resolved_tours[1:5]:
                        if other_tour and other_tour.style == tour.style or other_tour.category == tour.category:
                            similar_tours.append(f"• {other_tour.tour_name} ({other_tour.duration}, {other_tour.price})")
                    
                    if similar_tours:
                        reply += f"\n\n🔍 **Tour tương tự cùng phong cách:**\n" + "\n".join(similar_tours)
                else:
                    reply = "Không tìm thấy thông tin chi tiết cho tour này."
            else:
                reply = "Không tìm thấy tour nào. Vui lòng cung cấp tên tour hoặc mô tả chi tiết hơn."
            
            context.current_state = ConversationState.DETAIL_VIEW
            
        elif question_type == QuestionType.GENERAL_INFO and field_name:
            # Câu hỏi về field cụ thể từ knowledge.json
            if tour_indices:
                if len(tour_indices) == 1:
                    # Một tour cụ thể
                    tour = resolved_tours[0]
                    if tour:
                        field_value = EnhancedFieldDetectorV2.get_field_value(tour, field_name)
                        if field_value:
                            # Sử dụng template cụ thể nếu có
                            template_name = f'field_{field_name}'
                            if template_name in KnowledgeTemplateSystem.TEMPLATES:
                                if field_name == 'includes':
                                    includes_formatted = "\n".join([f"• {item}" for item in field_value])
                                    reply = KnowledgeTemplateSystem.render(template_name,
                                        tour_name=tour.tour_name,
                                        includes_formatted=includes_formatted
                                    )
                                else:
                                    reply = KnowledgeTemplateSystem.render(template_name,
                                        tour_name=tour.tour_name,
                                        **{field_name: field_value}
                                    )
                            else:
                                # Format chung
                                if isinstance(field_value, list):
                                    field_display = "\n".join([f"• {item}" for item in field_value])
                                else:
                                    field_display = str(field_value)
                                
                                field_display_name = {
                                    'tour_name': 'Tên tour',
                                    'price': 'Giá',
                                    'duration': 'Thời gian',
                                    'location': 'Địa điểm',
                                    'includes': 'Dịch vụ bao gồm',
                                    'style': 'Phong cách',
                                    'transport': 'Phương tiện',
                                    'accommodation': 'Chỗ ở',
                                    'meals': 'Ăn uống',
                                    'event_support': 'Hỗ trợ sự kiện',
                                    'summary': 'Tóm tắt',
                                    'notes': 'Lưu ý'
                                }.get(field_name, field_name.replace('_', ' ').upper())
                                
                                reply = f"**{field_display_name} của tour {tour.tour_name}:**\n{field_display}"
                        else:
                            reply = f"Tour {tour.tour_name} không có thông tin về {field_name.replace('_', ' ')}."
                    else:
                        reply = "Không tìm thấy tour."
                else:
                    # Nhiều tour - tổng hợp thông tin field
                    reply = f"**THÔNG TIN {field_name.replace('_', ' ').upper()} CHO CÁC TOUR:**\n\n"
                    for tour in resolved_tours[:5]:
                        field_value = EnhancedFieldDetectorV2.get_field_value(tour, field_name)
                        if field_value:
                            if isinstance(field_value, list):
                                field_display = ", ".join(field_value[:3]) + ("..." if len(field_value) > 3 else "")
                            else:
                                field_display = str(field_value)[:80] + ("..." if len(str(field_value)) > 80 else "")
                            
                            reply += f"• **{tour.tour_name}**: {field_display}\n"
                        else:
                            reply += f"• **{tour.tour_name}**: Không có thông tin\n"
                    
                    reply += f"\n💡 Có {len(tour_indices)} tour phù hợp. Để biết chi tiết về một tour cụ thể, vui lòng chọn tên tour."
            else:
                reply = f"Không tìm thấy tour nào để cung cấp thông tin về {field_name.replace('_', ' ')}."
            
            context.current_state = ConversationState.DETAIL_VIEW
            
        elif question_type == QuestionType.COMPARISON:
            if len(tour_indices) >= 2:
                # So sánh tối đa 3 tour từ knowledge.json
                tours_to_compare = resolved_tours[:3]
                
                # Tạo bảng so sánh với các field quan trọng
                comparison_rows = []
                
                # Các field so sánh từ knowledge.json
                comparison_fields = [
                    ('tour_name', 'Tên tour'),
                    ('price', 'Giá'),
                    ('duration', 'Thời gian'),
                    ('location', 'Địa điểm'),
                    ('style', 'Phong cách'),
                    ('transport', 'Phương tiện'),
                    ('accommodation', 'Chỗ ở'),
                    ('includes', 'Dịch vụ chính'),
                    ('rating', 'Đánh giá')
                ]
                
                for field_key, display_name in comparison_fields:
                    row = f"**{display_name}**: "
                    values = []
                    for tour in tours_to_compare:
                        val = EnhancedFieldDetectorV2.get_field_value(tour, field_key)
                        if val:
                            if isinstance(val, list):
                                val = ", ".join(val[:2]) if len(val) > 2 else ", ".join(val)
                            elif field_key == 'price' and len(str(val)) > 40:
                                val = str(val)[:40] + "..."
                            values.append(str(val))
                        else:
                            values.append("N/A")
                    row += " | ".join(values)
                    comparison_rows.append(row)
                
                comparison_table = "\n".join(comparison_rows)
                
                # Tạo summary và suggestion
                tour_names = [t.tour_name for t in tours_to_compare]
                summary = f"So sánh {len(tours_to_compare)} tour: {', '.join(tour_names)}"
                
                # Phân tích điểm mạnh của từng tour
                strengths = []
                for tour in tours_to_compare:
                    if tour.style:
                        strengths.append(f"• {tour.tour_name}: Mạnh về {tour.style}")
                    elif tour.category:
                        strengths.append(f"• {tour.tour_name}: Thuộc loại {tour.category}")
                
                suggestion = "Để chọn tour phù hợp nhất:\n"
                if strengths:
                    suggestion += "\n".join(strengths)
                suggestion += "\n\n📞 Liên hệ hotline 0332510486 để được tư vấn chi tiết."
                
                reply = KnowledgeTemplateSystem.render('comparison',
                    count=len(tours_to_compare),
                    comparison_table=comparison_table,
                    summary=summary,
                    suggestion=suggestion
                )
            else:
                reply = "Cần ít nhất 2 tour để so sánh. Vui lòng cung cấp tên các tour cần so sánh."
            
            context.current_state = ConversationState.COMPARISON
            
        elif question_type == QuestionType.RECOMMENDATION:
            if tour_indices:
                # Tính điểm recommendation dựa trên knowledge.json fields
                scored_tours = []
                for tour in resolved_tours:
                    score = 0
                    reasons = []
                    
                    # Điểm cho filter match
                    if mandatory_filters and not mandatory_filters.is_empty():
                        if mandatory_filters.location and mandatory_filters.location.lower() in tour.location.lower():
                            score += 3
                            reasons.append(f"Đúng địa điểm: {mandatory_filters.location}")
                        
                        if mandatory_filters.style and mandatory_filters.style.lower() in tour.style.lower():
                            score += 3
                            reasons.append(f"Đúng phong cách: {mandatory_filters.style}")
                        
                        if mandatory_filters.include_keywords:
                            matches = 0
                            for keyword in mandatory_filters.include_keywords:
                                if any(keyword in inc.lower() for inc in tour.includes):
                                    matches += 1
                            if matches > 0:
                                score += matches * 2
                                reasons.append(f"Có {matches} dịch vụ bạn cần")
                    
                    # Điểm cho field match
                    if field_name:
                        field_value = EnhancedFieldDetectorV2.get_field_value(tour, field_name)
                        if field_value:
                            score += 2
                            reasons.append(f"Có thông tin về {field_name.replace('_', ' ')}")
                    
                    # Điểm cho rating
                    if tour.rating:
                        score += tour.rating
                        reasons.append(f"Đánh giá {tour.rating}/5")
                    
                    # Điểm cho duration phù hợp
                    if mandatory_filters and (mandatory_filters.duration_min or mandatory_filters.duration_max):
                        if tour.duration_numeric:
                            if mandatory_filters.duration_min and tour.duration_numeric >= mandatory_filters.duration_min:
                                score += 1
                            if mandatory_filters.duration_max and tour.duration_numeric <= mandatory_filters.duration_max:
                                score += 1
                    
                    scored_tours.append({
                        'tour': tour,
                        'score': score,
                        'reasons': reasons[:3]
                    })
                
                # Sắp xếp theo điểm
                scored_tours.sort(key=lambda x: x['score'], reverse=True)
                
                if scored_tours:
                    # Lấy tour tốt nhất
                    best_tour = scored_tours[0]['tour']
                    best_reasons = scored_tours[0]['reasons']
                    
                    # Tạo alternatives
                    alternatives = []
                    for item in scored_tours[1:4]:
                        tour = item['tour']
                        alt_text = f"• {tour.tour_name}"
                        if tour.duration:
                            alt_text += f" ({tour.duration})"
                        if tour.price:
                            price_short = tour.price[:40] + "..." if len(tour.price) > 40 else tour.price
                            alt_text += f" - {price_short}"
                        alternatives.append(alt_text)
                    
                    # Format reasons
                    if not best_reasons:
                        best_reasons = ["Phù hợp với yêu cầu của bạn", "Được nhiều khách hàng lựa chọn"]
                    
                    reasons_text = "\n".join([f"• {r}" for r in best_reasons])
                    alternatives_text = "\n".join(alternatives) if alternatives else "Không có tour khác phù hợp"
                    
                    reply = KnowledgeTemplateSystem.render('recommendation',
                        recommended_tour=best_tour.tour_name,
                        reasons=reasons_text,
                        alternatives=alternatives_text
                    )
                else:
                    reply = "Không tìm thấy tour phù hợp để đề xuất."
            else:
                reply = KnowledgeTemplateSystem.render('no_results')
            
            context.current_state = ConversationState.RECOMMENDING
            
        elif question_type == QuestionType.UNKNOWN:
            # Fallback với LLM và knowledge context
            try:
                # Chuẩn bị knowledge context
                knowledge_context = []
                for tour in resolved_tours[:3]:
                    knowledge_context.append({
                        'name': tour.tour_name,
                        'summary': tour.summary,
                        'location': tour.location,
                        'price': tour.price,
                        'style': tour.style,
                        'includes': tour.includes[:3]
                    })
                
                # Tạo prompt với knowledge context
                prompt = _prepare_llm_prompt_with_knowledge(
                    user_message, 
                    knowledge_context,
                    {
                        'question_type': question_type.value,
                        'filters': mandatory_filters,
                        'field_name': field_name
                    }
                )
                
                # Gọi LLM
                llm_request_obj = LLMRequest(
                    prompt=prompt,
                    model="llama2",
                    temperature=0.7,
                    max_tokens=500
                )
                
                llm_response_text = llm_request(llm_request_obj)
                llm_response_parsed = parse_llm_response(llm_response_text)
                
                reply = llm_response_parsed.get('reply', '')
                
                if not reply:
                    reply = _generate_fallback_response_with_knowledge(user_message, resolved_tours)
                
                # Thêm thông tin tour nếu có
                if resolved_tours and 'tour' not in reply.lower():
                    tour_names = [t.tour_name for t in resolved_tours[:3]]
                    reply += f"\n\n🔍 **Một số tour Ruby Wings có thể bạn quan tâm:** {', '.join(tour_names)}"
                
            except Exception as e:
                logger.error(f"LLM fallback error: {e}")
                reply = _generate_fallback_response_with_knowledge(user_message, resolved_tours)
            
            context.current_state = ConversationState.INITIAL
        
        # 5.2 AUTO-VALIDATION với knowledge.json context
        if UpgradeFlags.is_enabled("9_AUTO_VALIDATION"):
            try:
                validation_context = {
                    'tours': [t.tour_name for t in resolved_tours[:3]],
                    'field_name': field_name,
                    'question_type': question_type.value
                }
                
                validated_reply = AutoValidator.safe_validate({'reply': reply, 'context': validation_context})
                if 'reply' in validated_reply and validated_reply['reply'] != reply:
                    reply = validated_reply['reply']
                    warnings.append("Phản hồi đã được tự động kiểm tra và điều chỉnh")
            except Exception as e:
                logger.warning(f"Auto-validation error: {e}")
        
        # 5.3 Đảm bảo có thông tin liên hệ
        if '0332510486' not in reply:
            reply += "\n\n📞 **Hotline tư vấn 24/7:** 0332510486"
        
        if 'www.rubywings.vn' not in reply and 'rubywings.vn' not in reply:
            reply += "\n🌐 **Website:** www.rubywings.vn"
        
        # 5.4 Formatting cuối cùng
        reply = reply.strip()
        
        # ================== 6. HẬU XỬ LÝ ==================
        # 6.1 Cập nhật conversation state
        state_machine = ConversationStateMachine(context.current_state)
        state_machine.update(user_message, reply[:100], tour_indices)
        context.current_state = state_machine.current_state
        
        # 6.2 Cập nhật last_tours_mentioned
        if tour_indices:
            for tour_id in tour_indices:
                if tour_id not in context.last_tours_mentioned:
                    context.last_tours_mentioned.append(tour_id)
            
            # Giới hạn 10 tour
            if len(context.last_tours_mentioned) > 10:
                context.last_tours_mentioned = context.last_tours_mentioned[-10:]
        
        # 6.3 Thêm bot response vào conversation history
        context.conversation_history.append({
            'role': 'assistant',
            'message': reply[:500],
            'timestamp': datetime.utcnow().isoformat(),
            'tour_indices': tour_indices[:5],
            'question_type': question_type.value,
            'field_name': field_name
        })
        
        # 6.4 Lưu cache
        if cache_key and UpgradeFlags.is_enabled("CACHE_SYSTEM"):
            cache_entry = CacheEntry(
                value={
                    'reply': reply,
                    'tour_indices': tour_indices,
                    'warnings': warnings,
                    'metadata': metadata
                },
                expiry=time.time() + CACHE_TTL
            )
            cache_system.set(cache_key, cache_entry)
            logger.info(f"💾 Cached response for key: {cache_key[:50]}...")
        
        # 6.5 Lưu session context
        save_session_context(session_id, context)
        
        # 6.6 Gửi CAPI event
        if CAPI_ENABLED:
            try:
                send_capi_event(session_id, user_message[:100], reply[:100])
            except Exception as e:
                logger.error(f"CAPI error: {e}")
        
        # ================== 7. TRẢ RESPONSE ==================
        processing_time = time.time() - start_time
        metadata['processing_time_ms'] = int(processing_time * 1000)
        metadata['from_cache'] = False
        
        # Tạo ChatResponse
        chat_response = ChatResponse(
            reply=reply,
            tour_indices=tour_indices,
            action="continue",
            context={
                "session_id": session_id,
                "question_type": question_type.value,
                "field_name": field_name,
                "confidence": confidence,
                "filters_applied": not mandatory_filters.is_empty() if mandatory_filters else False,
                "state": context.current_state.value,
                "tour_count": len(tour_indices)
            },
            warnings=warnings if warnings else None,
            metadata=metadata
        )
        
        logger.info(f"✅ Request processed in {processing_time:.2f}s | "
                   f"Tours: {len(tour_indices)} | "
                   f"Type: {question_type.value} | "
                   f"Confidence: {confidence:.2f}")
        
        return jsonify(asdict(chat_response))
        
    except Exception as e:
        logger.error(f"❌ Critical error in chat endpoint: {e}", exc_info=True)
        
        processing_time = time.time() - start_time
        
        # Tạo error response
        error_response = ChatResponse(
            reply="Xin lỗi, đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại hoặc liên hệ hotline 0332510486.",
            tour_indices=[],
            action="error",
            context={
                "error": str(e)[:100],
                "processing_time_ms": int(processing_time * 1000)
            },
            warnings=["Hệ thống gặp sự cố, vui lòng thử lại sau."],
            metadata={
                "error_type": type(e).__name__,
                "processing_time_ms": int(processing_time * 1000)
            }
        )
        
        return jsonify(asdict(error_response)), 500


# ==================== KNOWLEDGE-AWARE HELPER FUNCTIONS ====================

def _prepare_llm_prompt_with_knowledge(user_message: str, knowledge_context: List[Dict], extra_context: Dict) -> str:
    """
    Chuẩn bị prompt cho LLM với knowledge.json context
    """
    prompt = f"""Bạn là trợ lý AI của Ruby Wings Travel, chuyên về các tour trải nghiệm tại miền Trung Việt Nam.

THÔNG TIN TOUR HIỆN CÓ (từ knowledge.json):
{json.dumps(knowledge_context, indent=2, ensure_ascii=False)}

NGỮ CẢNH CUỘC HỘI THOẠI:
- Loại câu hỏi: {extra_context.get('question_type', 'unknown')}
- Field được hỏi: {extra_context.get('field_name', 'none')}
- Bộ lọc: {extra_context.get('filters', 'none')}

CÂU HỎI CỦA KHÁCH HÀNG: "{user_message}"

YÊU CẦU TRẢ LỜI:
1. Sử dụng thông tin từ knowledge.json ở trên
2. Trả lời thân thiện, chuyên nghiệp
3. Nếu không có thông tin, đề nghị liên hệ hotline
4. Luôn nhắc đến hotline 0332510486 và website www.rubywings.vn
5. Trả lời bằng tiếng Việt

TRẢ LỜI:"""
    
    return prompt


def _generate_fallback_response_with_knowledge(user_message: str, tours: List[Tour]) -> str:
    """
    Tạo fallback response với knowledge context
    """
    if tours:
        reply = f"Cảm ơn câu hỏi của bạn về: '{user_message}'\n\n"
        reply += "Dựa trên thông tin hiện có, đây là các tour Ruby Wings có thể phù hợp:\n\n"
        
        for i, tour in enumerate(tours[:4], 1):
            reply += f"{i}. **{tour.tour_name}**\n"
            if tour.duration:
                reply += f"   ⏱️ {tour.duration}\n"
            if tour.location:
                reply += f"   📍 {tour.location[:50]}...\n" if len(tour.location) > 50 else f"   📍 {tour.location}\n"
            if tour.summary:
                summary_short = tour.summary[:100] + "..." if len(tour.summary) > 100 else tour.summary
                reply += f"   📝 {summary_short}\n"
            reply += "\n"
        
        reply += "Để được tư vấn chi tiết và chính xác hơn, vui lòng:\n"
        reply += "• Cung cấp thêm thông tin về nhu cầu của bạn\n"
        reply += "• Gọi trực tiếp hotline 0332510486\n"
        reply += "• Truy cập website www.rubywings.vn\n\n"
        reply += "Ruby Wings có hơn 32 tour trải nghiệm đặc sắc tại Huế, Quảng Trị, Bạch Mã và Trường Sơn!"
    else:
        reply = f"Cảm ơn câu hỏi của bạn: '{user_message}'\n\n"
        reply += "Hiện Ruby Wings có các loại tour chính:\n\n"
        reply += "🏔️ **TOUR MẠO HIỂM & KHÁM PHÁ:**\n"
        reply += "• Trekking Bạch Mã, khám phá rừng nguyên sinh\n"
        reply += "• Khám phá Trường Sơn, di tích lịch sử\n\n"
        
        reply += "🕉️ **TOUR RETREAT & CHỮA LÀNH:**\n"
        reply += "• Thiền định, yoga tại Bạch Mã\n"
        reply += "• Retreat tĩnh tâm, chữa lành năng lượng\n\n"
        
        reply += "🏛️ **TOUR VĂN HÓA & LỊCH SỬ:**\n"
        reply += "• Di sản Huế, ẩm thực cung đình\n"
        reply += "• Di tích chiến tranh tại Quảng Trị\n\n"
        
        reply += "👥 **TOUR NHÓM & TEAM BUILDING:**\n"
        reply += "• Team building công ty, nhóm bạn\n"
        reply += "• Tour gia đình, đa thế hệ\n\n"
        
        reply += "📞 **Để biết thêm chi tiết và được tư vấn tour phù hợp nhất:**\n"
        reply += "• Hotline: 0332510486 (24/7)\n"
        reply += "• Website: www.rubywings.vn\n"
        reply += "• Email: rubywingslsa@gmail.com"
    
    return reply


# ==================== FLAG MANAGEMENT ====================

class UpgradeFlags:
    """Quản lý các tính năng nâng cao"""
    
    _flags = {
        "1_MANDATORY_FILTER": True,
        "2_DEDUPLICATION": True,
        "3_FIELD_DETECTION": True,
        "4_QUESTION_PIPELINE": True,
        "5_COMPLEX_QUERY": False,  # Tạm tắt
        "6_FUZZY_MATCHING": True,
        "7_STATE_MACHINE": True,
        "8_SEMANTIC_ANALYSIS": True,
        "9_AUTO_VALIDATION": True,
        "10_TEMPLATE_SYSTEM": True,
        "CACHE_SYSTEM": True,
        "LLM_FALLBACK": True
    }
    
    @classmethod
    def is_enabled(cls, flag_name: str) -> bool:
        return cls._flags.get(flag_name, False)
    
    @classmethod
    def enable(cls, flag_name: str):
        cls._flags[flag_name] = True
    
    @classmethod
    def disable(cls, flag_name: str):
        cls._flags[flag_name] = False
    
    @classmethod
    def get_all_flags(cls) -> Dict:
        return cls._flags.copy()


# ==================== BACKWARD COMPATIBILITY FUNCTIONS ====================

def get_session_context(session_id: str) -> ConversationContext:
    """Lấy context từ session - Tương thích với knowledge.json"""
    if session_id not in sessions:
        sessions[session_id] = ConversationContext(session_id=session_id)
    
    # Kiểm tra session timeout
    context = sessions[session_id]
    if time.time() - context.last_activity > SESSION_TIMEOUT:
        logger.info(f"Session {session_id} expired, creating new one")
        sessions[session_id] = ConversationContext(session_id=session_id)
    
    return sessions[session_id]


def save_session_context(session_id: str, context: ConversationContext):
    """Lưu context vào session"""
    sessions[session_id] = context


def extract_session_id(request_data: Dict, remote_addr: str) -> str:
    """Extract session ID từ request"""
    session_id = request_data.get("session_id")
    if not session_id:
        # Tạo session ID mới từ IP và timestamp
        session_hash = hashlib.md5(f"{remote_addr}_{time.time()}".encode()).hexdigest()[:16]
        session_id = f"session_{session_hash}"
    
    return session_id


def llm_request(request_data: LLMRequest) -> str:
    """Gửi request đến LLM"""
    try:
        response = requests.post(
            LLM_URL,
            json=asdict(request_data),
            timeout=30
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"LLM request error: {e}")
        return ""


def parse_llm_response(llm_response: str) -> Dict:
    """Parse LLM response"""
    try:
        # Đơn giản: trả về toàn bộ response
        return {"reply": llm_response}
    except:
        return {"reply": "Xin lỗi, không thể xử lý phản hồi từ AI."}


# ==================== INITIALIZATION ====================

def initialize_app():
    """Khởi tạo ứng dụng với knowledge.json"""
    global tours_db, tour_name_index, search_index
    
    try:
        # Load tours từ knowledge.json
        tours_db = KnowledgeLoader.build_tours_database()
        
        # Build tour name index
        tour_name_index = {tour.tour_name.lower(): tour_id for tour_id, tour in tours_db.items()}
        
        # Build search index
        build_index(force_rebuild=False)
        
        logger.info(f"✅ App initialized with {len(tours_db)} tours from knowledge.json")
        
        # Log số lượng tour theo category
        categories = {}
        for tour in tours_db.values():
            cat = tour.category or 'unknown'
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info(f"📊 Tour categories: {categories}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize app: {e}")
        tours_db = {}
        tour_name_index = {}


# Chạy khởi tạo khi import
initialize_app()

# ==================== PHẦN 9: ADDITIONAL ENDPOINTS ====================

@app.route('/api/filters/extract', methods=['POST'])
def extract_filters():
    """Extract filters from message"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        filters = MandatoryFilterSystemV2.extract_filters(message)
        
        return jsonify({
            "success": True,
            "filters": asdict(filters)
        })
    except Exception as e:
        logger.error(f"Filter extraction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/question/classify', methods=['POST'])
def classify_question():
    """Classify question type"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        qtype, confidence, metadata = KnowledgeAwareQuestionPipeline.classify_question(message)
        
        return jsonify({
            "success": True,
            "question_type": qtype.value,
            "confidence": confidence,
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Question classification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Cleanup expired sessions and cache"""
    try:
        cleanup_expired_sessions()
        cache_system._cleanup()
        
        return jsonify({
            "success": True,
            "message": f"Cleanup completed. Sessions: {len(sessions)}"
        })
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== PHẦN 10: ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting Ruby Wings Chatbot v5.0 on port {port}")
    logger.info(f"📊 Loaded {len(tours_db)} tours from knowledge.json")
    logger.info(f"🔍 Search index ready: {search_index is not None}")
    
    # Start cleanup thread
    def cleanup_thread():
        while True:
            time.sleep(300)  # 5 minutes
            cleanup_expired_sessions()
            cache_system._cleanup()
    
    threading.Thread(target=cleanup_thread, daemon=True).start()
    
    app.run(host='0.0.0.0', port=port, debug=debug)