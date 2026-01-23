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

import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
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

# ==================== PHẦN 8: MAIN CHAT ENDPOINT - PHIÊN BẢN CAO CẤP ====================

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    TRỢ LÝ AI THÔNG MINH RUBY WINGS - PHIÊN BẢN CAO CẤP
    Tích hợp đầy đủ 10 upgrades systems với knowledge.json
    Xử lý đa tầng, context-aware, real-time optimization
    """
    # ========== KHỞI TẠO BIẾN TOÀN CỤC ==========
    start_time = time.time()
    session_id = None
    context = None
    user_message = ""
    processing_phase = "initialization"
    
    try:
        # ========== PHASE 1: REQUEST PROCESSING & VALIDATION ==========
        processing_phase = "request_processing"
        
        # 1.1 Parse và validate request
        request_data = request.get_json()
        if not request_data:
            logger.warning("Empty request received")
            return jsonify({
                "reply": "Vui lòng gửi yêu cầu dưới dạng JSON với trường 'message'.",
                "tour_indices": [],
                "context": {"error": "invalid_request"},
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }), 400
        
        user_message = request_data.get("message", "").strip()
        if not user_message:
            # Trả về greeting template nếu message rỗng
            greeting_response = KnowledgeTemplateSystem.render('greeting')
            return jsonify({
                "reply": greeting_response,
                "tour_indices": [],
                "context": {"session_id": generate_session_id(), "action": "greeting"},
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "metadata": {"template": "greeting", "version": "4.2"}
            })
        
        # 1.2 Extract session information
        provided_session_id = request_data.get("session_id")
        client_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # 1.3 Generate or retrieve session ID
        session_id = extract_session_id(request_data, client_ip)
        logger.info(f"Session ID: {session_id}, Client IP: {client_ip}, User Agent: {user_agent[:50]}...")
        
        # ========== PHASE 2: SESSION & CONTEXT MANAGEMENT ==========
        processing_phase = "session_management"
        
        # 2.1 Lấy hoặc tạo mới conversation context
        context = get_session_context(session_id)
        
        # 2.2 Khởi tạo context nếu chưa có
        context_initialized = False
        if not hasattr(context, 'conversation_history'):
            context.conversation_history = []
            context_initialized = True
        
        if not hasattr(context, 'user_preferences'):
            context.user_preferences = {}
            context_initialized = True
        
        if not hasattr(context, 'last_tours_mentioned'):
            context.last_tours_mentioned = []
            context_initialized = True
        
        if not hasattr(context, 'current_state'):
            context.current_state = ConversationState.INITIAL
            context_initialized = True
        
        if context_initialized:
            logger.info(f"Initialized new context for session: {session_id}")
        
        # 2.3 Update activity tracking
        context.last_activity = time.time()
        context.session_id = session_id
        
        # 2.4 Add user message to conversation history với metadata
        message_entry = {
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat(),
            "timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "client_ip": client_ip,
            "user_agent": user_agent[:100]
        }
        
        context.conversation_history.append(message_entry)
        
        # 2.5 Limit conversation history để tránh memory leak
        if len(context.conversation_history) > 25:
            # Giữ lại 20 tin nhắn gần nhất
            context.conversation_history = context.conversation_history[-20:]
            logger.debug(f"Trimmed conversation history for session {session_id}")
        
        # 2.6 Kiểm tra cache trước khi xử lý
        cache_key = None
        cached_response = None
        
        if UpgradeFlags.is_enabled("ENABLE_CACHING"):
            # Tạo cache key từ message và context signature
            context_signature = hashlib.md5(
                json.dumps({
                    "last_tours": context.last_tours_mentioned[:3],
                    "state": context.current_state.value,
                    "preferences_hash": hashlib.md5(
                        json.dumps(context.user_preferences, sort_keys=True).encode()
                    ).hexdigest()[:8]
                }, sort_keys=True).encode()
            ).hexdigest()
            
            cache_key = cache_system.get_cache_key(user_message, context_signature)
            cached_response = cache_system.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for session {session_id}, key: {cache_key[:20]}...")
                # Update context với cached response
                context.conversation_history.append({
                    "role": "assistant",
                    "message": cached_response.get('reply', '')[:200] + "...",
                    "timestamp": datetime.now().isoformat(),
                    "cached": True
                })
                
                # Update processing time
                cached_response['processing_time_ms'] = int((time.time() - start_time) * 1000)
                cached_response['metadata']['cache_hit'] = True
                
                return jsonify(cached_response)
        
        # ========== PHASE 3: ADVANCED QUESTION ANALYSIS ==========
        processing_phase = "question_analysis"
        
        # 3.1 Phân tích câu hỏi với multiple layers
        message_lower = user_message.lower()
        message_length = len(user_message)
        word_count = len(user_message.split())
        
        logger.info(f"Analyzing message: '{user_message[:100]}...' (length: {message_length}, words: {word_count})")
        
        # 3.2 Phân loại câu hỏi với confidence scoring
        question_start_time = time.time()
        question_type, q_confidence, q_metadata = KnowledgeAwareQuestionPipeline.classify_question(user_message)
        question_analysis_time = int((time.time() - question_start_time) * 1000)
        
        context.last_question_type = question_type
        logger.info(f"Question classified as: {question_type.value} (confidence: {q_confidence:.2f}, time: {question_analysis_time}ms)")
        
        # 3.3 Trích xuất filters với knowledge.json support
        filter_start_time = time.time()
        filters = MandatoryFilterSystemV2.extract_filters(user_message)
        filter_analysis_time = int((time.time() - filter_start_time) * 1000)
        
        filter_applied = not filters.is_empty()
        if filter_applied:
            context.active_filters = filters
            logger.info(f"Filters extracted in {filter_analysis_time}ms: {filters}")
        
        # 3.4 Phát hiện field cụ thể được hỏi
        field_start_time = time.time()
        field_name, field_confidence, field_scores = EnhancedFieldDetectorV2.detect_field_with_confidence(user_message)
        field_analysis_time = int((time.time() - field_start_time) * 1000)
        
        if field_name and field_confidence > 0.5:
            logger.info(f"Field detected: {field_name} (confidence: {field_confidence:.2f}, time: {field_analysis_time}ms)")
            context.last_field_asked = field_name
        
        # 3.5 Semantic analysis và user profiling
        semantic_profile = {}
        semantic_analysis_time = 0
        
        if UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
            semantic_start_time = time.time()
            try:
                semantic_profile = SemanticAnalyzer.analyze_user_profile(user_message, context)
                semantic_analysis_time = int((time.time() - semantic_start_time) * 1000)
                
                if semantic_profile:
                    # Cập nhật user preferences với semantic insights
                    for key, value in semantic_profile.items():
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            context.user_preferences[key] = value
                    
                    logger.info(f"Semantic analysis completed in {semantic_analysis_time}ms, profile keys: {list(semantic_profile.keys())}")
            except Exception as e:
                logger.error(f"Semantic analysis error: {e}")
        
        # 3.6 Complexity analysis
        complexity_score = 0
        complexity_factors = {
            'word_count': min(word_count / 50, 2.0),  # Max 2 points
            'question_words': sum(1 for word in ['ai', 'gì', 'ở đâu', 'tại sao', 'thế nào', 'bao nhiêu'] if word in message_lower) * 0.3,
            'special_chars': len(re.findall(r'[?!]', user_message)) * 0.2,
            'multiple_clauses': len(re.findall(r'và|hoặc|nhưng|tuy nhiên', message_lower)) * 0.4
        }
        
        complexity_score = sum(complexity_factors.values())
        complexity_level = "SIMPLE" if complexity_score < 1.0 else "MODERATE" if complexity_score < 2.0 else "COMPLEX"
        
        logger.info(f"Complexity analysis: score={complexity_score:.2f}, level={complexity_level}")
        
        # ========== PHASE 4: INTELLIGENT TOUR SEARCH & MATCHING ==========
        processing_phase = "tour_search"
        
        tour_indices = []
        search_strategies = []
        search_metadata = {
            "strategies_used": [],
            "results_per_strategy": {},
            "total_time_ms": 0
        }
        
        search_start_time = time.time()
        
        # 4.1 STRATEGY 1: Direct Tour Name Matching (High Precision)
        strategy1_start = time.time()
        direct_matches = []
        
        # Tìm kiếm trực tiếp trong tour names
        for norm_name, idx in tour_name_index.items():
            # Kiểm tra exact match hoặc partial match
            if norm_name in message_lower:
                direct_matches.append(idx)
            else:
                # Kiểm tra từng từ trong tên tour
                name_words = norm_name.split()
                if any(word in message_lower for word in name_words if len(word) > 2):
                    direct_matches.append(idx)
        
        if direct_matches:
            direct_matches = list(set(direct_matches))[:10]  # Deduplicate và giới hạn
            tour_indices.extend(direct_matches)
            search_strategies.append("direct_name_match")
            search_metadata["results_per_strategy"]["direct_name_match"] = len(direct_matches)
            logger.info(f"Strategy 1 (Direct Name Match): Found {len(direct_matches)} tours")
        
        strategy1_time = int((time.time() - strategy1_start) * 1000)
        
        # 4.2 STRATEGY 2: Fuzzy Matching với nâng cấp
        if not tour_indices and UpgradeFlags.is_enabled("6_FUZZY_MATCHING"):
            strategy2_start = time.time()
            
            try:
                fuzzy_matcher = FuzzyMatcher(tours_db)
                
                # Tìm kiếm fuzzy trong tên tour
                fuzzy_results = fuzzy_matcher.find_similar_tours(user_message, tour_name_index)
                
                if fuzzy_results:
                    # Lọc với threshold thấp hơn cho fuzzy matching
                    fuzzy_indices = [idx for idx, score in fuzzy_results if score > 0.4]
                    fuzzy_indices = fuzzy_indices[:8]  # Giới hạn kết quả
                    
                    if fuzzy_indices:
                        tour_indices.extend(fuzzy_indices)
                        search_strategies.append("fuzzy_matching")
                        search_metadata["results_per_strategy"]["fuzzy_matching"] = len(fuzzy_indices)
                        logger.info(f"Strategy 2 (Fuzzy Matching): Found {len(fuzzy_indices)} tours")
            except Exception as e:
                logger.error(f"Fuzzy matching error: {e}")
            
            strategy2_time = int((time.time() - strategy2_start) * 1000)
        
        # 4.3 STRATEGY 3: Semantic Search với FAISS
        if not tour_indices and search_index is not None:
            strategy3_start = time.time()
            
            try:
                semantic_results = query_index(
                    user_message, 
                    top_k=15,  # Tăng số lượng kết quả
                    min_score=max(SEMANTIC_MIN_SCORE, 0.65)  # Điều chỉnh threshold
                )
                
                if semantic_results:
                    semantic_indices = []
                    for score, passage in semantic_results:
                        if 'tour_id' in passage:
                            tour_id = passage['tour_id']
                            if tour_id not in semantic_indices:
                                semantic_indices.append(tour_id)
                    
                    if semantic_indices:
                        # Ưu tiên các kết quả có score cao
                        semantic_indices = semantic_indices[:10]
                        tour_indices.extend(semantic_indices)
                        search_strategies.append("semantic_search")
                        search_metadata["results_per_strategy"]["semantic_search"] = len(semantic_indices)
                        logger.info(f"Strategy 3 (Semantic Search): Found {len(semantic_indices)} tours, top score: {semantic_results[0][0]:.3f}")
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
            
            strategy3_time = int((time.time() - strategy3_start) * 1000)
        
        # 4.4 STRATEGY 4: Knowledge.json Field Search
        if not tour_indices:
            strategy4_start = time.time()
            
            keyword_matches = []
            search_keywords = [word for word in message_lower.split() if len(word) > 2][:10]  # Lấy 10 từ khóa
            
            if search_keywords:
                for idx, tour in tours_db.items():
                    match_score = 0
                    
                    # Tìm kiếm trong multiple fields của knowledge.json
                    search_fields = [
                        tour.tour_name.lower(),
                        tour.summary.lower() if tour.summary else "",
                        tour.location.lower() if tour.location else "",
                        tour.style.lower() if tour.style else "",
                        " ".join(tour.includes).lower() if tour.includes else ""
                    ]
                    
                    field_weights = [2.0, 1.5, 1.2, 1.0, 0.8]  # Trọng số cho từng field
                    
                    for keyword in search_keywords:
                        for i, field_content in enumerate(search_fields):
                            if keyword in field_content:
                                match_score += field_weights[i]
                    
                    if match_score > 1.5:  # Ngưỡng tối thiểu
                        keyword_matches.append((idx, match_score))
                
                if keyword_matches:
                    # Sắp xếp theo match score
                    keyword_matches.sort(key=lambda x: x[1], reverse=True)
                    keyword_indices = [idx for idx, score in keyword_matches[:12]]
                    
                    tour_indices.extend(keyword_indices)
                    search_strategies.append("keyword_field_search")
                    search_metadata["results_per_strategy"]["keyword_field_search"] = len(keyword_indices)
                    logger.info(f"Strategy 4 (Keyword Field Search): Found {len(keyword_indices)} tours")
            
            strategy4_time = int((time.time() - strategy4_start) * 1000)
        
        # 4.5 STRATEGY 5: Context-based Search (sử dụng conversation history)
        if not tour_indices and len(context.conversation_history) > 1:
            strategy5_start = time.time()
            
            # Tìm trong previous mentions
            if context.last_tours_mentioned:
                tour_indices.extend(context.last_tours_mentioned[:5])
                search_strategies.append("context_based")
                search_metadata["results_per_strategy"]["context_based"] = len(context.last_tours_mentioned[:5])
                logger.info(f"Strategy 5 (Context-based): Using {len(context.last_tours_mentioned[:5])} previously mentioned tours")
            
            strategy5_time = int((time.time() - strategy5_start) * 1000)
        
        # 4.6 STRATEGY 6: Popular Tours Fallback
        if not tour_indices:
            strategy6_start = time.time()
            
            # Lấy các tour phổ biến (có thể dựa trên rating hoặc predefined list)
            popular_tours = []
            for idx, tour in tours_db.items():
                # Ưu tiên tour có rating cao và price hợp lý
                if (tour.rating or 0) >= 4.0 and (tour.price_numeric or float('inf')) < 3000000:
                    popular_tours.append(idx)
            
            if popular_tours:
                # Lấy ngẫu nhiên 5 tour phổ biến
                import random
                random_seed = int(hashlib.md5(user_message.encode()).hexdigest(), 16) % 1000
                random.seed(random_seed)
                popular_sample = random.sample(popular_tours, min(5, len(popular_tours)))
                
                tour_indices.extend(popular_sample)
                search_strategies.append("popular_fallback")
                search_metadata["results_per_strategy"]["popular_fallback"] = len(popular_sample)
                logger.info(f"Strategy 6 (Popular Fallback): Using {len(popular_sample)} popular tours")
            
            strategy6_time = int((time.time() - strategy6_start) * 1000)
        
        # 4.7 Áp dụng Mandatory Filters (nếu có)
        if filter_applied and filters and tour_indices:
            filter_start_time = time.time()
            
            try:
                # Áp dụng filters lên các tour đã tìm được
                filtered_indices = MandatoryFilterSystemV2.apply_filters(tours_db, filters)
                
                if filtered_indices:
                    # Tìm giao giữa kết quả tìm kiếm và filtered results
                    intersection = list(set(tour_indices) & set(filtered_indices))
                    
                    if intersection:
                        tour_indices = intersection[:MAX_TOURS_RETURN]
                        logger.info(f"Filter application: {len(intersection)} tours pass filters")
                        search_strategies.append("filter_applied")
                        search_metadata["filtered_from"] = len(tour_indices)
                        search_metadata["filtered_to"] = len(intersection)
                    else:
                        # Nếu không có giao, ưu tiên filtered results
                        tour_indices = filtered_indices[:MAX_TOURS_RETURN]
                        logger.info(f"No intersection, using filter results: {len(tour_indices)} tours")
                else:
                    logger.warning("No tours passed the filters")
                    # Vẫn giữ nguyên tour_indices nhưng sẽ thêm warning sau
            except Exception as e:
                logger.error(f"Filter application error: {e}")
                # Continue với tour_indices hiện tại
            
            filter_time = int((time.time() - filter_start_time) * 1000)
            search_metadata["filter_time_ms"] = filter_time
        
        # 4.8 Deduplication và Post-processing
        if tour_indices:
            # Remove duplicates
            tour_indices = list(dict.fromkeys(tour_indices))  # Giữ thứ tự
            
            # Apply deduplication engine
            if UpgradeFlags.is_enabled("2_DEDUPLICATION") and len(tour_indices) > 3:
                try:
                    dedup_start = time.time()
                    deduplicated = DeduplicationEngine.merge_similar_tours(tour_indices, tours_db)
                    tour_indices = deduplicated[:MAX_TOURS_RETURN]
                    dedup_time = int((time.time() - dedup_start) * 1000)
                    search_metadata["deduplication_time_ms"] = dedup_time
                    logger.info(f"Deduplication: {len(tour_indices)} unique tours after dedup")
                except Exception as e:
                    logger.error(f"Deduplication error: {e}")
            
            # Sort by relevance (kết hợp multiple factors)
            try:
                sort_start = time.time()
                
                def tour_relevance_score(idx):
                    tour = tours_db.get(idx)
                    if not tour:
                        return 0
                    
                    score = 0
                    
                    # Factor 1: Rating
                    score += (tour.rating or 3.5) * 100
                    
                    # Factor 2: Price (ưu tiên giá vừa phải)
                    if tour.price_numeric:
                        if 1000000 <= tour.price_numeric <= 3000000:
                            score += 50
                        elif tour.price_numeric < 1000000:
                            score += 30
                    
                    # Factor 3: Duration (ưu tiên tour 1-3 ngày)
                    if tour.duration_numeric:
                        if 1 <= tour.duration_numeric <= 3:
                            score += 40
                    
                    # Factor 4: Popularity (dựa trên position trong search results)
                    if idx in direct_matches:
                        score += 200
                    elif idx in tour_indices[:5]:
                        score += 100
                    
                    return score
                
                tour_indices.sort(key=tour_relevance_score, reverse=True)
                sort_time = int((time.time() - sort_start) * 1000)
                search_metadata["sorting_time_ms"] = sort_time
                
            except Exception as e:
                logger.error(f"Sorting error: {e}")
        
        # 4.9 Search performance logging
        total_search_time = int((time.time() - search_start_time) * 1000)
        search_metadata["total_time_ms"] = total_search_time
        search_metadata["strategies_used"] = search_strategies
        
        logger.info(f"""
        SEARCH PERFORMANCE SUMMARY:
        Total time: {total_search_time}ms
        Strategies used: {', '.join(search_strategies)}
        Total tours found: {len(tour_indices)}
        Final tour indices: {tour_indices[:10] if tour_indices else 'None'}
        """)
        
        # ========== PHASE 5: INTELLIGENT RESPONSE GENERATION ==========
        processing_phase = "response_generation"
        
        reply = ""
        warnings = []
        suggestions = []
        response_metadata = {
            "question_type": question_type.value,
            "question_confidence": q_confidence,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "field_detected": field_name,
            "field_confidence": field_confidence,
            "filter_applied": filter_applied,
            "tours_found": len(tour_indices),
            "search_performance": search_metadata
        }
        
        response_start_time = time.time()
        
        # 5.1 Xác định response strategy dựa trên question type và context
        response_strategy = None
        
        if question_type == QuestionType.GREETING:
            response_strategy = "greeting_template"
            reply = KnowledgeTemplateSystem.render('greeting')
            context.current_state = ConversationState.INITIAL
            
        elif question_type == QuestionType.FAREWELL:
            response_strategy = "farewell_template"
            reply = KnowledgeTemplateSystem.render('farewell')
            context.current_state = ConversationState.CLOSING
            
        elif question_type == QuestionType.LIST_TOURS:
            response_strategy = "tour_listing"
            
            if not tour_indices:
                reply = KnowledgeTemplateSystem.render('no_results')
                warnings.append("Không tìm thấy tour phù hợp")
            else:
                # Xây dựng danh sách tour chi tiết
                tour_items = []
                display_count = min(len(tour_indices), MAX_TOURS_RETURN)
                
                for i, idx in enumerate(tour_indices[:display_count], 1):
                    tour = tours_db.get(idx)
                    if tour:
                        # Format includes cho đẹp
                        includes_preview = ", ".join(tour.includes[:3])
                        if len(tour.includes) > 3:
                            includes_preview += f" và {len(tour.includes) - 3} dịch vụ khác"
                        
                        # Tạo tour item với đầy đủ thông tin từ knowledge.json
                        tour_item = KnowledgeTemplateSystem.render('tour_item',
                            idx=i,
                            tour_name=tour.tour_name,
                            location=tour.location,
                            duration=tour.duration,
                            price=tour.price,
                            summary=(tour.summary[:120] + "...") if len(tour.summary) > 120 else tour.summary,
                            includes_preview=includes_preview,
                            style=tour.style or "Không xác định",
                            category=tour.category or "general"
                        )
                        tour_items.append(tour_item)
                
                if tour_items:
                    reply = KnowledgeTemplateSystem.render('tour_list',
                        count=len(tour_indices),
                        tour_items="\n\n".join(tour_items),
                        filter_summary=f"📍 **Bộ lọc áp dụng:** {', '.join([f'{k}: {v}' for k, v in filters.__dict__.items() if v])}" if filter_applied else "",
                        suggestion="💡 **Gợi ý:** Gõ số thứ tự để xem chi tiết tour, hoặc hỏi thêm về tiêu chí cụ thể."
                    )
                    
                    # Cập nhật context
                    context.last_tours_mentioned = tour_indices[:display_count]
                    response_metadata["tours_displayed"] = display_count
                else:
                    reply = KnowledgeTemplateSystem.render('no_results')
        
        elif question_type == QuestionType.TOUR_DETAIL:
            response_strategy = "tour_detail"
            
            if not tour_indices:
                reply = KnowledgeTemplateSystem.render('no_results')
            else:
                # Hiển thị chi tiết đầy đủ cho tour đầu tiên
                primary_idx = tour_indices[0]
                tour = tours_db.get(primary_idx)
                
                if tour:
                    # Format includes với bullet points
                    includes_items = []
                    for i, item in enumerate(tour.includes, 1):
                        includes_items.append(f"{i}. {item}")
                    includes_formatted = "\n".join(includes_items)
                    
                    # Format additional information
                    additional_info = []
                    if tour.transport:
                        additional_info.append(f"🚌 **Phương tiện:** {tour.transport}")
                    if tour.accommodation:
                        additional_info.append(f"🏨 **Chỗ ở:** {tour.accommodation}")
                    if tour.meals:
                        additional_info.append(f"🍽️ **Ăn uống:** {tour.meals}")
                    if tour.event_support:
                        additional_info.append(f"🎪 **Hỗ trợ sự kiện:** {tour.event_support}")
                    
                    additional_formatted = "\n".join(additional_info)
                    
                    # Render template với đầy đủ fields từ knowledge.json
                    reply = KnowledgeTemplateSystem.render('tour_detail_full',
                        tour_name=tour.tour_name,
                        summary=tour.summary,
                        location=tour.location,
                        duration=tour.duration,
                        price=tour.price,
                        style=tour.style or "Đa dạng",
                        transport=tour.transport or "Xe du lịch đời mới",
                        accommodation=tour.accommodation or "Khách sạn 3 sao",
                        meals=tour.meals or "Theo chương trình",
                        includes_formatted=includes_formatted,
                        notes=tour.notes or "Vui lòng liên hệ để biết thêm chi tiết.",
                        event_support=tour.event_support or "Có sẵn theo yêu cầu",
                        category=tour.category or "general",
                        rating=tour.rating or 4.5,
                        additional_info=additional_formatted
                    )
                    
                    # Thêm đề xuất tour tương tự nếu có
                    if len(tour_indices) > 1:
                        reply += "\n\n🔍 **TOUR TƯƠNG TỰ CÓ THỂ BẠN QUAN TÂM:**\n"
                        for idx in tour_indices[1:4]:
                            similar_tour = tours_db.get(idx)
                            if similar_tour:
                                reply += f"• **{similar_tour.tour_name}** ({similar_tour.duration}, {similar_tour.location})\n"
                    
                    # Cập nhật context
                    context.current_tour = primary_idx
                    context.last_tours_mentioned = [primary_idx]
                    response_metadata["current_tour"] = primary_idx
                    response_metadata["tour_name"] = tour.tour_name
                else:
                    reply = "❌ **Không tìm thấy thông tin chi tiết về tour này.**\n\nVui lòng kiểm tra lại tên tour hoặc liên hệ hotline 0332510486 để được hỗ trợ."
                    warnings.append("Tour not found in database")
        
        elif question_type == QuestionType.GENERAL_INFO:
            response_strategy = "field_specific_info"
            
            if field_name and field_confidence > 0.5 and tour_indices:
                # Xử lý câu hỏi về field cụ thể
                field_responses = []
                
                for idx in tour_indices[:3]:  # Hiển thị cho 3 tour đầu
                    tour = tours_db.get(idx)
                    if tour:
                        field_value = EnhancedFieldDetectorV2.get_field_value(tour, field_name)
                        
                        if field_value:
                            if isinstance(field_value, list):
                                if field_value:
                                    # Format list thành string đẹp
                                    if len(field_value) <= 5:
                                        value_str = ", ".join(field_value)
                                    else:
                                        value_str = ", ".join(field_value[:5]) + f" và {len(field_value) - 5} mục khác"
                                else:
                                    value_str = "Không có thông tin"
                            else:
                                value_str = str(field_value)
                            
                            field_responses.append(f"**{tour.tour_name}:** {value_str}")
                
                if field_responses:
                    # Sử dụng field-specific template nếu có
                    template_key = f'field_{field_name}'
                    if template_key in KnowledgeTemplateSystem.TEMPLATES:
                        primary_tour = tours_db.get(tour_indices[0])
                        if primary_tour:
                            field_value = EnhancedFieldDetectorV2.get_field_value(primary_tour, field_name)
                            
                            if isinstance(field_value, list):
                                includes_formatted = "\n".join([f"• {item}" for item in field_value])
                                reply = KnowledgeTemplateSystem.render(template_key,
                                    tour_name=primary_tour.tour_name,
                                    **{field_name: field_value},
                                    includes_formatted=includes_formatted
                                )
                            else:
                                reply = KnowledgeTemplateSystem.render(template_key,
                                    tour_name=primary_tour.tour_name,
                                    **{field_name: field_value}
                                )
                    else:
                        # Fallback to general field response
                        field_display_name = field_name.replace('_', ' ').title()
                        reply = f"📋 **THÔNG TIN {field_display_name.upper()}**\n\n"
                        reply += "\n\n".join(field_responses)
                        
                        # Thêm giải thích về field nếu cần
                        field_explanations = {
                            'includes': "Các dịch vụ đã bao gồm trong giá tour.",
                            'price': "Giá tour đã bao gồm thuế và phí dịch vụ.",
                            'duration': "Thời gian tính từ lúc khởi hành đến khi kết thúc.",
                            'style': "Phong cách và loại hình của tour."
                        }
                        
                        if field_name in field_explanations:
                            reply += f"\n\n💡 **Lưu ý:** {field_explanations[field_name]}"
                else:
                    reply = f"Không tìm thấy thông tin về **{field_name}** cho các tour được đề cập."
            else:
                # General information request
                response_strategy = "general_info_fallback"
                
                available_fields = [
                    "tên tour (tour_name)",
                    "giá (price)", 
                    "thời gian (duration)",
                    "địa điểm (location)",
                    "dịch vụ bao gồm (includes)",
                    "phong cách (style)",
                    "phương tiện (transport)",
                    "chỗ ở (accommodation)",
                    "ăn uống (meals)",
                    "ghi chú (notes)"
                ]
                
                reply = KnowledgeTemplateSystem.render('general_fallback',
                    user_message=user_message,
                    available_fields=", ".join(available_fields[:5]) + ", ...",
                    suggestion="Vui lòng hỏi cụ thể về một field hoặc một tour nhất định."
                )
        
        elif question_type == QuestionType.COMPARISON:
            response_strategy = "tour_comparison"
            
            if len(tour_indices) >= 2:
                # So sánh 2-3 tour
                comparison_tours = []
                for idx in tour_indices[:3]:
                    tour = tours_db.get(idx)
                    if tour:
                        comparison_tours.append(tour)
                
                if len(comparison_tours) >= 2:
                    # Tạo bảng so sánh chi tiết
                    comparison_rows = []
                    
                    for i, tour in enumerate(comparison_tours, 1):
                        # Format includes cho ngắn gọn
                        includes_preview = ", ".join(tour.includes[:2])
                        if len(tour.includes) > 2:
                            includes_preview += f" (+{len(tour.includes) - 2})"
                        
                        row = f"**{i}. {tour.tour_name}**\n"
                        row += f"   📍 **Địa điểm:** {tour.location}\n"
                        row += f"   ⏱ **Thời gian:** {tour.duration}\n"
                        row += f"   💰 **Giá:** {tour.price}\n"
                        row += f"   🎨 **Phong cách:** {tour.style or 'Đa dạng'}\n"
                        row += f"   ✅ **Bao gồm:** {includes_preview}\n"
                        
                        # Thêm điểm đặc biệt nếu có
                        special_features = []
                        if tour.event_support and "có" in tour.event_support.lower():
                            special_features.append("Hỗ trợ sự kiện")
                        if tour.accommodation and "resort" in tour.accommodation.lower():
                            special_features.append("Resort cao cấp")
                        
                        if special_features:
                            row += f"   ✨ **Đặc điểm:** {', '.join(special_features)}"
                        
                        comparison_rows.append(row)
                    
                    # Phân tích điểm khác biệt
                    differences = []
                    
                    if len(comparison_tours) == 2:
                        t1, t2 = comparison_tours[0], comparison_tours[1]
                        
                        # So sánh giá
                        if t1.price_numeric and t2.price_numeric:
                            price_diff = abs(t1.price_numeric - t2.price_numeric)
                            if price_diff > 500000:  # Chênh lệch > 500k
                                cheaper = t1 if t1.price_numeric < t2.price_numeric else t2
                                expensive = t2 if cheaper == t1 else t1
                                differences.append(f"💰 **Giá cả:** {cheaper.tour_name} rẻ hơn {expensive.tour_name} khoảng {price_diff/1000000:.1f} triệu VND")
                        
                        # So sánh thời gian
                        if t1.duration_numeric and t2.duration_numeric:
                            if t1.duration_numeric != t2.duration_numeric:
                                differences.append(f"⏱ **Thời gian:** {t1.tour_name} ({t1.duration}) vs {t2.tour_name} ({t2.duration})")
                        
                        # So sánh phong cách
                        if t1.style != t2.style:
                            differences.append(f"🎨 **Phong cách:** {t1.tour_name} ({t1.style}) vs {t2.tour_name} ({t2.style})")
                        
                        # So sánh địa điểm
                        if t1.location != t2.location:
                            differences.append(f"📍 **Địa điểm:** {t1.tour_name} ({t1.location}) vs {t2.tour_name} ({t2.location})")
                    
                    # Tạo đề xuất thông minh
                    suggestion = ""
                    if comparison_tours:
                        # Dựa vào semantic profile nếu có
                        if semantic_profile:
                            if semantic_profile.get('preferred_budget') == 'low':
                                # Tìm tour rẻ nhất
                                cheapest_tour = min(comparison_tours, 
                                                   key=lambda t: t.price_numeric or float('inf'))
                                suggestion = f"Với ngân sách thấp, nên chọn **{cheapest_tour.tour_name}**."
                            elif semantic_profile.get('preferred_duration') == 'short':
                                # Tìm tour ngắn nhất
                                shortest_tour = min(comparison_tours,
                                                   key=lambda t: t.duration_numeric or float('inf'))
                                suggestion = f"Với thời gian hạn chế, nên chọn **{shortest_tour.tour_name}**."
                            else:
                                suggestion = "Nên chọn tour phù hợp nhất với sở thích và điều kiện của bạn."
                        else:
                            suggestion = "💡 **Gợi ý:** Chọn tour phù hợp với ngân sách, thời gian và sở thích cá nhân."
                    
                    reply = KnowledgeTemplateSystem.render('comparison',
                        count=len(comparison_tours),
                        comparison_table="\n\n".join(comparison_rows),
                        summary="\n".join(differences) if differences else "Các tour có chất lượng dịch vụ tương đương, khác biệt chủ yếu về phong cách và địa điểm.",
                        suggestion=suggestion
                    )
                    
                    response_metadata["compared_tours"] = [tour.id for tour in comparison_tours]
                    response_metadata["comparison_points"] = len(differences)
                else:
                    reply = "❌ **Cần ít nhất 2 tour để so sánh.**\n\nVui lòng chỉ định tên tour cụ thể (ví dụ: 'so sánh tour A và tour B')."
            else:
                reply = "❌ **Không đủ thông tin để so sánh.**\n\nVui lòng cung cấp tên ít nhất 2 tour hoặc mô tả rõ hơn về các tour bạn muốn so sánh."
        
        elif question_type == QuestionType.RECOMMENDATION:
            response_strategy = "smart_recommendation"
            
            if not tour_indices:
                # Thử semantic recommendation nếu không có kết quả tìm kiếm
                if semantic_profile and UpgradeFlags.is_enabled("8_SEMANTIC_ANALYSIS"):
                    try:
                        semantic_recommendations = SemanticAnalyzer.match_tours_to_profile(semantic_profile, tours_db)
                        if semantic_recommendations:
                            tour_indices = [idx for idx, score in semantic_recommendations[:5]]
                            logger.info(f"Semantic recommendations: {len(tour_indices)} tours")
                            response_metadata["recommendation_source"] = "semantic_analysis"
                    except Exception as e:
                        logger.error(f"Semantic recommendation error: {e}")
            
            if tour_indices:
                # Lấy tour được đề xuất cao nhất
                primary_idx = tour_indices[0]
                primary_tour = tours_db.get(primary_idx)
                
                if primary_tour:
                    # Tìm lý do đề xuất thông minh
                    recommendation_reasons = []
                    
                    # Reason 1: Phù hợp với filters
                    if filters.location and filters.location.lower() in primary_tour.location.lower():
                        recommendation_reasons.append(f"📍 **Địa điểm phù hợp:** {primary_tour.location}")
                    
                    if filters.style and filters.style.lower() in primary_tour.style.lower():
                        recommendation_reasons.append(f"🎨 **Phong cách phù hợp:** {primary_tour.style}")
                    
                    if filters.include_keywords:
                        matched_includes = [inc for inc in filters.include_keywords 
                                          if any(inc in tour_inc.lower() for tour_inc in primary_tour.includes)]
                        if matched_includes:
                            recommendation_reasons.append(f"✅ **Có dịch vụ bạn cần:** {', '.join(matched_includes)}")
                    
                    # Reason 2: Phù hợp với semantic profile
                    if semantic_profile:
                        if semantic_profile.get('preferred_budget') == 'low' and primary_tour.price_numeric and primary_tour.price_numeric < 2000000:
                            recommendation_reasons.append("💰 **Ngân sách phù hợp:** Giá tour dưới 2 triệu")
                        elif semantic_profile.get('preferred_budget') == 'high' and primary_tour.price_numeric and primary_tour.price_numeric > 3000000:
                            recommendation_reasons.append("💰 **Dịch vụ cao cấp:** Giá tour trên 3 triệu")
                    
                    # Reason 3: Ưu điểm của tour
                    if not recommendation_reasons:
                        # Default reasons based on tour features
                        if primary_tour.rating and primary_tour.rating >= 4.5:
                            recommendation_reasons.append("⭐ **Đánh giá xuất sắc:** 4.5/5 từ khách hàng")
                        
                        if primary_tour.includes and len(primary_tour.includes) >= 5:
                            recommendation_reasons.append("✅ **Nhiều dịch vụ bao gồm:** Đầy đủ tiện nghi")
                        
                        if primary_tour.duration_numeric and 2 <= primary_tour.duration_numeric <= 4:
                            recommendation_reasons.append("⏱ **Thời gian lý tưởng:** 2-4 ngày phù hợp cho kỳ nghỉ")
                    
                    # Tìm alternatives
                    alternative_tours = []
                    for idx in tour_indices[1:4]:
                        tour = tours_db.get(idx)
                        if tour:
                            alt_text = f"• **{tour.tour_name}**"
                            if tour.duration:
                                alt_text += f" ({tour.duration})"
                            if tour.price:
                                price_preview = tour.price[:40] + "..." if len(tour.price) > 40 else tour.price
                                alt_text += f" - {price_preview}"
                            alternative_tours.append(alt_text)
                    
                    # Tạo recommended tour display
                    recommended_tour_display = KnowledgeTemplateSystem.render('tour_item',
                        idx=1,
                        tour_name=primary_tour.tour_name,
                        location=primary_tour.location,
                        duration=primary_tour.duration,
                        price=primary_tour.price,
                        summary=(primary_tour.summary[:100] + "...") if len(primary_tour.summary) > 100 else primary_tour.summary,
                        includes_preview=", ".join(primary_tour.includes[:3]) if primary_tour.includes else "Nhiều dịch vụ"
                    )
                    
                    reply = KnowledgeTemplateSystem.render('recommendation',
                        recommended_tour=recommended_tour_display,
                        reasons="\n".join(recommendation_reasons),
                        alternatives="\n".join(alternative_tours) if alternative_tours else "• Liên hệ hotline để được tư vấn thêm các lựa chọn khác",
                        personal_note="Dựa trên phân tích nhu cầu của bạn, tôi tin rây đây là lựa chọn tốt nhất."
                    )
                    
                    # Cập nhật context
                    context.last_recommended_tours = tour_indices[:3]
                    response_metadata["recommendation_reasons"] = recommendation_reasons
                    response_metadata["recommendation_score"] = "high" if len(recommendation_reasons) >= 3 else "medium"
                else:
                    reply = KnowledgeTemplateSystem.render('no_results')
            else:
                reply = KnowledgeTemplateSystem.render('no_results')
        
        elif question_type == QuestionType.UNKNOWN:
            response_strategy = "llm_fallback"
            
            # Sử dụng LLM fallback với context phong phú
            try:
                # Chuẩn bị rich context cho LLM
                llm_context = {
                    "user_message": user_message,
                    "detected_intent": "unknown",
                    "conversation_history": context.conversation_history[-3:],
                    "available_tours_count": len(tours_db),
                    "relevant_tours_found": len(tour_indices),
                    "filters_applied": filter_applied,
                    "field_detected": field_name,
                    "complexity_level": complexity_level,
                    "user_preferences": context.user_preferences
                }
                
                # Thêm thông tin về tours nếu có
                if tour_indices:
                    tours_info = []
                    for idx in tour_indices[:3]:
                        tour = tours_db.get(idx)
                        if tour:
                            tours_info.append({
                                "name": tour.tour_name,
                                "summary": tour.summary[:150],
                                "price": tour.price,
                                "duration": tour.duration
                            })
                    llm_context["relevant_tours"] = tours_info
                
                # Tạo prompt thông minh
                prompt = _prepare_llm_prompt(user_message, [], llm_context)
                
                # Gọi LLM với timeout
                llm_timeout = 10  # seconds
                llm_response = ""
                
                try:
                    llm_request_obj = LLMRequest(
                        prompt=prompt,
                        model="llama2",
                        temperature=0.7,
                        max_tokens=500,
                        stream=False
                    )
                    
                    # Trong thực tế, đây là nơi gọi LLM API
                    # llm_response = call_llm_api(llm_request_obj, timeout=llm_timeout)
                    
                    # Tạm thời dùng fallback response
                    llm_response = _generate_fallback_response(user_message, [], tour_indices)
                    
                except TimeoutError:
                    logger.warning(f"LLM timeout after {llm_timeout} seconds")
                    llm_response = _generate_fallback_response(user_message, [], tour_indices)
                except Exception as e:
                    logger.error(f"LLM API error: {e}")
                    llm_response = _generate_fallback_response(user_message, [], tour_indices)
                
                if llm_response:
                    # Parse và clean response
                    parsed_response = parse_llm_response(llm_response)
                    reply = parsed_response.get("reply", "").strip()
                    
                    # Auto-validation
                    if UpgradeFlags.is_enabled("9_AUTO_VALIDATION"):
                        validated = AutoValidator.validate_response(reply)
                        reply = validated
                    
                    # Đảm bảo response có chất lượng
                    if len(reply) < 80 or "xin chào" in reply.lower() and "tour" not in reply.lower():
                        # Fallback nếu response quá ngắn hoặc không liên quan
                        reply = _generate_fallback_response(user_message, [], tour_indices)
                    
                    response_metadata["llm_used"] = True
                    response_metadata["llm_model"] = "llama2"
                    response_metadata["llm_fallback"] = True
                else:
                    reply = _generate_fallback_response(user_message, [], tour_indices)
                    response_metadata["llm_failed"] = True
                    
            except Exception as e:
                logger.error(f"LLM fallback system error: {e}")
                reply = _generate_fallback_response(user_message, [], tour_indices)
                response_metadata["error"] = str(e)[:100]
        
        # 5.2 Post-process response
        if reply:
            # Auto-validation
            if UpgradeFlags.is_enabled("9_AUTO_VALIDATION"):
                try:
                    validation_result = AutoValidator.safe_validate({
                        "reply": reply,
                        "tour_indices": tour_indices,
                        "question_type": question_type.value
                    })
                    
                    reply = validation_result.get("reply", reply)
                    
                    if validation_result.get("warnings"):
                        warnings.extend(validation_result["warnings"])
                    
                    if validation_result.get("suggestions"):
                        suggestions.extend(validation_result["suggestions"])
                        
                except Exception as e:
                    logger.error(f"Auto-validation error: {e}")
            
            # Ensure contact information is present
            if not any(keyword in reply.lower() for keyword in ["0332510486", "hotline", "liên hệ", "điện thoại"]):
                reply += "\n\n📞 **Hotline tư vấn 24/7: 0332510486**"
                response_metadata["contact_added"] = True
            
            if not any(keyword in reply.lower() for keyword in ["rubywings.vn", "website", "trang web"]):
                reply += "\n🌐 **Website chính thức: www.rubywings.vn**"
                response_metadata["website_added"] = True
            
            # Add filter summary if filters were applied
            if filter_applied and filters:
                filter_summary_parts = []
                if filters.location:
                    filter_summary_parts.append(f"📍 {filters.location}")
                if filters.style:
                    filter_summary_parts.append(f"🎨 {filters.style}")
                if filters.min_price or filters.max_price:
                    price_range = []
                    if filters.min_price:
                        price_range.append(f"từ {filters.min_price:,.0f} VND")
                    if filters.max_price:
                        price_range.append(f"đến {filters.max_price:,.0f} VND")
                    if price_range:
                        filter_summary_parts.append(f"💰 {' '.join(price_range)}")
                
                if filter_summary_parts:
                    reply += f"\n\n🔍 **Bộ lọc đã áp dụng:** {', '.join(filter_summary_parts)}"
            
            # Add context-aware follow-up suggestions
            if len(tour_indices) > 0 and question_type not in [QuestionType.FAREWELL, QuestionType.GREETING]:
                # Tạo follow-up questions dựa trên context
                follow_up_suggestions = []
                
                if question_type == QuestionType.LIST_TOURS:
                    if len(tour_indices) > 1:
                        follow_up_suggestions.append("• 'So sánh tour 1 và tour 2'")
                    follow_up_suggestions.append("• 'Tour 1 giá bao nhiêu?'")
                
                elif question_type == QuestionType.TOUR_DETAIL:
                    primary_tour = tours_db.get(tour_indices[0]) if tour_indices else None
                    if primary_tour:
                        follow_up_suggestions.append(f"• 'Tour {primary_tour.tour_name} có những dịch vụ gì?'")
                        follow_up_suggestions.append("• 'Có tour tương tự nào không?'")
                
                if follow_up_suggestions:
                    reply += f"\n\n💡 **Bạn cũng có thể hỏi:**\n" + "\n".join(follow_up_suggestions)
            
            # Format và clean up response
            # Remove excessive empty lines
            import re
            reply = re.sub(r'\n{3,}', '\n\n', reply)
            
            # Ensure proper spacing
            reply = reply.strip()
            
            # Truncate if too long (rare case)
            max_response_length = 4000
            if len(reply) > max_response_length:
                logger.warning(f"Response too long: {len(reply)} chars, truncating...")
                
                # Try to cut at a paragraph boundary
                last_paragraph = reply.rfind('\n\n', 0, max_response_length - 200)
                if last_paragraph > max_response_length // 2:
                    reply = reply[:last_paragraph] + "\n\n📞 **Thông tin còn tiếp. Vui lòng liên hệ hotline 0332510486 để biết thêm chi tiết.**"
                else:
                    reply = reply[:max_response_length - 200] + "...\n\n📞 **Vui lòng liên hệ hotline để biết thêm chi tiết.**"
        
        response_time = int((time.time() - response_start_time) * 1000)
        response_metadata["response_generation_time_ms"] = response_time
        response_metadata["response_strategy"] = response_strategy
        
        logger.info(f"Response generated in {response_time}ms using strategy: {response_strategy}")
        
        # ========== PHASE 6: POST-PROCESSING & UPDATES ==========
        processing_phase = "post_processing"
        
        # 6.1 Update conversation state machine
        state_machine = ConversationStateMachine(context.current_state)
        state_machine.update(user_message, reply[:100] + "...", tour_indices)
        context.current_state = state_machine.current_state
        
        # 6.2 Add assistant response to conversation history
        assistant_entry = {
            "role": "assistant",
            "message": reply[:500] + "..." if len(reply) > 500 else reply,
            "timestamp": datetime.now().isoformat(),
            "tour_indices": tour_indices[:5],
            "question_type": question_type.value,
            "response_strategy": response_strategy,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
        context.conversation_history.append(assistant_entry)
        
        # 6.3 Update last tours mentioned
        if tour_indices:
            context.last_tours_mentioned = tour_indices[:5]
        
        # 6.4 Update user preferences based on this interaction
        if question_type in [QuestionType.RECOMMENDATION, QuestionType.LIST_TOURS]:
            # Ghi nhận loại tour user quan tâm
            if tour_indices:
                tour_categories = []
                for idx in tour_indices[:3]:
                    tour = tours_db.get(idx)
                    if tour and tour.category:
                        if tour.category not in tour_categories:
                            tour_categories.append(tour.category)
                
                if tour_categories:
                    context.user_preferences['interested_categories'] = list(set(
                        context.user_preferences.get('interested_categories', []) + tour_categories
                    ))[:5]
        
        # 6.5 Save to cache
        if UpgradeFlags.is_enabled("ENABLE_CACHING") and cache_key:
            try:
                cache_entry = {
                    "reply": reply,
                    "tour_indices": tour_indices,
                    "context": {
                        "session_id": session_id,
                        "current_state": context.current_state.value,
                        "last_tours_mentioned": context.last_tours_mentioned[:3]
                    },
                    "metadata": response_metadata,
                    "warnings": warnings if warnings else None,
                    "suggestions": suggestions if suggestions else None,
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "cached_at": time.time(),
                    "expiry": time.time() + CACHE_TTL
                }
                
                cache_system.set(cache_key, cache_entry, expiry=CACHE_TTL)
                logger.info(f"Response cached with key: {cache_key[:25]}... (expires in {CACHE_TTL}s)")
            except Exception as e:
                logger.error(f"Caching error: {e}")
        
        # 6.6 Save session context
        save_session_context(session_id, context)
        
        # 6.7 Send CAPI event if enabled
        if CAPI_ENABLED:
            try:
                capi_payload = {
                    "session_id": session_id,
                    "user_message": user_message[:200],
                    "bot_response": reply[:300],
                    "tour_count": len(tour_indices),
                    "question_type": question_type.value,
                    "timestamp": datetime.now().isoformat()
                }
                
                send_capi_event(session_id, user_message[:200], reply[:300])
                logger.info(f"CAPI event sent for session {session_id}")
            except Exception as e:
                logger.error(f"CAPI event error: {e}")
        
        # 6.8 Cleanup old sessions periodically
        if random.random() < 0.1:  # 10% chance on each request
            cleanup_expired_sessions()
        
        # ========== PHASE 7: FINAL RESPONSE PREPARATION ==========
        processing_phase = "final_preparation"
        
        total_processing_time = int((time.time() - start_time) * 1000)
        
        # 7.1 Prepare final response object
        final_response = {
            "reply": reply,
            "tour_indices": tour_indices,
            "action": "continue",
            "context": {
                "session_id": session_id,
                "current_state": context.current_state.value,
                "question_type": question_type.value,
                "tours_found": len(tour_indices),
                "processing_time_ms": total_processing_time,
                "conversation_length": len(context.conversation_history)
            },
            "warnings": warnings if warnings else None,
            "suggestions": suggestions if suggestions else None,
            "metadata": {
                **response_metadata,
                "total_processing_time_ms": total_processing_time,
                "cache_hit": False,
                "system_version": "RubyWings AI v4.2",
                "knowledge_base_version": "knowledge.json v2.0",
                "processing_phases": [
                    "request_processing",
                    "session_management", 
                    "question_analysis",
                    "tour_search",
                    "response_generation",
                    "post_processing",
                    "final_preparation"
                ],
                "performance_metrics": {
                    "question_analysis_ms": question_analysis_time,
                    "filter_analysis_ms": filter_analysis_time,
                    "field_analysis_ms": field_analysis_time,
                    "semantic_analysis_ms": semantic_analysis_time,
                    "search_total_ms": search_metadata.get("total_time_ms", 0),
                    "response_generation_ms": response_time,
                    "total_ms": total_processing_time
                }
            }
        }
        
        # 7.2 Log completion
        logger.info(f"""
        ✅ CHAT ENDPOINT PROCESSING COMPLETE
        ⏱  Total time: {total_processing_time}ms
        👤 Session: {session_id[:12]}...
        ❓ Question: {question_type.value} (confidence: {q_confidence:.2f})
        🗺️  Tours found: {len(tour_indices)}
        🔍 Search strategies: {', '.join(search_strategies)}
        🎯 Response strategy: {response_strategy}
        📊 Response length: {len(reply)} characters
        ⚠️  Warnings: {len(warnings) if warnings else 0}
        💡 Suggestions: {len(suggestions) if suggestions else 0}
        """)
        
        # 7.3 Return final response
        return jsonify(final_response)
        
    except Exception as e:
        # ========== PHASE 8: ERROR HANDLING ==========
        error_time = time.time()
        total_processing_time = int((error_time - start_time) * 1000)
        
        logger.critical(f"""
        ❌ CRITICAL ERROR in chat endpoint
        Phase: {processing_phase}
        Error: {str(e)}
        Traceback: {traceback.format_exc()}
        Session ID: {session_id or 'Unknown'}
        User message: {user_message[:200] if user_message else 'Empty'}
        Processing time: {total_processing_time}ms
        """)
        
        # Prepare comprehensive error response
        error_id = hashlib.md5(f"{str(e)}{time.time()}".encode()).hexdigest()[:8]
        
        error_reply = f"""⚡ **XIN LỖI VÌ SỰ BẤT TIỆN**

Hệ thống gặp sự cố kỹ thuật khi xử lý yêu cầu của bạn. Đội ngũ kỹ thuật đã được thông báo.

**MÃ LỖI:** RW-{error_id}
**THỜI GIAN:** {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
**TÌNH TRẠNG:** Đang khắc phục

**VUI LÒNG THỬ MỘT TRONG CÁC CÁCH SAU:**

1. **📞 GỌI NGAY HOTLINE:** 0332510486
   • Tư vấn trực tiếp, nhanh chóng
   • Hỗ trợ 24/7, kể cả cuối tuần

2. **🌐 TRUY CẬP WEBSITE:** www.rubywings.vn
   • Xem danh sách tour đầy đủ
   • Đặt tour trực tuyến
   • Tìm hiểu thông tin chi tiết

3. **📱 LIÊN HỆ QUA ZALO:** @rubywings
   • Chat với nhân viên tư vấn
   • Nhận báo giá nhanh

4. **🔄 THỬ LẠI CÂU HỎI ĐƠN GIẢN HƠN:**
   • "Tour Bạch Mã giá bao nhiêu?"
   • "Có tour nào đi Huế 2 ngày không?"
   • "Tour gia đình phù hợp cho trẻ em"

**THÔNG TIN KỸ THUẬT (DÀNH CHO KỸ THUẬT VIÊN):**
• Lỗi: {type(e).__name__}
• Pha lỗi: {processing_phase}
• Thời gian xử lý: {total_processing_time}ms
• Session: {session_id or 'N/A'}

Chúng tôi chân thành xin lỗi vì sự cố này và đang nỗ lực khắc phục trong thời gian sớm nhất."""

        # Prepare error response object
        error_response = {
            "reply": error_reply,
            "tour_indices": [],
            "action": "error",
            "context": {
                "session_id": session_id or generate_session_id(),
                "error": True,
                "error_id": f"RW-{error_id}",
                "error_type": type(e).__name__,
                "processing_phase": processing_phase,
                "processing_time_ms": total_processing_time
            },
            "warnings": ["system_error", "technical_issue", "please_contact_support"],
            "suggestions": [
                "Thử lại với câu hỏi đơn giản hơn",
                "Gọi hotline 0332510486 để được hỗ trợ ngay",
                "Truy cập website www.rubywings.vn"
            ],
            "metadata": {
                "error_details": str(e)[:500],
                "error_timestamp": datetime.now().isoformat(),
                "system_status": "degraded",
                "recommended_action": "contact_support",
                "support_channels": ["hotline: 0332510486", "website: www.rubywings.vn", "zalo: @rubywings"]
            }
        }
        
        # Try to save error to error log
        try:
            error_log_entry = {
                "error_id": f"RW-{error_id}",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "processing_phase": processing_phase,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "user_message": user_message[:500] if user_message else "",
                "traceback": traceback.format_exc()[:1000],
                "processing_time_ms": total_processing_time,
                "system_version": "RubyWings AI v4.2"
            }
            
            # In production, this would save to a database or error tracking service
            logger.critical(f"ERROR LOG ENTRY: {json.dumps(error_log_entry, ensure_ascii=False)}")
        except:
            pass
        
        return jsonify(error_response), 500

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