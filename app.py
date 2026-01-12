# app.py - Ruby Wings Chatbot v5.2 (Enhanced with Location Filter, State Machine & Improved Intents)
# =========== IMPORTS & CONFIGURATION ===========
import os
import sys
import json
import time
import threading
import logging
import re
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache
from collections import defaultdict
from enum import Enum

# ===== Platform detection =====
import platform
IS_WINDOWS = platform.system().lower().startswith("win")

# ===== Flask =====
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# ===== Safe Gunicorn Import (Linux only) =====
if not IS_WINDOWS:
    try:
        from gunicorn.app.base import BaseApplication
    except Exception:
        BaseApplication = None
else:
    BaseApplication = None

# ===== Memory optimization =====
import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ruby-wings-enhanced")

# =========== ENVIRONMENT VARIABLES ===========
# RAM Profile
RAM_PROFILE = os.environ.get("RAM_PROFILE", "512").strip()
IS_LOW_RAM = RAM_PROFILE == "512"
logger.info(f"🧠 RAM Profile: {RAM_PROFILE}MB | Low RAM: {IS_LOW_RAM}")

# Feature Toggles
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "false").lower() == "true" and not IS_LOW_RAM
ENABLE_INTENT_DETECTION = os.environ.get("ENABLE_INTENT_DETECTION", "true").lower() == "true"
ENABLE_PHONE_DETECTION = os.environ.get("ENABLE_PHONE_DETECTION", "true").lower() == "true"
ENABLE_LEAD_CAPTURE = os.environ.get("ENABLE_LEAD_CAPTURE", "true").lower() == "true"
ENABLE_LLM_FALLBACK = os.environ.get("ENABLE_LLM_FALLBACK", "true").lower() == "true"
ENABLE_CACHING = os.environ.get("ENABLE_CACHING", "true").lower() == "true" and not IS_LOW_RAM
ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "true").lower() == "true"
ENABLE_META_CAPI = os.environ.get("ENABLE_META_CAPI", "true").lower() == "true"

# Core Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
MAX_CACHE_SIZE = 50 if IS_LOW_RAM else 200

# =========== LAZY IMPORTS ===========
# Import only what's needed based on feature toggles
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️ NumPy not available")

# Import FAISS only if enabled
FAISS_AVAILABLE = False
if FAISS_ENABLED:
    try:
        import faiss
        FAISS_AVAILABLE = True
        logger.info("✅ FAISS loaded")
    except ImportError:
        logger.warning("⚠️ FAISS not available")

# Import OpenAI only if key exists
OPENAI_AVAILABLE = False
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        logger.info("✅ OpenAI loaded")
    except ImportError as e:
        logger.error(f"❌ OpenAI import error: {e}")
        OPENAI_AVAILABLE = False

# Import entities (required)
try:
    from entities import (
        Intent, detect_intent, detect_phone_number,
        Tour, UserProfile, ConversationContext, LeadData,
        EnhancedJSONEncoder
    )
    logger.info("✅ Entities loaded")
except ImportError as e:
    logger.error(f"❌ Failed to import entities: {e}")
    sys.exit(1)

# =========== FLASK APP ===========
app = Flask(__name__)
app.json_encoder = EnhancedJSONEncoder
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

# =========== LOCATION REGION MAPPING ===========
# BƯỚC 3: Thêm mapping khu vực để đề xuất tour tương đương
LOCATION_REGION_MAPPING = {
    "đà nẵng": "Miền Trung",
    "huế": "Miền Trung",
    "quảng trị": "Miền Trung",
    "bạch mã": "Miền Trung",
    "hà nội": "Miền Bắc",
    "hạ long": "Miền Bắc",
    "sapa": "Miền Bắc",
    "hồ chí minh": "Miền Nam",
    "cần thơ": "Miền Nam",
    "phú quốc": "Miền Nam",
    "nha trang": "Miền Nam"
}

REGION_TOURS = {
    "Miền Trung": ["Bạch Mã", "Huế", "Quảng Trị", "Đà Nẵng", "Hội An"],
    "Miền Bắc": ["Hà Nội", "Hạ Long", "Sapa", "Ninh Bình"],
    "Miền Nam": ["TP.HCM", "Cần Thơ", "Phú Quốc", "Nha Trang", "Đà Lạt"]
}

# =========== STATE MACHINE CONSTANTS ===========
# BƯỚC 4: Thêm state machine
class ConversationStage(Enum):
    EXPLORE = "explore"
    SUGGEST = "suggest"
    COMPARE = "compare"
    SELECT = "select"
    BOOK = "book"
    LEAD = "lead"
    CALLBACK = "callback"

# =========== MEMORY OPTIMIZED GLOBAL STATE ===========
class MemoryOptimizedState:
    """Memory-optimized global state management with enhanced session storage"""
    
    def __init__(self):
        self._tours_db = {}  # Dict[int, Tour]
        self._tour_name_to_index = {}
        self._session_contexts = {}
        self._response_cache = {}
        self._embedding_cache = {}
        self._knowledge_loaded = False
        self._index_loaded = False
        self._lock = threading.RLock()
        
        # Memory limits
        self.max_sessions = 100 if IS_LOW_RAM else 500
        self.max_cache_items = MAX_CACHE_SIZE
        self.max_embedding_cache = 50 if IS_LOW_RAM else 200
    
    # Tours DB methods
    def get_tour(self, index: int) -> Optional[Tour]:
        with self._lock:
            return self._tours_db.get(index)
    
    def get_tours_by_indices(self, indices: List[int]) -> List[Tour]:
        with self._lock:
            return [self._tours_db.get(idx) for idx in indices if idx in self._tours_db]
    
    def get_all_tour_indices(self) -> List[int]:
        with self._lock:
            return list(self._tours_db.keys())
    
    def add_tour(self, index: int, tour: Tour):
        with self._lock:
            self._tours_db[index] = tour
    
    def clear_tours(self):
        with self._lock:
            self._tours_db.clear()
            self._tour_name_to_index.clear()
    
    # Tour name lookup
    def find_tour_by_name(self, name: str) -> Optional[int]:
        with self._lock:
            return self._tour_name_to_index.get(name.lower().strip())
    
    def add_tour_name(self, name: str, index: int):
        with self._lock:
            self._tour_name_to_index[name.lower().strip()] = index
    
    # Session management with enhanced state machine
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session with enhanced state machine fields"""
        with self._lock:
            if session_id not in self._session_contexts:
                # BƯỚC 4: Thêm state machine fields vào session
                self._session_contexts[session_id] = {
                    'stage': ConversationStage.EXPLORE.value,
                    'selected_tour_id': None,
                    'lead_phone': None,
                    'last_intent': None,
                    'conversation_history': [],
                    'last_location_filter': None,
                    'last_updated': datetime.utcnow().isoformat(),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Cleanup if too many sessions
                if len(self._session_contexts) > self.max_sessions:
                    self._cleanup_old_sessions()
            
            return self._session_contexts[session_id]
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session fields"""
        with self._lock:
            if session_id in self._session_contexts:
                self._session_contexts[session_id].update(updates)
                self._session_contexts[session_id]['last_updated'] = datetime.utcnow().isoformat()
    
    def update_session_stage(self, session_id: str, new_stage: ConversationStage, metadata: Dict[str, Any] = None):
        """Update session stage with metadata"""
        updates = {
            'stage': new_stage.value,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        if metadata:
            updates.update(metadata)
        
        self.update_session(session_id, updates)
    
    def _cleanup_old_sessions(self):
        """Remove oldest sessions when limit exceeded"""
        if len(self._session_contexts) <= self.max_sessions:
            return
        
        # Sort by last_updated
        sorted_sessions = sorted(
            self._session_contexts.items(),
            key=lambda x: x[1].get('last_updated', '')
        )
        
        # Remove oldest 20%
        remove_count = max(1, len(sorted_sessions) // 5)
        for session_id, _ in sorted_sessions[:remove_count]:
            del self._session_contexts[session_id]
        
        logger.info(f"🧹 Cleaned up {remove_count} old sessions")
    
    # Cache management
    def get_cache(self, key: str):
        with self._lock:
            entry = self._response_cache.get(key)
            if entry:
                if time.time() - entry['timestamp'] < CACHE_TTL_SECONDS:
                    return entry['value']
                else:
                    del self._response_cache[key]
            return None
    
    def set_cache(self, key: str, value: Any):
        with self._lock:
            self._response_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            
            # Cleanup if cache too large
            if len(self._response_cache) > self.max_cache_items:
                sorted_items = sorted(
                    self._response_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                for old_key, _ in sorted_items[:self.max_cache_items//2]:
                    del self._response_cache[old_key]
    
    # Embedding cache
    def get_embedding(self, text: str):
        with self._lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            return self._embedding_cache.get(text_hash)
    
    def set_embedding(self, text: str, embedding: List[float]):
        with self._lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self._embedding_cache[text_hash] = embedding
            
            # Cleanup if cache too large
            if len(self._embedding_cache) > self.max_embedding_cache:
                items = list(self._embedding_cache.items())
                for key, _ in items[:self.max_embedding_cache//2]:
                    del self._embedding_cache[key]

# Initialize global state
state = MemoryOptimizedState()

# =========== KNOWLEDGE LOADING ===========
def load_knowledge_lazy():
    """Lazy load knowledge base with memory optimization"""
    if state._knowledge_loaded:
        return True
    
    try:
        logger.info(f"📚 Loading knowledge from {KNOWLEDGE_PATH}...")
        
        with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        tours_data = knowledge.get('tours', [])
        
        for idx, tour_data in enumerate(tours_data):
            try:
                # Create Tour object from data
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
                    style=tour_data.get('style', '')
                )
                
                state.add_tour(idx, tour)
                
                # Index by name
                if tour.name:
                    state.add_tour_name(tour.name, idx)
                
            except Exception as e:
                logger.error(f"Error loading tour {idx}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(state._tours_db)} tours")
        state._knowledge_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge: {e}")
        return False

# =========== BƯỚC 3: LOCATION FILTER FUNCTIONS ===========
def apply_location_filter(tours: List[Tour], location: str) -> Tuple[List[Tour], str]:
    """
    Apply hard location filter. If no tours found, suggest similar tours.
    Returns: (filtered_tours, response_message)
    """
    if not location:
        return tours, ""
    
    location_lower = location.lower().strip()
    
    # BƯỚC 3: Apply hard filter
    filtered_tours = []
    for tour in tours:
        if tour and tour.location and location_lower in tour.location.lower():
            filtered_tours.append(tour)
    
    if filtered_tours:
        # Found tours at exact location
        return filtered_tours, f"Tìm thấy {len(filtered_tours)} tour tại {location}:"
    
    # BƯỚC 3: No tours found, suggest similar ones
    return find_similar_tours(location_lower, tours)

def find_similar_tours(requested_location: str, all_tours: List[Tour]) -> Tuple[List[Tour], str]:
    """
    Suggest similar tours when no exact location match
    Logic: find tours in same region, similar duration and price range
    """
    # Determine region
    region = None
    for loc, reg in LOCATION_REGION_MAPPING.items():
        if loc in requested_location:
            region = reg
            break
    
    if not region:
        region = "Miền Trung"  # Default
    
    # Get region tours keywords
    region_keywords = REGION_TOURS.get(region, [])
    
    # Find tours in same region
    similar_tours = []
    for tour in all_tours:
        if not tour or not tour.location:
            continue
        
        # Check if tour location contains region keywords
        tour_location_lower = tour.location.lower()
        for keyword in region_keywords:
            if keyword.lower() in tour_location_lower:
                similar_tours.append(tour)
                break
    
    # If still no tours, return some popular tours
    if not similar_tours:
        similar_tours = all_tours[:3]  # Return first 3 tours
        message = f"Không tìm thấy tour tại '{requested_location}'. Dưới đây là một số tour phổ biến của Ruby Wings:"
    else:
        similar_tours = similar_tours[:3]  # Limit to 3 tours
        message = f"Không có tour tại '{requested_location}', bạn có muốn tham khảo các tour tương tự tại {region} không?"
    
    return similar_tours, message

def extract_location_from_query(query: str) -> Optional[str]:
    """
    Extract location from user query using keyword matching
    """
    query_lower = query.lower()
    
    # Check for location keywords
    location_keywords = [
        "đà nẵng", "huế", "quảng trị", "bạch mã", "hà nội", 
        "hạ long", "sapa", "hồ chí minh", "sài gòn", "cần thơ",
        "phú quốc", "nha trang", "hội an", "ninh bình", "đà lạt"
    ]
    
    for location in location_keywords:
        if location in query_lower:
            return location
    
    # Check for "tại", "ở" + location pattern
    location_patterns = [
        r'tại\s+([a-zA-ZÀ-ỹ\s]+)',
        r'ở\s+([a-zA-ZÀ-ỹ\s]+)',
        r'location\s+([a-zA-ZÀ-ỹ\s]+)'
    ]
    
    for pattern in location_patterns:
        matches = re.search(pattern, query_lower)
        if matches:
            location = matches.group(1).strip()
            if location and len(location) > 1:
                return location
    
    return None

# =========== INTENT HANDLERS ===========
class IntentHandler:
    """Handler for new intents with state machine integration"""
    
    @staticmethod
    def handle_provide_phone(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle PROVIDE_PHONE intent with state machine update"""
        phone = metadata.get("phone_number")
        if not phone:
            return {
                "reply": "Bạn vui lòng cung cấp số điện thoại của bạn nhé!",
                "context": context,
                "sources": [],
                "tour_indices": []
            }
        
        # BƯỚC 4: Update state machine
        state.update_session_stage(
            session_id, 
            ConversationStage.LEAD,
            {
                'lead_phone': phone,
                'last_intent': Intent.PROVIDE_PHONE.value
            }
        )
        
        # Save phone to context
        context.user_preferences['phone'] = phone
        context.last_updated = datetime.utcnow()
        
        # Lead capture
        if ENABLE_LEAD_CAPTURE:
            lead_data = LeadData(
                phone=phone,
                source_channel="Chatbot",
                action_type="Provide Phone",
                note="User provided phone number in chat"
            )
            _save_lead_async(lead_data)
        
        return {
            "reply": f"Cảm ơn bạn đã cung cấp số điện thoại {phone}. Đội ngũ Ruby Wings sẽ liên hệ với bạn sớm nhất! 📞",
            "context": context,
            "sources": [],
            "tour_indices": []
        }
    
    @staticmethod
    def handle_callback_request(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle CALLBACK_REQUEST intent with state machine update"""
        phone = metadata.get("phone_number")
        
        if not phone and 'phone' in context.user_preferences:
            phone = context.user_preferences['phone']
        
        # BƯỚC 4: Update state machine
        state.update_session_stage(
            session_id,
            ConversationStage.CALLBACK,
            {
                'lead_phone': phone,
                'last_intent': Intent.CALLBACK_REQUEST.value
            }
        )
        
        if phone:
            # Confirm callback
            reply = f"Đội ngũ Ruby Wings sẽ gọi lại cho bạn số {phone} trong vòng 5-10 phút! 📞"
            
            # Lead capture with callback request
            if ENABLE_LEAD_CAPTURE:
                lead_data = LeadData(
                    phone=phone,
                    source_channel="Chatbot",
                    action_type="Callback Request",
                    note="User requested callback"
                )
                _save_lead_async(lead_data)
        else:
            # Ask for phone
            reply = "Để chúng tôi có thể gọi lại tư vấn cho bạn, vui lòng cung cấp số điện thoại của bạn nhé!"
        
        return {
            "reply": reply,
            "context": context,
            "sources": [],
            "tour_indices": []
        }
    
    @staticmethod
    def handle_booking_confirm(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle BOOKING_CONFIRM intent with state machine update"""
        # BƯỚC 4: Check if we have selected tour
        session_data = state.get_session(session_id)
        selected_tour_id = session_data.get('selected_tour_id')
        
        # Extract booking details if available
        phone = metadata.get("phone_number")
        
        if phone:
            context.user_preferences['phone'] = phone
        
        if selected_tour_id is not None:
            # We have a selected tour, confirm booking for that tour
            tour = state.get_tour(selected_tour_id)
            tour_name = tour.name if tour else "tour đã chọn"
            
            reply = f"Cảm ơn bạn đã xác nhận đặt {tour_name}! Đội ngũ Ruby Wings sẽ liên hệ với bạn sớm để hoàn tất thủ tục. 📞"
            
            # Update state machine
            state.update_session_stage(
                session_id,
                ConversationStage.BOOK,
                {
                    'last_intent': Intent.BOOKING_CONFIRM.value
                }
            )
        else:
            # Ask for booking details
            reply = "Cảm ơn bạn đã quan tâm đặt tour Ruby Wings! Để xác nhận booking, vui lòng cung cấp:\n\n"
            reply += "1. Họ tên của bạn\n"
            reply += "2. Số lượng người tham gia\n"
            reply += "3. Ngày dự kiến đi\n"
            reply += "4. Tour bạn quan tâm (nếu có)\n\n"
            reply += "Hoặc gọi ngay hotline 0332510486 để được hỗ trợ nhanh nhất! 📞"
        
        return {
            "reply": reply,
            "context": context,
            "sources": [],
            "tour_indices": []
        }
    
    @staticmethod
    def handle_modify_request(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle MODIFY_REQUEST intent"""
        phone = metadata.get("phone_number")
        
        if phone:
            context.user_preferences['phone'] = phone
        
        reply = "Để hỗ trợ bạn thay đổi/thông tin booking, vui lòng cung cấp:\n\n"
        reply += "1. Mã booking hoặc số điện thoại đặt tour\n"
        reply += "2. Loại thay đổi bạn cần (ngày, số người, hủy tour...)\n"
        reply += "3. Thông tin mới (nếu có)\n\n"
        reply += "Hoặc liên hệ hotline 0332510486 để được hỗ trợ ngay! 📞"
        
        return {
            "reply": reply,
            "context": context,
            "sources": [],
            "tour_indices": []
        }
    
    @staticmethod
    def handle_smalltalk(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle SMALLTALK intent"""
        message_lower = metadata.get("original_message", "").lower()
        
        if any(word in message_lower for word in ["cảm ơn", "thanks", "thank you"]):
            reply = "Cảm ơn bạn! Rất vui được hỗ trợ bạn. Bạn cần tư vấn thêm về tour nào không? 😊"
        elif any(word in message_lower for word in ["chào", "hello", "hi", "xin chào"]):
            reply = "Xin chào! Tôi là trợ lý AI của Ruby Wings. Tôi có thể giúp gì cho bạn về các tour trải nghiệm? 🌿"
        elif "khỏe không" in message_lower:
            reply = "Cảm ơn bạn! Tôi vẫn khỏe và sẵn sàng hỗ trợ bạn. Bạn có khỏe không? Cần tư vấn tour nào không? 😊"
        else:
            reply = "Rất vui được trò chuyện với bạn! Bạn có cần tư vấn về tour du lịch trải nghiệm Ruby Wings không? 🌟"
        
        return {
            "reply": reply,
            "context": context,
            "sources": [],
            "tour_indices": []
        }
    
    @staticmethod
    def handle_lead_captured(context: ConversationContext, metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle LEAD_CAPTURED intent with state machine update"""
        phone = metadata.get("phone_number")
        has_phone = metadata.get("has_phone", False)
        
        if phone:
            context.user_preferences['phone'] = phone
            has_phone = True
        
        # BƯỚC 4: Update state machine
        state.update_session_stage(
            session_id,
            ConversationStage.LEAD,
            {
                'lead_phone': phone,
                'last_intent': Intent.LEAD_CAPTURED.value
            }
        )
        
        # Lead capture
        if ENABLE_LEAD_CAPTURE and has_phone:
            lead_data = LeadData(
                phone=phone or "Unknown",
                source_channel="Chatbot",
                action_type="Lead Captured",
                service_interest="Tour Inquiry",
                note="User showed interest in tours"
            )
            _save_lead_async(lead_data)
        
        # Transition to tour recommendation
        if has_phone:
            reply = "Cảm ơn bạn đã quan tâm đến Ruby Wings! Dưới đây là một số tour phổ biến:\n\n"
            
            # Get some tours
            tour_indices = state.get_all_tour_indices()[:3]
            tours = state.get_tours_by_indices(tour_indices)
            
            for i, tour in enumerate(tours, 1):
                if tour:
                    reply += f"{i}. **{tour.name}**\n"
                    if tour.duration:
                        reply += f"   ⏱️ {tour.duration}\n"
                    if tour.price:
                        reply += f"   💰 {tour.price}\n"
                    reply += "\n"
            
            reply += "Bạn quan tâm tour nào nhất? Hoặc gọi 0332510486 để được tư vấn ngay! 📞"
        else:
            reply = "Cảm ơn bạn đã quan tâm! Để tôi tư vấn tốt hơn, bạn có thể cho tôi biết số điện thoại của bạn được không?"
        
        return {
            "reply": reply,
            "context": context,
            "sources": [],
            "tour_indices": tour_indices if 'tour_indices' in locals() else []
        }

# =========== SEARCH ENGINE (FIXED) ===========
class MemoryOptimizedSearch:
    """Memory-optimized search engine with fixes"""
    
    def __init__(self):
        self.index = None
        self.mapping = []
        self.dim = 0
        self._loaded = False
    
    def load_index(self):
        """Lazy load index - FIXED for numpy file error"""
        if self._loaded:
            return True
        
        try:
            # FIX 1: Check if FAISS is enabled and available
            if FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(FAISS_INDEX_PATH):
                logger.info("📦 Loading FAISS index...")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                self.dim = self.index.d
                logger.info(f"✅ FAISS index loaded: {self.dim} dimensions")
            elif NUMPY_AVAILABLE and os.path.exists(FALLBACK_VECTORS_PATH):
                logger.info("📦 Loading numpy vectors...")
                try:
                    # FIX 2: Properly load numpy file
                    data = np.load(FALLBACK_VECTORS_PATH, allow_pickle=True)
                    
                    # Check if 'vectors' key exists
                    if 'vectors' in data:
                        self.index = data['vectors']
                        self.dim = self.index.shape[1]
                        logger.info(f"✅ Numpy vectors loaded: {self.index.shape}")
                    elif 'mat' in data:
                        # Alternative key name
                        self.index = data['mat']
                        self.dim = self.index.shape[1]
                        logger.info(f"✅ Numpy vectors loaded (mat): {self.index.shape}")
                    else:
                        # Try to load first array
                        first_key = list(data.keys())[0]
                        self.index = data[first_key]
                        self.dim = self.index.shape[1]
                        logger.info(f"✅ Numpy vectors loaded ({first_key}): {self.index.shape}")
                except Exception as e:
                    logger.error(f"❌ Failed to load numpy vectors: {e}")
                    self.index = None
            else:
                logger.warning("⚠️ No index found, using text search")
                self.index = None
            
            # Load mapping
            if os.path.exists(FAISS_MAPPING_PATH):
                with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                logger.info(f"✅ Loaded {len(self.mapping)} mapping entries")
            else:
                logger.warning("⚠️ No mapping file found")
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            self.index = None
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for relevant passages"""
        if not self._loaded:
            self.load_index()
        
        # Text fallback search
        if self.index is None or len(self.mapping) == 0:
            return self._text_search(query, top_k)
        
        # Try vector search if available
        try:
            # Get embedding
            embedding = self._get_embedding(query)
            if not embedding:
                return self._text_search(query, top_k)
            
            # Search
            if FAISS_ENABLED and FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                # FAISS search
                query_vec = np.array([embedding], dtype='float32')
                scores, indices = self.index.search(query_vec, top_k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self.mapping):
                        results.append((float(score), self.mapping[idx]))
                return results
            elif NUMPY_AVAILABLE and isinstance(self.index, np.ndarray):
                # Numpy search
                query_vec = np.array(embedding, dtype='float32').reshape(1, -1)
                index_norm = self.index / (np.linalg.norm(self.index, axis=1, keepdims=True) + 1e-12)
                query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
                
                similarities = np.dot(index_norm, query_norm.T).flatten()
                top_indices = np.argsort(-similarities)[:top_k]
                
                results = []
                for idx in top_indices:
                    if 0 <= idx < len(self.mapping):
                        score = float(similarities[idx])
                        results.append((score, self.mapping[idx]))
                return results
            else:
                return self._text_search(query, top_k)
                
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return self._text_search(query, top_k)
    
    def _text_search(self, query: str, top_k: int) -> List[Tuple[float, Dict]]:
        """Simple text-based search fallback - IMPROVED"""
        query_words = set(query.lower().split())
        results = []
        
        for entry in self.mapping[:200]:  # Increased limit for better results
            text = entry.get('text', '').lower()
            
            # Simple keyword matching
            score = 0
            for word in query_words:
                if len(word) > 2 and word in text:
                    score += 1
            
            # Bonus for exact matches
            if any(word in query.lower() for word in ['tour', 'hành trình', 'chuyến đi']):
                if 'tour' in text or 'hành trình' in text:
                    score += 2
            
            if score > 0:
                results.append((score, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching - FIXED OpenAI client"""
        if not text:
            return None
        
        # Check cache
        cached = state.get_embedding(text)
        if cached:
            return cached
        
        # Generate embedding
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                # FIX 3: Create OpenAI client WITHOUT proxies parameter
                client = OpenAI(api_key=OPENAI_API_KEY)
                
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:2000]
                )
                embedding = response.data[0].embedding
                state.set_embedding(text, embedding)
                return embedding
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                # Don't fall through, try hash-based embedding
        
        # Fallback: simple hash-based embedding
        embedding = self._hash_embedding(text)
        state.set_embedding(text, embedding)
        return embedding
    
    def _hash_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """Simple hash-based embedding for fallback"""
        import random
        random.seed(hash(text))
        return [random.random() for _ in range(dim)]

# Initialize search engine
search_engine = MemoryOptimizedSearch()

# =========== RESPONSE GENERATOR (FIXED) ===========
class ResponseGenerator:
    """Generate responses based on intent and search results"""
    
    @staticmethod
    def generate_llm_response(query: str, search_results: List, context: Dict) -> str:
        """Generate response using LLM - FIXED OpenAI client"""
        if not OPENAI_AVAILABLE or not ENABLE_LLM_FALLBACK:
            return ResponseGenerator.generate_template_response(search_results, context)
        
        try:
            # FIX 4: Create OpenAI client WITHOUT proxies parameter
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Prepare prompt
            prompt = f"""Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.

THÔNG TIN NGỮ CẢNH:
- Người dùng: {context.get('user_info', 'Khách hàng mới')}

DỮ LIỆU LIÊN QUAN:
"""
            if search_results:
                for i, (score, passage) in enumerate(search_results[:3], 1):
                    text = passage.get('text', '')[:200]
                    prompt += f"\n[{i}] {text}\n"
            else:
                prompt += "\nKhông tìm thấy thông tin cụ thể trong cơ sở dữ liệu."
            
            prompt += f"""

CÂU HỎI: {query}

HƯỚNG DẪN:
1. Trả lời dựa trên thông tin từ dữ liệu trên
2. Nếu không có thông tin, đề xuất các tour phù hợp
3. Giữ câu trả lời ngắn gọn, thân thiện
4. Kết thúc bằng lời mời liên hệ hotline 0332510486

TRẢ LỜI:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ResponseGenerator.generate_template_response(search_results, context)
    
    @staticmethod
    def generate_template_response(search_results: List, context: Dict, location_filtered: bool = False, 
                                  location_message: str = "") -> str:
        """Generate template-based response - IMPROVED with location filtering"""
        # BƯỚC 3: Add location filter message if applicable
        if location_filtered and location_message:
            reply = location_message + "\n\n"
        else:
            reply = ""
        
        if not search_results:
            # Try to get tours from knowledge base
            tour_indices = state.get_all_tour_indices()[:3]
            tours = state.get_tours_by_indices(tour_indices)
            
            if tours:
                if not location_filtered:
                    reply += "Hiện tôi có thông tin về các tour sau:\n\n"
                
                for i, tour in enumerate(tours, 1):
                    if tour:
                        # Add labels
                        label = ""
                        if i == 1:
                            label = "🏆 Phù hợp nhất: "
                        elif i == 2:
                            label = "⭐ Phổ biến: "
                        else:
                            label = "💰 Giá tốt: "
                        
                        reply += f"{i}. {label}**{tour.name}**\n"
                        if tour.duration:
                            reply += f"   ⏱️ {tour.duration}\n"
                        if tour.location:
                            reply += f"   📍 {tour.location}\n"
                        if tour.price and len(tour.price) < 100:  # Only show if not too long
                            reply += f"   💰 {tour.price[:80]}{'...' if len(tour.price) > 80 else ''}\n"
                        reply += "\n"
                reply += "Bạn quan tâm tour nào? Tôi sẽ cung cấp thêm chi tiết! 😊"
                return reply
            else:
                return "Xin lỗi, tôi hiện chưa tìm thấy thông tin cụ thể. " \
                       "Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp!"
        
        # Build response from top results
        if not location_filtered:
            reply += "Tôi tìm thấy một số thông tin liên quan:\n\n"
        
        for i, (score, passage) in enumerate(search_results[:3], 1):
            text = passage.get('text', '')
            # Extract tour name if available
            tour_name = ""
            if 'Tên tour:' in text:
                lines = text.split('\n')
                for line in lines:
                    if 'Tên tour:' in line:
                        tour_name = line.replace('Tên tour:', '').strip()
                        break
            
            # Add labels
            label = ""
            if i == 1:
                label = "🏆 "
            elif i == 2:
                label = "⭐ "
            else:
                label = "💰 "
            
            if tour_name:
                reply += f"{i}. {label}**{tour_name}**\n"
                # Try to extract more details
                lines = text.split('\n')
                if 'Thời lượng:' in text:
                    for line in lines:
                        if 'Thời lượng:' in line:
                            duration = line.replace('Thời lượng:', '').strip()
                            reply += f"   ⏱️ {duration}\n"
                            break
                if 'Địa điểm:' in text:
                    for line in lines:
                        if 'Địa điểm:' in line:
                            location = line.replace('Địa điểm:', '').strip()
                            reply += f"   📍 {location}\n"
                            break
                if 'Giá:' in text:
                    for line in lines:
                        if 'Giá:' in line:
                            price = line.replace('Giá:', '').strip()
                            if len(price) < 100:  # Only show if not too long
                                reply += f"   💰 {price[:80]}{'...' if len(price) > 80 else ''}\n"
                            break
            else:
                reply += f"{i}. {label}{text[:150]}...\n"
            
            reply += "\n"
        
        reply += "\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết và đặt tour*"
        
        return reply

# =========== ASYNC HELPERS ===========
def _save_lead_async(lead_data: LeadData):
    """Save lead data asynchronously"""
    if not ENABLE_LEAD_CAPTURE:
        return
    
    def save_task():
        try:
            # Google Sheets
            if ENABLE_GOOGLE_SHEETS:
                try:
                    import gspread
                    from google.oauth2.service_account import Credentials
                    
                    # Load service account
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
                            logger.info(f"✅ Lead saved to Google Sheets: {lead_data.phone}")
                except Exception as e:
                    logger.error(f"Google Sheets error: {e}")
            
            # Meta CAPI
            if ENABLE_META_CAPI:
                try:
                    meta_token = os.environ.get("META_CAPI_TOKEN")
                    pixel_id = os.environ.get("META_PIXEL_ID")
                    
                    if meta_token and pixel_id:
                        import requests
                        
                        event_data = {
                            "event_name": "Lead",
                            "event_time": int(time.time()),
                            "event_id": hashlib.md5(f"{lead_data.phone}{time.time()}".encode()).hexdigest(),
                            "user_data": {
                                "ph": hashlib.sha256(lead_data.phone.encode()).hexdigest(),
                                "client_ip_address": "",
                                "client_user_agent": ""
                            },
                            "action_source": "website"
                        }
                        
                        # Send to Meta (simplified)
                        # In production, use proper Meta CAPI library
                        logger.info(f"📊 Meta CAPI event prepared for: {lead_data.phone}")
                except Exception as e:
                    logger.error(f"Meta CAPI error: {e}")
            
            # Fallback storage
            fallback_path = os.environ.get("FALLBACK_STORAGE_PATH", "leads_fallback.json")
            try:
                leads = []
                if os.path.exists(fallback_path):
                    with open(fallback_path, 'r', encoding='utf-8') as f:
                        leads = json.load(f)
                
                leads.append(lead_data.to_dict())
                
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(leads, f, ensure_ascii=False, indent=2)
                
                logger.info(f"📝 Lead saved to fallback: {lead_data.phone}")
            except Exception as e:
                logger.error(f"Fallback storage error: {e}")
                
        except Exception as e:
            logger.error(f"Lead save error: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=save_task, daemon=True)
    thread.start()

# =========== MAIN CHAT PROCESSOR ===========
def process_chat_message(user_message: str, session_id: str) -> Dict[str, Any]:
    """Main chat processing pipeline with enhanced features"""
    start_time = time.time()
    
    try:
        # Get session data (enhanced with state machine)
        session_data = state.get_session(session_id)
        
        # Get or create conversation context
        context = ConversationContext(session_id=session_id)
        
        # Generate cache key
        cache_key = None
        if ENABLE_CACHING:
            cache_key = f"chat:{session_id}:{hashlib.md5(user_message.encode()).hexdigest()[:16]}"
            cached = state.get_cache(cache_key)
            if cached:
                logger.info("💾 Using cached response")
                cached['processing_time_ms'] = int((time.time() - start_time) * 1000)
                cached['from_cache'] = True
                return cached
        
        # BƯỚC 5: ENHANCED INTENT DETECTION
        intent = Intent.UNKNOWN
        metadata = {}
        
        if ENABLE_INTENT_DETECTION:
            intent, metadata = detect_intent(user_message)
            metadata['original_message'] = user_message
            
            # BƯỚC 5: Enhanced phone detection
            if ENABLE_PHONE_DETECTION:
                phone = detect_phone_number(user_message)
                if phone and 'phone_number' not in metadata:
                    metadata['phone_number'] = phone
                    logger.info(f"📱 Phone detected: {phone}")
            
            logger.info(f"🎯 Intent detected: {intent.value} | Stage: {session_data.get('stage')}")
            
            # BƯỚC 4: Update session with intent
            state.update_session(session_id, {
                'last_intent': intent.value,
                'last_message': user_message
            })
        
        # BƯỚC 3: Check for location filter
        location = extract_location_from_query(user_message)
        location_filtered = False
        location_message = ""
        
        # HANDLE NEW INTENTS WITH STATE MACHINE
        if intent in [Intent.PROVIDE_PHONE, Intent.CALLBACK_REQUEST, Intent.BOOKING_CONFIRM,
                     Intent.MODIFY_REQUEST, Intent.SMALLTALK, Intent.LEAD_CAPTURED]:
            
            handler_map = {
                Intent.PROVIDE_PHONE: IntentHandler.handle_provide_phone,
                Intent.CALLBACK_REQUEST: IntentHandler.handle_callback_request,
                Intent.BOOKING_CONFIRM: IntentHandler.handle_booking_confirm,
                Intent.MODIFY_REQUEST: IntentHandler.handle_modify_request,
                Intent.SMALLTALK: IntentHandler.handle_smalltalk,
                Intent.LEAD_CAPTURED: IntentHandler.handle_lead_captured,
            }
            
            handler = handler_map.get(intent)
            if handler:
                result = handler(context, metadata, session_id)
                
                # Update context
                state.update_session(session_id, {'context_updated': True})
                
                # Prepare response
                response = {
                    "reply": result['reply'],
                    "sources": result['sources'],
                    "context": {
                        "session_id": session_id,
                        "intent": intent.value,
                        "tour_indices": result['tour_indices'],
                        "has_phone": bool(metadata.get('phone_number')),
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "from_cache": False,
                        "stage": session_data.get('stage'),
                        "location_filtered": location_filtered
                    }
                }
                
                # Cache if enabled
                if ENABLE_CACHING and cache_key:
                    state.set_cache(cache_key, response)
                
                return response
        
        # LEGACY INTENTS & TOUR INQUIRIES
        # Load knowledge if needed
        if not state._knowledge_loaded:
            load_knowledge_lazy()
        
        # BƯỚC 3: Apply location filter if specified
        all_tours = state.get_tours_by_indices(state.get_all_tour_indices())
        filtered_tours = all_tours
        
        if location:
            # Apply location filter
            filtered_tours, location_message = apply_location_filter(all_tours, location)
            location_filtered = True
            
            # Save location filter to session
            state.update_session(session_id, {
                'last_location_filter': location
            })
            
            logger.info(f"📍 Location filter applied: '{location}' -> {len(filtered_tours)} tours")
        
        # Search for relevant information
        search_results = search_engine.search(user_message, top_k=3)
        
        # Generate response
        if intent in [Intent.GREETING, Intent.FAREWELL]:
            # Use simple responses for greetings/farewells
            if intent == Intent.GREETING:
                reply = "Xin chào! Tôi là trợ lý AI của Ruby Wings. Tôi có thể giúp gì cho bạn về các tour trải nghiệm thiên nhiên và chữa lành? 🌿"
            else:
                reply = "Cảm ơn bạn đã trò chuyện! Hy vọng sớm được đồng hành cùng bạn trong hành trình trải nghiệm. Liên hệ hotline 0332510486 nếu cần hỗ trợ thêm! ✨"
        else:
            # Generate response based on search results and location filter
            if location_filtered and filtered_tours:
                # Create custom search results from filtered tours
                custom_results = []
                for tour in filtered_tours[:3]:  # Limit to 3 tours
                    if tour:
                        # Create a passage-like entry
                        passage = {
                            'text': f"Tên tour: {tour.name}\nĐịa điểm: {tour.location}\nThời lượng: {tour.duration}\nGiá: {tour.price}",
                            'path': f"tours[{tour.index}]"
                        }
                        custom_results.append((1.0, passage))
                
                if custom_results:
                    search_results = custom_results
            
            reply = ResponseGenerator.generate_llm_response(
                user_message, 
                search_results,
                {
                    'user_info': f"Session {session_id}",
                    'intent': intent.value,
                    'stage': session_data.get('stage'),
                    'has_location_filter': location_filtered
                }
            )
        
        # Extract tour indices from search results
        tour_indices = []
        for score, passage in search_results:
            path = passage.get('path', '')
            if 'tours[' in path:
                match = re.search(r'tours\[(\d+)\]', path)
                if match:
                    idx = int(match.group(1))
                    if idx not in tour_indices:
                        tour_indices.append(idx)
        
        # Also try to find tours by name in the query
        if not tour_indices:
            # Simple tour name matching
            for idx in state.get_all_tour_indices():
                tour = state.get_tour(idx)
                if tour and tour.name:
                    tour_name_lower = tour.name.lower()
                    query_lower = user_message.lower()
                    if any(word in query_lower for word in tour_name_lower.split()):
                        if idx not in tour_indices:
                            tour_indices.append(idx)
        
        # BƯỚC 4: Update state machine if tour is selected
        if tour_indices and len(tour_indices) == 1:
            # Single tour selected
            selected_tour_id = tour_indices[0]
            state.update_session_stage(
                session_id,
                ConversationStage.SELECT,
                {
                    'selected_tour_id': selected_tour_id,
                    'selected_tour_name': state.get_tour(selected_tour_id).name if state.get_tour(selected_tour_id) else None
                }
            )
            logger.info(f"🎯 Tour selected: {selected_tour_id}")
        
        # Update context with tour information
        if tour_indices:
            context.last_tour_indices = tour_indices
            context.mentioned_tours.update(tour_indices)
        
        context.last_question = user_message
        context.last_response = reply
        context.last_updated = datetime.utcnow()
        
        # Update conversation history in session
        history = session_data.get('conversation_history', [])
        history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'user': user_message,
            'bot': reply[:100] + '...' if len(reply) > 100 else reply,
            'intent': intent.value
        })
        state.update_session(session_id, {
            'conversation_history': history[-10:]  # Keep last 10 messages
        })
        
        # Prepare response
        response = {
            "reply": reply,
            "sources": [passage for _, passage in search_results],
            "context": {
                "session_id": session_id,
                "intent": intent.value,
                "tour_indices": tour_indices,
                "has_phone": bool(metadata.get('phone_number')),
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "from_cache": False,
                "stage": session_data.get('stage'),
                "selected_tour_id": session_data.get('selected_tour_id'),
                "location_filtered": location_filtered,
                "location": location
            }
        }
        
        # Cache if enabled
        if ENABLE_CACHING and cache_key:
            state.set_cache(cache_key, response)
        
        logger.info(f"⏱️ Processing time: {(time.time() - start_time):.2f}s | "
                   f"Intent: {intent.value} | Stage: {session_data.get('stage')} | "
                   f"Tours: {len(tour_indices)} | Location: {location or 'N/A'}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat processing error: {e}\n{traceback.format_exc()}")
        
        return {
            "reply": "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. "
                    "Vui lòng thử lại sau hoặc liên hệ hotline 0332510486.",
            "sources": [],
            "context": {
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        }

# =========== API ENDPOINTS ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Main chat endpoint"""
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        
        if not user_message:
            return jsonify({
                "reply": "Xin chào! Tôi có thể giúp gì cho bạn về các tour Ruby Wings?",
                "sources": [],
                "context": {"error": "Empty message"}
            })
        
        # Get session ID
        session_id = data.get("session_id", "")
        if not session_id:
            # Generate from IP and timestamp
            ip = request.remote_addr or "0.0.0.0"
            timestamp = int(time.time() / 60)  # Change every minute
            session_id = hashlib.md5(f"{ip}_{timestamp}".encode()).hexdigest()[:12]
        
        # Process message
        result = process_chat_message(user_message, session_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({
            "reply": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
            "sources": [],
            "context": {"error": str(e)}
        }), 500

@app.route("/")
def home():
    """Home endpoint"""
    return jsonify({
        "status": "ok",
        "version": "5.2",
        "memory_optimized": True,
        "ram_profile": RAM_PROFILE,
        "features": {
            "intent_detection": ENABLE_INTENT_DETECTION,
            "phone_detection": ENABLE_PHONE_DETECTION,
            "lead_capture": ENABLE_LEAD_CAPTURE,
            "llm_fallback": ENABLE_LLM_FALLBACK,
            "caching": ENABLE_CACHING,
            "faiss_enabled": FAISS_ENABLED and FAISS_AVAILABLE,
            "tours_loaded": state._knowledge_loaded,
            "tours_count": len(state._tours_db) if state._knowledge_loaded else 0,
            "state_machine": True,
            "location_filter": True
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "memory": {
            "ram_profile": RAM_PROFILE,
            "sessions": len(state._session_contexts),
            "tours": len(state._tours_db),
            "cache_items": len(state._response_cache)
        },
        "services": {
            "chatbot": "running",
            "openai": "available" if OPENAI_AVAILABLE else "unavailable",
            "faiss": "available" if FAISS_AVAILABLE else "unavailable",
            "numpy": "available" if NUMPY_AVAILABLE else "unavailable"
        }
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    """Rebuild index (admin only)"""
    secret = request.headers.get("X-Admin-Key", "")
    if secret != os.environ.get("ADMIN_SECRET", ""):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        # Clear and reload
        state.clear_tours()
        state._knowledge_loaded = False
        search_engine._loaded = False
        
        load_knowledge_lazy()
        search_engine.load_index()
        
        return jsonify({
            "ok": True,
            "tours": len(state._tours_db),
            "mappings": len(search_engine.mapping)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lead capture endpoint
@app.route("/api/save-lead", methods=["POST"])
def save_lead():
    """Save lead data"""
    if not ENABLE_LEAD_CAPTURE:
        return jsonify({"error": "Lead capture disabled"}), 400
    
    try:
        data = request.get_json() or {}
        phone = data.get("phone", "").strip()
        
        if not phone:
            return jsonify({"error": "Phone required"}), 400
        
        lead_data = LeadData(
            phone=phone,
            source_channel=data.get("source", "API"),
            action_type=data.get("action", "Lead"),
            contact_name=data.get("name", ""),
            service_interest=data.get("interest", ""),
            note=data.get("note", "")
        )
        
        _save_lead_async(lead_data)
        
        return jsonify({
            "success": True,
            "message": "Lead saved",
            "phone": phone
        })
        
    except Exception as e:
        logger.error(f"Save lead error: {e}")
        return jsonify({"error": str(e)}), 500

# =========== UTILITY FUNCTIONS ===========
def create_vectors_npz_from_faiss():
    """Utility function to create vectors.npz from FAISS index"""
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.error("FAISS index file not found")
            return False
        
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return False
        
        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Get vectors from FAISS index
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
        vectors = vectors.reshape(index.ntotal, index.d)
        
        # Save as numpy file
        np.savez_compressed(FALLBACK_VECTORS_PATH, vectors=vectors)
        
        logger.info(f"✅ Created vectors.npz with {vectors.shape} vectors")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create vectors.npz: {e}")
        return False

# =========== INITIALIZATION ===========
def initialize():
    """Initialize application"""
    logger.info("🚀 Starting Ruby Wings Chatbot v5.2 (Enhanced with Location Filter & State Machine)")
    logger.info(f"🔧 Active features:")
    logger.info(f"   • Intent Detection: {ENABLE_INTENT_DETECTION}")
    logger.info(f"   • Phone Detection: {ENABLE_PHONE_DETECTION}")
    logger.info(f"   • Lead Capture: {ENABLE_LEAD_CAPTURE}")
    logger.info(f"   • LLM Fallback: {ENABLE_LLM_FALLBACK}")
    logger.info(f"   • FAISS: {FAISS_ENABLED and FAISS_AVAILABLE}")
    logger.info(f"   • Caching: {ENABLE_CACHING}")
    logger.info(f"   • State Machine: Enabled")
    logger.info(f"   • Location Filter: Enabled")
    
    # Check if vectors.npz exists, if not try to create it
    if not os.path.exists(FALLBACK_VECTORS_PATH) and FAISS_AVAILABLE:
        logger.info("🔄 Attempting to create vectors.npz from FAISS index...")
        create_vectors_npz_from_faiss()
    
    logger.info("✅ Initialization complete")

# =========== APPLICATION START ===========
if __name__ == "__main__":
    initialize()
    
    # Load knowledge in background
    def preload_knowledge():
        time.sleep(2)
        if load_knowledge_lazy():
            logger.info("✅ Knowledge preloaded")
        search_engine.load_index()
    
    threading.Thread(target=preload_knowledge, daemon=True).start()
    
    # Start server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "10000"))
    
    logger.info(f"🌐 Server starting on {host}:{port}")
    
    # Use simple server for development
    if os.environ.get("FLASK_ENV") == "development":
        app.run(host=host, port=port, debug=True, threaded=True)
    else:
        # ===== Production / Dev runner =====
        import platform

        IS_WINDOWS = platform.system().lower().startswith("win")

        if not IS_WINDOWS:
            # ===== Linux / Render: Gunicorn =====
            from gunicorn.app.base import BaseApplication

            class FlaskApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.application = app
                    self.options = options or {}
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key, value)

                def load(self):
                    return self.application

            options = {
                'bind': f'{host}:{port}',
                'workers': 1 if IS_LOW_RAM else 2,
                'threads': 2,
                'timeout': 30,
                'worker_class': 'sync',
                'loglevel': 'info'
            }

            FlaskApplication(app, options).run()

        else:
            # ===== Windows: Flask dev server =====
            initialize()
            app.run(host="0.0.0.0", port=10000, debug=False, use_reloader=False)

