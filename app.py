# app.py - Ruby Wings Chatbot v5.0 (RAM Optimized + 6 New Intents)
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

# Flask
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# =========== MEMORY OPTIMIZATION ===========
# Disable unused imports for low RAM
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ruby-wings-ram")

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
    except ImportError:
        logger.warning("⚠️ OpenAI not available")

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
    raise

# =========== FLASK APP ===========
app = Flask(__name__)
# Force Flask into WSGI-only mode (disable self runner when used by uvicorn)
app.run = lambda *args, **kwargs: None

app.json_encoder = EnhancedJSONEncoder
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

# =========== MEMORY OPTIMIZED GLOBAL STATE ===========
class MemoryOptimizedState:
    """Memory-optimized global state management"""
    
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
    
    # Session management
    def get_session(self, session_id: str) -> ConversationContext:
        with self._lock:
            if session_id not in self._session_contexts:
                self._session_contexts[session_id] = ConversationContext(session_id=session_id)
                
                # Cleanup if too many sessions
                if len(self._session_contexts) > self.max_sessions:
                    self._cleanup_old_sessions()
            
            return self._session_contexts[session_id]
    
    def update_session(self, session_id: str, context: ConversationContext):
        with self._lock:
            self._session_contexts[session_id] = context
    
    def _cleanup_old_sessions(self):
        """Remove oldest sessions when limit exceeded"""
        if len(self._session_contexts) <= self.max_sessions:
            return
        
        # Sort by last_updated
        sorted_sessions = sorted(
            self._session_contexts.items(),
            key=lambda x: x[1].last_updated if hasattr(x[1], 'last_updated') else datetime.min
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

# =========== INTENT HANDLERS ===========
class IntentHandler:
    """Handler for new intents"""
    
    @staticmethod
    def handle_provide_phone(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle PROVIDE_PHONE intent"""
        phone = metadata.get("phone_number")
        if not phone:
            return {
                "reply": "Bạn vui lòng cung cấp số điện thoại của bạn nhé!",
                "context": context,
                "sources": [],
                "tour_indices": []
            }
        
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
    def handle_callback_request(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CALLBACK_REQUEST intent"""
        phone = metadata.get("phone_number")
        
        if not phone and 'phone' in context.user_preferences:
            phone = context.user_preferences['phone']
        
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
    def handle_booking_confirm(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle BOOKING_CONFIRM intent"""
        # Extract booking details if available
        phone = metadata.get("phone_number")
        
        if phone:
            context.user_preferences['phone'] = phone
        
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
    def handle_modify_request(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
    def handle_smalltalk(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
    def handle_lead_captured(context: ConversationContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LEAD_CAPTURED intent"""
        phone = metadata.get("phone_number")
        has_phone = metadata.get("has_phone", False)
        
        if phone:
            context.user_preferences['phone'] = phone
            has_phone = True
        
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

# =========== SEARCH ENGINE ===========
class MemoryOptimizedSearch:
    """Memory-optimized search engine"""
    
    def __init__(self):
        self.index = None
        self.mapping = []
        self.dim = 0
        self._loaded = False
    
    def load_index(self):
        """Lazy load index"""
        if self._loaded:
            return True
        
        try:
            if FAISS_ENABLED and FAISS_AVAILABLE and os.path.exists(FAISS_INDEX_PATH):
                logger.info("📦 Loading FAISS index...")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                self.dim = self.index.d
                logger.info(f"✅ FAISS index loaded: {self.dim} dimensions")
            elif NUMPY_AVAILABLE and os.path.exists(FALLBACK_VECTORS_PATH):
                logger.info("📦 Loading numpy vectors...")
                data = np.load(FALLBACK_VECTORS_PATH)
                self.index = data['vectors']
                self.dim = self.index.shape[1]
                logger.info(f"✅ Numpy vectors loaded: {self.index.shape}")
            else:
                logger.warning("⚠️ No index found, using text search")
                self.index = None
            
            # Load mapping
            if os.path.exists(FAISS_MAPPING_PATH):
                with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                logger.info(f"✅ Loaded {len(self.mapping)} mapping entries")
            
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
                index_norm = self.index / np.linalg.norm(self.index, axis=1, keepdims=True)
                query_norm = query_vec / np.linalg.norm(query_vec)
                
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
        """Simple text-based search fallback"""
        query_words = set(query.lower().split())
        results = []
        
        for entry in self.mapping[:100]:  # Limit for memory
            text = entry.get('text', '').lower()
            text_words = set(text.split())
            
            common_words = query_words.intersection(text_words)
            if common_words:
                score = len(common_words) / max(len(query_words), 1)
                results.append((score, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching"""
        if not text:
            return None
        
        # Check cache
        cached = state.get_embedding(text)
        if cached:
            return cached
        
        # Generate embedding
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
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

# =========== RESPONSE GENERATOR ===========
class ResponseGenerator:
    """Generate responses based on intent and search results"""
    
    @staticmethod
    def generate_llm_response(query: str, search_results: List, context: Dict) -> str:
        """Generate response using LLM"""
        if not OPENAI_AVAILABLE or not ENABLE_LLM_FALLBACK:
            return ResponseGenerator.generate_template_response(search_results, context)
        
        try:
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
    def generate_template_response(search_results: List, context: Dict) -> str:
        """Generate template-based response"""
        if not search_results:
            return "Xin lỗi, tôi chưa tìm thấy thông tin cụ thể về yêu cầu của bạn. " \
                   "Vui lòng liên hệ hotline 0332510486 để được tư vấn trực tiếp!"
        
        # Build response from top results
        response_parts = ["Tôi tìm thấy một số thông tin liên quan:"]
        
        for i, (score, passage) in enumerate(search_results[:3], 1):
            text = passage.get('text', '')
            # Extract tour name if available
            tour_name = ""
            if 'tour_name' in text:
                lines = text.split('\n')
                for line in lines:
                    if 'Tên tour:' in line:
                        tour_name = line.replace('Tên tour:', '').strip()
                        break
            
            if tour_name:
                response_parts.append(f"\n{i}. **{tour_name}**")
            else:
                response_parts.append(f"\n{i}. {text[:150]}...")
        
        response_parts.append("\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết và đặt tour*")
        
        return "\n".join(response_parts)

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
    """Main chat processing pipeline"""
    start_time = time.time()
    
    try:
        # Get or create session context
        context = state.get_session(session_id)
        
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
        
        # INTENT DETECTION
        intent = Intent.UNKNOWN
        metadata = {}
        
        if ENABLE_INTENT_DETECTION:
            intent, metadata = detect_intent(user_message)
            metadata['original_message'] = user_message
            logger.info(f"🎯 Intent detected: {intent.value}")
        
        # PHONE DETECTION
        if ENABLE_PHONE_DETECTION:
            phone = detect_phone_number(user_message)
            if phone and 'phone_number' not in metadata:
                metadata['phone_number'] = phone
                logger.info(f"📱 Phone detected: {phone}")
        
        # HANDLE NEW INTENTS
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
                result = handler(context, metadata)
                
                # Update context
                state.update_session(session_id, result['context'])
                
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
                        "from_cache": False
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
            # Generate response based on search results
            reply = ResponseGenerator.generate_llm_response(
                user_message, 
                search_results,
                {
                    'user_info': f"Session {session_id}",
                    'intent': intent.value
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
        
        # Update context with tour information
        if tour_indices:
            context.last_tour_indices = tour_indices
            context.mentioned_tours.update(tour_indices)
        
        context.last_question = user_message
        context.last_response = reply
        context.last_updated = datetime.utcnow()
        
        # Update session
        state.update_session(session_id, context)
        
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
                "from_cache": False
            }
        }
        
        # Cache if enabled
        if ENABLE_CACHING and cache_key:
            state.set_cache(cache_key, response)
        
        logger.info(f"⏱️ Processing time: {(time.time() - start_time):.2f}s | "
                   f"Intent: {intent.value} | Tours: {len(tour_indices)}")
        
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
        "version": "5.0",
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
            "tours_count": len(state._tours_db) if state._knowledge_loaded else 0
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

# =========== INITIALIZATION ===========
def initialize():
    """Initialize application"""
    logger.info("🚀 Starting Ruby Wings Chatbot v5.0 (RAM Optimized)")
    logger.info(f"🔧 Active features:")
    logger.info(f"   • Intent Detection: {ENABLE_INTENT_DETECTION}")
    logger.info(f"   • Phone Detection: {ENABLE_PHONE_DETECTION}")
    logger.info(f"   • Lead Capture: {ENABLE_LEAD_CAPTURE}")
    logger.info(f"   • LLM Fallback: {ENABLE_LLM_FALLBACK}")
    logger.info(f"   • FAISS: {FAISS_ENABLED and FAISS_AVAILABLE}")
    logger.info(f"   • Caching: {ENABLE_CACHING}")
    
    # Lazy load knowledge on first request
    # This saves memory on startup
    
    logger.info("✅ Initialization complete")

# =========== APPLICATION START ===========
if __name__ == "__main__":
    # Chỉ dùng khi chạy trực tiếp: python app.py (Linux / Render)
    initialize()

    def preload_knowledge():
        time.sleep(2)
        if load_knowledge_lazy():
            logger.info("✅ Knowledge preloaded")
        search_engine.load_index()

    threading.Thread(target=preload_knowledge, daemon=True).start()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "10000"))
    logger.info(f"🌐 Server starting on {host}:{port}")

    # Chỉ chạy gunicorn trên Linux
    if os.name != "nt":
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
        # Windows: không tự chạy server
        pass

else:
    # Khi bị uvicorn import → chỉ init, KHÔNG exit, KHÔNG start server
    initialize()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)