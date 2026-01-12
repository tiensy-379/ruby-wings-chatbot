# entities.py - Core Data Models for Ruby Wings Chatbot v4.0
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
import re

# ===== ENUMS =====
class QuestionType(Enum):
    """Types of user questions"""
    INFORMATION = "information"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    LISTING = "listing"
    CALCULATION = "calculation"
    CONFIRMATION = "confirmation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    COMPLEX = "complex"

class ConversationState(Enum):
    """Conversation states for state machine"""
    INITIAL = "initial"
    TOUR_SELECTED = "tour_selected"
    COMPARING = "comparing"
    ASKING_DETAILS = "asking_details"
    RECOMMENDATION = "recommendation"
    BOOKING = "booking"
    FAREWELL = "farewell"

class PriceLevel(Enum):
    """Price level categories"""
    BUDGET = "budget"
    MIDRANGE = "midrange"
    PREMIUM = "premium"

class DurationType(Enum):
    """Duration categories"""
    SHORT = "short"      # 1 day
    MEDIUM = "medium"    # 2-3 days
    LONG = "long"        # 4+ days

# ===== INTENT CLASSIFICATION =====
class Intent(Enum):
    """User intent classification"""
    # Existing intents
    GREETING = "greeting"
    FAREWELL = "farewell"
    TOUR_INQUIRY = "tour_inquiry"
    TOUR_COMPARISON = "tour_comparison"
    TOUR_RECOMMENDATION = "tour_recommendation"
    PRICE_ASK = "price_ask"
    BOOKING_INQUIRY = "booking_inquiry"
    
    # New intents
    PROVIDE_PHONE = "provide_phone"
    CALLBACK_REQUEST = "callback_request"
    BOOKING_CONFIRM = "booking_confirm"
    MODIFY_REQUEST = "modify_request"
    SMALLTALK = "smalltalk"
    LEAD_CAPTURED = "lead_captured"
    
    # Fallback
    UNKNOWN = "unknown"

# Intent keywords for matching
INTENT_KEYWORDS = {
    Intent.PROVIDE_PHONE: [
        "số điện thoại", "điện thoại", "số của tôi", "số tôi", "phone", "số",
        "liên lạc qua", "gọi cho tôi số", "số liên hệ", "sdt", "đt",
        "số của tôi là", "số tôi là", "số phone", "điện thoại của tôi",
        "090", "091", "092", "093", "094", "096", "097", "098", "032", "033", "034", "035", "036", "037", "038", "039"
    ],
    
    Intent.CALLBACK_REQUEST: [
        "gọi lại", "gọi cho tôi", "callback", "liên hệ lại", "gọi điện",
        "tư vấn qua điện thoại", "gọi ngay", "gọi lại cho tôi",
        "cho tôi xin cuộc gọi", "muốn được gọi", "nhờ gọi lại",
        "alô", "alo", "call back", "call lại", "phone lại"
    ],
    
    Intent.BOOKING_CONFIRM: [
        "xác nhận đặt", "đã đặt", "booking confirm", "xác nhận booking",
        "confirm tour", "xác nhận chuyến đi", "đặt tour thành công",
        "đã thanh toán", "đã book", "book rồi", "đặt rồi",
        "xác nhận lịch trình", "confirm lịch trình", "đặt xong"
    ],
    
    Intent.MODIFY_REQUEST: [
        "thay đổi", "chỉnh sửa", "modify", "đổi", "hủy", "cancel",
        "đổi tour", "thay đổi booking", "chỉnh sửa đặt chỗ",
        "hoãn tour", "dời lịch", "đổi ngày", "đổi lịch trình",
        "hủy đặt", "cancel booking", "chỉnh sửa thông tin"
    ],
    
    Intent.SMALLTALK: [
        "chào", "hello", "hi", "bạn khỏe", "cảm ơn", "thanks", "tạm biệt",
        "khỏe không", "ổn không", "good morning", "good afternoon", "good evening",
        "xin chào", "chào bạn", "chào admin", "cám ơn", "thank you",
        "bye", "goodbye", "hẹn gặp", "chúc ngủ ngon", "chúc vui vẻ"
    ],
    
    Intent.LEAD_CAPTURED: [
        "đăng ký", "tư vấn", "lead", "nhận thông tin", "muốn biết thêm",
        "liên hệ tư vấn", "cần tư vấn", "muốn đăng ký", "đăng ký tour",
        "để lại thông tin", "lưu thông tin", "lead capture",
        "tôi muốn đặt", "tôi muốn book", "cần book tour"
    ],
    
    Intent.GREETING: [
        "xin chào", "chào", "hello", "hi", "chào bạn", "chào anh", "chào chị"
    ],
    
    Intent.FAREWELL: [
        "tạm biệt", "bye", "goodbye", "hẹn gặp", "cảm ơn", "thanks"
    ],
    
    Intent.TOUR_INQUIRY: [
        "tour", "du lịch", "chuyến đi", "trải nghiệm", "giá", "thông tin"
    ]
}

def detect_phone_number(text: str) -> Optional[str]:
    """
    Detect phone number using regex pattern (9-11 digits)
    Vietnamese phone formats: 09x xxxx xxx, 03x xxxx xxx, +84 3x xxxx xxx
    """
    # Pattern for Vietnamese phone numbers (9-11 digits, may include +84 or 0 prefix)
    patterns = [
        r'(?:\+84|0)(?:3|5|7|8|9)(?:\d{8})',  # Standard Vietnam mobile
        r'(?:\+84|0)(?:\d{9,10})',  # General 9-10 digits
        r'(?:0\d{9,10})',  # Local format
        r'(?:\d{10,11})'  # Just digits
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean the phone number
            phone = matches[0]
            # Remove non-digits
            phone_digits = re.sub(r'\D', '', phone)
            # Check length (9-11 digits)
            if 9 <= len(phone_digits) <= 11:
                # Format to standard Vietnam format: 0xxxxxxxxx
                if phone_digits.startswith('84'):
                    phone_digits = '0' + phone_digits[2:]
                elif len(phone_digits) == 9:
                    phone_digits = '0' + phone_digits
                return phone_digits
    
    return None

def detect_intent(text: str) -> Tuple[Intent, Dict[str, Any]]:
    """
    Detect user intent from text using keyword matching
    Returns: (intent, metadata)
    """
    text_lower = text.lower().strip()
    
    # First check for phone number
    phone_number = detect_phone_number(text_lower)
    
    # Then check keywords
    detected_intent = Intent.UNKNOWN
    confidence = 0.0
    metadata = {"phone_number": phone_number}
    
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_intent = intent
                confidence = max(confidence, 0.8)  # High confidence for keyword match
                break
    
    # Special cases
    if phone_number:
        if detected_intent == Intent.UNKNOWN:
            detected_intent = Intent.PROVIDE_PHONE
            confidence = 0.9
        else:
            # If phone found but other intent detected, mark as lead captured
            if detected_intent in [Intent.BOOKING_CONFIRM, Intent.TOUR_INQUIRY, Intent.LEAD_CAPTURED]:
                metadata["has_phone"] = True
    
    # Fallback: check for question patterns
    if detected_intent == Intent.UNKNOWN:
        question_patterns = ["bao nhiêu", "thế nào", "là gì", "ở đâu", "khi nào"]
        if any(pattern in text_lower for pattern in question_patterns):
            detected_intent = Intent.TOUR_INQUIRY
            confidence = 0.6
    
    metadata["confidence"] = confidence
    return detected_intent, metadata

# ===== CORE DATA MODELS =====
@dataclass
class Tour:
    """Tour data model"""
    index: int
    name: str = ""
    duration: str = ""
    location: str = ""
    price: str = ""
    summary: str = ""
    includes: List[str] = field(default_factory=list)
    accommodation: str = ""
    meals: str = ""
    transport: str = ""
    notes: str = ""
    style: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    popularity_score: float = 0.5
    last_mentioned: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "index": self.index,
            "tour_name": self.name,
            "duration": self.duration,
            "location": self.location,
            "price": self.price,
            "summary": self.summary,
            "includes": self.includes,
            "accommodation": self.accommodation,
            "meals": self.meals,
            "transport": self.transport,
            "notes": self.notes,
            "style": self.style,
            "tags": self.tags,
            "completeness_score": self.completeness_score,
            "popularity_score": self.popularity_score,
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None
        }
    
    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'Tour':
        """Create Tour from dictionary"""
        return cls(
            index=index,
            name=data.get("tour_name", ""),
            duration=data.get("duration", ""),
            location=data.get("location", ""),
            price=data.get("price", ""),
            summary=data.get("summary", ""),
            includes=data.get("includes", []),
            accommodation=data.get("accommodation", ""),
            meals=data.get("meals", ""),
            transport=data.get("transport", ""),
            notes=data.get("notes", ""),
            style=data.get("style", ""),
            tags=data.get("tags", []),
            completeness_score=data.get("completeness_score", 0.0),
            popularity_score=data.get("popularity_score", 0.5)
        )

@dataclass
class UserProfile:
    """User profile for semantic analysis"""
    age_group: Optional[str] = None  # young, middle_aged, senior, family_with_kids
    group_type: Optional[str] = None  # solo, couple, family, friends, corporate
    interests: List[str] = field(default_factory=list)  # nature, history, culture, spiritual, wellness, adventure, food
    budget_level: Optional[str] = None  # budget, midrange, premium
    physical_level: Optional[str] = None  # easy, moderate, challenging
    special_needs: List[str] = field(default_factory=list)
    
    # Confidence scores
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def get_overall_confidence(self) -> float:
        """Calculate overall confidence"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
    
    def to_summary(self) -> str:
        """Get summary string"""
        parts = []
        if self.age_group:
            parts.append(f"Độ tuổi: {self.age_group}")
        if self.group_type:
            parts.append(f"Nhóm: {self.group_type}")
        if self.interests:
            parts.append(f"Sở thích: {', '.join(self.interests)}")
        if self.budget_level:
            parts.append(f"Ngân sách: {self.budget_level}")
        return "; ".join(parts)

@dataclass
class SearchResult:
    """Search result from vector index"""
    score: float
    text: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tour_index(self) -> Optional[int]:
        """Extract tour index from path"""
        import re
        match = re.search(r'tours\[(\d+)\]', self.path)
        if match:
            return int(match.group(1))
        return None

@dataclass
class ConversationContext:
    """Conversation context for state management"""
    session_id: str

    # Core state
    current_tours: List[int] = field(default_factory=list)
    last_tour_indices: List[int] = field(default_factory=list)   # FIX: required by StateMachine
    last_successful_tours: List[int] = field(default_factory=list)

    # Conversation memory
    last_question: Optional[str] = None
    last_response: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # User modeling
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    mentioned_tours: Set[int] = field(default_factory=set)

    # Dialogue control
    current_focus: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update context with new interaction"""
        self.last_updated = datetime.utcnow()
        self.last_question = user_message
        self.last_response = bot_response
        
        # Update conversation history
        self.conversation_history.append({
            'timestamp': self.last_updated.isoformat(),
            'user': user_message,
            'bot': bot_response,
            'tours': tour_indices or []
        })
        
        # Keep only last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Update mentioned tours
        if tour_indices:
            self.mentioned_tours.update(tour_indices)
            self.current_tours = tour_indices
            self.last_successful_tours = tour_indices

@dataclass
class FilterSet:
    """Filter set for tour filtering"""
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    location: Optional[str] = None
    near_location: Optional[str] = None
    month: Optional[int] = None
    weekend: bool = False
    holiday: Optional[str] = None
    group_type: Optional[str] = None
    
    def is_empty(self) -> bool:
        """Check if filter set is empty"""
        return all(
            getattr(self, field) is None or getattr(self, field) is False
            for field in self.__dataclass_fields__
            if field not in ['weekend']
        )
    
    def to_dict(self) -> Dict[str, Any]:
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

@dataclass
class LLMRequest:
    """LLM request data"""
    user_message: str
    context: Dict[str, Any]
    search_results: List[SearchResult]
    tour_indices: List[int]
    question_type: QuestionType
    requested_field: Optional[str]
    user_profile: UserProfile
    
    def build_prompt(self) -> str:
        """Build prompt for LLM"""
        prompt_parts = [
            "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.",
            "HƯỚNG DẪN QUAN TRỌNG:",
            "1. LUÔN sử dụng thông tin từ dữ liệu nội bộ được cung cấp",
            "2. Nếu thiếu thông tin chi tiết, tổng hợp từ thông tin chung",
            "3. KHÔNG BAO GIỜ nói 'không có thông tin', 'không biết', 'không rõ'",
            "4. Luôn giữ thái độ nhiệt tình, hữu ích, chuyên nghiệp",
            "5. Nếu không tìm thấy thông tin chính xác, đưa ra thông tin tổng quát",
            "6. KHÔNG tự ý bịa thông tin không có trong dữ liệu",
            "",
            "THÔNG TIN NGỮ CẢNH:",
        ]
        
        # Add user profile if available
        if self.user_profile.to_summary():
            prompt_parts.append(f"- Sở thích người dùng: {self.user_profile.to_summary()}")
        
        # Add current tours if available
        if self.tour_indices:
            prompt_parts.append(f"- Tour đang thảo luận: {len(self.tour_indices)} tour")
        
        prompt_parts.append("")
        prompt_parts.append("DỮ LIỆU NỘI BỘ RUBY WINGS:")
        
        if self.search_results:
            for i, result in enumerate(self.search_results[:5], 1):
                prompt_parts.append(f"\n[{i}] (Độ liên quan: {result.score:.2f})")
                prompt_parts.append(f"{result.text[:300]}...")
        else:
            prompt_parts.append("Không tìm thấy dữ liệu liên quan trực tiếp.")
        
        prompt_parts.append("")
        prompt_parts.append("TRẢ LỜI:")
        prompt_parts.append("1. Dựa trên dữ liệu trên, trả lời câu hỏi người dùng")
        prompt_parts.append("2. Nếu có thông tin từ dữ liệu, trích dẫn nó")
        prompt_parts.append("3. Giữ câu trả lời ngắn gọn, rõ ràng, hữu ích")
        prompt_parts.append("4. Kết thúc bằng lời mời liên hệ hotline 0332510486")
        
        return "\n".join(prompt_parts)

@dataclass
class ChatResponse:
    """Chatbot response"""
    reply: str
    sources: List[Dict[str, Any]]
    context: Dict[str, Any]
    tour_indices: List[int]
    processing_time_ms: int
    from_memory: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "reply": self.reply,
            "sources": self.sources,
            "context": {
                "tour_indices": self.tour_indices,
                "processing_time_ms": self.processing_time_ms,
                "from_memory": self.from_memory,
                **self.context
            }
        }

@dataclass
class LeadData:
    """Lead data for Google Sheets and Meta CAPI"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_channel: str = "Website"
    action_type: str = "Click Call"
    page_url: str = ""
    contact_name: str = ""
    phone: str = ""
    service_interest: str = ""
    note: str = ""
    status: str = "New"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "source_channel": self.source_channel,
            "action_type": self.action_type,
            "page_url": self.page_url,
            "contact_name": self.contact_name,
            "phone": self.phone,
            "service_interest": self.service_interest,
            "note": self.note,
            "status": self.status
        }
    
    def to_row(self) -> List[str]:
        """Convert to Google Sheets row"""
        return [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(self.timestamp, datetime) else str(self.timestamp),
            self.source_channel,
            self.action_type,
            self.page_url,
            self.contact_name,
            self.phone,
            self.service_interest,
            self.note,
            self.status
        ]
    
    def to_meta_event(self, request, event_name: str = "Lead") -> Dict[str, Any]:
        """Convert to Meta CAPI event"""
        return {
            "event_name": event_name,
            "event_time": int(self.timestamp.timestamp()) if isinstance(self.timestamp, datetime) else int(time.time()),
            "event_id": str(hash(f"{self.phone}{self.timestamp}")),
            "event_source_url": self.page_url,
            "action_source": "website",
            "user_data": {
                "ph": self._hash_phone(self.phone) if self.phone else "",
                "client_ip_address": request.remote_addr if hasattr(request, 'remote_addr') else "",
                "client_user_agent": request.headers.get("User-Agent", "") if hasattr(request, 'headers') else ""
            },
            "custom_data": {
                "value": 200000,
                "currency": "VND",
                "content_name": "Ruby Wings Lead"
            }
        }
    
    @staticmethod
    def _hash_phone(phone: str) -> str:
        """Hash phone number for Meta"""
        if not phone:
            return ""
        cleaned = phone.strip().lower()
        return hashlib.sha256(cleaned.encode()).hexdigest()

@dataclass
class CacheEntry:
    """Cache entry for response caching"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

# ===== SERIALIZATION HELPERS =====
class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for custom objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# ===== BUILD ENTITY INDEX =====
def build_entity_index(mapping: List[Dict[str, Any]], out_path: str = "tour_entities.json") -> Dict[str, Any]:
    """
    Build entity index from mapping for quick lookup
    """
    entity_index = {
        "tours_by_name": {},
        "tours_by_location": {},
        "tours_by_duration": {},
        "tours_by_price_range": {},
        "keywords": {}
    }
    
    for item in mapping:
        if "tours[" in item["path"]:
            # Extract tour index
            match = re.search(r'tours\[(\d+)\]', item["path"])
            if match:
                tour_idx = int(match.group(1))
                
                # Get text content
                text = item.get("text", "")
                
                # Parse tour name from text
                if "Tên tour:" in text:
                    lines = text.split("\n")
                    for line in lines:
                        if line.startswith("Tên tour:"):
                            tour_name = line.replace("Tên tour:", "").strip()
                            entity_index["tours_by_name"][tour_name] = tour_idx
                            break
                
                # Extract location
                if "Địa điểm:" in text:
                    lines = text.split("\n")
                    for line in lines:
                        if line.startswith("Địa điểm:"):
                            location = line.replace("Địa điểm:", "").strip()
                            if location not in entity_index["tours_by_location"]:
                                entity_index["tours_by_location"][location] = []
                            entity_index["tours_by_location"][location].append(tour_idx)
                            break
                
                # Extract duration
                if "Thời lượng:" in text:
                    lines = text.split("\n")
                    for line in lines:
                        if line.startswith("Thời lượng:"):
                            duration = line.replace("Thời lượng:", "").strip()
                            if duration not in entity_index["tours_by_duration"]:
                                entity_index["tours_by_duration"][duration] = []
                            entity_index["tours_by_duration"][duration].append(tour_idx)
                            break
                
                # Extract price and create price ranges
                if "Giá:" in text:
                    lines = text.split("\n")
                    for line in lines:
                        if line.startswith("Giá:"):
                            price_text = line.replace("Giá:", "").strip()
                            # Simple price range detection
                            if "triệu" in price_text or "VND" in price_text:
                                entity_index["tours_by_price_range"]["premium"] = entity_index["tours_by_price_range"].get("premium", [])
                                entity_index["tours_by_price_range"]["premium"].append(tour_idx)
                            elif "nghìn" in price_text or price_text.endswith("k"):
                                entity_index["tours_by_price_range"]["midrange"] = entity_index["tours_by_price_range"].get("midrange", [])
                                entity_index["tours_by_price_range"]["midrange"].append(tour_idx)
                            break
    
    # Save to file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entity_index, f, ensure_ascii=False, indent=2)
    
    return entity_index

# ===== EXPORTS =====
__all__ = [
    'QuestionType',
    'ConversationState',
    'PriceLevel',
    'DurationType',
    'Intent',
    'INTENT_KEYWORDS',
    'detect_phone_number',
    'detect_intent',
    'Tour',
    'UserProfile',
    'SearchResult',
    'ConversationContext',
    'FilterSet',
    'LLMRequest',
    'ChatResponse',
    'LeadData',
    'CacheEntry',
    'EnhancedJSONEncoder',
    'build_entity_index'
]