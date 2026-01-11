# entities.py - Core Data Models for Ruby Wings Chatbot v4.0
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from enum import Enum
import json
import hashlib

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
    current_tours: List[int] = field(default_factory=list)
    last_tour_indices: list[int] = field(default_factory=list)
    last_question: Optional[str] = None
    last_response: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    mentioned_tours: Set[int] = field(default_factory=set)
    current_focus: Optional[str] = None
    last_successful_tours: List[int] = field(default_factory=list)
   
   

    
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
    """Lead data for Google Sheets/Meta CAPI"""
    timestamp: datetime
    source_channel: str
    action_type: str
    page_url: str
    contact_name: str
    phone: str
    service_interest: str
    note: str
    status: str
    
    def to_row(self) -> List[str]:
        """Convert to Google Sheets row"""
        return [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
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
            "event_time": int(self.timestamp.timestamp()),
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

# ===== EXPORTS =====
__all__ = [
    'QuestionType',
    'ConversationState',
    'PriceLevel',
    'DurationType',
    'Tour',
    'UserProfile',
    'SearchResult',
    'ConversationContext',
    'FilterSet',
    'LLMRequest',
    'ChatResponse',
    'LeadData',
    'CacheEntry',
    'EnhancedJSONEncoder'
]