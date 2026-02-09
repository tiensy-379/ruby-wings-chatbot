"""
Entities module for Ruby Wings Chatbot v4.0
All dataclasses and enums in one place for easy import
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import json

# =========== ENUMS ===========
class QuestionType(Enum):
    INFORMATION = "information"
    LISTING = "listing"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    CALCULATION = "calculation"
    COMPLEX = "complex"

class ConversationState(Enum):
    INITIAL = "initial"
    ASKING_DETAILS = "asking_details"
    TOUR_SELECTED = "tour_selected"
    COMPARING = "comparing"
    RECOMMENDATION = "recommendation"
    BOOKING = "booking"
    FAREWELL = "farewell"

class PriceLevel(Enum):
    BUDGET = "budget"
    MIDRANGE = "midrange"
    PREMIUM = "premium"

class DurationType(Enum):
    DAY_TRIP = "day_trip"
    OVERNIGHT = "overnight"
    MULTIDAY = "multiday"

# =========== DATACLASSES ===========
@dataclass
class Tour:
    """Tour dataclass"""
    index: int = 0
    name: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    price: Optional[str] = None
    summary: Optional[str] = None
    includes: List[str] = field(default_factory=list)
    accommodation: Optional[str] = None
    meals: Optional[str] = None
    transport: Optional[str] = None
    notes: Optional[str] = None
    style: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class UserProfile:
    """User profile dataclass"""
    age_group: Optional[str] = None
    group_type: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    budget_level: Optional[str] = None
    physical_level: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0

@dataclass
class SearchResult:
    """Search result dataclass"""
    text: str = ""
    path: str = ""
    score: float = 0.0

@dataclass
class ConversationContext:
    """Conversation context dataclass"""
    session_id: str = ""
    current_tours: List[int] = field(default_factory=list)
    last_successful_tours: List[int] = field(default_factory=list)
    mentioned_tours: Set[int] = field(default_factory=set)
    conversation_history: List[Dict] = field(default_factory=list)
    user_profile: UserProfile = field(default_factory=UserProfile)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None):
        """Update context with new interaction"""
        self.last_updated = datetime.utcnow()
        
        if tour_indices:
            self.current_tours = tour_indices
            self.last_successful_tours = tour_indices
            self.mentioned_tours.update(tour_indices)

@dataclass
class FilterSet:
    """Filter set dataclass"""
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
        """Check if filter set is empty"""
        return all(
            getattr(self, field) in (None, False, '', 0)
            for field in ['price_min', 'price_max', 'duration_min', 'duration_max', 
                         'location', 'near_location', 'month', 'weekend', 'holiday', 'group_type']
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LLMRequest:
    """LLM request dataclass"""
    model: str = "gpt-4o-mini"
    messages: List[Dict] = field(default_factory=list)
    temperature: float = 0.6
    max_tokens: int = 500

@dataclass
class ChatResponse:
    """Chat response dataclass"""
    reply: str = ""
    sources: List[Dict] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    tour_indices: List[int] = field(default_factory=list)
    processing_time_ms: int = 0
    from_memory: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LeadData:
    """Lead data dataclass"""
    timestamp: str = ""
    phone: str = ""
    name: str = ""
    email: str = ""
    tour_interest: str = ""
    page_url: str = ""
    note: str = ""
    source: str = ""

@dataclass
class CacheEntry:
    """Cache entry dataclass"""
    key: str = ""
    value: Any = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

# =========== JSON ENCODER ===========
class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for dataclasses and enums"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# =========== EXPORTS ===========
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
    'EnhancedJSONEncoder',
]