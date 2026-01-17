# entities.py - Core Data Models for Ruby Wings Chatbot v5.2 (Enhanced with State Machine & New Intents)
# Đồng bộ hóa với app.py v5.2
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
import re
import time

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
    """User intent classification - ĐÃ MỞ RỘNG VỚI 6 INTENT MỚI"""
    # Existing intents
    GREETING = "greeting"
    FAREWELL = "farewell"
    TOUR_INQUIRY = "tour_inquiry"
    TOUR_COMPARISON = "tour_comparison"
    TOUR_RECOMMENDATION = "tour_recommendation"
    PRICE_ASK = "price_ask"
    BOOKING_INQUIRY = "booking_inquiry"
    
    # New intents (added in v5.2)
    PROVIDE_PHONE = "provide_phone"
    CALLBACK_REQUEST = "callback_request"
    BOOKING_CONFIRM = "booking_confirm"
    MODIFY_REQUEST = "modify_request"
    SMALLTALK = "smalltalk"
    LEAD_CAPTURED = "lead_captured"
    
    # Fallback
    UNKNOWN = "unknown"

# Intent keywords for matching - ĐÃ MỞ RỘNG VỚI TỪ KHÓA CHO INTENT MỚI
INTENT_KEYWORDS = {
    Intent.PROVIDE_PHONE: [
        "số điện thoại", "điện thoại", "số của tôi", "số tôi", "phone", "số",
        "liên lạc qua", "gọi cho tôi số", "số liên hệ", "sdt", "đt",
        "số của tôi là", "số tôi là", "số phone", "điện thoại của tôi",
        "090", "091", "092", "093", "094", "096", "097", "098", "032", "033", "034", "035", "036", "037", "038", "039",
        "085", "086", "088", "089", "070", "076", "077", "078", "079",
        "số tôi đây", "đây là số tôi", "số của em", "số của anh", "số của chị",
        "số mình", "số của mình", "liên hệ số", "gọi số"
    ],
    
    Intent.CALLBACK_REQUEST: [
        "gọi lại", "gọi cho tôi", "callback", "liên hệ lại", "gọi điện",
        "tư vấn qua điện thoại", "gọi ngay", "gọi lại cho tôi",
        "cho tôi xin cuộc gọi", "muốn được gọi", "nhờ gọi lại",
        "alô", "alo", "call back", "call lại", "phone lại",
        "tư vấn điện thoại", "gọi tư vấn", "để tôi gọi lại",
        "xin gọi lại", "vui lòng gọi lại", "có thể gọi lại không",
        "gọi cho tôi với", "gọi tôi nhé", "gọi tôi đi"
    ],
    
    Intent.BOOKING_CONFIRM: [
        "xác nhận đặt", "đã đặt", "booking confirm", "xác nhận booking",
        "confirm tour", "xác nhận chuyến đi", "đặt tour thành công",
        "đã thanh toán", "đã book", "book rồi", "đặt rồi",
        "xác nhận lịch trình", "confirm lịch trình", "đặt xong",
        "đặt tour này", "tôi muốn đặt", "book tour", "đặt ngay",
        "tôi đồng ý đặt", "ok đặt tour", "đặt cho tôi", "tôi đặt tour",
        "tôi book tour", "đặt giúp tôi", "xác nhận đặt tour",
        "tôi xác nhận đặt", "tôi confirm booking"
    ],
    
    Intent.MODIFY_REQUEST: [
        "thay đổi", "chỉnh sửa", "modify", "đổi", "hủy", "cancel",
        "đổi tour", "thay đổi booking", "chỉnh sửa đặt chỗ",
        "hoãn tour", "dời lịch", "đổi ngày", "đổi lịch trình",
        "hủy đặt", "cancel booking", "chỉnh sửa thông tin",
        "thay đổi thông tin", "sửa booking", "sửa đặt tour",
        "đổi lịch", "hoãn lại", "dời ngày", "thay đổi ngày đi",
        "hủy tour", "cancel tour", "đổi tour khác"
    ],
    
    Intent.SMALLTALK: [
        "chào", "hello", "hi", "bạn khỏe", "cảm ơn", "thanks", "tạm biệt",
        "khỏe không", "ổn không", "good morning", "good afternoon", "good evening",
        "xin chào", "chào bạn", "chào admin", "cám ơn", "thank you",
        "bye", "goodbye", "hẹn gặp", "chúc ngủ ngon", "chúc vui vẻ",
        "hello bạn", "hi bạn", "chào anh", "chào chị", "chào em",
        "bạn ơi", "admin ơi", "anh ơi", "chị ơi",
        "dạ", "vâng", "ừ", "ok", "oke", "okay",
        "hôm nay thế nào", "có khỏe không", "mọi thứ ổn chứ"
    ],
    
    Intent.LEAD_CAPTURED: [
        "đăng ký", "tư vấn", "lead", "nhận thông tin", "muốn biết thêm",
        "liên hệ tư vấn", "cần tư vấn", "muốn đăng ký", "đăng ký tour",
        "để lại thông tin", "lưu thông tin", "lead capture",
        "tôi muốn đặt", "tôi muốn book", "cần book tour",
        "tôi quan tâm", "tôi cần tư vấn", "cho tôi thông tin",
        "tôi muốn tham gia", "đăng ký cho tôi", "tôi muốn đi tour",
        "cần hỗ trợ", "cần tư vấn tour", "tư vấn giúp tôi"
    ],
    
    Intent.GREETING: [
        "xin chào", "chào", "hello", "hi", "chào bạn", "chào anh", "chào chị",
        "chào em", "chào admin", "chào ruby wings", "chào chatbot"
    ],
    
    Intent.FAREWELL: [
        "tạm biệt", "bye", "goodbye", "hẹn gặp", "cảm ơn", "thanks",
        "tạm biệt nhé", "bye bạn", "goodbye bạn", "hẹn gặp lại",
        "cảm ơn bạn", "thanks bạn", "cám ơn", "thank you"
    ],
    
    Intent.TOUR_INQUIRY: [
        "tour", "du lịch", "chuyến đi", "trải nghiệm", "giá", "thông tin",
        "hành trình", "tour nào", "có tour nào", "tour du lịch",
        "trải nghiệm ruby wings", "tour ruby wings", "chương trình",
        "tour gì", "có những tour nào", "giới thiệu tour"
    ],
    
    Intent.TOUR_COMPARISON: [
        "so sánh", "khác nhau", "giống nhau", "compare", "khác gì",
        "so với", "hơn", "tốt hơn", "khác biệt", "nên chọn"
    ],
    
    Intent.TOUR_RECOMMENDATION: [
        "gợi ý", "đề xuất", "recommend", "nên đi", "phù hợp",
        "tư vấn", "chọn tour", "tour nào hay", "tour nào tốt"
    ],
    
    Intent.PRICE_ASK: [
        "giá", "bao nhiêu", "chi phí", "price", "cost",
        "giá tour", "giá bao nhiêu", "hết bao nhiêu"
    ]
}

# Phone detection patterns - ĐÃ CẢI THIỆN REGEX
PHONE_PATTERNS = [
    # Vietnamese mobile: 09x, 03x, 05x, 07x, 08x, 09x
    r'(?:\+?84|0)(?:3[2-9]|5[2689]|7[06-9]|8[1-9]|9[0-9])\d{7}',
    # General 9-11 digits
    r'(?:\+?84|0)\d{9,10}',
    # Just digits 9-11
    r'\b\d{9,11}\b',
    # With spaces/dashes
    r'(?:\+?84\s?|0)(?:\d\s?){9,10}',
    r'(?:\d[- ]?){9,11}'
]

def detect_phone_number(text: str) -> Optional[str]:
    """
    Detect phone number using improved regex patterns (9-11 digits)
    Vietnamese phone formats: 09x xxx xxx, 03x xxx xxx, +84 3x xxx xxx, 84-90-xxx-xxx
    """
    text_cleaned = re.sub(r'[^\d\s\+\-]', '', text)  # Remove non-digit/non-phone chars
    
    for pattern in PHONE_PATTERNS:
        try:
            matches = re.findall(pattern, text_cleaned)
            if matches:
                for phone in matches:
                    # Clean the phone number
                    phone_digits = re.sub(r'\D', '', phone)
                    
                    # Check length (9-11 digits)
                    if 9 <= len(phone_digits) <= 11:
                        # Format to standard Vietnam format: 0xxxxxxxxx
                        if phone_digits.startswith('84'):
                            phone_digits = '0' + phone_digits[2:]
                        elif len(phone_digits) == 9:
                            phone_digits = '0' + phone_digits
                        
                        # Additional validation: check if it's a valid Vietnam mobile prefix
                        valid_prefixes = ['03', '05', '07', '08', '09', '032', '033', '034', '035', 
                                        '036', '037', '038', '039', '070', '076', '077', 
                                        '078', '079', '081', '082', '083', '084', '085', 
                                        '086', '087', '088', '089', '090', '091', '092', 
                                        '093', '094', '096', '097', '098', '099']
                        
                        if any(phone_digits.startswith(prefix) for prefix in valid_prefixes):
                            return phone_digits
        except Exception:
            continue
    
    # Fallback: look for common phone number patterns in text
    phone_words = ["điện thoại", "số", "sdt", "đt", "phone", "liên hệ", "call", "gọi"]
    for word in phone_words:
        if word in text.lower():
            # Try to extract digits near the phone word
            word_index = text.lower().find(word)
            if word_index >= 0:
                context = text[max(0, word_index-20):min(len(text), word_index+30)]
                digit_matches = re.findall(r'\b\d{9,11}\b', context)
                if digit_matches:
                    phone_digits = digit_matches[0]
                    if 9 <= len(phone_digits) <= 11:
                        if phone_digits.startswith('84'):
                            phone_digits = '0' + phone_digits[2:]
                        elif len(phone_digits) == 9:
                            phone_digits = '0' + phone_digits
                        return phone_digits
    
    return None

def detect_intent(text: str) -> Tuple[Intent, float, Dict[str, Any]]:
    """
    Detect user intent from text using keyword matching - ĐÃ CẢI THIỆN LOGIC
    Returns: (intent, confidence, metadata)
    """
    text_lower = text.lower().strip()
    
    # First check for phone number (priority)
    phone_number = detect_phone_number(text_lower)
    metadata = {"phone_number": phone_number, "original_message": text}
    
    # Special case: if text contains only digits (likely phone number)
    if re.fullmatch(r'\d{9,11}', text_lower.strip()):
        return Intent.PROVIDE_PHONE, 0.95, metadata
    
    # Check keywords with priority
    detected_intent = Intent.UNKNOWN
    confidence = 0.0
    keyword_matches = []
    
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Score based on keyword position and length
                score = 0.8  # Base score
                
                # Boost score for exact match at beginning
                if text_lower.startswith(keyword):
                    score += 0.1
                
                # Boost for longer keywords
                if len(keyword) > 5:
                    score += 0.05
                
                keyword_matches.append((intent, score, keyword))
    
    # Sort by score and get highest
    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        detected_intent, confidence, matched_keyword = keyword_matches[0]
        metadata["matched_keyword"] = matched_keyword
        metadata["confidence"] = confidence
        
        # Special handling for phone-related intents
        if phone_number:
            if detected_intent == Intent.PROVIDE_PHONE:
                confidence = max(confidence, 0.95)  # Very high confidence
            elif detected_intent in [Intent.CALLBACK_REQUEST, Intent.BOOKING_CONFIRM, Intent.LEAD_CAPTURED]:
                metadata["has_phone"] = True
                confidence = max(confidence, 0.9)
    
    # Fallback logic
    if detected_intent == Intent.UNKNOWN:
        # Check for question patterns
        question_patterns = [
            ("bao nhiêu", Intent.TOUR_INQUIRY),
            ("thế nào", Intent.TOUR_INQUIRY),
            ("là gì", Intent.TOUR_INQUIRY),
            ("ở đâu", Intent.TOUR_INQUIRY),
            ("khi nào", Intent.TOUR_INQUIRY),
            ("có không", Intent.TOUR_INQUIRY),
            ("như thế nào", Intent.TOUR_INQUIRY)
        ]
        
        for pattern, intent in question_patterns:
            if pattern in text_lower:
                detected_intent = intent
                confidence = 0.7
                break
        
        # If still unknown and has phone, default to provide_phone
        if detected_intent == Intent.UNKNOWN and phone_number:
            detected_intent = Intent.PROVIDE_PHONE
            confidence = 0.85
            metadata["auto_detected"] = True
    
    # Handle smalltalk that might be misclassified
    if detected_intent != Intent.SMALLTALK:
        smalltalk_indicators = ["ạ", "nhé", "nhỉ", "hả", "à", "ơi"]
        if any(indicator in text_lower for indicator in smalltalk_indicators) and len(text_lower.split()) <= 5:
            # Likely smalltalk
            detected_intent = Intent.SMALLTALK
            confidence = 0.8
    
    metadata["confidence"] = confidence
    return detected_intent, confidence, metadata

# ===== STATE MACHINE ENUMS (ĐỒNG BỘ VỚI APP.PY) =====
class ConversationStage(Enum):
    """Conversation stages for state machine - ĐỒNG BỘ VỚI APP.PY"""
    EXPLORE = "explore"
    SUGGEST = "suggest"
    COMPARE = "compare"
    SELECT = "select"
    BOOK = "book"
    LEAD = "lead"
    CALLBACK = "callback"

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
    event_support: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    popularity_score: float = 0.5
    last_mentioned: Optional[datetime] = None
    region: Optional[str] = None  # Thêm trường region để hỗ trợ location filter
    
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
            "event_support": self.event_support,
            "tags": self.tags,
            "completeness_score": self.completeness_score,
            "popularity_score": self.popularity_score,
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None,
            "region": self.region
        }
    
    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'Tour':
        """Create Tour from dictionary"""
        # Auto-detect region from location
        region = None
        location = data.get("location", "").lower()
        if any(keyword in location for keyword in ["huế", "quảng trị", "bạch mã", "đà nẵng", "hội an"]):
            region = "Miền Trung"
        elif any(keyword in location for keyword in ["hà nội", "hạ long", "sapa", "ninh bình"]):
            region = "Miền Bắc"
        elif any(keyword in location for keyword in ["hồ chí minh", "sài gòn", "cần thơ", "phú quốc", "nha trang", "đà lạt"]):
            region = "Miền Nam"
        
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
            event_support=data.get("event_support", ""),
            tags=data.get("tags", []),
            completeness_score=data.get("completeness_score", 0.0),
            popularity_score=data.get("popularity_score", 0.5),
            region=region
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
    
    # State machine integration
    current_stage: str = ConversationStage.EXPLORE.value
    selected_tour_id: Optional[int] = None
    lead_phone: Optional[str] = None
    last_intent: Optional[str] = None
    
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
        if self.current_stage:
            parts.append(f"Giai đoạn: {self.current_stage}")
        return "; ".join(parts)
    
    def update_stage(self, new_stage: ConversationStage, metadata: Dict[str, Any] = None):
        """Update user stage with metadata"""
        self.current_stage = new_stage.value
        
        if metadata:
            if 'selected_tour_id' in metadata:
                self.selected_tour_id = metadata['selected_tour_id']
            if 'lead_phone' in metadata:
                self.lead_phone = metadata['lead_phone']
            if 'last_intent' in metadata:
                self.last_intent = metadata['last_intent']

@dataclass
class SearchResult:
    """Search result from vector index"""
    score: float
    text: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tour_index(self) -> Optional[int]:
        """Extract tour index from path"""
        match = re.search(r'tours\[(\d+)\]', self.path)
        if match:
            return int(match.group(1))
        return None

@dataclass
class ConversationContext:
    """Conversation context for state management - ĐÃ TÍCH HỢP STATE MACHINE"""
    session_id: str

    # Core state
    current_tours: List[int] = field(default_factory=list)
    last_tour_indices: List[int] = field(default_factory=list)
    last_successful_tours: List[int] = field(default_factory=list)

    # Conversation memory
    last_question: Optional[str] = None
    last_response: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # User modeling
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    mentioned_tours: Set[int] = field(default_factory=set)
    
    # State machine fields (đồng bộ với app.py)
    current_stage: str = ConversationStage.EXPLORE.value
    selected_tour_id: Optional[int] = None
    lead_phone: Optional[str] = None
    last_intent: Optional[str] = None
    location_filter: Optional[str] = None

    # Dialogue control
    current_focus: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, user_message: str, bot_response: str, tour_indices: List[int] = None,
               intent: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Update context with new interaction"""
        self.last_updated = datetime.utcnow()
        self.last_question = user_message
        self.last_response = bot_response
        
        if intent:
            self.last_intent = intent
        
        if metadata:
            if 'phone_number' in metadata and metadata['phone_number']:
                self.lead_phone = metadata['phone_number']
            if 'location' in metadata and metadata['location']:
                self.location_filter = metadata['location']
        
        # Update conversation history
        self.conversation_history.append({
            'timestamp': self.last_updated.isoformat(),
            'user': user_message,
            'bot': bot_response,
            'tours': tour_indices or [],
            'intent': intent,
            'stage': self.current_stage
        })
        
        # Keep only last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Update mentioned tours
        if tour_indices:
            self.mentioned_tours.update(tour_indices)
            self.current_tours = tour_indices
            self.last_successful_tours = tour_indices
            
            # Auto-update stage if single tour selected
            if len(tour_indices) == 1:
                self.selected_tour_id = tour_indices[0]
                self.current_stage = ConversationStage.SELECT.value
    
    def update_stage(self, new_stage: ConversationStage, metadata: Dict[str, Any] = None):
        """Update conversation stage"""
        self.current_stage = new_stage.value
        self.last_updated = datetime.utcnow()
        
        if metadata:
            if 'selected_tour_id' in metadata:
                self.selected_tour_id = metadata['selected_tour_id']
            if 'lead_phone' in metadata:
                self.lead_phone = metadata['lead_phone']
            if 'last_intent' in metadata:
                self.last_intent = metadata['last_intent']
            if 'location' in metadata:
                self.location_filter = metadata['location']
    
    def get_stage_summary(self) -> str:
        """Get stage summary"""
        stage_map = {
            ConversationStage.EXPLORE.value: "Khám phá tour",
            ConversationStage.SUGGEST.value: "Đang đề xuất",
            ConversationStage.COMPARE.value: "So sánh tour",
            ConversationStage.SELECT.value: f"Đã chọn tour #{self.selected_tour_id}" if self.selected_tour_id else "Đã chọn tour",
            ConversationStage.BOOK.value: "Xác nhận đặt tour",
            ConversationStage.LEAD.value: f"Đã thu thập lead: {self.lead_phone}" if self.lead_phone else "Thu thập thông tin",
            ConversationStage.CALLBACK.value: "Yêu cầu gọi lại"
        }
        return stage_map.get(self.current_stage, "Khám phá")

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
    
    # New fields for location filter
    region: Optional[str] = None
    strict_location: bool = False  # If True, only exact location matches
    
    def is_empty(self) -> bool:
        """Check if filter set is empty"""
        return all(
            getattr(self, field) is None or getattr(self, field) is False
            for field in self.__dataclass_fields__
            if field not in ['weekend', 'strict_location']
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
            'group_type': self.group_type,
            'region': self.region,
            'strict_location': self.strict_location
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
        
        # Add current stage if available
        if self.context.get('stage'):
            prompt_parts.append(f"- Giai đoạn hội thoại: {self.context.get('stage')}")
        
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
                "stage": self.context.get("stage"),
                "selected_tour_id": self.context.get("selected_tour_id"),
                "has_phone": self.context.get("has_phone", False),
                "location_filtered": self.context.get("location_filtered", False),
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
    
    # New fields for better tracking
    session_id: str = ""
    intent: str = ""
    tour_id: Optional[int] = None
    stage: str = ""
    
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
            "status": self.status,
            "session_id": self.session_id,
            "intent": self.intent,
            "tour_id": self.tour_id,
            "stage": self.stage
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
            self.status,
            self.session_id,
            self.intent,
            str(self.tour_id) if self.tour_id else "",
            self.stage
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
                "content_name": "Ruby Wings Lead",
                "content_category": self.service_interest,
                "content_ids": [str(self.tour_id)] if self.tour_id else []
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

# ===== LOCATION HELPERS =====
def extract_location_from_query(text: str) -> Optional[str]:
    """
    Extract location from user query - ĐỒNG BỘ VỚI APP.PY
    """
    text_lower = text.lower()
    
    # Location keywords mapping
    location_keywords = {
        "đà nẵng": "Đà Nẵng",
        "huế": "Huế",
        "quảng trị": "Quảng Trị",
        "bạch mã": "Bạch Mã",
        "hà nội": "Hà Nội",
        "hạ long": "Hạ Long",
        "sapa": "Sapa",
        "hồ chí minh": "Hồ Chí Minh",
        "sài gòn": "Sài Gòn",
        "cần thơ": "Cần Thơ",
        "phú quốc": "Phú Quốc",
        "nha trang": "Nha Trang",
        "hội an": "Hội An",
        "ninh bình": "Ninh Bình",
        "đà lạt": "Đà Lạt"
    }
    
    # Check exact matches
    for keyword, location in location_keywords.items():
        if keyword in text_lower:
            return location
    
    # Check patterns
    patterns = [
        r'tại\s+([a-zA-ZÀ-ỹ\s]+)',
        r'ở\s+([a-zA-ZÀ-ỹ\s]+)',
        r'location\s+([a-zA-ZÀ-ỹ\s]+)',
        r'tour\s+([a-zA-ZÀ-ỹ\s]+)\?',
        r'du lịch\s+([a-zA-ZÀ-ỹ\s]+)'
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text_lower)
        if matches:
            location = matches.group(1).strip()
            if location and len(location) > 1:
                # Map to standard location
                for keyword, std_location in location_keywords.items():
                    if keyword in location.lower():
                        return std_location
                return location.title()
    
    return None

def get_region_from_location(location: str) -> Optional[str]:
    """Get region from location"""
    location_lower = location.lower()
    
    if any(keyword in location_lower for keyword in ["huế", "quảng trị", "bạch mã", "đà nẵng", "hội an"]):
        return "Miền Trung"
    elif any(keyword in location_lower for keyword in ["hà nội", "hạ long", "sapa", "ninh bình"]):
        return "Miền Bắc"
    elif any(keyword in location_lower for keyword in ["hồ chí minh", "sài gòn", "cần thơ", "phú quốc", "nha trang", "đà lạt"]):
        return "Miền Nam"
    
    return None

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
        "tours_by_region": {},
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
                            
                            # Also add to region index
                            region = get_region_from_location(location)
                            if region:
                                if region not in entity_index["tours_by_region"]:
                                    entity_index["tours_by_region"][region] = []
                                entity_index["tours_by_region"][region].append(tour_idx)
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
                            if "triệu" in price_text.lower() or "vnd" in price_text.lower():
                                entity_index["tours_by_price_range"]["premium"] = entity_index["tours_by_price_range"].get("premium", [])
                                entity_index["tours_by_price_range"]["premium"].append(tour_idx)
                            elif "nghìn" in price_text.lower() or price_text.lower().endswith("k"):
                                entity_index["tours_by_price_range"]["midrange"] = entity_index["tours_by_price_range"].get("midrange", [])
                                entity_index["tours_by_price_range"]["midrange"].append(tour_idx)
                            else:
                                entity_index["tours_by_price_range"]["budget"] = entity_index["tours_by_price_range"].get("budget", [])
                                entity_index["tours_by_price_range"]["budget"].append(tour_idx)
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
    'ConversationStage',
    'detect_phone_number',
    'detect_intent',
    'extract_location_from_query',
    'get_region_from_location',
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