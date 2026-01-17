#!/usr/bin/env python3
"""
response_guard.py v5.2 - Ruby Wings Chatbot

Enhanced "expert guard" to validate & format final answers before sending to user.
Fully integrated with entities.py v5.2 and tour_entities.json structure.

Responsibilities:
- Ensure answers cite sources (e.g., [1], [2]) or attach retrieved snippets if LLM hallucinated
- Ensure answer content is consistent with retrieved evidence (token overlap check)
- Ensure requested_field is respected (if provided) by preferring passages for that field
- Enforce friendly "healing travel" tone with sanitization heuristics
- Provide deterministic fallback using only retrieved passages when LLM output fails checks
- State-based response templates for different conversation stages
- Location-aware response formatting with region suggestions
- Tour response formatting with labels (🏆, ⭐, 💰)
- Intent-specific response templates

ĐỒNG BỘ: entities.py v5.2, knowledge.json, tour_entities.json, app.py v5.2

Usage:
  from response_guard import validate_and_format_answer
  out = validate_and_format_answer(
      llm_text=llm_text,
      top_passages=top_passages,            # List[Tuple[score, mapping_entry]]
      requested_field=requested_field,      # optional string
      tour_indices=tour_indices,            # optional list[int]
      max_tokens=700,
      context={}                            # conversation context
  )

Return value:
  {
    "answer": "<final text to send user>",
    "sources": ["root.tours[2].price", ...],
    "guard_passed": True/False,
    "reason": "ok" | "no_evidence" | "mismatch_field" | ...,
    "state": "explore" | "suggest" | ...,
    "tour_labels": [],
    "location_filtered": False
  }
"""

import re
import html
import time
import random
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import Counter
from datetime import datetime

# Import from entities.py (ĐỒNG BỘ)
try:
    from entities import (
        ConversationStage,
        Intent,
        extract_location_from_query,
        get_region_from_location
    )
except ImportError:
    # Fallback definitions if entities.py not available
    logging.warning("⚠️ Could not import from entities.py, using fallback definitions")
    
    class ConversationStage:
        """Fallback ConversationStage"""
        EXPLORE = "explore"
        SUGGEST = "suggest"
        COMPARE = "compare"
        SELECT = "select"
        BOOK = "book"
        LEAD = "lead"
        CALLBACK = "callback"
    
    class Intent:
        """Fallback Intent"""
        PROVIDE_PHONE = "provide_phone"
        CALLBACK_REQUEST = "callback_request"
        BOOKING_CONFIRM = "booking_confirm"
        MODIFY_REQUEST = "modify_request"
        SMALLTALK = "smalltalk"
        LEAD_CAPTURED = "lead_captured"
        GREETING = "greeting"
        FAREWELL = "farewell"
        TOUR_INQUIRY = "tour_inquiry"
        TOUR_COMPARISON = "tour_comparison"
        TOUR_RECOMMENDATION = "tour_recommendation"
        PRICE_ASK = "price_ask"
        BOOKING_INQUIRY = "booking_inquiry"
        UNKNOWN = "unknown"
    
    def extract_location_from_query(text: str) -> Optional[str]:
        return None
    
    def get_region_from_location(location: str) -> Optional[str]:
        location_lower = location.lower()
        if any(k in location_lower for k in ["huế", "quảng trị", "bạch mã", "đà nẵng", "hội an"]):
            return "Miền Trung"
        elif any(k in location_lower for k in ["hà nội", "hạ long", "sapa", "ninh bình"]):
            return "Miền Bắc"
        elif any(k in location_lower for k in ["hồ chí minh", "sài gòn", "cần thơ", "phú quốc"]):
            return "Miền Nam"
        return None

# Setup logging
logger = logging.getLogger("response_guard")

# ==================== CONSTANTS ====================

# Citation regex
SRC_RE = re.compile(r"\[\d+\]")  # detect [1], [2] style citations

# Guard parameters
MIN_OVERLAP_RATIO = 0.12   # minimal overlap between LLM text and evidence to accept
MIN_FIELD_MENTION_RATIO = 0.02  # small threshold to allow field-specific match via text overlap
MAX_ANSWER_CHARS = 1500
BANNED_PHRASES = ["i think", "i guess", "maybe", "probably", "as far as i know", "i'm not sure"]

# ==================== RESPONSE TEMPLATES ====================

# State-based templates (ĐỒNG BỘ VỚI ENTITIES.PY)
STATE_TEMPLATES = {
    ConversationStage.EXPLORE: [
        "Tôi có thể giúp gì cho bạn về tour du lịch trải nghiệm Ruby Wings? 🌿",
        "Bạn muốn tìm hiểu về tour du lịch nào của Ruby Wings? 😊",
        "Chào bạn! Tôi có thể tư vấn cho bạn về các hành trình trải nghiệm của Ruby Wings.",
        "Ruby Wings có nhiều tour trải nghiệm độc đáo. Bạn quan tâm đến tour nào?"
    ],
    
    ConversationStage.SUGGEST: [
        "Dựa trên yêu cầu của bạn, tôi đề xuất các tour sau:",
        "Tôi tìm thấy một số tour phù hợp với bạn:",
        "Dưới đây là các tour Ruby Wings bạn có thể quan tâm:",
        "Đây là những gợi ý tour phù hợp nhất cho bạn:"
    ],
    
    ConversationStage.COMPARE: [
        "Để so sánh các tour, tôi tóm tắt thông tin chính:",
        "Dưới đây là thông tin so sánh giữa các tour:",
        "Tôi sẽ giúp bạn so sánh các tour để chọn phù hợp nhất:",
        "So sánh chi tiết các tour:"
    ],
    
    ConversationStage.SELECT: [
        "Bạn đã chọn tour **{tour_name}**. Bạn muốn đặt tour này không?",
        "Tour **{tour_name}** rất phù hợp với bạn! Bạn muốn tiếp tục đặt tour không?",
        "Tuyệt vời! Tour **{tour_name}** đã được chọn. Bạn có muốn đặt ngay không?",
        "**{tour_name}** là lựa chọn tuyệt vời! Bạn cần thêm thông tin gì về tour này?"
    ],
    
    ConversationStage.BOOK: [
        "Tour **{tour_name}** đã được đặt. Vui lòng cung cấp số điện thoại để chúng tôi liên hệ xác nhận.",
        "Booking thành công! Chúng tôi sẽ liên hệ với bạn qua số điện thoại để xác nhận chi tiết.",
        "Đã xác nhận đặt tour **{tour_name}**. Vui lòng cho chúng tôi số điện thoại để hoàn tất thủ tục.",
        "Cảm ơn bạn đã chọn **{tour_name}**! Để hoàn tất đặt tour, vui lòng cung cấp số điện thoại."
    ],
    
    ConversationStage.LEAD: [
        "Đã lưu số **{phone}**. Chúng tôi sẽ gọi lại cho bạn trong 30 phút. 📞",
        "Cảm ơn bạn đã cung cấp số điện thoại **{phone}**. Đội ngũ Ruby Wings sẽ liên hệ sớm nhất!",
        "Số điện thoại **{phone}** đã được ghi nhận. Chúng tôi sẽ liên hệ tư vấn cho bạn sớm.",
        "Đã nhận số **{phone}**. Bộ phận tư vấn Ruby Wings sẽ liên hệ bạn trong thời gian sớm nhất."
    ],
    
    ConversationStage.CALLBACK: [
        "Đã ghi nhận yêu cầu gọi lại. Chúng tôi sẽ liên hệ số **{phone}** trong ngày hôm nay.",
        "Yêu cầu gọi lại đã được xác nhận. Chúng tôi sẽ gọi số **{phone}** trong vòng 2 giờ.",
        "Chúng tôi đã ghi nhận cần gọi lại số **{phone}**. Sẽ liên hệ với bạn sớm nhất có thể.",
        "Cảm ơn bạn! Chúng tôi sẽ gọi lại số **{phone}** theo yêu cầu của bạn."
    ]
}

# Intent-based templates (ĐỒNG BỘ VỚI ENTITIES.PY)
INTENT_TEMPLATES = {
    Intent.PROVIDE_PHONE: [
        "Cảm ơn bạn đã cung cấp số điện thoại **{phone}**. Chúng tôi sẽ liên hệ sớm nhất! 📞",
        "Đã nhận số điện thoại **{phone}**. Đội ngũ Ruby Wings sẽ gọi tư vấn cho bạn!",
        "Cảm ơn bạn! Số **{phone}** đã được lưu lại. Chúng tôi sẽ liên hệ trong thời gian sớm nhất.",
        "Số **{phone}** đã được ghi nhận. Chúng tôi rất mong được tư vấn trực tiếp cho bạn!"
    ],
    
    Intent.CALLBACK_REQUEST: [
        "Bạn muốn chúng tôi gọi lại khi nào? (sáng/chiều/tối)",
        "Vui lòng cho biết khung giờ phù hợp để chúng tôi gọi lại cho bạn?",
        "Để thuận tiện cho bạn, bạn muốn được gọi lại vào khoảng thời gian nào trong ngày?",
        "Khung giờ nào thuận tiện để chúng tôi liên hệ với bạn?"
    ],
    
    Intent.BOOKING_CONFIRM: [
        "Tuyệt vời! Để xác nhận đặt tour **{tour_name}**, vui lòng cung cấp số điện thoại.",
        "Bạn đã sẵn sàng đặt tour **{tour_name}**. Vui lòng cho chúng tôi số điện thoại để xác nhận.",
        "Cảm ơn bạn đã chọn **{tour_name}**! Để hoàn tất booking, vui lòng cung cấp số điện thoại.",
        "Đặt tour **{tour_name}** ngay! Vui lòng cung cấp số điện thoại để chúng tôi xác nhận."
    ],
    
    Intent.MODIFY_REQUEST: [
        "Bạn muốn thay đổi thông tin tour? Vui lòng cho biết chi tiết.",
        "Tôi sẽ giúp bạn chỉnh sửa thông tin. Bạn muốn thay đổi gì?",
        "Để hỗ trợ bạn thay đổi, vui lòng cho biết cụ thể bạn muốn điều chỉnh gì?",
        "Bạn cần thay đổi thông tin nào? Tôi sẽ hỗ trợ ngay."
    ],
    
    Intent.SMALLTALK: [
        "Xin chào! Tôi là Ruby Wings AI, rất vui được hỗ trợ bạn. 😊",
        "Chào bạn! Tôi ở đây để giúp bạn tìm tour trải nghiệm phù hợp nhất.",
        "Rất vui được trò chuyện với bạn! Bạn cần tư vấn về tour nào không?",
        "Hello! Tôi có thể giúp gì cho bạn về các tour Ruby Wings?"
    ],
    
    Intent.GREETING: [
        "Xin chào! Tôi là trợ lý AI của Ruby Wings, chuyên tư vấn tour trải nghiệm thiên nhiên và chữa lành. 🌿",
        "Chào bạn! Rất vui được gặp bạn. Tôi có thể giúp gì cho bạn về các tour Ruby Wings?",
        "Hello! Tôi là chatbot Ruby Wings, sẵn sàng hỗ trợ bạn tìm tour phù hợp nhất.",
        "Chào mừng bạn đến với Ruby Wings! Tôi có thể tư vấn tour nào cho bạn?"
    ],
    
    Intent.FAREWELL: [
        "Cảm ơn bạn đã trò chuyện! Hy vọng sớm được đồng hành cùng bạn trong hành trình trải nghiệm. ✨",
        "Tạm biệt bạn! Liên hệ hotline **0332510486** nếu cần hỗ trợ thêm nhé!",
        "Chúc bạn một ngày tốt lành! Mong sớm được gặp lại bạn trong tour Ruby Wings.",
        "Hẹn gặp lại bạn! Đừng quên hotline **0332510486** khi cần tư vấn."
    ],
    
    Intent.TOUR_INQUIRY: [
        "Tôi có thể giúp bạn tìm hiểu về tour nào?",
        "Bạn quan tâm đến loại tour nào? Tôi sẽ tư vấn chi tiết.",
        "Ruby Wings có nhiều tour độc đáo. Bạn muốn biết về tour nào?",
        "Cho tôi biết bạn quan tâm tour gì, tôi sẽ giới thiệu chi tiết."
    ],
    
    Intent.PRICE_ASK: [
        "Tôi sẽ cung cấp thông tin giá tour cho bạn:",
        "Dưới đây là thông tin về giá các tour:",
        "Giá tour như sau:",
        "Thông tin chi phí tour:"
    ]
}

# Location templates
LOCATION_TEMPLATES = {
    "no_tour_exact": [
        "Hiện chưa có tour chính xác tại **{location}**. Bạn có muốn tham khảo các tour tương tự tại **{region}** không?",
        "Ruby Wings chưa có tour ở **{location}**. Tôi có thể đề xuất tour ở khu vực **{region}** cho bạn.",
        "Không tìm thấy tour tại **{location}**. Bạn có quan tâm đến các tour tại **{region}** không?",
        "Tour tại **{location}** hiện chưa có. Tôi có thể giới thiệu tour ở **{region}** thay thế?"
    ],
    
    "tour_found": [
        "Tìm thấy **{count}** tour tại **{location}**:",
        "Dưới đây là **{count}** tour Ruby Wings tại **{location}**:",
        "Có **{count}** tour phù hợp tại **{location}** bạn có thể tham khảo:",
        "**{count}** tour tại **{location}** đang chờ bạn khám phá:"
    ],
    
    "region_fallback": [
        "Các tour tại khu vực **{region}**:",
        "Tour ở **{region}** bạn có thể quan tâm:",
        "Gợi ý tour tại **{region}**:",
        "Khám phá **{region}** cùng các tour:"
    ]
}

# ==================== HELPER FUNCTIONS ====================

def extract_source_tokens(text: str) -> List[str]:
    """Return list of citation tokens like [1] found in text."""
    return SRC_RE.findall(text or "")

def normalize_for_overlap(s: str) -> List[str]:
    """Normalize text for overlap comparison."""
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks

def overlap_ratio(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Calculate overlap ratio between two token lists."""
    if not a_tokens or not b_tokens:
        return 0.0
    ca = Counter(a_tokens)
    cb = Counter(b_tokens)
    common = sum(min(ca[t], cb.get(t, 0)) for t in ca)
    return common / max(len(a_tokens), 1)

def collect_passage_texts(top_passages: List[Tuple[float, Dict]]) -> List[str]:
    """Collect text from passages."""
    return [m.get("text", "") for _, m in (top_passages or [])]

def collect_passage_paths(top_passages: List[Tuple[float, Dict]]) -> List[str]:
    """Collect paths from passages."""
    return [m.get("path", "") for _, m in (top_passages or [])]

def safe_shorten(text: str, max_chars: int = 1200) -> str:
    """Safely shorten text to max_chars, trying to cut at sentence boundary."""
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    # Try to cut at sentence boundary
    cut = t[:max_chars].rfind(".")
    if cut > int(max_chars * 0.5):
        return t[:cut + 1]
    return t[:max_chars].rstrip() + "..."

def get_random_template(template_dict: Dict[str, List[str]], key: str, default: str = "") -> str:
    """Get random template from dict."""
    templates = template_dict.get(key, [default])
    return random.choice(templates) if templates else default

def mask_phone(phone: str) -> str:
    """Mask phone number for display."""
    if not phone or len(phone) < 4:
        return phone
    return f"{phone[:4]}***{phone[-2:]}"

# ==================== TOUR FORMATTING ====================

def extract_tour_info_from_passages(passages: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Extract structured tour information from passages.
    Compatible with tour_entities.json structure.
    """
    tours = {}
    
    for score, passage in passages:
        text = passage.get("text", "")
        path = passage.get("path", "")
        
        # Extract tour index from path
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            continue
        
        tour_idx = int(tour_match.group(1))
        
        # Initialize tour dict if not exists
        if tour_idx not in tours:
            tours[tour_idx] = {
                "index": tour_idx,
                "tour_name": "",
                "location": "",
                "duration": "",
                "price": "",
                "summary": "",
                "score": 0.0
            }
        
        # Update tour info based on text content
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Tên tour:"):
                tours[tour_idx]["tour_name"] = line.replace("Tên tour:", "").strip()
            elif line.startswith("Địa điểm:"):
                tours[tour_idx]["location"] = line.replace("Địa điểm:", "").strip()
            elif line.startswith("Thời lượng:"):
                tours[tour_idx]["duration"] = line.replace("Thời lượng:", "").strip()
            elif line.startswith("Giá:"):
                price_text = line.replace("Giá:", "").strip()
                # Truncate long prices
                if len(price_text) > 100:
                    price_text = price_text[:100] + "..."
                tours[tour_idx]["price"] = price_text
            elif line.startswith("Tóm tắt:"):
                tours[tour_idx]["summary"] = line.replace("Tóm tắt:", "").strip()
        
        # Update score (highest score for this tour)
        tours[tour_idx]["score"] = max(tours[tour_idx]["score"], score)
    
    # Convert to list and sort by score
    tour_list = list(tours.values())
    tour_list.sort(key=lambda x: x["score"], reverse=True)
    
    return tour_list

def format_tour_response(tours: List[Dict[str, Any]], max_tours: int = 3, 
                        include_summary: bool = False) -> Tuple[str, List[str]]:
    """
    Format tours with labels and structured information.
    Returns: (formatted_text, tour_labels)
    """
    if not tours:
        return "", []
    
    # Limit to max_tours
    tours = tours[:max_tours]
    tour_labels = []
    formatted_parts = []
    
    # Define labels based on position (ĐỒNG BỘ VỚI APP.PY)
    label_map = {
        0: "🏆 Phù hợp nhất",
        1: "⭐ Phổ biến",
        2: "💰 Giá tốt"
    }
    
    for i, tour in enumerate(tours):
        if not tour:
            continue
        
        # Get label
        label = label_map.get(i, f"**{i+1}.**")
        tour_labels.append(label)
        
        # Build tour block
        tour_block = f"{label} **{tour.get('tour_name', 'Tour')}**\n"
        
        # Add details if available
        if tour.get('location'):
            tour_block += f"   📍 Địa điểm: {tour['location']}\n"
        if tour.get('duration'):
            tour_block += f"   ⏱️ Thời lượng: {tour['duration']}\n"
        if tour.get('price'):
            tour_block += f"   💰 Giá: {tour['price']}\n"
        if include_summary and tour.get('summary'):
            summary = tour['summary'][:150] + "..." if len(tour['summary']) > 150 else tour['summary']
            tour_block += f"   📝 {summary}\n"
        
        formatted_parts.append(tour_block)
    
    return "\n".join(formatted_parts), tour_labels

# ==================== TEMPLATE GENERATION ====================

def generate_intent_response(intent: str, context: Dict[str, Any]) -> Optional[str]:
    """Generate intent-specific response."""
    if intent not in INTENT_TEMPLATES:
        return None
    
    template = random.choice(INTENT_TEMPLATES[intent])
    
    # Fill template variables
    phone = context.get("phone") or context.get("lead_phone") or ""
    tour_name = context.get("selected_tour_name") or context.get("tour_name") or "tour đã chọn"
    
    # Format phone for display
    if phone:
        phone_display = mask_phone(phone)
    else:
        phone_display = ""
    
    try:
        # Try to format template
        if "{phone}" in template and phone_display:
            return template.format(phone=phone_display)
        elif "{tour_name}" in template and tour_name:
            return template.format(tour_name=tour_name)
        else:
            return template
    except KeyError:
        # If template has variables but context doesn't have them, return plain template
        return template

def generate_state_fallback(state: str, context: Dict[str, Any], 
                           top_passages: List[Tuple[float, Dict[str, Any]]], 
                           requested_field: Optional[str] = None) -> str:
    """Generate state-based fallback response."""
    
    # Try to get state template
    if state in STATE_TEMPLATES:
        template = random.choice(STATE_TEMPLATES[state])
        
        # Fill template variables
        phone = context.get("phone") or context.get("lead_phone") or ""
        tour_name = context.get("selected_tour_name") or context.get("tour_name") or ""
        location = context.get("location") or ""
        
        # Format phone
        if phone:
            phone = mask_phone(phone)
        
        try:
            if "{tour_name}" in template and tour_name:
                template = template.format(tour_name=tour_name)
            if "{phone}" in template and phone:
                template = template.format(phone=phone)
            if "{location}" in template and location:
                template = template.format(location=location)
        except KeyError:
            pass  # Use template as-is if formatting fails
        
        # Add tour information for SUGGEST/COMPARE states
        if state in [ConversationStage.SUGGEST, ConversationStage.COMPARE]:
            tours_info = extract_tour_info_from_passages(top_passages)
            if tours_info:
                formatted_tours, _ = format_tour_response(tours_info, max_tours=3)
                if formatted_tours:
                    return template + "\n\n" + formatted_tours + "\n\n💡 *Liên hệ hotline **0332510486** để biết thêm chi tiết*"
        
        return template + "\n\n💡 *Liên hệ hotline **0332510486** để biết thêm chi tiết*"
    
    # Default to deterministic fallback
    return deterministic_fallback_answer(top_passages, requested_field, context=context)

def add_state_template(text: str, state: str, context: Dict[str, Any]) -> str:
    """Add state-appropriate template prefix to text."""
    if state not in STATE_TEMPLATES:
        return text
    
    # Only add template for certain states
    if state in [ConversationStage.SUGGEST, ConversationStage.COMPARE]:
        template = random.choice(STATE_TEMPLATES[state])
        
        # Check if template already present
        if not any(template_part in text for template_part in STATE_TEMPLATES[state]):
            text = template + "\n\n" + text
    
    return text

def add_location_context(text: str, location: str, tour_count: int, region: Optional[str] = None) -> str:
    """Add location context to response."""
    if not location:
        return text
    
    # Get region if not provided
    if not region:
        region = get_region_from_location(location) or "khu vực tương tự"
    
    # Check if location info already in text
    location_lower = location.lower()
    text_lower = text.lower()
    
    if location_lower not in text_lower and "địa điểm" not in text_lower:
        if tour_count > 0:
            template = random.choice(LOCATION_TEMPLATES["tour_found"])
            prefix = template.format(count=tour_count, location=location)
        else:
            template = random.choice(LOCATION_TEMPLATES["no_tour_exact"])
            prefix = template.format(location=location, region=region)
        
        text = prefix + "\n\n" + text
    
    return text

# ==================== DETERMINISTIC FALLBACK ====================

def deterministic_fallback_answer(
    top_passages: List[Tuple[float, Dict[str, Any]]], 
    requested_field: Optional[str] = None, 
    max_snippets: int = 3,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build a safe answer using only retrieved passages.
    Short, friendly, cites indexed sources [1],[2].
    If requested_field provided, prioritize passages whose path mentions that field.
    """
    context = context or {}
    
    if not top_passages:
        return "Xin lỗi — hiện không có thông tin trong tài liệu về yêu cầu của bạn.\n\n💡 *Liên hệ hotline **0332510486** để được tư vấn trực tiếp*"

    # Prioritize field passages
    prioritized = []
    others = []
    
    for score, m in top_passages:
        p = m.get("path", "")
        if requested_field and (p.endswith(f".{requested_field}") or f".{requested_field}" in p):
            prioritized.append((score, m))
        else:
            others.append((score, m))
    
    chosen = (prioritized + others)[:max_snippets]

    pieces = []
    for i, (score, m) in enumerate(chosen, start=1):
        text = m.get("text", "").strip()
        text = safe_shorten(text, 800)
        pieces.append(f"[{i}] {text}")

    # Build header
    header = ""
    if requested_field:
        header = f'Về "{requested_field}", tôi tìm thấy thông tin sau (trích từ tài liệu Ruby Wings):\n\n'
    else:
        header = "Tôi tìm thấy thông tin sau từ dữ liệu Ruby Wings:\n\n"

    footer = "\n\n💡 *Liên hệ hotline **0332510486** để biết thêm chi tiết và đặt tour*"
    
    return header + "\n\n".join(pieces) + footer

# ==================== MAIN VALIDATION FUNCTION ====================

def validate_and_format_answer(
    llm_text: str,
    top_passages: List[Tuple[float, Dict[str, Any]]],
    requested_field: Optional[str] = None,
    tour_indices: Optional[List[int]] = None,
    max_chars: int = MAX_ANSWER_CHARS,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate LLM answer against retrieved top_passages.
    If fails safety checks, return deterministic aggregated snippets instead.
    
    Enhanced with state-based templates, location-aware responses, and improved formatting.
    Fully integrated with entities.py v5.2 and tour_entities.json structure.
    
    Parameters:
      - llm_text: text returned by LLM (may be empty)
      - top_passages: list of (score, mapping_entry) where mapping_entry has 'path' and 'text'
      - requested_field: if provided, ensure answer addresses that field
      - tour_indices: list of tour indices in context (optional)
      - max_chars: maximum characters for answer
      - context: conversation context dict with state, intent, location, etc.
    
    Returns:
      - dict with answer, sources, guard_passed, reason, state, etc.
    """
    start = time.time()
    context = context or {}
    
    # Extract context values (ĐỒNG BỘ VỚI ENTITIES.PY)
    state = context.get("stage", ConversationStage.EXPLORE)
    intent = context.get("intent")
    location = context.get("location")
    location_filtered = context.get("location_filtered", False)
    has_phone = context.get("has_phone", False)
    phone = context.get("phone") or context.get("lead_phone")
    selected_tour_name = context.get("selected_tour_name") or context.get("tour_name")
    
    passages = collect_passage_texts(top_passages)
    paths = collect_passage_paths(top_passages)

    # Sanitize LLM text first
    candidate = (llm_text or "").strip()
    candidate = html.unescape(candidate)
    candidate = re.sub(r"\s+\n", "\n", candidate)
    candidate = safe_shorten(candidate, max_chars)

    # NEW: Handle intent-specific responses first
    if intent and intent in INTENT_TEMPLATES:
        intent_response = generate_intent_response(intent, context)
        if intent_response:
            return {
                "answer": intent_response,
                "sources": [],
                "guard_passed": True,
                "reason": "intent_template",
                "state": state,
                "intent": intent,
                "elapsed": time.time() - start
            }

    # 1) If no retrieved evidence at all -> state-based fallback
    if not passages:
        fallback = generate_state_fallback(state, context, top_passages, requested_field)
        return {
            "answer": fallback,
            "sources": [],
            "guard_passed": False,
            "reason": "no_evidence",
            "state": state,
            "elapsed": time.time() - start
        }

    # 2) Check for explicit citation tokens in LLM text
    cited_tokens = extract_source_tokens(candidate)
    if cited_tokens:
        # Map numeric citation tokens to mapping paths: [1] -> top_passages[0], etc.
        cited_paths = []
        for tok in cited_tokens:
            try:
                idx = int(tok.strip("[]")) - 1
                if 0 <= idx < len(top_passages):
                    cited_paths.append(paths[idx])
            except Exception:
                pass
        
        # Basic evidence overlap check
        evidence_concat = " ".join(passages[:5])
        if overlap_ratio(normalize_for_overlap(candidate), normalize_for_overlap(evidence_concat)) >= MIN_OVERLAP_RATIO:
            # Add state template if appropriate
            if state in [ConversationStage.SUGGEST, ConversationStage.COMPARE]:
                candidate = add_state_template(candidate, state, context)
            
            # Add location context if filtered
            if location_filtered and location:
                candidate = add_location_context(candidate, location, len(passages))
            
            return {
                "answer": candidate,
                "sources": cited_paths or paths[:3],
                "guard_passed": True,
                "reason": "ok_with_citations",
                "state": state,
                "location_filtered": location_filtered,
                "elapsed": time.time() - start
            }

    # 3) Token-overlap heuristic between LLM output and evidence
    evidence_concat = " ".join(passages[:5])
    ov = overlap_ratio(normalize_for_overlap(candidate), normalize_for_overlap(evidence_concat))
    
    if ov >= MIN_OVERLAP_RATIO:
        # 3a) If requested_field is provided, ensure candidate mentions field-specific content
        if requested_field:
            # Find passages matching requested_field by path suffix
            field_passages = [
                m.get("text", "") for _, m in top_passages 
                if (m.get("path", "").endswith(f".{requested_field}") or f".{requested_field}" in m.get("path", ""))
            ]
            
            if field_passages:
                field_ov = overlap_ratio(
                    normalize_for_overlap(candidate), 
                    normalize_for_overlap(" ".join(field_passages[:4]))
                )
                
                if field_ov < MIN_FIELD_MENTION_RATIO:
                    # Mismatch: LLM didn't address requested field sufficiently
                    fallback = generate_state_fallback(state, context, top_passages, requested_field)
                    return {
                        "answer": fallback,
                        "sources": collect_passage_paths(top_passages)[:3],
                        "guard_passed": False,
                        "reason": "mismatch_field",
                        "state": state,
                        "elapsed": time.time() - start
                    }
        
        # 3b) Ban hedging phrases to enforce professional tone
        low = candidate.lower()
        banned_found = []
        for banned in BANNED_PHRASES:
            if banned in low:
                banned_found.append(banned)
                low = low.replace(banned, "")
        
        # If too many banned phrases, use fallback
        if len(banned_found) > 2:
            logger.warning(f"Too many banned phrases in LLM response: {banned_found}")
            fallback = generate_state_fallback(state, context, top_passages, requested_field)
            return {
                "answer": fallback,
                "sources": collect_passage_paths(top_passages)[:3],
                "guard_passed": False,
                "reason": "too_many_banned_phrases",
                "state": state,
                "elapsed": time.time() - start
            }
        
        candidate = safe_shorten(candidate, max_chars)
        
        # Add location context if applicable
        if location_filtered and location:
            candidate = add_location_context(candidate, location, len(passages))
        
        # Add state template
        candidate = add_state_template(candidate, state, context)
        
        return {
            "answer": candidate,
            "sources": collect_passage_paths(top_passages)[:3],
            "guard_passed": True,
            "reason": "ok",
            "overlap": round(ov, 3),
            "state": state,
            "location_filtered": location_filtered,
            "elapsed": time.time() - start
        }

    # 4) Low overlap -> LLM likely hallucinated -> state-based deterministic fallback
    logger.warning(f"Low overlap detected: {ov:.3f} < {MIN_OVERLAP_RATIO}")
    
    fallback = generate_state_fallback(state, context, top_passages, requested_field)
    
    # Extract tour info for formatting
    tours_info = extract_tour_info_from_passages(top_passages)
    formatted_tours, tour_labels = format_tour_response(tours_info, max_tours=3)
    
    # Add formatted tours to fallback if available
    if formatted_tours and state in [ConversationStage.SUGGEST, ConversationStage.COMPARE, ConversationStage.EXPLORE]:
        if not fallback.endswith("\n\n"):
            fallback += "\n\n"
        fallback += formatted_tours
        fallback += "\n\n💡 *Liên hệ hotline **0332510486** để biết thêm chi tiết*"
    
    return {
        "answer": fallback,
        "sources": collect_passage_paths(top_passages)[:3],
        "guard_passed": False,
        "reason": "low_overlap",
        "overlap": round(ov, 3),
        "state": state,
        "tour_labels": tour_labels,
        "location_filtered": location_filtered,
        "elapsed": time.time() - start
    }

# ==================== UTILITY FUNCTIONS ====================

def sanitize_answer(text: str) -> str:
    """Sanitize answer text for safe output."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    
    # Trim
    text = text.strip()
    
    return text

def add_hotline_cta(text: str) -> str:
    """Add hotline CTA if not already present."""
    if "0332510486" in text or "hotline" in text.lower():
        return text
    
    return text + "\n\n💡 *Liên hệ hotline **0332510486** để biết thêm chi tiết*"

def format_price_text(price_text: str, max_length: int = 100) -> str:
    """Format price text for display."""
    if not price_text:
        return ""
    
    # Truncate if too long
    if len(price_text) > max_length:
        price_text = price_text[:max_length] + "..."
    
    return price_text

# ==================== TESTING ====================

def test_response_guard():
    """Test response guard functionality."""
    print("=" * 60)
    print("TESTING RESPONSE GUARD v5.2")
    print("=" * 60)
    
    # Sample passages
    sample_passages = [
        (1.0, {"path": "root.tours[0].price", "text": "Tên tour: Non nước Bạch Mã\nĐịa điểm: Vườn quốc gia Bạch Mã\nThời lượng: 1 ngày\nGiá: 890.000 VNĐ/khách"}),
        (0.9, {"path": "root.tours[0].transport", "text": "Phương tiện: Xe 7-16 chỗ đời mới"}),
        (0.8, {"path": "root.tours[1].tour_name", "text": "Tên tour: Mưa Đỏ và Trường Sơn – Hành Trình Khát Vọng\nĐịa điểm: Quảng Trị\nThời lượng: 2 ngày 1 đêm\nGiá: 1.700.000 – 2.300.000 VNĐ/người"})
    ]
    
    # Test 1: With context
    print("\n### Test 1: State-based response (SUGGEST)")
    context1 = {
        "stage": ConversationStage.SUGGEST,
        "intent": Intent.TOUR_INQUIRY,
        "location": "Huế",
        "location_filtered": True
    }
    
    llm_good = "Giá tour Bạch Mã là 890.000 VNĐ/khách. [1]"
    result1 = validate_and_format_answer(llm_good, sample_passages, context=context1)
    print(f"Guard passed: {result1['guard_passed']}")
    print(f"Reason: {result1['reason']}")
    print(f"Answer preview: {result1['answer'][:200]}...")
    
    # Test 2: Intent template
    print("\n### Test 2: Intent-based response (PROVIDE_PHONE)")
    context2 = {
        "intent": Intent.PROVIDE_PHONE,
        "phone": "0909123456",
        "stage": ConversationStage.LEAD
    }
    
    result2 = validate_and_format_answer("", sample_passages, context=context2)
    print(f"Guard passed: {result2['guard_passed']}")
    print(f"Answer: {result2['answer']}")
    
    # Test 3: Tour formatting
    print("\n### Test 3: Tour formatting")
    tours_info = extract_tour_info_from_passages(sample_passages)
    formatted, labels = format_tour_response(tours_info)
    print(f"Formatted tours:\n{formatted}")
    print(f"Labels: {labels}")
    
    # Test 4: Low overlap (hallucination)
    print("\n### Test 4: Low overlap detection")
    llm_bad = "Bạn chỉ cần mang 10 triệu và mọi thứ sẽ ổn."
    result4 = validate_and_format_answer(llm_bad, sample_passages, context=context1)
    print(f"Guard passed: {result4['guard_passed']}")
    print(f"Reason: {result4['reason']}")
    print(f"Overlap: {result4.get('overlap', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

# ==================== EXPORTS ====================

__all__ = [
    # Main function
    'validate_and_format_answer',
    
    # Helper functions
    'extract_tour_info_from_passages',
    'format_tour_response',
    'generate_intent_response',
    'generate_state_fallback',
    'add_state_template',
    'add_location_context',
    'deterministic_fallback_answer',
    'sanitize_answer',
    'add_hotline_cta',
    
    # Test function
    'test_response_guard'
]

# ==================== MAIN ====================

if __name__ == "__main__":
    # Run tests
    test_response_guard()