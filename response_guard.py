#!/usr/bin/env python3
"""
response_guard.py v5.2

Enhanced "expert guard" to validate & format final answers before sending to user.
Now includes state-based templates, location-aware responses, and improved tour formatting.

Responsibilities:
- Ensure answers cite sources (e.g., [1], [2]) or attach retrieved snippets if LLM hallucinated.
- Ensure answer content is consistent with retrieved evidence (simple token overlap check).
- Ensure requested_field is respected (if provided) by preferring passages for that field.
- Enforce friendly "healing travel" tone with short sanitization heuristics.
- Provide deterministic fallback that only uses retrieved passages when LLM output fails checks.
- NEW: State-based response templates for different conversation stages.
- NEW: Location-aware response formatting with region suggestions.
- NEW: Tour response formatting with labels (🏆, ⭐, 💰).
- NEW: Intent-specific response templates.

Usage (minimal):
  from response_guard import validate_and_format_answer
  out = validate_and_format_answer(
      llm_text=llm_text,
      top_passages=top_passages,            # List[Tuple[score, mapping_entry]]
      requested_field=requested_field,      # optional string
      tour_indices=tour_indices,            # optional list[int]
      max_tokens=700,
      context={}                           # NEW: conversation context
  )
  return jsonify(out)

Return value:
  {
    "answer": "<final text to send user>",
    "sources": ["root.tours[2].price", ...],   # list of mapping paths used
    "guard_passed": True/False,
    "reason": "ok" | "no_evidence" | "mismatch_field" | ...,
    "state": "explore" | "suggest" | ...,
    "tour_labels": [],  # NEW: tour labels used in response
    "location_filtered": False  # NEW: if location filter was applied
  }
"""

import re
import html
import time
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import Counter
from datetime import datetime

# Import necessary enums from entities (simplified)
class ConversationStage:
    """Simplified ConversationStage for response_guard"""
    EXPLORE = "explore"
    SUGGEST = "suggest"
    COMPARE = "compare"
    SELECT = "select"
    BOOK = "book"
    LEAD = "lead"
    CALLBACK = "callback"

class Intent:
    """Simplified Intent for response_guard"""
    PROVIDE_PHONE = "provide_phone"
    CALLBACK_REQUEST = "callback_request"
    BOOKING_CONFIRM = "booking_confirm"
    MODIFY_REQUEST = "modify_request"
    SMALLTALK = "smalltalk"
    LEAD_CAPTURED = "lead_captured"
    GREETING = "greeting"
    FAREWELL = "farewell"
    TOUR_INQUIRY = "tour_inquiry"
    UNKNOWN = "unknown"

# --- Simple helpers ---
SRC_RE = re.compile(r"\[\d+\]")  # detect [1], [2] style citations

def extract_source_tokens(text: str) -> List[str]:
    """Return list of citation tokens like [1] found in text."""
    return SRC_RE.findall(text or "")

def normalize_for_overlap(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks

def overlap_ratio(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    ca = Counter(a_tokens)
    cb = Counter(b_tokens)
    common = sum(min(ca[t], cb.get(t, 0)) for t in ca)
    return common / max(len(a_tokens), 1)

def collect_passage_texts(top_passages: List[Tuple[float, Dict]]) -> List[str]:
    return [m.get("text","") for _, m in (top_passages or [])]

def collect_passage_paths(top_passages: List[Tuple[float, Dict]]) -> List[str]:
    return [m.get("path","") for _, m in (top_passages or [])]

def safe_shorten(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    # try to cut at sentence boundary
    cut = t[:max_chars].rfind(".")
    if cut > int(max_chars*0.5):
        return t[:cut+1]
    return t[:max_chars].rstrip() + "..."

# --- Guard rules / parameters ---
MIN_OVERLAP_RATIO = 0.12   # minimal overlap between LLM text and evidence to accept
MIN_FIELD_MENTION_RATIO = 0.02  # small threshold to allow field-specific match via text overlap
MAX_ANSWER_CHARS = 1500
BANNED_PHRASES = ["i think", "i guess", "maybe", "probably", "as far as i know", "i'm not sure"]

# --- NEW: Response Templates by State ---
STATE_TEMPLATES = {
    ConversationStage.EXPLORE: [
        "Tôi có thể giúp gì cho bạn về tour du lịch trải nghiệm Ruby Wings? 🌿",
        "Bạn muốn tìm hiểu về tour du lịch nào của Ruby Wings? 😊",
        "Chào bạn! Tôi có thể tư vấn cho bạn về các hành trình trải nghiệm của Ruby Wings."
    ],
    
    ConversationStage.SUGGEST: [
        "Dựa trên yêu cầu của bạn, tôi đề xuất 3 tour sau:",
        "Tôi tìm thấy một số tour phù hợp với bạn:",
        "Dưới đây là các tour Ruby Wings bạn có thể quan tâm:"
    ],
    
    ConversationStage.COMPARE: [
        "Để so sánh các tour, tôi tóm tắt thông tin chính:",
        "Dưới đây là thông tin so sánh giữa các tour:",
        "Tôi sẽ giúp bạn so sánh các tour để chọn phù hợp nhất:"
    ],
    
    ConversationStage.SELECT: [
        "Bạn đã chọn tour {tour_name}. Bạn muốn đặt tour này không?",
        "Tour {tour_name} rất phù hợp với bạn! Bạn muốn tiếp tục đặt tour không?",
        "Tuyệt vời! Tour {tour_name} đã được chọn. Bạn có muốn đặt ngay không?"
    ],
    
    ConversationStage.BOOK: [
        "Tour {tour_name} đã được đặt. Vui lòng cung cấp số điện thoại để chúng tôi liên hệ xác nhận.",
        "Booking thành công! Chúng tôi sẽ liên hệ với bạn qua số điện thoại để xác nhận chi tiết.",
        "Đã xác nhận đặt tour {tour_name}. Vui lòng cho chúng tôi số điện thoại để hoàn tất thủ tục."
    ],
    
    ConversationStage.LEAD: [
        "Đã lưu số {phone}. Chúng tôi sẽ gọi lại cho bạn trong 30 phút.",
        "Cảm ơn bạn đã cung cấp số điện thoại {phone}. Đội ngũ Ruby Wings sẽ liên hệ sớm nhất!",
        "Số điện thoại {phone} đã được ghi nhận. Chúng tôi sẽ liên hệ tư vấn cho bạn sớm."
    ],
    
    ConversationStage.CALLBACK: [
        "Đã ghi nhận yêu cầu gọi lại. Chúng tôi sẽ liên hệ số {phone} trong ngày hôm nay.",
        "Yêu cầu gọi lại đã được xác nhận. Chúng tôi sẽ gọi số {phone} trong vòng 2 giờ.",
        "Chúng tôi đã ghi nhận cần gọi lại số {phone}. Sẽ liên hệ với bạn sớm nhất có thể."
    ]
}

# --- NEW: Intent Templates ---
INTENT_TEMPLATES = {
    Intent.PROVIDE_PHONE: [
        "Cảm ơn bạn đã cung cấp số điện thoại {phone}. Chúng tôi sẽ liên hệ sớm nhất! 📞",
        "Đã nhận số điện thoại {phone}. Đội ngũ Ruby Wings sẽ gọi tư vấn cho bạn!",
        "Cảm ơn bạn! Số {phone} đã được lưu lại. Chúng tôi sẽ liên hệ trong thời gian sớm nhất."
    ],
    
    Intent.CALLBACK_REQUEST: [
        "Bạn muốn chúng tôi gọi lại khi nào? (sáng/chiều/tối)",
        "Vui lòng cho biết khung giờ phù hợp để chúng tôi gọi lại cho bạn?",
        "Để thuận tiện cho bạn, bạn muốn được gọi lại vào khoảng thời gian nào trong ngày?"
    ],
    
    Intent.SMALLTALK: [
        "Xin chào! Tôi là Ruby Wings AI, rất vui được hỗ trợ bạn. 😊",
        "Chào bạn! Tôi ở đây để giúp bạn tìm tour trải nghiệm phù hợp nhất.",
        "Rất vui được trò chuyện với bạn! Bạn cần tư vấn về tour nào không?"
    ],
    
    Intent.GREETING: [
        "Xin chào! Tôi là trợ lý AI của Ruby Wings, chuyên tư vấn tour trải nghiệm thiên nhiên và chữa lành. 🌿",
        "Chào bạn! Rất vui được gặp bạn. Tôi có thể giúp gì cho bạn về các tour Ruby Wings?",
        "Hello! Tôi là chatbot Ruby Wings, sẵn sàng hỗ trợ bạn tìm tour phù hợp nhất."
    ],
    
    Intent.FAREWELL: [
        "Cảm ơn bạn đã trò chuyện! Hy vọng sớm được đồng hành cùng bạn trong hành trình trải nghiệm. ✨",
        "Tạm biệt bạn! Liên hệ hotline 0332510486 nếu cần hỗ trợ thêm nhé!",
        "Chúc bạn một ngày tốt lành! Mong sớm được gặp lại bạn trong tour Ruby Wings."
    ]
}

# --- NEW: Location Templates ---
LOCATION_TEMPLATES = {
    "no_tour_exact": [
        "Không có tour tại {location}. Bạn có muốn tham khảo các tour tương tự tại {region} không?",
        "Hiện chưa có tour nào tại {location}. Tôi có thể đề xuất tour ở khu vực {region} nhé?",
        "Ruby Wings chưa có tour ở {location}. Bạn có quan tâm đến tour tại {region} không?"
    ],
    
    "tour_found": [
        "Tìm thấy {count} tour tại {location}:",
        "Dưới đây là các tour Ruby Wings tại {location}:",
        "Có {count} tour phù hợp tại {location} bạn có thể tham khảo:"
    ]
}

# --- NEW: Region Mapping ---
REGION_MAPPING = {
    "đà nẵng": "Miền Trung",
    "huế": "Miền Trung",
    "quảng trị": "Miền Trung",
    "bạch mã": "Miền Trung",
    "hội an": "Miền Trung",
    "hà nội": "Miền Trung Bắc",  # Special case
    "hạ long": "Miền Bắc",
    "sapa": "Miền Bắc",
    "ninh bình": "Miền Bắc",
    "hồ chí minh": "Miền Nam",
    "sài gòn": "Miền Nam",
    "cần thơ": "Miền Nam",
    "phú quốc": "Miền Nam",
    "nha trang": "Miền Nam",
    "đà lạt": "Miền Nam"
}

# --- NEW: Tour Formatting Helpers ---
def format_tour_response(tours: List[Dict[str, Any]], max_tours: int = 3) -> Tuple[str, List[str]]:
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
    
    # Define labels based on position
    label_map = {
        0: "🏆 Phù hợp nhất",
        1: "⭐ Phổ biến",
        2: "💰 Giá tốt"
    }
    
    for i, tour in enumerate(tours):
        if not tour:
            continue
            
        # Get label
        label = label_map.get(i, f"{i+1}.")
        tour_labels.append(label)
        
        # Build tour line
        tour_line = f"{label} **{tour.get('tour_name', 'Tour')}**\n"
        
        # Add details if available
        if tour.get('location'):
            tour_line += f"   📍 {tour['location']}\n"
        if tour.get('duration'):
            tour_line += f"   ⏱️ {tour['duration']}\n"
        if tour.get('price'):
            price = tour['price']
            if len(price) > 100:  # Truncate very long prices
                price = price[:100] + "..."
            tour_line += f"   💰 {price}\n"
        
        formatted_parts.append(tour_line)
    
    return "\n".join(formatted_parts), tour_labels

def extract_tour_info_from_passages(passages: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract structured tour information from passages."""
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
                "score": 0.0
            }
        
        # Update tour info based on text content
        if "Tên tour:" in text:
            for line in text.split('\n'):
                if line.startswith("Tên tour:"):
                    tours[tour_idx]["tour_name"] = line.replace("Tên tour:", "").strip()
                    break
        elif "Địa điểm:" in text:
            for line in text.split('\n'):
                if line.startswith("Địa điểm:"):
                    tours[tour_idx]["location"] = line.replace("Địa điểm:", "").strip()
                    break
        elif "Thời lượng:" in text:
            for line in text.split('\n'):
                if line.startswith("Thời lượng:"):
                    tours[tour_idx]["duration"] = line.replace("Thời lượng:", "").strip()
                    break
        elif "Giá:" in text:
            for line in text.split('\n'):
                if line.startswith("Giá:"):
                    tours[tour_idx]["price"] = line.replace("Giá:", "").strip()
                    break
        
        # Update score (highest score for this tour)
        tours[tour_idx]["score"] = max(tours[tour_idx]["score"], score)
    
    # Convert to list and sort by score
    tour_list = list(tours.values())
    tour_list.sort(key=lambda x: x["score"], reverse=True)
    
    return tour_list

def get_random_template(template_dict: Dict[str, List[str]], key: str, default: str = "") -> str:
    """Get random template from dict."""
    templates = template_dict.get(key, [default])
    return random.choice(templates) if templates else default

# --- Core function (Enhanced) ---
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
    
    NEW: Supports state-based templates, location-aware responses, and improved formatting.
    
    Parameters:
      - llm_text: text returned by LLM (may be empty)
      - top_passages: list of (score, mapping_entry) where mapping_entry has 'path' and 'text'
      - requested_field: if provided, ensure answer addresses that field
      - tour_indices: list of tour indices in context (optional)
      - context: conversation context dict with state, intent, location, etc.
    """
    start = time.time()
    context = context or {}
    
    # Extract context values
    state = context.get("stage", ConversationStage.EXPLORE)
    intent = context.get("intent")
    location = context.get("location")
    location_filtered = context.get("location_filtered", False)
    has_phone = context.get("has_phone", False)
    phone = context.get("phone") or context.get("lead_phone")
    selected_tour_name = context.get("selected_tour_name")
    
    passages = collect_passage_texts(top_passages)
    paths = collect_passage_paths(top_passages)

    # sanitize LLM text first
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
        # map numeric citation tokens to mapping paths if possible: assume [1] -> top_passages[0], ...
        cited_paths = []
        for tok in cited_tokens:
            try:
                idx = int(tok.strip("[]")) - 1
                if 0 <= idx < len(top_passages):
                    cited_paths.append(paths[idx])
            except Exception:
                pass
        # basic evidence overlap check
        evidence_concat = " ".join(passages[:5])
        if overlap_ratio(normalize_for_overlap(candidate), normalize_for_overlap(evidence_concat)) >= MIN_OVERLAP_RATIO:
            # NEW: Add state template if appropriate
            if state in [ConversationStage.SUGGEST, ConversationStage.COMPARE]:
                candidate = add_state_template(candidate, state, context)
            
            return {
                "answer": candidate,
                "sources": cited_paths or paths[:3],
                "guard_passed": True,
                "reason": "ok",
                "state": state,
                "elapsed": time.time() - start
            }

    # 3) Token-overlap heuristic between LLM output and evidence
    evidence_concat = " ".join(passages[:5])
    ov = overlap_ratio(normalize_for_overlap(candidate), normalize_for_overlap(evidence_concat))
    if ov >= MIN_OVERLAP_RATIO:
        # 3a) if requested_field is provided ensure candidate mentions field-specific content from passages
        if requested_field:
            # find passages matching requested_field by path suffix
            field_passages = [m.get("text","") for _, m in top_passages if (m.get("path","").endswith(f".{requested_field}") or f".{requested_field}" in m.get("path",""))]
            if field_passages:
                field_ov = overlap_ratio(normalize_for_overlap(candidate), normalize_for_overlap(" ".join(field_passages[:4])))
                if field_ov < MIN_FIELD_MENTION_RATIO:
                    # mismatch: LLM didn't address requested field sufficiently
                    fallback = generate_state_fallback(state, context, top_passages, requested_field)
                    return {
                        "answer": fallback,
                        "sources": collect_passage_paths(top_passages)[:3],
                        "guard_passed": False,
                        "reason": "mismatch_field",
                        "state": state,
                        "elapsed": time.time() - start
                    }
        # 3b) ban hedging phrases to enforce professional tone where possible
        low = candidate.lower()
        for banned in BANNED_PHRASES:
            if banned in low:
                # remove banned phrase and continue; if too many banned phrases, fallback
                low = low.replace(banned, "")
        
        candidate = safe_shorten(candidate, max_chars)
        
        # NEW: Add location context if applicable
        if location_filtered and location:
            candidate = add_location_context(candidate, location, len(passages))
        
        # NEW: Add state template
        candidate = add_state_template(candidate, state, context)
        
        return {
            "answer": candidate,
            "sources": collect_passage_paths(top_passages)[:3],
            "guard_passed": True,
            "reason": "ok",
            "overlap": ov,
            "state": state,
            "location_filtered": location_filtered,
            "elapsed": time.time() - start
        }

    # 4) Low overlap -> LLM likely hallucinated -> state-based deterministic fallback
    fallback = generate_state_fallback(state, context, top_passages, requested_field)
    
    # NEW: Extract tour info for formatting
    tours_info = extract_tour_info_from_passages(top_passages)
    formatted_tours, tour_labels = format_tour_response(tours_info, max_tours=3)
    
    # Add formatted tours to fallback if available
    if formatted_tours and state in [ConversationStage.SUGGEST, ConversationStage.COMPARE, ConversationStage.EXPLORE]:
        if not fallback.endswith("\n\n"):
            fallback += "\n\n"
        fallback += formatted_tours
    
    return {
        "answer": fallback,
        "sources": collect_passage_paths(top_passages)[:3],
        "guard_passed": False,
        "reason": "low_overlap",
        "overlap": ov,
        "state": state,
        "tour_labels": tour_labels,
        "location_filtered": location_filtered,
        "elapsed": time.time() - start
    }

# --- NEW: Template Generation Functions ---
def generate_intent_response(intent: str, context: Dict[str, Any]) -> Optional[str]:
    """Generate intent-specific response."""
    templates = INTENT_TEMPLATES.get(intent)
    if not templates:
        return None
    
    template = random.choice(templates)
    
    # Fill template variables
    phone = context.get("phone") or context.get("lead_phone") or ""
    
    if intent == Intent.PROVIDE_PHONE and phone:
        return template.format(phone=phone)
    elif intent == Intent.CALLBACK_REQUEST and phone:
        return template + f"\n\nSố điện thoại của bạn là {phone} đúng không?"
    elif intent == Intent.BOOKING_CONFIRM:
        tour_name = context.get("selected_tour_name") or "tour đã chọn"
        return template.format(tour_name=tour_name)
    else:
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
        tour_name = context.get("selected_tour_name") or ""
        location = context.get("location") or ""
        
        if state == ConversationStage.SELECT and tour_name:
            return template.format(tour_name=tour_name)
        elif state == ConversationStage.BOOK and tour_name:
            return template.format(tour_name=tour_name)
        elif state == ConversationStage.LEAD and phone:
            return template.format(phone=phone)
        elif state == ConversationStage.CALLBACK and phone:
            return template.format(phone=phone)
        else:
            return template
    
    # Default to deterministic fallback
    return deterministic_fallback_answer(top_passages, requested_field)

def add_state_template(text: str, state: str, context: Dict[str, Any]) -> str:
    """Add state-appropriate template to text."""
    if state not in STATE_TEMPLATES:
        return text
    
    # Only add template for certain states
    if state in [ConversationStage.SUGGEST, ConversationStage.COMPARE]:
        template = random.choice(STATE_TEMPLATES[state])
        
        # Check if template already present
        if not any(template_part in text for template_part in STATE_TEMPLATES[state]):
            text = template + "\n\n" + text
    
    return text

def add_location_context(text: str, location: str, tour_count: int) -> str:
    """Add location context to response."""
    if not location:
        return text
    
    region = REGION_MAPPING.get(location.lower(), "khu vực tương tự")
    
    # Check if location info already in text
    location_lower = location.lower()
    text_lower = text.lower()
    
    if location_lower not in text_lower and "location" not in text_lower and "địa điểm" not in text_lower:
        if tour_count > 0:
            template = random.choice(LOCATION_TEMPLATES["tour_found"])
            prefix = template.format(count=tour_count, location=location)
        else:
            template = random.choice(LOCATION_TEMPLATES["no_tour_exact"])
            prefix = template.format(location=location, region=region)
        
        text = prefix + "\n\n" + text
    
    return text

# --- Deterministic fallback builder (Enhanced) ---
def deterministic_fallback_answer(
    top_passages: List[Tuple[float, Dict[str, Any]]], 
    requested_field: Optional[str] = None, 
    max_snippets: int = 3,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build a safe answer using only retrieved passages. Short, friendly, cites indexed sources [1],[2].
    If requested_field provided, prioritize passages whose path mentions that field.
    """
    if not top_passages:
        return "Xin lỗi — hiện không có thông tin trong tài liệu về yêu cầu của bạn."

    # prioritize field passages
    prioritized = []
    others = []
    for score, m in top_passages:
        p = m.get("path","")
        if requested_field and (p.endswith(f".{requested_field}") or f".{requested_field}" in p):
            prioritized.append((score, m))
        else:
            others.append((score, m))
    chosen = (prioritized + others)[:max_snippets]

    pieces = []
    for i, (score, m) in enumerate(chosen, start=1):
        text = m.get("text","").strip()
        text = safe_shorten(text, 800)
        pieces.append(f"[{i}] {text}")

    header = ""
    if requested_field:
        header = f'Về "{requested_field}", tôi tìm thấy thông tin sau (trích từ tài liệu Ruby Wings):\n\n'
    else:
        header = "Tôi tìm thấy thông tin sau từ dữ liệu Ruby Wings:\n\n"

    footer = "\n\n💡 *Liên hệ hotline 0332510486 để biết thêm chi tiết và đặt tour*"
    return header + "\n\n".join(pieces) + footer

# --- Small CLI for quick manual tests ---
if __name__ == "__main__":
    # quick smoke test with new features
    sample_passages = [
        (1.0, {"path": "root.tours[0].price", "text": "Giá tour: 2.500.000 VNĐ/khách (tham khảo)."}),
        (0.9, {"path": "root.tours[0].transport", "text": "Phương tiện: Xe 16 chỗ đời mới."}),
        (0.8, {"path": "root.tours[1].tour_name", "text": "Dấu ấn Vĩ tuyến – Kết nối thế hệ"})
    ]
    
    # Test with context
    context = {
        "stage": ConversationStage.SUGGEST,
        "intent": Intent.TOUR_INQUIRY,
        "location": "Huế",
        "location_filtered": True
    }
    
    llm_good = "Giá tour là 2.500.000 VNĐ/khách. [1]"
    llm_bad = "Bạn chỉ cần mang 10 triệu và mọi thứ sẽ ổn."  # hallucination
    
    print("=== TEST WITH CONTEXT ===")
    print("GOOD:", validate_and_format_answer(
        llm_good, sample_passages, requested_field="price", context=context
    ))
    
    print("\n=== TEST INTENT TEMPLATE ===")
    intent_context = {
        "intent": Intent.PROVIDE_PHONE,
        "phone": "0909123456",
        "stage": ConversationStage.LEAD
    }
    print("INTENT RESPONSE:", generate_intent_response(Intent.PROVIDE_PHONE, intent_context))
    
    print("\n=== TEST TOUR FORMATTING ===")
    tours_info = [
        {"tour_name": "Non nước Bạch Mã", "location": "Huế", "duration": "1 ngày", "price": "890.000 VNĐ"},
        {"tour_name": "Mưa Đỏ và Trường Sơn", "location": "Quảng Trị", "duration": "2 ngày 1 đêm", "price": "1.500.000 VNĐ"},
        {"tour_name": "Ký ức Lịch Sử", "location": "Quảng Trị - Huế", "duration": "2 ngày 1 đêm", "price": "2.200.000 VNĐ"}
    ]
    formatted, labels = format_tour_response(tours_info)
    print("FORMATTED TOURS:\n", formatted)
    print("LABELS:", labels)
    
    print("\n=== TEST LOCATION TEMPLATE ===")
    text = "Tour rất thú vị"
    enhanced = add_location_context(text, "Đà Nẵng", 3)
    print("ORIGINAL:", text)
    print("ENHANCED:", enhanced)