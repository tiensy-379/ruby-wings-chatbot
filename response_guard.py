#!/usr/bin/env python3
"""
response_guard.py

Lightweight "expert guard" to validate & format final answers before sending to user.

Responsibilities:
- Ensure answers cite sources (e.g., [1], [2]) or attach retrieved snippets if LLM hallucinated.
- Ensure answer content is consistent with retrieved evidence (simple token overlap check).
- Ensure requested_field is respected (if provided) by preferring passages for that field.
- Enforce friendly "healing travel" tone with short sanitization heuristics.
- Provide deterministic fallback that only uses retrieved passages when LLM output fails checks.

Usage (minimal):
  from response_guard import validate_and_format_answer
  out = validate_and_format_answer(
      llm_text=llm_text,
      top_passages=top_passages,            # List[Tuple[score, mapping_entry]]
      requested_field=requested_field,      # optional string
      tour_indices=tour_indices,            # optional list[int]
      max_tokens=700
  )
  return jsonify(out)

Return value:
  {
    "answer": "<final text to send user>",
    "sources": ["root.tours[2].price", ...],   # list of mapping paths used
    "guard_passed": True/False,
    "reason": "ok" | "no_evidence" | "mismatch_field" | ...
  }
"""

import re
import html
import time
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

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

# --- Core function ---
def validate_and_format_answer(
    llm_text: str,
    top_passages: List[Tuple[float, Dict[str, Any]]],
    requested_field: Optional[str] = None,
    tour_indices: Optional[List[int]] = None,
    max_chars: int = MAX_ANSWER_CHARS
) -> Dict[str, Any]:
    """
    Validate LLM answer against retrieved top_passages.
    If fails safety checks, return deterministic aggregated snippets instead.

    Parameters:
      - llm_text: text returned by LLM (may be empty)
      - top_passages: list of (score, mapping_entry) where mapping_entry has 'path' and 'text'
      - requested_field: if provided, ensure answer addresses that field
      - tour_indices: list of tour indices in context (optional)
    """
    start = time.time()
    passages = collect_passage_texts(top_passages)
    paths = collect_passage_paths(top_passages)

    # sanitize LLM text first
    candidate = (llm_text or "").strip()
    candidate = html.unescape(candidate)
    candidate = re.sub(r"\s+\n", "\n", candidate)
    candidate = safe_shorten(candidate, max_chars)

    # 1) If no retrieved evidence at all -> deterministic fallback
    if not passages:
        fallback = deterministic_fallback_answer(top_passages, requested_field)
        return {"answer": fallback, "sources": [], "guard_passed": False, "reason": "no_evidence"}

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
            return {"answer": candidate, "sources": cited_paths or paths[:3], "guard_passed": True, "reason": "ok", "elapsed": time.time()-start}
        # else: citations present but text not grounded enough -> fall through to further checks

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
                    fallback = deterministic_fallback_answer(top_passages, requested_field)
                    return {"answer": fallback, "sources": collect_passage_paths(top_passages)[:3], "guard_passed": False, "reason": "mismatch_field"}
        # 3b) ban hedging phrases to enforce professional tone where possible
        low = candidate.lower()
        for banned in BANNED_PHRASES:
            if banned in low:
                # remove banned phrase and continue; if too many banned phrases, fallback
                low = low.replace(banned, "")
        candidate = safe_shorten(candidate, max_chars)
        return {"answer": candidate, "sources": collect_passage_paths(top_passages)[:3], "guard_passed": True, "reason": "ok", "overlap": ov, "elapsed": time.time()-start}

    # 4) Low overlap -> LLM likely hallucinated -> deterministic fallback
    fallback = deterministic_fallback_answer(top_passages, requested_field)
    return {"answer": fallback, "sources": collect_passage_paths(top_passages)[:3], "guard_passed": False, "reason": "low_overlap", "overlap": ov, "elapsed": time.time()-start}

# --- Deterministic fallback builder ---
def deterministic_fallback_answer(top_passages: List[Tuple[float, Dict[str, Any]]], requested_field: Optional[str] = None, max_snippets: int = 3) -> str:
    """
    Build a safe answer using only retrieved passages. Short, friendly, cites indexed sources [1],[2].
    If requested_field provided, prioritize passages whose path mentions that field.
    """
    if not top_passages:
        return "Xin lỗi — hiện không có thông tin trong tài liệu."

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
        header = f'Về "{requested_field}", tôi tìm thấy thông tin sau (trích từ tài liệu):\n\n'
    else:
        header = "Tôi tìm thấy thông tin sau (trích từ tài liệu):\n\n"

    footer = "\n\nNếu bạn cần chi tiết hơn về mục nào, hãy cho tôi biết tên tour hoặc hỏi cụ thể trường (ví dụ: 'giá', 'chương trình')."
    return header + "\n\n".join(pieces) + footer

# --- Small CLI for quick manual tests ---
if __name__ == "__main__":
    # quick smoke test
    sample_passages = [
        (1.0, {"path": "root.tours[0].price", "text": "Giá tour: 2.500.000 VNĐ/khách (tham khảo)."}),
        (0.9, {"path": "root.tours[0].transport", "text": "Phương tiện: Xe 16 chỗ đời mới."}),
        (0.8, {"path": "root.tours[1].tour_name", "text": "Dấu ấn Vĩ tuyến – Kết nối thế hệ"})
    ]
    llm_good = "Giá tour là 2.500.000 VNĐ/khách. [1]"
    llm_bad = "Bạn chỉ cần mang 10 triệu và mọi thứ sẽ ổn."  # hallucination
    print("GOOD:", validate_and_format_answer(llm_good, sample_passages, requested_field="price"))
    print("BAD :", validate_and_format_answer(llm_bad, sample_passages, requested_field="price"))