#!/usr/bin/env python3
"""
entities.py - Build and query a simple tour <-> alias index.

Features:
- Extract aliases from mapping entries (tour_name, location, summary, includes)
- Save/load tour_entities.json
- Fuzzy matching using rapidfuzz (optional)
- Optional semantic fallback via provided function (e.g., query_index)
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import os

# reuse normalization from common_utils
try:
    from common_utils import normalize_text_simple
except Exception:
    # fallback local normalizer
    def normalize_text_simple(s: str) -> str:
        import unicodedata, re
        if not s:
            return ""
        s = s.lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

# optional dependency
try:
    from rapidfuzz import process, fuzz  # type: ignore
except Exception:
    process = None
    fuzz = None

# default path can be overridden by env TOUR_ENTITIES_PATH
ENTITY_PATH_DEFAULT = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")

def extract_ngrams(tokens: List[str], min_n: int = 1, max_n: int = 3) -> Set[str]:
    out = set()
    n_tokens = len(tokens)
    for n in range(min_n, max_n + 1):
        for i in range(0, n_tokens - n + 1):
            ng = " ".join(tokens[i:i+n])
            out.add(ng)
    return out

def build_entity_index(mapping: List[dict], out_path: str = ENTITY_PATH_DEFAULT,
                       min_ngram: int = 1, max_ngram: int = 3,
                       fields_to_scan: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Build index: alias_norm -> {"tours": [indices], "examples": [paths], "aliases": [originals]}
    mapping: list of {"path":..., "text": ...}
    fields_to_scan: list of field substrings to consider (default tour_name, location, summary, includes)
    """
    if fields_to_scan is None:
        fields_to_scan = ["tour_name", "location", "summary", "includes", "itinerary", "location_name"]

    alias_to_tours = defaultdict(set)
    alias_examples = defaultdict(set)
    alias_raw = defaultdict(set)

    # helper to extract index int from path like root.tours[2].tour_name
    def extract_index(path: str) -> Optional[int]:
        m = re.search(r"\[(\d+)\]", path)
        if m:
            return int(m.group(1))
        return None

    for entry in mapping:
        path = entry.get("path","")
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        path_lower = path.lower()
        # only scan entries that look like tour-related fields
        interesting = any(f.lower() in path_lower for f in fields_to_scan)
        if not interesting:
            continue
        ti = extract_index(path)
        norm_text = normalize_text_simple(text)
        if not norm_text:
            continue
        tokens = norm_text.split()
        # add whole text as alias
        alias_raw[norm_text].add(text)
        if ti is not None:
            alias_to_tours[norm_text].add(ti)
            alias_examples[norm_text].add(path)
        # add ngrams as potential aliases
        for ng in extract_ngrams(tokens, min_n=min_ngram, max_n=max_ngram):
            alias_raw[ng].add(ng)
            if ti is not None:
                alias_to_tours[ng].add(ti)
                alias_examples[ng].add(path)

    # build final dict
    idx_map: Dict[str, dict] = {}
    for a in alias_to_tours:
        idx_map[a] = {
            "tours": sorted(list(alias_to_tours[a])),
            "examples": sorted(list(alias_examples.get(a, []))),
            "aliases": sorted(list(alias_raw.get(a, [])))
        }

    # persist to out_path
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(idx_map, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return idx_map

def load_entity_index(path: str = ENTITY_PATH_DEFAULT) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return {}

def find_tours_by_place(query: str,
                        index: Dict[str, dict],
                        top_k: int = 5,
                        fuzzy_threshold: int = 70,
                        use_fuzzy: bool = True,
                        semantic_fallback_fn = None) -> List[Tuple[int, float, List[str]]]:
    """
    Return list of tuples: (tour_index, score, matched_paths)
    Steps:
     - normalize query
     - exact match on keys
     - substring match on keys
     - fuzzy match via rapidfuzz (partial_ratio) if available
     - if no result and semantic_fallback_fn provided, call it (should return list of (score, mapping_entry))
    """
    qn = normalize_text_simple(query)
    if not qn:
        return []

    results = {}
    # exact match
    if qn in index:
        for ti in index[qn]["tours"]:
            results[ti] = max(results.get(ti, 0.0), 100.0)

    # substring matches on keys (fast)
    if not results:
        for k in index.keys():
            if qn in k:
                for ti in index[k]["tours"]:
                    results[ti] = max(results.get(ti, 0.0), 90.0)

    # fuzzy match
    if use_fuzzy and process is not None and not results:
        choices = list(index.keys())
        try:
            extracted = process.extract(qn, choices, scorer=fuzz.partial_ratio, limit=top_k)
            for choice, score, _ in extracted:
                if score >= fuzzy_threshold:
                    for ti in index[choice]["tours"]:
                        results[ti] = max(results.get(ti, 0.0), float(score))
        except Exception:
            # fallback: ignore fuzzy if rapidfuzz errors
            pass

    # semantic fallback
    if not results and semantic_fallback_fn is not None:
        try:
            sem = semantic_fallback_fn(query, top_k)
            for score, m in sem:
                p = m.get("path","")
                mm = re.search(r"\[(\d+)\]", p)
                if mm:
                    m_idx = int(mm.group(1))
                    # convert score to 0-100 like scale if necessary
                    s = float(score) * 100.0 if score <= 1.0 else float(score)
                    results[m_idx] = max(results.get(m_idx, 0.0), s)
        except Exception:
            pass

    # prepare sorted list
    # collect examples per tour from index (if available)
    tour_examples = {}
    for k, v in index.items():
        for ti in v.get("tours", []):
            tour_examples.setdefault(ti, set()).update(v.get("examples", []))

    out_list = sorted(((ti, s, sorted(list(tour_examples.get(ti, [])))) for ti, s in results.items()), key=lambda x: -x[1])
    # dedupe & limit
    final = []
    seen = set()
    for ti, score, examples in out_list:
        if ti in seen:
            continue
        seen.add(ti)
        final.append((ti, score, examples))
        if len(final) >= top_k:
            break
    return final