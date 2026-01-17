#!/usr/bin/env python3
# build_index.py — build embeddings/faiss index + mapping + tour_entities.json (compatible with app.py v5.2 & entities.py v5.2)
# Usage:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY="sk-..."
#   python build_index.py

import os
import sys
import json
import time
import datetime
import re
from typing import Any, List, Optional, Tuple, Dict
import numpy as np

# try imports with helpful fallbacks
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# New OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========== NEW: CÁC HÀM XỬ LÝ MỚI CHO CẤU TRÚC TOUR_ENTITIES ===========

def extract_region(location_text: str) -> str:
    """
    Trích xuất region (Miền Bắc/Trung/Nam) từ location string
    """
    if not location_text:
        return "Không xác định"
    
    location_lower = location_text.lower()
    
    # Mapping các keyword cho từng miền (đồng bộ với entities.py)
    north_keywords = ["hà nội", "sapa", "hạ long", "ninh bình", "tam đảo", "mộc châu", "phú thọ"]
    central_keywords = [
        "đà nẵng", "huế", "quảng trị", "nha trang", "hội an", "đông hà", 
        "cửa việt", "cồn cỏ", "quảng bình", "bạch mã", "hiền lương", "khe sanh",
        "hướng hóa", "hướng hoá", "vĩ tuyến 17", "đôi bờ hiền lương", 
        "vườn quốc gia bạch mã", "vĩnh linh", "gio linh", "thị xã quảng trị",
        "ngũ hồ", "thác đỗ quyên"
    ]
    south_keywords = [
        "phú quốc", "cần thơ", "cà mau", "sài gòn", "thành phố hồ chí minh", 
        "vũng tàu", "đà lạt", "buôn ma thuột", "nha trang", "phan thiết"
    ]
    
    # Đếm số lần xuất hiện của từng miền
    north_count = sum(1 for kw in north_keywords if kw in location_lower)
    central_count = sum(1 for kw in central_keywords if kw in location_lower)
    south_count = sum(1 for kw in south_keywords if kw in location_lower)
    
    # Chọn miền có số lần xuất hiện nhiều nhất
    counts = {"Miền Bắc": north_count, "Miền Trung": central_count, "Miền Nam": south_count}
    region = max(counts, key=counts.get)
    
    return region if counts[region] > 0 else "Miền Trung"  # Default Miền Trung cho Ruby Wings

def extract_tags(tour_data: Dict[str, Any]) -> List[str]:
    """
    Trích xuất tags từ style, includes, notes của tour
    """
    tags = []
    
    # Lấy các field cần thiết - XỬ LÝ CẢ STRING VÀ LIST
    style = tour_data.get("style", "")
    if isinstance(style, str):
        style = style.lower()
    else:
        style = str(style).lower()
    
    # Xử lý includes - có thể là list hoặc string
    includes_raw = tour_data.get("includes", [])
    if isinstance(includes_raw, list):
        includes = " ".join(str(item) for item in includes_raw).lower()
    else:
        includes = str(includes_raw).lower()
    
    # Xử lý notes - CÓ THỂ LÀ LIST HOẶC STRING
    notes_raw = tour_data.get("notes", "")
    if isinstance(notes_raw, list):
        notes = " ".join(str(item) for item in notes_raw).lower()
    else:
        notes = str(notes_raw).lower()
    
    summary = tour_data.get("summary", "")
    if isinstance(summary, str):
        summary = summary.lower()
    else:
        summary = str(summary).lower()
    
    tour_name = tour_data.get("tour_name", "")
    if isinstance(tour_name, str):
        tour_name = tour_name.lower()
    else:
        tour_name = str(tour_name).lower()
    
    # Danh sách keyword mapping theo knowledge.json thực tế
    keyword_mapping = {
        "retreat": ["retreat", "nghỉ dưỡng", "thư giãn", "tĩnh tâm", "chữa lành", "tái tạo năng lượng", "tĩnh tại"],
        "tâm_linh": ["tâm linh", "thiền", "chánh niệm", "tịnh tâm", "cầu nguyện", "nội tâm", "thiền định"],
        "lịch_sử": ["lịch sử", "tri ân", "di tích", "chiến tranh", "cựu chiến binh", "ký ức", "kháng chiến", "khát vọng"],
        "biển_đảo": ["biển", "đảo", "bãi biển", "cồn cỏ", "cửa việt", "ven biển", "bờ biển"],
        "văn_hóa": ["văn hóa", "bản địa", "dân tộc", "cộng đồng", "vân kiều", "pa kô", "cồng chiêng", "đàn ta lư"],
        "team_building": ["team building", "công ty", "doanh nghiệp", "tập thể", "corporate", "đoàn viên"],
        "gia_đình": ["gia đình", "trẻ em", "trẻ nhỏ", "phù hợp gia đình"],
        "thanh_niên": ["thanh niên", "học sinh", "sinh viên", "đoàn viên", "trẻ trung"],
        "người_lớn_tuổi": ["người lớn tuổi", "người già", "senior", "người cao tuổi"],
        "thiền": ["thiền", "khí công", "chánh niệm", "yoga", "tập luyện tinh thần", "thực hành thiền"],
        "thiên_nhiên": ["rừng", "núi", "suối", "thiên nhiên", "bạch mã", "nguyên sinh", "cây cỏ", "rừng nguyên sinh", "ngũ hồ"],
        "mạo_hiểm": ["trekking", "leo núi", "khám phá", "mạo hiểm", "thử thách"],
        "trải_nghiệm": ["trải nghiệm", "hành trình", "khám phá", "thực tế", "gắn kết"],
        "du_lịch_xanh": ["xanh", "bền vững", "môi trường", "sinh thái", "hành trình xanh"],
        "lửa_trại": ["lửa trại", "đốt lửa", "giao lưu đêm", "cồng chiêng"],
        "picnic": ["picnic", "ăn ngoài trời", "thuần chay"],
        "1_ngày": ["1 ngày", "một ngày"],
        "2_ngày": ["2 ngày", "hai ngày", "1 đêm"],
        "giá_rẻ": ["890.000", "dưới 1 triệu", "tiết kiệm"],
        "cao_cấp": ["cao cấp", "premium", "chất lượng cao", "nâng cao"]
    }
    
    # Kiểm tra từng keyword
    all_text = f"{tour_name} {style} {includes} {notes} {summary}"
    for tag, keywords in keyword_mapping.items():
        if any(keyword in all_text for keyword in keywords):
            tags.append(tag)
    
    # Đảm bảo unique tags
    return list(set(tags))

def parse_duration(duration_text: str) -> int:
    """
    Parse duration text thành số ngày
    Ví dụ: "2 ngày 1 đêm" → 2, "1 ngày" → 1
    """
    if not duration_text:
        return 1
    
    duration_lower = duration_text.lower().strip()
    
    # Tìm số trong text (ưu tiên số đầu tiên trước "ngày")
    # Pattern: "2 ngày", "1 ngày", etc.
    day_match = re.search(r'(\d+)\s*ngày', duration_lower)
    if day_match:
        try:
            return int(day_match.group(1))
        except:
            pass
    
    # Fallback: tìm số bất kỳ
    numbers = re.findall(r'\d+', duration_text)
    if numbers:
        try:
            return int(numbers[0])
        except:
            pass
    
    return 1  # Mặc định 1 ngày

def parse_price(price_text: str) -> Tuple[int, int, int]:
    """
    Parse price text thành min_price, max_price, avg_price
    Ví dụ: 
    - "1.700.000 – 2.300.000 VNĐ/người" → (1700000, 2300000, 2000000)
    - "890.000 VNĐ/khách" → (890000, 890000, 890000)
    """
    if not price_text:
        return 1000000, 2000000, 1500000
    
    price_lower = price_text.lower().replace(',', '').replace(' ', '')
    
    # Tìm tất cả số (bỏ dấu chấm phân cách ngàn)
    # Pattern: 1.700.000, 890.000, etc.
    numbers_raw = re.findall(r'[\d\.]+', price_text)
    
    clean_numbers = []
    for num_str in numbers_raw:
        try:
            # Loại bỏ dấu chấm phân cách ngàn
            clean_num_str = num_str.replace('.', '')
            clean_num = int(clean_num_str)
            
            # Chỉ lấy số >= 1000 (tránh số nhỏ như năm, số người)
            if clean_num >= 1000:
                clean_numbers.append(clean_num)
        except:
            continue
    
    if len(clean_numbers) >= 2:
        # Có khoảng giá: lấy min, max
        min_price = min(clean_numbers)
        max_price = max(clean_numbers)
        avg_price = (min_price + max_price) // 2
    elif len(clean_numbers) == 1:
        # Chỉ có 1 giá: giả sử đó là giá cơ bản
        base_price = clean_numbers[0]
        
        # Nếu trong text có "gói" hoặc "theo đoàn" thì có thể có range
        if any(word in price_lower for word in ["gói", "theo", "tuỳ", "chi tiết"]):
            # Ước lượng range: ±30%
            min_price = int(base_price * 0.7)
            max_price = int(base_price * 1.3)
            avg_price = base_price
        else:
            # Giá cố định
            min_price = base_price
            max_price = base_price
            avg_price = base_price
    else:
        # Không parse được: ước lượng từ text
        if "triệu" in price_lower:
            # Tìm số triệu
            million_match = re.search(r'(\d+)\s*triệu', price_lower)
            if million_match:
                try:
                    million_val = int(million_match.group(1))
                    base_price = million_val * 1000000
                    min_price = base_price
                    max_price = int(base_price * 1.5)
                    avg_price = int((min_price + max_price) / 2)
                except:
                    min_price, max_price, avg_price = 2000000, 3000000, 2500000
            else:
                min_price, max_price, avg_price = 2000000, 3000000, 2500000
        elif any(word in price_lower for word in ["nghìn", "k"]):
            min_price, max_price, avg_price = 500000, 1500000, 1000000
        else:
            # Default cho Ruby Wings
            min_price, max_price, avg_price = 1000000, 2000000, 1500000
    
    return int(min_price), int(max_price), int(avg_price)

def create_embedding_text(tour_data: Dict[str, Any]) -> str:
    """
    Tạo text cho embedding từ các field quan trọng
    """
    def safe_str(val):
        if val is None:
            return ""
        if isinstance(val, list):
            return " ".join(str(item) for item in val)
        if isinstance(val, dict):
            return " ".join(f"{k}: {v}" for k, v in val.items())
        return str(val)
    
    fields = [
        safe_str(tour_data.get("tour_name", "")),
        safe_str(tour_data.get("summary", "")),
        safe_str(tour_data.get("location", "")),
        safe_str(tour_data.get("style", "")),
        safe_str(tour_data.get("includes", "")),
        safe_str(tour_data.get("notes", "")),
        safe_str(tour_data.get("duration", "")),
        safe_str(tour_data.get("price", "")),
        safe_str(tour_data.get("accommodation", "")),
        safe_str(tour_data.get("meals", "")),
        safe_str(tour_data.get("transport", "")),
        safe_str(tour_data.get("event_support", ""))
    ]
    return " ".join([field for field in fields if field and field.strip()])

def calculate_popularity_score(tour_index: int, total_tours: int) -> float:
    """
    Tính popularity score dựa trên vị trí tour (giả định tour đầu popular hơn)
    """
    if total_tours <= 1:
        return 0.8
    
    # Tour đầu tiên có score cao nhất, giảm dần
    base_score = 0.7
    position_factor = (total_tours - tour_index) / total_tours  # từ 1 đến 0
    return base_score + (0.3 * position_factor)

def calculate_value_score(min_price: int, max_price: int, duration_days: int) -> float:
    """
    Tính value score dựa trên giá và số ngày (giá thấp + ngày nhiều = value cao)
    """
    if duration_days == 0 or max_price == 0:
        return 0.5
    
    avg_price = (min_price + max_price) / 2
    price_per_day = avg_price / duration_days
    
    # Normalize: giá mỗi ngày dưới 1 triệu -> score cao
    if price_per_day < 1000000:
        return 0.8
    elif price_per_day < 2000000:
        return 0.6
    else:
        return 0.4

# =========== HÀM FLATTEN JSON (CẬP NHẬT) ===========

def flatten_json(path: str) -> List[dict]:
    """
    Flatten knowledge.json thành list of passages cho FAISS
    Mỗi tour = 1 passage duy nhất với tất cả thông tin
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = []
    
    # 1. Xử lý about_company
    about = data.get("about_company", {})
    for key, value in about.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.about_company.{key}",
                "text": value
            })
    
    # 2. Xử lý tours - MỖI TOUR LÀ 1 PASSAGE DUY NHẤT
    tours = data.get("tours", [])
    for i, tour in enumerate(tours):
        tour_text_parts = []
        
        # Các trường quan trọng cần index
        fields_to_include = [
            ("tour_name", "Tên tour"),
            ("summary", "Tóm tắt"),
            ("location", "Địa điểm"),
            ("duration", "Thời lượng"),
            ("price", "Giá"),
            ("notes", "Lưu ý"),
            ("style", "Phong cách"),
            ("transport", "Phương tiện"),
            ("accommodation", "Chỗ ở"),
            ("meals", "Bữa ăn"),
            ("event_support", "Hỗ trợ sự kiện")
        ]
        
        for field_key, field_label in fields_to_include:
            if field_key in tour:
                value = tour[field_key]
                if isinstance(value, list):
                    tour_text_parts.append(f"{field_label}: {', '.join(str(v) for v in value)}")
                elif value and str(value).strip():
                    tour_text_parts.append(f"{field_label}: {value}")
        
        # Xử lý includes
        if "includes" in tour and tour["includes"]:
            includes_text = "Dịch vụ bao gồm: " + "; ".join(str(item) for item in tour["includes"])
            tour_text_parts.append(includes_text)
        
        # Gộp thành 1 passage
        full_tour_text = "\n".join(tour_text_parts)
        
        mapping.append({
            "path": f"root.tours[{i}]",
            "text": full_tour_text
        })
    
    # 3. Xử lý FAQ (nếu có)
    faq = data.get("faq", {})
    for key, value in faq.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.faq.{key}",
                "text": value
            })
    
    # 4. Xử lý contact (nếu có)
    contact = data.get("contact", {})
    for key, value in contact.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.contact.{key}",
                "text": value
            })
    
    return mapping

# =========== HÀM TẠO TOUR_ENTITIES ===========

def create_tour_entities(tours_data: List[Dict[str, Any]], mapping: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tạo tour_entities.json với cấu trúc v5.2
    Đồng bộ với entities.py và app.py
    """
    tour_entities = {}
    total_tours = len(tours_data)
    
    for i, tour in enumerate(tours_data):
        tour_id = f"tour_{i:03d}"
        
        # Parse các thông tin cơ bản
        tour_name = tour.get("tour_name", "")
        location = tour.get("location", "")
        duration_text = tour.get("duration", "")
        price_text = tour.get("price", "")
        
        # Extract metadata
        region = extract_region(location)
        tags = extract_tags(tour)
        duration_days = parse_duration(duration_text)
        min_price, max_price, avg_price = parse_price(price_text)
        
        # Tạo embedding text
        embedding_text = create_embedding_text(tour)
        
        # Tính các score
        popularity_score = calculate_popularity_score(i, total_tours)
        value_score = calculate_value_score(min_price, max_price, duration_days)
        
        # Kiểm tra các flag
        family_friendly = "gia_đình" in tags
        senior_friendly = "người_lớn_tuổi" in tags or (
            "mạo_hiểm" not in tags and 
            "trekking" not in embedding_text.lower() and
            duration_days <= 2
        )
        corporate_friendly = "team_building" in tags or "thanh_niên" in tags
        
        # Tạo tour entity
        tour_entities[tour_id] = {
            "tour_id": tour_id,
            "index": i,
            "tour_name": tour_name,
            "location": location,
            "region": region,
            
            "tags": tags,
            
            "duration": duration_text,
            "duration_days": duration_days,
            
            "price_text": price_text,
            "min_price": min_price,
            "max_price": max_price,
            "avg_price": avg_price,
            
            "embedding_text": embedding_text,
            
            # Metadata cho ranking
            "popularity_score": round(popularity_score, 2),
            "value_score": round(value_score, 2),
            "family_friendly": family_friendly,
            "senior_friendly": senior_friendly,
            "corporate_friendly": corporate_friendly,
            
            # Các field từ knowledge.json
            "summary": tour.get("summary", ""),
            "style": tour.get("style", ""),
            "includes": tour.get("includes", []),
            "notes": tour.get("notes", ""),
            "transport": tour.get("transport", ""),
            "accommodation": tour.get("accommodation", ""),
            "meals": tour.get("meals", ""),
            "event_support": tour.get("event_support", ""),
            
            # Timestamps
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
    
    return tour_entities

# =========== EMBEDDING FUNCTIONS ===========

def synthetic_embedding(text: str, dim: int = 1536) -> List[float]:
    """Generate synthetic embedding for fallback"""
    h = abs(hash(text)) % (10 ** 12)
    return [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]

def call_embeddings_with_retry(inputs: List[str], model: str) -> List[List[float]]:
    """Call OpenAI embeddings API with retry logic"""
    if not OPENAI_KEY or OpenAI is None:
        print("⚠️ OpenAI API key not found, using synthetic embeddings", file=sys.stderr)
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in inputs]

    client = OpenAI(api_key=OPENAI_KEY)
    attempt = 0
    
    while attempt <= RETRY_LIMIT:
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            if getattr(resp, "data", None):
                out = [r.embedding for r in resp.data]
                print(f"✅ Generated {len(out)} embeddings (model={model})", flush=True)
                return out
            else:
                raise ValueError("Empty response from OpenAI embeddings API")
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"❌ Embedding API failed after {RETRY_LIMIT} attempts: {e}", file=sys.stderr)
                print("⚠️ Falling back to synthetic embeddings", file=sys.stderr)
                dim = 1536 if "3-small" in model else 3072
                return [synthetic_embedding(t, dim) for t in inputs]
            
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"⚠️ Embedding API error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
    
    # Final fallback
    dim = 1536 if "3-small" in model else 3072
    return [synthetic_embedding(t, dim) for t in inputs]

# =========== CONFIG ===========

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")
TOUR_ENTITIES_PATH = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")

EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

TMP_EMB_FILE = "emb_tmp.bin"

# =========== MAIN BUILD FLOW ===========

def build_index():
    print("=" * 60)
    print("BUILDING INDEX FOR RUBY WINGS v5.2")
    print("=" * 60)
    
    # 1. Đọc knowledge.json
    print(f"\n📚 Reading knowledge from {KNOW_PATH}...")
    if not os.path.exists(KNOW_PATH):
        print(f"❌ Error: {KNOW_PATH} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(KNOW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tours_data = data.get("tours", [])
    print(f"✅ Found {len(tours_data)} tours")
    
    if len(tours_data) == 0:
        print("❌ No tours found in knowledge.json", file=sys.stderr)
        sys.exit(1)
    
    # 2. Flatten knowledge.json thành mapping cho FAISS
    print("\n🔄 Flattening knowledge.json for FAISS mapping...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m.get("text", "") for m in mapping]
    n = len(texts)
    print(f"✅ Created {n} passages for FAISS indexing")
    
    if n == 0:
        print("❌ No passages to index -> exit", file=sys.stderr)
        sys.exit(1)
    
    # 3. Tạo tour_entities.json
    print("\n🏗️  Creating tour_entities.json with enhanced structure...")
    tour_entities = create_tour_entities(tours_data, mapping)
    
    # Lưu tour_entities.json
    try:
        with open(TOUR_ENTITIES_PATH, "w", encoding="utf-8") as f:
            json.dump(tour_entities, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved enhanced tour_entities.json to {TOUR_ENTITIES_PATH}")
        print(f"   - Contains {len(tour_entities)} tours with fields:")
        print(f"     • region, tags, duration_days, min/max/avg_price")
        print(f"     • popularity_score, value_score")
        print(f"     • family_friendly, senior_friendly, corporate_friendly")
    except Exception as e:
        print(f"❌ Failed to save tour_entities.json: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 4. Tạo embeddings
    print("\n🧠 Creating embeddings for FAISS index...")
    print(f"   Using model: {EMBEDDING_MODEL}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Remove tmp if exists
    if os.path.exists(TMP_EMB_FILE):
        try:
            os.remove(TMP_EMB_FILE)
        except Exception:
            pass

    dim: Optional[int] = None
    total_rows = 0
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, n, BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        print(f"   Embedding batch {start//BATCH_SIZE + 1}/{batches} ({len(inputs)} texts)...", flush=True)
        vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL)

        # Ensure no None entries
        for j, v in enumerate(vecs):
            if v is None:
                vecs[j] = synthetic_embedding(inputs[j], 1536 if "3-small" in EMBEDDING_MODEL else 3072)

        if dim is None and vecs:
            dim = len(vecs[0])

        arr = np.array(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / (norms + 1e-12)

        with open(TMP_EMB_FILE, "ab") as f:
            f.write(arr.tobytes())

        total_rows += arr.shape[0]

    if total_rows == 0 or dim is None:
        print("❌ No embeddings created -> exit", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Generated {total_rows} embeddings with dimension {dim}")
    
    # 5. Load embeddings và build FAISS index
    print("\n🔍 Building FAISS index...")
    try:
        emb = np.memmap(TMP_EMB_FILE, dtype="float32", mode="r", shape=(total_rows, dim))
    except Exception:
        # Fallback: load entire array into memory
        raw = np.fromfile(TMP_EMB_FILE, dtype="float32")
        emb = raw.reshape((total_rows, dim))

    # Build FAISS index if available
    HAS_FAISS_local = False
    if HAS_FAISS:
        try:
            index = faiss.IndexFlatIP(dim)
            index.add(np.asarray(emb))
            try:
                faiss.write_index(index, FAISS_INDEX_PATH)
                print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
                HAS_FAISS_local = True
            except Exception as e:
                print(f"⚠️ Failed to persist FAISS index: {e}", file=sys.stderr)
                HAS_FAISS_local = False
        except Exception as e:
            print(f"⚠️ FAISS index build failed: {e}", file=sys.stderr)
            HAS_FAISS_local = False
    else:
        print("⚠️ FAISS not available, skipping FAISS index creation")

    # 6. Luôn lưu fallback vectors (npz) cho numpy fallback
    try:
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        print(f"✅ Saved fallback vectors to {FALLBACK_VECTORS_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save fallback vectors: {e}", file=sys.stderr)

    # 7. Lưu mapping (list of {"path","text"}) expected by app.py
    print(f"\n🗂️  Saving mapping to {FAISS_MAPPING_PATH}...")
    try:
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(mapping)} mapping entries")
    except Exception as e:
        print(f"❌ Failed to save mapping: {e}", file=sys.stderr)
        sys.exit(1)

    # 8. Write metadata
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_passages": int(total_rows),
        "num_tours": len(tours_data),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": int(dim),
        "faiss_available": bool(HAS_FAISS_local),
        "system_version": "v5.2",
        "notes": "Built with enhanced tour_entities.json structure for Ruby Wings v5.2",
        "features": {
            "region_extraction": True,
            "tags_extraction": True,
            "price_parsing": True,
            "duration_parsing": True,
            "popularity_scoring": True,
            "value_scoring": True,
            "event_support_field": True
        }
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved metadata to {META_PATH}")
    except Exception:
        print(f"⚠️ Failed to save metadata", file=sys.stderr)

    # 9. Cleanup temp file
    try:
        os.remove(TMP_EMB_FILE)
        print(f"✅ Cleaned up temporary file: {TMP_EMB_FILE}")
    except Exception:
        pass

    # 10. Summary
    print("\n" + "=" * 60)
    print("🎉 BUILD COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   • Tours processed: {len(tours_data)}")
    print(f"   • FAISS passages: {total_rows}")
    print(f"   • Embedding dimension: {dim}")
    print(f"   • Embedding model: {EMBEDDING_MODEL}")
    print(f"\n📁 Files created:")
    print(f"   1. tour_entities.json: {TOUR_ENTITIES_PATH}")
    print(f"      - Enhanced structure with region, tags, pricing, scores")
    print(f"   2. FAISS index: {FAISS_INDEX_PATH if HAS_FAISS_local else '(skipped - FAISS not available)'}")
    print(f"   3. FAISS mapping: {FAISS_MAPPING_PATH}")
    print(f"   4. Fallback vectors: {FALLBACK_VECTORS_PATH}")
    print(f"   5. Metadata: {META_PATH}")
    
    # Hiển thị sample của tour đầu tiên
    if tour_entities:
        sample_id = list(tour_entities.keys())[0]
        sample_tour = tour_entities[sample_id]
        print(f"\n📝 Sample tour structure (first tour):")
        print(f"   • Tour ID: {sample_id}")
        print(f"   • Name: {sample_tour.get('tour_name', 'N/A')[:60]}...")
        print(f"   • Location: {sample_tour.get('location', 'N/A')[:60]}")
        print(f"   • Region: {sample_tour.get('region', 'N/A')}")
        print(f"   • Tags: {', '.join(sample_tour.get('tags', [])[:5])}")
        if len(sample_tour.get('tags', [])) > 5:
            print(f"            (and {len(sample_tour.get('tags', [])) - 5} more...)")
        print(f"   • Duration: {sample_tour.get('duration', 'N/A')} ({sample_tour.get('duration_days', 'N/A')} days)")
        print(f"   • Price range: {sample_tour.get('min_price', 0):,} - {sample_tour.get('max_price', 0):,} VND")
        print(f"   • Avg price: {sample_tour.get('avg_price', 0):,} VND")
        print(f"   • Popularity score: {sample_tour.get('popularity_score', 0)}")
        print(f"   • Value score: {sample_tour.get('value_score', 0)}")
        print(f"   • Family friendly: {sample_tour.get('family_friendly', False)}")
        print(f"   • Senior friendly: {sample_tour.get('senior_friendly', False)}")
        print(f"   • Corporate friendly: {sample_tour.get('corporate_friendly', False)}")
    
    # Hiển thị thống kê tags
    if tour_entities:
        all_tags = []
        for tour_id, tour in tour_entities.items():
            all_tags.extend(tour.get('tags', []))
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        print(f"\n🏷️  Tag statistics (top 10):")
        for tag, count in tag_counts.most_common(10):
            print(f"   • {tag}: {count} tours")
    
    # Hiển thị thống kê regions
    if tour_entities:
        regions = {}
        for tour_id, tour in tour_entities.items():
            region = tour.get('region', 'Không xác định')
            regions[region] = regions.get(region, 0) + 1
        
        print(f"\n🗺️  Region distribution:")
        for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {region}: {count} tours")
    
    print("\n✅ Index ready for Ruby Wings v5.2 system!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        build_index()
    except KeyboardInterrupt:
        print("\n\n⚠️ Build interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR building index: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)