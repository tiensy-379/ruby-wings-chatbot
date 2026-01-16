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
    Ví dụ: "Hà Nội – Đông Hà – Thành Cổ – Cửa Việt – Cồn Cỏ" → "Miền Trung"
    """
    if not location_text:
        return "Không xác định"
    
    location_lower = location_text.lower()
    
    # Mapping các keyword cho từng miền (đã cập nhật với knowledge.json mẫu)
    north_keywords = ["hà nội", "sapa", "hạ long", "ninh bình", "tam đảo", "mộc châu", "phú thọ"]
    central_keywords = [
        "đà nẵng", "huế", "quảng trị", "nha trang", "hội an", "đông hà", 
        "cửa việt", "cồn cỏ", "quảng bình", "bạch mã", "hiền lương", "khe sanh",
        "hướng hóa", "vĩ tuyến 17", "đôi bờ hiền lương", "vườn quốc gia bạch mã"
    ]
    south_keywords = [
        "phú quốc", "cần thơ", "cà mau", "sài gòn", "thành phố hồ chí minh", 
        "vũng tàu", "đà lạt", "buôn ma thuột", "nha trang"
    ]
    
    # Đếm số lần xuất hiện của từng miền
    north_count = sum(1 for kw in north_keywords if kw in location_lower)
    central_count = sum(1 for kw in central_keywords if kw in location_lower)
    south_count = sum(1 for kw in south_keywords if kw in location_lower)
    
    # Chọn miền có số lần xuất hiện nhiều nhất
    counts = {"Miền Bắc": north_count, "Miền Trung": central_count, "Miền Nam": south_count}
    region = max(counts, key=counts.get)
    
    return region if counts[region] > 0 else "Không xác định"

def extract_tags(tour_data: Dict[str, Any]) -> List[str]:
    """
    Trích xuất tags từ style, includes, notes của tour
    """
    tags = []
    
    # Lấy các field cần thiết
    style = tour_data.get("style", "").lower()
    includes = " ".join(tour_data.get("includes", [])).lower() if isinstance(tour_data.get("includes"), list) else str(tour_data.get("includes", "")).lower()
    notes = tour_data.get("notes", "").lower()
    summary = tour_data.get("summary", "").lower()
    tour_name = tour_data.get("tour_name", "").lower()
    
    # Danh sách keyword mapping cập nhật theo knowledge.json mẫu
    keyword_mapping = {
        "retreat": ["retreat", "nghỉ dưỡng", "thư giãn", "tĩnh tâm", "chữa lành", "tái tạo năng lượng"],
        "tâm_linh": ["tâm linh", "thiền", "chánh niệm", "tịnh tâm", "cầu nguyện", "nội tâm"],
        "lịch_sử": ["lịch sử", "tri ân", "di tích", "chiến tranh", "cựu chiến binh", "ký ức", "kháng chiến"],
        "biển_đảo": ["biển", "đảo", "bãi biển", "cồn cỏ", "cửa việt", "ven biển"],
        "văn_hóa": ["văn hóa", "bản địa", "dân tộc", "cộng đồng", "vân kiều", "pa kô", "cồng chiêng"],
        "team_building": ["team building", "công ty", "doanh nghiệp", "tập thể", "corporate"],
        "gia_đình": ["gia đình", "trẻ em", "trẻ nhỏ", "phù hợp gia đình"],
        "người_lớn_tuổi": ["người lớn tuổi", "người già", "senior", "người cao tuổi"],
        "thiền": ["thiền", "khí công", "chánh niệm", "yoga", "tập luyện tinh thần"],
        "thiên_nhiên": ["rừng", "núi", "suối", "thiên nhiên", "bạch mã", "nguyên sinh", "cây cỏ"],
        "mạo_hiểm": ["trekking", "leo núi", "khám phá", "mạo hiểm", "thử thách"],
        "trải_nghiệm": ["trải nghiệm", "hành trình", "khám phá", "thực tế"],
        "du_lịch_tâm_linh": ["tâm linh", "thiền định", "chữa lành", "tinh thần"],
        "du_lịch_lịch_sử": ["lịch sử", "di sản", "văn hóa", "truyền thống"],
        "du_lịch_nghỉ_dưỡng": ["nghỉ dưỡng", "thư giãn", "spa", "wellness"],
        "du_lịch_sinh_thái": ["sinh thái", "môi trường", "xanh", "bền vững"]
    }
    
    # Kiểm tra từng keyword
    all_text = f"{tour_name} {style} {includes} {notes} {summary}"
    for tag, keywords in keyword_mapping.items():
        if any(keyword in all_text for keyword in keywords):
            tags.append(tag)
    
    return list(set(tags))

def parse_duration(duration_text: str) -> int:
    """
    Parse duration text thành số ngày
    Ví dụ: "3 ngày 2 đêm" → 3, "1 ngày" → 1
    """
    if not duration_text:
        return 1
    
    # Tìm số trong text (ưu tiên số đầu tiên)
    numbers = re.findall(r'\d+', duration_text)
    if numbers:
        try:
            return int(numbers[0])
        except:
            pass
    
    # Fallback: dựa vào keyword
    duration_lower = duration_text.lower()
    if "ngày" in duration_lower or "day" in duration_lower:
        if "2" in duration_lower or "hai" in duration_lower:
            return 2
        elif "3" in duration_lower or "ba" in duration_lower:
            return 3
        elif "4" in duration_lower or "bốn" in duration_lower:
            return 4
        elif "5" in duration_lower or "năm" in duration_lower:
            return 5
        elif "6" in duration_lower or "sáu" in duration_lower:
            return 6
        elif "7" in duration_lower or "bảy" in duration_lower:
            return 7
        else:
            return 1
    
    return 1  # Mặc định 1 ngày

def parse_price(price_text: str) -> Tuple[int, int, int]:
    """
    Parse price text thành min_price, max_price, avg_price
    Ví dụ: "2.200.000 đến 4.500.000 VNĐ/người" → (2200000, 4500000, 3350000)
    """
    if not price_text:
        return 1000000, 2000000, 1500000
    
    price_lower = price_text.lower()
    
    # Tìm tất cả số trong text (đã bỏ dấu chấm phân cách ngàn)
    # Pattern tìm số có dạng: 1.000.000, 2,500,000, 1500000
    numbers = re.findall(r'[\d\,\.]+', price_lower)
    
    clean_numbers = []
    for num in numbers:
        try:
            # Loại bỏ dấu chấm và phẩy phân cách ngàn
            clean_num_str = num.replace('.', '').replace(',', '')
            clean_num = int(clean_num_str)
            
            # Chỉ lấy số lớn hơn 1000 (tránh số nhỏ như năm, số lượng người)
            if clean_num >= 1000:
                clean_numbers.append(clean_num)
        except:
            continue
    
    if len(clean_numbers) >= 2:
        # Loại bỏ outliers (số quá lớn so với các số khác)
        if len(clean_numbers) > 2:
            avg_val = sum(clean_numbers) / len(clean_numbers)
            clean_numbers = [n for n in clean_numbers if n <= avg_val * 3]
        
        min_price = min(clean_numbers)
        max_price = max(clean_numbers)
        avg_price = (min_price + max_price) // 2
    elif len(clean_numbers) == 1:
        min_price = clean_numbers[0]
        # Ước tính max_price dựa trên context
        if "triệu" in price_lower:
            max_price = min_price * 2 if min_price < 5000000 else min_price + 2000000
        else:
            max_price = min_price * 1.5
        avg_price = (min_price + max_price) // 2
    else:
        # Nếu không parse được, ước lượng từ text
        if "triệu" in price_lower:
            # Kiểm tra số triệu
            million_match = re.search(r'(\d+)\s*(triệu|tr)', price_lower)
            if million_match:
                try:
                    million_val = int(million_match.group(1))
                    base_price = million_val * 1000000
                    min_price = base_price
                    max_price = base_price + 2000000
                    avg_price = (min_price + max_price) // 2
                except:
                    min_price, max_price, avg_price = 2000000, 4000000, 3000000
            else:
                # Giả sử 2-4 triệu
                min_price, max_price, avg_price = 2000000, 4000000, 3000000
        elif "nghìn" in price_lower or "k" in price_lower:
            # Giả sử 500k-1.5 triệu
            min_price, max_price, avg_price = 500000, 1500000, 1000000
        else:
            # Mặc định 1-2 triệu
            min_price, max_price, avg_price = 1000000, 2000000, 1500000
    
    return int(min_price), int(max_price), int(avg_price)

def create_embedding_text(tour_data: Dict[str, Any]) -> str:
    """
    Tạo text cho embedding từ các field quan trọng
    """
    fields = [
        tour_data.get("tour_name", ""),
        tour_data.get("summary", ""),
        tour_data.get("location", ""),
        tour_data.get("style", ""),
        " ".join(tour_data.get("includes", [])) if isinstance(tour_data.get("includes"), list) else str(tour_data.get("includes", "")),
        tour_data.get("notes", ""),
        str(tour_data.get("duration", "")),
        str(tour_data.get("price", "")),
        str(tour_data.get("accommodation", "")),
        str(tour_data.get("meals", "")),
        str(tour_data.get("transport", "")),
        str(tour_data.get("event_support", ""))
    ]
    return " ".join([field for field in fields if field and str(field).strip()])

def calculate_popularity_score(tour_index: int, total_tours: int) -> float:
    """
    Tính popularity score dựa trên vị trí tour (giả định tour đầu popular hơn)
    """
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

# =========== HÀM FLATTEN JSON HIỆN TẠI (ĐÃ CẬP NHẬT) ===========

def flatten_json(path: str) -> List[dict]:
    # fallback simple flattener - mỗi tour là 1 passage duy nhất
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = []
    
    # Xử lý about_company (giữ nguyên)
    about = data.get("about_company", {})
    for key, value in about.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.about_company.{key}",
                "text": value
            })
    
    # Xử lý tours - MỖI TOUR LÀ 1 PASSAGE DUY NHẤT
    tours = data.get("tours", [])
    for i, tour in enumerate(tours):
        tour_text_parts = []
        
        # Thêm các trường quan trọng
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
    
    # Xử lý FAQ và contact (giữ nguyên)
    faq = data.get("faq", {})
    for key, value in faq.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.faq.{key}",
                "text": value
            })
    
    contact = data.get("contact", {})
    for key, value in contact.items():
        if isinstance(value, str) and value.strip():
            mapping.append({
                "path": f"root.contact.{key}",
                "text": value
            })
    
    return mapping

# =========== HÀM TẠO TOUR_ENTITIES VỚI CẤU TRÚC MỚI ===========

def create_tour_entities(tours_data: List[Dict[str, Any]], mapping: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tạo tour_entities.json với cấu trúc mới tương thích với hệ thống v5.2
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
        
        # Thêm các field mới với các hàm xử lý mới
        region = extract_region(location)
        tags = extract_tags(tour)
        duration_days = parse_duration(duration_text)
        min_price, max_price, avg_price = parse_price(price_text)
        
        # Tạo embedding text
        embedding_text = create_embedding_text(tour)
        
        # Tính các score
        popularity_score = calculate_popularity_score(i, total_tours)
        value_score = calculate_value_score(min_price, max_price, duration_days)
        
        # Kiểm tra family/senior/corporate friendly từ tags
        family_friendly = "gia_đình" in tags
        senior_friendly = "người_lớn_tuổi" in tags or all(word not in embedding_text.lower() for word in ["mạo hiểm", "trekking", "leo núi", "thử thách"])
        corporate_friendly = "team_building" in tags
        
        # Tạo tour entity với cấu trúc mới
        tour_entities[tour_id] = {
            "tour_id": tour_id,
            "index": i,  # Thêm index để tương thích với app.py
            "tour_name": tour_name,
            "location": location,
            "region": region,  # MỚI: cho fallback suggestion
            
            "tags": tags,  # MỚI: cho filtering và classification
            
            "duration": duration_text,
            "duration_days": duration_days,  # MỚI: số ngày dạng số
            
            "price_text": price_text,
            "min_price": min_price,  # MỚI: giá nhỏ nhất
            "max_price": max_price,  # MỚI: giá lớn nhất
            "avg_price": avg_price,  # MỚI: giá trung bình
            
            "embedding_text": embedding_text,  # Cho FAISS embedding
            
            # Metadata cho ranking và filtering
            "popularity_score": round(popularity_score, 2),
            "value_score": round(value_score, 2),
            "family_friendly": family_friendly,
            "senior_friendly": senior_friendly,
            "corporate_friendly": corporate_friendly,
            
            # Các field khác từ mapping (cho response guard)
            "summary": tour.get("summary", ""),
            "style": tour.get("style", ""),
            "includes": tour.get("includes", []),
            "notes": tour.get("notes", ""),
            "transport": tour.get("transport", ""),
            "accommodation": tour.get("accommodation", ""),
            "meals": tour.get("meals", ""),
            "event_support": tour.get("event_support", ""),
            
            # Metadata bổ sung
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
    
    return tour_entities

# =========== CÁC HÀM CŨ (GIỮ NGUYÊN VỚI TÍCH HỢP MỚI) ===========

def synthetic_embedding(text: str, dim: int = 1536) -> List[float]:
    h = abs(hash(text)) % (10 ** 12)
    return [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]

def call_embeddings_with_retry(inputs: List[str], model: str) -> List[List[float]]:
    if not OPENAI_KEY or OpenAI is None:
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
                dim = 1536 if "3-small" in model else 3072
                return [synthetic_embedding(t, dim) for t in inputs]
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"⚠️ Embedding API error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
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

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

TMP_EMB_FILE = "emb_tmp.bin"

# =========== MAIN BUILD FLOW (ĐÃ CẬP NHẬT) ===========

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
    
    # 2. Flatten knowledge.json thành mapping cho FAISS
    print("\n🔄 Flattening knowledge.json for FAISS mapping...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m.get("text", "") for m in mapping]
    n = len(texts)
    print(f"✅ Created {n} passages for FAISS indexing")
    
    if n == 0:
        print("❌ No passages to index -> exit", file=sys.stderr)
        sys.exit(1)
    
    # 3. Tạo tour_entities.json với cấu trúc mới
    print("\n🏗️ Creating tour_entities.json with enhanced structure...")
    tour_entities = create_tour_entities(tours_data, mapping)
    
    # Lưu tour_entities.json
    try:
        with open(TOUR_ENTITIES_PATH, "w", encoding="utf-8") as f:
            json.dump(tour_entities, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved enhanced tour_entities.json to {TOUR_ENTITIES_PATH}")
        print(f"   - Contains {len(tour_entities)} tours with new fields:")
        print(f"     • region, tags, duration_days, min/max/avg_price")
        print(f"     • popularity_score, value_score")
        print(f"     • family_friendly, senior_friendly, corporate_friendly")
    except Exception as e:
        print(f"❌ Failed to save tour_entities.json: {e}", file=sys.stderr)
    
    # 4. Tạo embeddings cho FAISS
    print("\n🧠 Creating embeddings for FAISS index...")
    
    # remove tmp if exists
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
        print(f"   Embedding batch {start//BATCH_SIZE + 1}/{batches} ...", flush=True)
        vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL)

        # ensure no None entries
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
        # fallback: load entire array into memory
        raw = np.fromfile(TMP_EMB_FILE, dtype="float32")
        emb = raw.reshape((total_rows, dim))

    # Build FAISS index if available
    if HAS_FAISS:
        try:
            index = faiss.IndexFlatIP(dim)
            index.add(np.asarray(emb))
            try:
                faiss.write_index(index, FAISS_INDEX_PATH)
                print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
            except Exception as e:
                print(f"⚠️ Failed to persist FAISS index: {e}", file=sys.stderr)
                HAS_FAISS_local = False
        except Exception as e:
            print(f"⚠️ FAISS index build failed: {e}", file=sys.stderr)
            HAS_FAISS_local = False
        else:
            HAS_FAISS_local = True
    else:
        HAS_FAISS_local = False
        print("⚠️ FAISS not available, skipping FAISS index creation")

    # 6. Luôn lưu fallback vectors (npz) cho numpy fallback
    try:
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        print(f"✅ Saved fallback vectors to {FALLBACK_VECTORS_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save fallback vectors: {e}", file=sys.stderr)

    # 7. Lưu mapping (list of {"path","text"}) expected by app.py
    print(f"\n🗂️ Saving mapping to {FAISS_MAPPING_PATH} ...")
    try:
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(mapping)} mapping entries")
    except Exception as e:
        print(f"❌ Failed to save mapping: {e}", file=sys.stderr)

    # 8. Tạo faiss_mapping.json cho tương thích
    print(f"\n📋 Creating FAISS mapping index...")
    try:
        # Tạo mapping từ index FAISS sang tour_id
        faiss_mapping = {}
        for i in range(len(tours_data)):
            faiss_mapping[str(i)] = f"tour_{i:03d}"
        
        with open("faiss_mapping.json", "w", encoding="utf-8") as f:
            json.dump(faiss_mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved FAISS mapping index")
    except Exception as e:
        print(f"⚠️ Failed to save FAISS mapping index: {e}", file=sys.stderr)

    # 9. Write metadata
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
            "value_scoring": True
        }
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved metadata to {META_PATH}")
    except Exception:
        print(f"⚠️ Failed to save metadata", file=sys.stderr)

    # 10. Cleanup temp file
    try:
        os.remove(TMP_EMB_FILE)
    except Exception:
        pass

    # 11. Summary
    print("\n" + "=" * 60)
    print("🎉 BUILD COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   • Tours processed: {len(tours_data)}")
    print(f"   • FAISS passages: {total_rows}")
    print(f"   • Embedding dimension: {dim}")
    print(f"\n📁 Files created:")
    print(f"   • tour_entities.json: {TOUR_ENTITIES_PATH} (enhanced structure)")
    print(f"   • FAISS index: {FAISS_INDEX_PATH if HAS_FAISS_local else '(not available)'}")
    print(f"   • FAISS mapping: {FAISS_MAPPING_PATH}")
    print(f"   • Fallback vectors: {FALLBACK_VECTORS_PATH}")
    print(f"   • FAISS mapping index: faiss_mapping.json")
    print(f"   • Metadata: {META_PATH}")
    
    # Hiển thị sample của 1 tour trong tour_entities.json
    if tour_entities:
        sample_id = list(tour_entities.keys())[0]
        sample_tour = tour_entities[sample_id]
        print(f"\n📝 Sample tour structure (first tour):")
        print(f"   Tour ID: {sample_id}")
        print(f"   Name: {sample_tour.get('tour_name', 'N/A')[:50]}...")
        print(f"   Location: {sample_tour.get('location', 'N/A')}")
        print(f"   Region: {sample_tour.get('region', 'N/A')}")
        print(f"   Tags: {', '.join(sample_tour.get('tags', []))}")
        print(f"   Duration: {sample_tour.get('duration', 'N/A')} ({sample_tour.get('duration_days', 'N/A')} days)")
        print(f"   Price range: {sample_tour.get('min_price', 'N/A'):,} - {sample_tour.get('max_price', 'N/A'):,} VND")
        print(f"   Popularity score: {sample_tour.get('popularity_score', 'N/A')}")
        print(f"   Value score: {sample_tour.get('value_score', 'N/A')}")
    
    print("\n✅ Index ready for Ruby Wings v5.2 system!")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print(f"\n❌ ERROR building index: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)