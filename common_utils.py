# common_utils.py
from typing import Any, Dict

def flatten_json(data):
    """Convert tours to searchable passages"""
    passages = []
    
    for tour in data.get("tours", []):
        # Lấy tất cả trường có giá trị
        parts = []
        
        # Các trường quan trọng
        fields = [
            "tour_name", "summary", "location", "duration", 
            "price", "includes", "notes", "style", 
            "transport", "accommodation", "meals", "event_support"
        ]
        
        for field in fields:
            value = tour.get(field)
            if value:
                if isinstance(value, list):
                    parts.append(f"{field}: {', '.join(str(v) for v in value)}")
                else:
                    parts.append(f"{field}: {value}")
        
        if parts:  # Chỉ thêm nếu có dữ liệu
            passages.append(" | ".join(parts))
    
    print(f"[DEBUG] Created {len(passages)} passages from {len(data.get('tours', []))} tours")
    return passages
