# common_utils.py
from typing import Any, Dict

import json

def flatten_json(data):
    """Convert tours to searchable passages"""
    # FIX: Handle both file path (string) and loaded JSON data
    if isinstance(data, str):
        # If data is a string, assume it's a file path
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded JSON from file")
    
    # Ensure data is a dictionary with 'tours' key
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    
    tours = data.get("tours", [])
    passages = []
    
    for tour in tours:
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
    
    print(f"[DEBUG] Created {len(passages)} passages from {len(tours)} tours")
    return passages
