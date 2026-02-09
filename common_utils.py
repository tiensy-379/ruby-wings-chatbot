# common_utils.py
from typing import Any, Dict

import json

import json

def flatten_json(data):
    """Convert tours to searchable passages - FIXED VERSION"""
    print(f"[DEBUG] flatten_json called with data type: {type(data)}")
    
    # Handle both file path and loaded JSON
    if isinstance(data, str):
        try:
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[DEBUG] Loaded JSON from file")
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {e}")
            return []
    
    # Check if data is a dict with tours
    if not isinstance(data, dict):
        print(f"[ERROR] Expected dict, got {type(data)}")
        return []
    
    tours = data.get("tours", [])
    if not tours:
        print(f"[WARNING] No tours found in data")
        return []
    
    print(f"[DEBUG] Processing {len(tours)} tours")
    
    mapping = []
    
    for i, tour in enumerate(tours):
        try:
            parts = []
            
            # Always include tour_name if available
            tour_name = tour.get("tour_name", "").strip()
            if tour_name:
                parts.append(f"tour_name: {tour_name}")
            
            # Include other fields if they exist
            fields_to_check = [
                "summary", "location", "duration", "price",
                "includes", "notes", "style", "transport",
                "accommodation", "meals", "event_support"
            ]
            
            for field in fields_to_check:
                value = tour.get(field)
                if value:
                    if isinstance(value, list):
                        if value:  # Only add non-empty lists
                            parts.append(f"{field}: {', '.join(str(v) for v in value)}")
                    elif isinstance(value, str) and value.strip():
                        parts.append(f"{field}: {value.strip()}")
            
            # Create passage - even if just tour_name
            if parts:
                passage = " | ".join(parts)
                mapping.append({
                    "text": passage,
                    "tour_name": tour_name,
                    "location": tour.get("location", ""),
                    "duration": tour.get("duration", ""),
                    "price": tour.get("price", "")
                })
                if i < 2:  # Log first 2 tours for debugging
                    print(f"[DEBUG] Tour {i}: {tour_name} -> {len(passage)} chars")
            else:
                print(f"[WARNING] Tour {i} has no data")
                
        except Exception as e:
            print(f"[ERROR] Error processing tour {i}: {e}")
            continue
    
    print(f"[DEBUG] Created {len(mapping)} passages")
    return mapping
