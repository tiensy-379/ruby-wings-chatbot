# common_utils.py
from typing import Any, Dict

def flatten_json(data: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}"
            items.update(flatten_json(v, new_key, sep))
    else:
        items[parent_key] = data
    return items
