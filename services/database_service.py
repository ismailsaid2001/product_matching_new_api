import requests
from typing import List, Dict
import config as _cfg


def _normalize_suggestions(payload) -> List[Dict]:
    """
    Normalize the find_suggestions API response.
    """
    # Extract the suggestions list
    if isinstance(payload, dict):
        suggestions = payload.get("nature_product_suggestions", [])
    elif isinstance(payload, list):
        suggestions = payload
    else:
        suggestions = []

    # Filter and sort by descending similarity_score
    valid_suggestions = [s for s in suggestions if isinstance(s, dict) and s.get("similarity_score", 0) > 0]
    valid_suggestions.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    
    return valid_suggestions


def get_database_suggestions(designation: str) -> List[Dict]:
    """
    Call the find_suggestions endpoint with simplified structure.
    """
    try:
        payload = {
            "designation": designation
        }
        
        response = requests.post(
            _cfg.API_URL,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return _normalize_suggestions(response.json())
    except Exception as e:
        print(f"API Error (find_suggestions): {e}")
        return []
