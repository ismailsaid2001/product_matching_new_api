import requests
from config import API_URL


def _normalize_suggestions(payload):
    """
    Normalise la réponse de l'API find_similar.
    La réponse est directement une liste:
    [
      {
        "nature_product": str,
        "similarity_score": float,
        "nature_product_id": int,
        "category": int,
        ...
      },
      ...
    ]
    """
    # La réponse est déjà une liste normalisée
    if isinstance(payload, list):
        suggestions = payload
    elif isinstance(payload, dict):
        # Fallback au cas où l'API enveloppe dans un objet
        suggestions = (
            payload.get("nature_product_suggestions")
            or payload.get("suggestions")
            or payload.get("results")
            or payload.get("items")
            or []
        )
    else:
        suggestions = []

    # Trier par similarity_score décroissant
    valid_suggestions = [s for s in suggestions if isinstance(s, dict)]
    valid_suggestions.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    
    return valid_suggestions


def get_database_suggestions(designation: str):
    """
    Appelle l'endpoint find_similar avec la structure requise.
    Seul le champ 'designation' est rempli, les autres sont des placeholders.
    """
    try:
        payload = {
            "hub": "",
            "id_user": 0,
            "id_document": 0,
            "created_at": "",
            "supplier_name": "",
            "supplier_unique_id": "",
            "items": [
                {
                    "id_item": 0,
                    "item_code": "",
                    "designation": designation,
                    "unit_price": 0,
                    "uom": ""
                }
            ]
        }
        
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return _normalize_suggestions(response.json())
    except Exception as e:
        print(f"Erreur API (find_similar): {e}")
        return []
