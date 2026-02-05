import requests
from config import API_URL


def _normalize_suggestions(payload):
    """
    Normalise la réponse de l'API find_suggestions.
    La réponse a la structure:
    {
      "nature_product_suggestions": [
        {
          "nature_product": str,
          "nature_product_group": str,
          "similarity_score": float
        },
        ...
      ]
    }
    """
    # Extraire la liste des suggestions
    if isinstance(payload, dict):
        suggestions = payload.get("nature_product_suggestions", [])
    elif isinstance(payload, list):
        # Fallback si la réponse est directement une liste
        suggestions = payload
    else:
        suggestions = []

    # Filtrer et trier par similarity_score décroissant
    valid_suggestions = [s for s in suggestions if isinstance(s, dict) and s.get("similarity_score", 0) > 0]
    valid_suggestions.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    
    return valid_suggestions


def get_database_suggestions(designation: str):
    """
    Appelle l'endpoint find_suggestions avec la structure simplifiée.
    """
    try:
        payload = {
            "designation": designation
        }
        
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return _normalize_suggestions(response.json())
    except Exception as e:
        print(f"Erreur API (find_suggestions): {e}")
        return []
