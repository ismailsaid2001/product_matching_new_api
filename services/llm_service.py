from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import config as _cfg

class OrchestratorService:
    def __init__(self):
        # Récupération des clés API (si absentes, on lève une exception gérée en amont)
        openai_key = getattr(_cfg, "OPENAI_API_KEY", None)
        tavily_key = getattr(_cfg, "TAVILY_API_KEY", None)
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans config")

        # Configuration du cerveau (GPT-5-nano)
        self.llm = ChatOpenAI(
            model="gpt-5-nano", 
            temperature=0,
            api_key=openai_key
        ).with_config({"return_usage_metadata": True})  # Get token usage without stream_options
        # Configuration de l'enquêteur (Web Search)
        if tavily_key:
            self.search_tool = TavilySearchResults(api_key=tavily_key)
        else:
            self.search_tool = None

    def search_web(self, query: str):
        """Lance une recherche web pour identifier un produit inconnu."""
        if not self.search_tool:
            raise RuntimeError("TAVILY_API_KEY manquante pour la recherche web")
        results = self.search_tool.invoke({"query": query})
        # On concatène les résultats pour le contexte
        return "\n".join([r['content'] for r in results])

    def calculate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        """
        Calculate the cost of a GPT-5-mini API call.
        
        Pricing (per 1M tokens):
        - Input: $0.050
        - Cached Input: $0.005
        - Output: $0.400
        
        Returns:
            dict with cost breakdown in USD
        """
        # Convert to millions
        input_cost = (input_tokens - cached_tokens) * 0.05 / 1_000_000
        cached_cost = cached_tokens * 0.005 / 1_000_000
        output_cost = output_tokens * 0.4 / 1_000_000
        total_cost = input_cost + cached_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "cached_cost_usd": round(cached_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }

    def arbitrate(self, description, t5_suggestion, t5_confidence, api_suggestions, web_context=None):
        """Le prompt final qui prend la décision."""
        
        system_prompt = """### RÔLE
        Tu es un expert en Normalisation de Données Logistiques (Master Data Management). Ta mission est de convertir une description de facture brute en un "nature_product" : un nom canonique, générique, précis et TOUJOURS EN FRANÇAIS.

        ### LOGIQUE DE DÉCISION GÉNÉRALE
        1.  **Identifier la catégorie** du produit.
        2.  **Arbitrer sur la Quantité/Volume** :
            * Si le volume est la **norme standard** de la catégorie (ex: 75cl pour le vin) -> **SUPPRIMER**.
            * Si le volume est **atypique ou définit un format logistique** (ex: fût, magnum, sac de 25kg) -> **CONSERVER**.
        3.  **Prioriser l'Exact Match** : Si une "Suggestion Nomenclature API" est sémantiquement correcte, utilise son libellé exact.

        ### TABLE DES STANDARDS MÉTIER (Règles d'élagage)
        | Catégorie | Valeurs Standards (À SUPPRIMER) | Exceptions Logistiques (À CONSERVER) |
        | :--- | :--- | :--- |
        | **Vins / Champagne** | 70cl, 75cl | 37.5cl (Demi), 1.5L (Magnum), 3L+, Cubi, BIB |
        | **Spiritueux** | 70cl, 1L | Mignonnettes, 1.5L, 3L |
        | **Bières** | 25cl, 33cl, 50cl | Fût (6L, 20L, 30L), 75cl (Artisanale) |
        | **Softs / Eaux** | 33cl, 50cl, 1L, 1.5L | Post-mix, Fontaine, 25cl (Verre consigné) |
        | **Épicerie Sèche** | 500g, 1kg | Sacs vrac 5kg, 10kg, 25kg |
        | **Liquides (Huile/Lait)**| 1L | Bidon 5L, 10L, 20L |
        | **Produits Frais** | 125g, 250g, 500g | Seau 5kg, Format industriel |

        ### RÈGLES DE NETTOYAGE (Bruit)
        - **Langue** : Traduire systématiquement en Français (ex: "Tomato" -> "Tomate").
        - **Conditionnement** : Supprimer le bruit type "X12", "Pack", "Carton", "C6" sauf si c'est un format de vente indivisible (ex: "Oeufs x30").
        - **Marques** : Supprimer les marques commerciales (ex: "Evian", "Heineken") sauf si elles définissent la nature unique du produit (ex: "Coca-Cola").
        - **Style** : Pas d'articles (Le/La), "Sentence case" (Majuscule en début de mot uniquement).

        ### EXEMPLES DE RÉFÉRENCE
        - "CHATEAU PETRUS 2015 75CL" -> "Vin rouge Bordeaux"
        - "CHATEAU PETRUS 2015 1.5L" -> "Vin rouge Bordeaux 1.5L"
        - "HEINEKEN BOUTEILLE 33CL" -> "Bière blonde"
        - "HEINEKEN FUT 30L" -> "Bière blonde en fût 30L"
        - "HUILE FRITURE CUISINOR BIDON 20L" -> "Huile friture 20L"
        - "YAOURT NATURE PEUPLIERS 125G" -> "Yaourt nature"
        - "FARINE DE BLE SAC 25KG" -> "Farine de blé 25kg"
        - "bol saveurs ocean 115cm 28cl" -> "Bol"
        - "carcasse poulet vrac fr 9420g" -> "Carcasse poulet"
        - "celeri branche kg italie" -> "Céleri branche"
        - "celeri rave 6p plt 7k be" -> "Céleri rave"
        - "petit nova nat 84pcent lfr30gx6x8 nova" -> "petit suisse nature"
        - "poudre de piment despelette aop par pot de 50 g" -> "Piment Espelette poudre"
        ### MISSION FINALE
        1. Applique la règle du volume standard selon la catégorie identifiée.
        2. Produis le nom canonique français le plus court et pertinent sur la description brute.

        RÉPONDS UNIQUEMENT AVEC LE LIBELLÉ FINAL."""

        user_content = f"""### SOURCES POUR L'ARBITRAGE
        - **Description brute** : {description}
        - **Suggestions Nomenclature API** : {api_suggestions}
        - **Contexte Web** : {web_context if web_context else 'N/A'}

        Quel est le nature_product final ?"""

        response = self.llm.invoke([
            ("system", system_prompt),
            ("user", user_content)
        ])
        
        # Extract token usage from response
        usage = response.response_metadata.get("token_usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        
        # Calculate cost
        cost_info = self.calculate_cost(input_tokens, output_tokens, cached_tokens)
        
        # Log the cost
        print(f"💰 Coût GPT-5-nano: ${cost_info['total_cost_usd']:.6f} "
              f"(Input: {input_tokens} tokens, Output: {output_tokens} tokens)")
        
        return response.content, cost_info