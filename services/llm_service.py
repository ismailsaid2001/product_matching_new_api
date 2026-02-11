from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional, Dict, List, Tuple
import logging
import config as _cfg


class OrchestratorService:
    """
    Simplified LLM service focused on orchestrating product classification.
    """
    
    def __init__(self, enable_prompt_logging: Optional[bool] = None):
        """Initialize the orchestrator service."""
        self.service_name = "llm_orchestrator"
        self.logger = self._setup_logger()
        
        # Store configuration
        self.enable_prompt_logging = (
            enable_prompt_logging 
            if enable_prompt_logging is not None 
            else _cfg.ENABLE_LLM_PROMPT_LOGGING
        )
        
        self.model_name = "openai/gpt-oss-safeguard-20b"
        
        # Initialize components
        self._initialize()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this service"""
        logger = logging.getLogger(f'{self.service_name}_service')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.service_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize(self) -> None:
        """Initialize the LLM and search stuff"""
        try:
            self.logger.info("Starting up LLM service")
            
            # check if we have the groq key
            if not _cfg.GROQ_API_KEY:
                raise RuntimeError("Need GROQ_API_KEY for this to work")
            
            # setup groq client
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=0,
                api_key=_cfg.GROQ_API_KEY
            )
            self.logger.info(f"Groq LLM ready with: {self.model_name}")
            
            # setup web search if we have the key
            if _cfg.TAVILY_API_KEY:
                self.search_tool = TavilySearchResults(api_key=_cfg.TAVILY_API_KEY)
                self.logger.info("Web search ready")
            else:
                self.search_tool = None
                self.logger.warning("No TAVILY_API_KEY - web search disabled")
            
            self.logger.info("LLM service initialization done")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LLM service: {e}")
            raise
    
    def search_web(self, query: str) -> str:
        """Launch web search to identify an unknown product."""
        if not self.search_tool:
            raise RuntimeError("Web search is not available - TAVILY_API_KEY missing")
        
        try:
            self.logger.info(f"Performing web search for: {query}")
            results = self.search_tool.invoke({"query": query})
            search_content = "\n".join([r['content'] for r in results])
            self.logger.info(f"Web search completed, found {len(results)} results")
            return search_content
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            raise
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> dict:
        """Calculate cost for Groq API usage."""
        input_cost = input_tokens * 0.075 / 1_000_000
        output_cost = output_tokens * 0.3 / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }
    
    def arbitrate(self, 
                 description: str, 
                 t5_suggestion: str, 
                 t5_confidence: float, 
                 api_suggestions: List[Dict], 
                 web_context: Optional[str] = None) -> Tuple[str, dict]:
        """
        Orchestrate product classification using LLM with API suggestions and T5 input.
        """
        try:
            self.logger.info(f"Starting arbitration for: {description[:50]}...")
            
            # Prepare the system prompt
            system_prompt = """You are an expert in Logistics Data Normalization (Master Data Management). Your mission is to convert raw invoice descriptions into standardized "nature_product" names: canonical, generic, precise names ALWAYS IN FRENCH.

DECISION LOGIC:
1. HIGH CONFIDENCE MATCH (API suggestion ≥ 0.9): Give it a chance ,if it perfectly matches the product type, brand, and unit specifications use the EXACT API suggestion .
2. LOW CONFIDENCE MATCH (< 0.82): Create a new standardized nature_product following normalization rules

NORMALIZATION RULES:
• Language: French only, no articles (le/la/les/des)
• Volume/Size: Remove standard volumes (33cl beer, 75cl wine) EXCEPT special formats
• Brand: Keep brand names when they're the primary identifier
• Product type: Always include the core product category
• Specifications: Keep important characteristics (organic, aged, size variants)
• Format: Shortest meaningful canonical name

CATEGORY-SPECIFIC RULES:

BEVERAGES:
- Remove standard volumes: 33cl (beer), 75cl (wine), 25cl, 50cl
- Keep special formats: 150cl, 300cl, 5L, etc.
- Examples: "heineken 33cl" → "heineken" | "champagne dom perignon 150cl" → "champagne dom perignon 150cl"

MEAT/CHARCUTERIE:
- Specify cut/type + characteristics
- Keep aging/preparation method
- Examples: "jambon cru 24 mois" → "jambon cru 24 mois" | "saucisson sec" → "saucisson sec"

CHEESE/DAIRY:
- Include aging when relevant (12 mois, 24 mois)
- Specify format when important (bloc, râpé, tranches)
- Examples: "comte 24 mois" → "comte 24 mois" | "emmental rapes" → "emmental râpé"

CLEANING/HOUSEHOLD:
- Include product type + brand + key characteristics
- Examples: "liquide vaisselle citron" → "liquide vaisselle citron"

FOOD ITEMS:
- Keep preparation method (cuit, cru, fumé, bio)
- Include packaging when it affects usage (conserve, frais, surgelé)

TRANSFORMATION EXAMPLES:

KITCHEN EQUIPMENT:
"couteau doffice 11 cm acier plastique unie" → "couteau office 11cm"
"araignee de buyer 16cm" → "araignee buyer"
"spatule bois" → "spatule bois"
"pince feuille de chene inox 23cm" → "pince feuille chene inox 23cm"
"planche a decouper bar poly 35x25 cm" → "planche polyethylene"
"balance 10kg 10g a sect dbl affich" → "balance inox"

COOKING VESSELS:
"moule alu tartes t1" → "moule tarte t1"
"casserole inox o18 cm moyenne" → "casserole"
"braisiere ix d28x185cm 11l ecoplus" → "braisiere inox"
"poele alu a ad 4 couches0280" → "poele aluminium"
"cocotte ovale 31cm signature meringue" → "cocotte"

STORAGE & CONTAINERS:
"bac allibert ra 1120" → "bac plastique"
"bocal weck 200 ml bigarrade dapple betteravepopotte du chef 120g" → "bocal weck"
"bac inox gn1 4 p 15cm 41l" → "bac inox/blanc"
"godet aluminium 75 x 40 90 ml x 100" → "godet aluminium"

PASTRY EQUIPMENT:
"poche jetable gm chaude 30x54cm x100" → "poche patissiere 30x54"
"plaque souple flexipan 24 cakes all 120x40mm prof20mm 585x385mm pour plaques 600x400" → "plaque souple flexipan cakes all"
"toile patissiere fiberlux dim 400x300 mm" → "toile cuisson"
"bte 9 decoupoirs ix ronds canneles" → "decoupoir rond cannele"

BEVERAGE EQUIPMENT:
"saupoudreuse parmesan inox" → "saupoudreuse parmesan"
"shaker 50cl inox 2 pieces hypinox" → "shaker"
"machine espresso sage the dual boiler unite" → "machine a cafe"
"moulin comandante noir unite" → "moulin a cafe"

SPECIALTY ITEMS:
"archibaltic porter 30 l key kegbiere de type baltic porter 75pcent alc au poivre long de javafut 30 l a usage unique" → "keykeg"
"2bouchon nikele champagne a vi" → "bouchon champagne"
"tube naturco2 10kg l2pi orange" → "co2"
"distributeur pressable transp 709cl" → "distributeur sauce"

CLEANING & MAINTENANCE:
"pulverisateur d epaule pulsen pour produit chimique" → "pulverisateur"
"colle instant loctite 401 20gr" → "colle instantanee"
"distributeur de savon blanc 245x11x99" → "distributeur de savon ou gel hydroalcoolique"

PLUMBING & HARDWARE:
"mamelon laiton m m 20 27 15 21" → "mamelon"
"mitigeur evier premier" → "mitigeur"
"presse etoupe pg 16" → "presse etoupe"
"raccord gaz mm20x150" → "raccord"

WHAT NOT TO DO:
- Don't add articles: "le jambon" → "jambon"
- Don't keep redundant info: "fromage comte fromage" → "comté"
- Don't over-specify common items: "eau plate 1.5L evian" → "eau evian 1.5L"
- Don't translate brand names: "Coca-Cola" stays "coca cola"

QUALITY CHECKS:
1. Is it in French?
2. No articles?
3. Essential info preserved?
4. Shortest meaningful form?
5. Category-appropriate normalization?

PROCESS:
1. Check API suggestion confidence (≥0.82 = use exact match if appropriate)
2. Identify product category
3. Apply category-specific rules
4. Verify normalization quality
5. Output final canonical name

RESPOND ONLY WITH THE FINAL STANDARDIZED LABEL."""
            
            # Format suggestions
            top_suggestions = api_suggestions[:3] if api_suggestions else []
            suggestions_text = ""
            if top_suggestions:
                suggestions_text = " | ".join([
                    f"{s.get('nature_product', '')} ({s.get('similarity_score', 0):.2f})" 
                    for s in top_suggestions
                ])
            
            # Create user content
            user_content = f"Description: {description}\nSuggestions: {suggestions_text}\nT5: {t5_suggestion} ({t5_confidence:.2f})"
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]
            
            # Make LLM call
            response = self.llm.invoke(messages)
            
            # Extract usage and calculate cost
            usage = response.response_metadata.get("token_usage", {})
            cost = self.calculate_cost(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )
            
            # Process response
            final_response = response.content.strip()
            if not final_response:
                final_response = "Produit non identifie"
                self.logger.warning(f"Empty response from LLM for: {description}")
            
            self.logger.info(f"Arbitration completed: {final_response}")
            return final_response, cost
            
        except Exception as e:
            error_msg = f"LLM arbitration failed for: {description}"
            self.logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg)