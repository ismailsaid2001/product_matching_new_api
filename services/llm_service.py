import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import json
import config as _cfg
from rapidfuzz import fuzz
import logging
import datetime

class OrchestratorService:
    def __init__(self, enable_prompt_logging=None):
        # Prompt logging configuration
        self.enable_prompt_logging = enable_prompt_logging if enable_prompt_logging is not None else getattr(_cfg, "ENABLE_LLM_PROMPT_LOGGING", True)
        self.max_prompt_length = getattr(_cfg, "MAX_PROMPT_LOG_LENGTH", 10000)
        self.log_level = getattr(_cfg, "LLM_PROMPT_LOG_LEVEL", "INFO")
        
        self.logger = logging.getLogger('llm_prompts')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        
        # API keys retrieval
        groq_key = getattr(_cfg, "GROQ_API_KEY", None)
        tavily_key = getattr(_cfg, "TAVILY_API_KEY", None)
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY manquante dans config")

        # LLM configuration with Groq
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=groq_key
        )
        
        # Web search configuration
        self.search_tool = TavilySearchResults(api_key=tavily_key) if tavily_key else None
        
        # Available tools list
        self.tools = []

    def search_web(self, query: str):
        """Launch web search to identify an unknown product."""
        if not self.search_tool:
            raise RuntimeError("TAVILY_API_KEY manquante pour la recherche web")
        results = self.search_tool.invoke({"query": query})
        return "\n".join([r['content'] for r in results])

    def calculate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        """
        Calcule le coût d'un appel Groq avec openai/gpt-oss-120b.
        
        Tarification Groq (par 1M tokens):
        - Input: $0.50
        - Output: $0.50
        Note: Groq ne supporte pas le cache de tokens
        """
        input_cost = input_tokens * 0.075 / 1_000_000
        output_cost = output_tokens * 0.3 / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "cached_tokens": 0,  # Groq does not support cache
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "cached_cost_usd": 0.0,
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }

    def log_prompt(self, messages, call_type="INITIAL", context=""):
        """Log prompts sent to LLM with detailed formatting."""
        if not self.enable_prompt_logging:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PROMPT LLM - {call_type} - {timestamp}")
        if context:
            self.logger.info(f"Contexte: {context}")
        self.logger.info(f"{'='*80}")
        
        for i, message in enumerate(messages, 1):
            if hasattr(message, 'type'):
                msg_type = message.type.upper()
            else:
                msg_type = type(message).__name__.upper()
            
            self.logger.info(f"\nMESSAGE {i} - TYPE: {msg_type}")
            self.logger.info(f"{'-'*50}")
            
            if hasattr(message, 'content') and message.content:
                content = str(message.content)
                if len(content) > self.max_prompt_length:
                    content = content[:self.max_prompt_length-100] + f"\n... [TRONQUE - {len(message.content) - (self.max_prompt_length-100)} caracteres supplementaires]"
                self.logger.info(content)
            
            if hasattr(message, 'tool_call_id'):
                self.logger.info(f"Tool Call ID: {message.tool_call_id}")
        
        self.logger.info(f"\n{'='*80}\n")

    def save_prompt_to_file(self, messages, description, filename_prefix="prompt"):
        """Save complete prompt to file for detailed analysis."""
        if not self.enable_prompt_logging:
            return None
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean description for filename
        safe_desc = "".join(c for c in description[:50] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        filename = f"{filename_prefix}_{safe_desc}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"PROMPT LLM - {timestamp}\n")
                f.write(f"Description: {description}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, message in enumerate(messages, 1):
                    msg_type = getattr(message, 'type', type(message).__name__).upper()
                    f.write(f"MESSAGE {i} - TYPE: {msg_type}\n")
                    f.write("-" * 50 + "\n")
                    
                    if hasattr(message, 'content') and message.content:
                        f.write(str(message.content) + "\n")
                    
                    if hasattr(message, 'tool_call_id'):
                        f.write(f"Tool Call ID: {message.tool_call_id}\n")
                    
                    f.write("\n")
                    
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving prompt: {e}")
            return None

    def arbitrate(self, description, t5_suggestion, t5_confidence, api_suggestions, web_context=None):
        """Optimized agent logic - only API suggestions or creation."""
        
        system_prompt = """You are an expert in Logistics Data Normalization (Master Data Management). Your mission is to convert a raw invoice description into a "nature_product": a canonical, generic, precise name ALWAYS IN FRENCH

LOGIC:
1. If API suggestions score ≥ 0.82 and really corresponds to the description → use the EXACT suggestion
2. Otherwise → create a new nature product

CREATION RULES:
- French, no articles (le/la)
- Remove standard volumes (33cl, 75cl) except special formats
- Keep important info: product type + key characteristics

CREATION - EXAMPLES:
descriptions → nature_product
- "cote detallonee angus boeuf angus" → "Boeuf angus côte détallonnée"
- "kolors mousse" → "Mousse nettoyante"
- "HEINEKEN BOUTEILLE 33CL" → "Bière blonde"
- "HUILE OLIVE BIDON 5L" → "Huile olive 5L"
- "jambon fume demi par piece de 3.7 kg" → "Jambon cru demi"

In summary:
1. Check API suggestions, if a suggestion has a high similarity score (≥ 0.82) and corresponds well to the description, use this EXACT suggestion.
2. Apply the standard volume rule according to the identified category.
3. Produce the shortest and most relevant French canonical name from the raw description.

RESPOND ONLY WITH THE FINAL LABEL."""

        # Format only the best suggestions (top 3 max)
        top_suggestions = api_suggestions[:3] if api_suggestions else []
        suggestions_text = ""
        if top_suggestions:
            suggestions_text = " | ".join([f"{s.get('nature_product', '')} ({s.get('similarity_score', 0):.2f})" for s in top_suggestions])

        user_content = f"Description: {description}\nSuggestions: {suggestions_text}\nT5: {t5_suggestion} ({t5_confidence:.2f})"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        # LLM call with Groq
        response = self.llm.invoke(messages)
        
        # Extract usage metadata for Groq
        usage = response.response_metadata.get("token_usage", {})
        
        # Calculate costs for Groq
        cost = self.calculate_cost(
            usage.get("prompt_tokens", 0), 
            usage.get("completion_tokens", 0),
            0
        )

        final_response = response.content.strip()
        if not final_response:
            final_response = "Produit non identifie"

        return final_response, cost