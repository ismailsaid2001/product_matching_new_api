"""
Cost calculation service for various LLM providers.
Handles cost calculations previously embedded in OrchestratorService.
"""
from typing import Dict, Union
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class CostCalculator:
    """Centralized cost calculation for different LLM providers."""
    
    # Pricing per 1M tokens (USD)
    PRICING = {
        LLMProvider.GROQ: {
            "openai/gpt-oss-safeguard-20b": {
                "input": 0.075,
                "output": 0.3,
                "supports_cache": False
            },
            "llama-3.1-70b-versatile": {
                "input": 0.59,
                "output": 0.79,
                "supports_cache": False
            }
        },
        LLMProvider.OPENAI: {
            "gpt-4o": {
                "input": 2.50,
                "output": 10.0,
                "supports_cache": True,
                "cache_discount": 0.5
            },
            "gpt-4o-mini": {
                "input": 0.15,
                "output": 0.6,
                "supports_cache": True,
                "cache_discount": 0.5
            }
        }
    }
    
    @classmethod
    def calculate_cost(cls, 
                      provider: Union[LLMProvider, str],
                      model: str,
                      input_tokens: int,
                      output_tokens: int,
                      cached_tokens: int = 0) -> Dict:
        """
        Calculate cost for LLM usage.
        
        Args:
            provider: LLM provider (groq, openai, etc.)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens (if supported)
            
        Returns:
            Dictionary with cost breakdown
            
        Raises:
            ValueError: If provider/model not supported
        """
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        
        if provider not in cls.PRICING:
            raise ValueError(f"No pricing data for provider: {provider}")
        
        if model not in cls.PRICING[provider]:
            raise ValueError(f"No pricing data for model: {model} on provider: {provider}")
        
        pricing = cls.PRICING[provider][model]
        
        # Calculate input cost
        input_cost = input_tokens * pricing["input"] / 1_000_000
        
        # Calculate cached tokens cost (if supported)
        cached_cost = 0.0
        if cached_tokens > 0 and pricing.get("supports_cache", False):
            cache_discount = pricing.get("cache_discount", 0.5)
            cached_cost = cached_tokens * pricing["input"] * cache_discount / 1_000_000
            # Reduce input cost by cached amount
            input_cost = (input_tokens - cached_tokens) * pricing["input"] / 1_000_000
        
        # Calculate output cost
        output_cost = output_tokens * pricing["output"] / 1_000_000
        
        total_cost = input_cost + cached_cost + output_cost
        
        return {
            "provider": provider.value,
            "model": model,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens if pricing.get("supports_cache", False) else 0,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "cached_cost_usd": round(cached_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "supports_cache": pricing.get("supports_cache", False)
        }
    
    @classmethod
    def calculate_groq_cost(cls, 
                           input_tokens: int, 
                           output_tokens: int,
                           model: str = "openai/gpt-oss-safeguard-20b") -> Dict:
        """
        Convenience method for Groq cost calculation.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Groq model name
            
        Returns:
            Cost breakdown dictionary
        """
        return cls.calculate_cost(
            provider=LLMProvider.GROQ,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0  # Groq doesn't support caching
        )
    
    @classmethod
    def calculate_openai_cost(cls,
                             input_tokens: int,
                             output_tokens: int,
                             cached_tokens: int = 0,
                             model: str = "gpt-4o-mini") -> Dict:
        """
        Convenience method for OpenAI cost calculation.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens
            model: OpenAI model name
            
        Returns:
            Cost breakdown dictionary
        """
        return cls.calculate_cost(
            provider=LLMProvider.OPENAI,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens
        )
    
    @classmethod
    def add_custom_pricing(cls,
                          provider: Union[LLMProvider, str],
                          model: str,
                          input_price: float,
                          output_price: float,
                          supports_cache: bool = False,
                          cache_discount: float = 0.5) -> None:
        """
        Add custom pricing for a new model.
        
        Args:
            provider: LLM provider
            model: Model name
            input_price: Price per 1M input tokens (USD)
            output_price: Price per 1M output tokens (USD)
            supports_cache: Whether the model supports token caching
            cache_discount: Discount factor for cached tokens
        """
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        
        if provider not in cls.PRICING:
            cls.PRICING[provider] = {}
        
        cls.PRICING[provider][model] = {
            "input": input_price,
            "output": output_price,
            "supports_cache": supports_cache,
            "cache_discount": cache_discount
        }