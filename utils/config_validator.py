"""
Configuration validation and management for the Product Match API.
Provides centralized configuration loading, validation, and defaults.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from utils.exceptions import ConfigurationError

# Load environment variables from .env file
load_dotenv()


class ConfigValidator:
    """Validates and manages application configuration."""
    
    # Required environment variables
    REQUIRED_VARS = {
        "GROQ_API_KEY": "Groq API key for LLM processing"
    }
    
    # Optional environment variables with defaults
    OPTIONAL_VARS = {
        "OPENAI_API_KEY": None,
        "TAVILY_API_KEY": None,
        "HUGGINGFACE_TOKEN": None,
        "ENABLE_LLM_PROMPT_LOGGING": "false",
        "MAX_PROMPT_LOG_LENGTH": "2000",
        "LLM_PROMPT_LOG_LEVEL": "INFO"
    }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate all configuration and return validated config dict.
        
        Returns:
            Dictionary of validated configuration values
            
        Raises:
            ConfigurationError: If required configuration is missing
        """
        config = {}
        errors = []
        
        # Check required variables
        for var_name, description in cls.REQUIRED_VARS.items():
            value = os.getenv(var_name)
            if not value:
                errors.append(f"Missing required environment variable: {var_name} ({description})")
            else:
                config[var_name] = value
        
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )
        
        # Set optional variables with defaults
        for var_name, default in cls.OPTIONAL_VARS.items():
            config[var_name] = os.getenv(var_name, default)
        
        # Convert string booleans
        config["ENABLE_LLM_PROMPT_LOGGING"] = config["ENABLE_LLM_PROMPT_LOGGING"].lower() == "true"
        
        # Convert string integers
        try:
            config["MAX_PROMPT_LOG_LENGTH"] = int(config["MAX_PROMPT_LOG_LENGTH"])
        except ValueError:
            config["MAX_PROMPT_LOG_LENGTH"] = 2000
        
        return config


# Validate configuration on import
try:
    validated_config = ConfigValidator.validate_config()
except ConfigurationError as e:
    # Re-raise with more context
    raise ConfigurationError(f"Failed to load configuration: {e.message}")

# ==================== API KEYS ====================
OPENAI_API_KEY = validated_config.get("OPENAI_API_KEY")
GROQ_API_KEY = validated_config["GROQ_API_KEY"]
TAVILY_API_KEY = validated_config.get("TAVILY_API_KEY")
HF_TOKEN = validated_config.get("HUGGINGFACE_TOKEN")

# External API endpoint
API_URL = "http://178.33.46.169:8012/find_suggestions"

# ==================== MODEL CONFIGURATION ====================
MODEL_PATH = "./checkpoint-11004"
BASE_MODEL_ID = "google/t5gemma-b-b-prefixlm"

# ==================== CRITICAL THRESHOLDS ====================
THRESHOLD_DATABASE = 0.94
THRESHOLD_T5_CONF = 0.95

# ==================== LOGGING CONFIGURATION ====================
ENABLE_LLM_PROMPT_LOGGING = validated_config["ENABLE_LLM_PROMPT_LOGGING"]
MAX_PROMPT_LOG_LENGTH = validated_config["MAX_PROMPT_LOG_LENGTH"]
LLM_PROMPT_LOG_LEVEL = validated_config["LLM_PROMPT_LOG_LEVEL"].upper()


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration (excluding sensitive data).
    
    Returns:
        Configuration summary dictionary
    """
    return {
        "api_keys_configured": {
            "groq": bool(GROQ_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
            "huggingface": bool(HF_TOKEN)
        },
        "model_config": {
            "model_path": MODEL_PATH,
            "base_model_id": BASE_MODEL_ID
        },
        "thresholds": {
            "database": THRESHOLD_DATABASE,
            "t5_confidence": THRESHOLD_T5_CONF
        },
        "logging": {
            "enabled": ENABLE_LLM_PROMPT_LOGGING,
            "max_length": MAX_PROMPT_LOG_LENGTH,
            "level": LLM_PROMPT_LOG_LEVEL
        }
    }


def validate_model_path() -> bool:
    """
    Validate that the model path exists.
    
    Returns:
        True if model path exists, False otherwise
    """
    return os.path.exists(MODEL_PATH)


def get_missing_optional_config() -> list:
    """
    Get list of optional configuration that is missing.
    
    Returns:
        List of missing optional configuration keys
    """
    missing = []
    
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY (required for web search)")
    if not HF_TOKEN:
        missing.append("HUGGINGFACE_TOKEN")
    
    return missing