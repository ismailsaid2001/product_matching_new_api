"""
Base service class providing common functionality for all services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from utils.logging_service import LLMLoggingService
from utils.cost_calculator import CostCalculator
from utils.exceptions import ServiceInitializationError


class BaseService(ABC):
    """
    Abstract base class for all services in the product matching API.
    Provides common functionality like logging, error handling, and configuration.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the base service.
        
        Args:
            service_name: Name of the service for logging and error handling
        """
        self.service_name = service_name
        self.logger = self._setup_logger()
        self._initialized = False
        
    def _setup_logger(self) -> logging.Logger:
        """Setup service-specific logger."""
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
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the service. Must be implemented by subclasses.
        Should set self._initialized = True when complete.
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized
    
    def ensure_initialized(self) -> None:
        """Ensure the service is initialized, initialize if not."""
        if not self.is_initialized:
            self.initialize()
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message."""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def handle_initialization_error(self, error: Exception, context: str = "") -> None:
        """
        Handle initialization errors consistently across services.
        
        Args:
            error: The exception that occurred
            context: Additional context about what was being initialized
        """
        error_msg = f"Failed to initialize {self.service_name}"
        if context:
            error_msg += f" ({context})"
        
        self.log_error(error_msg, error)
        raise ServiceInitializationError(
            message=error_msg,
            service_name=self.service_name
        )


class LLMServiceMixin:
    """Mixin for services that use LLM functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_logger: Optional[LLMLoggingService] = None
        self.cost_calculator = CostCalculator()
    
    def setup_llm_logging(self, 
                         enable_logging: bool = True,
                         max_prompt_length: int = 10000,
                         log_level: str = "INFO") -> None:
        """Setup LLM-specific logging."""
        self.llm_logger = LLMLoggingService(
            enable_logging=enable_logging,
            max_prompt_length=max_prompt_length,
            log_level=log_level
        )
    
    def log_llm_prompt(self, messages, call_type: str = "INITIAL", context: str = "") -> None:
        """Log LLM prompts if logging is enabled."""
        if self.llm_logger:
            self.llm_logger.log_prompt(messages, call_type, context)
    
    def save_llm_prompt(self, messages, description: str, filename_prefix: str = "prompt") -> Optional[str]:
        """Save LLM prompt to file if logging is enabled."""
        if self.llm_logger:
            return self.llm_logger.save_prompt_to_file(messages, description, filename_prefix)
        return None


class APIServiceMixin:
    """Mixin for services that make external API calls."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_timeout = 10  # Default timeout in seconds
    
    def set_api_timeout(self, timeout: int) -> None:
        """Set API timeout for external calls."""
        self.api_timeout = timeout
    
    def handle_api_error(self, error: Exception, api_name: str) -> None:
        """Handle API errors consistently."""
        from utils.exceptions import APIConnectionError
        
        error_msg = f"API call to {api_name} failed"
        self.log_error(error_msg, error)
        
        status_code = getattr(error, 'response', {}).get('status_code', None)
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code
        
        raise APIConnectionError(
            message=error_msg,
            api_name=api_name,
            status_code=status_code
        )