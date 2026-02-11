"""
Custom exceptions for the product matching API.
Provides structured error handling across the application.
"""


class ProductMatchAPIError(Exception):
    """Base exception for all product matching API errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(ProductMatchAPIError):
    """Raised when there's an issue with application configuration."""
    
    def __init__(self, message: str, missing_key: str = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details={"missing_key": missing_key} if missing_key else {}
        )


class ModelLoadError(ProductMatchAPIError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_type: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details={"model_type": model_type} if model_type else {}
        )


class APIConnectionError(ProductMatchAPIError):
    """Raised when external API connection fails."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        super().__init__(
            message=message,
            error_code="API_CONNECTION_ERROR",
            details={
                "api_name": api_name,
                "status_code": status_code
            }
        )


class LLMProcessingError(ProductMatchAPIError):
    """Raised when LLM processing fails."""
    
    def __init__(self, message: str, provider: str = None):
        super().__init__(
            message=message,
            error_code="LLM_PROCESSING_ERROR",
            details={"provider": provider} if provider else {}
        )


class ValidationError(ProductMatchAPIError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None, field_value=None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field_name": field_name,
                "field_value": field_value
            }
        )


class ServiceInitializationError(ProductMatchAPIError):
    """Raised when service initialization fails."""
    
    def __init__(self, message: str, service_name: str = None):
        super().__init__(
            message=message,
            error_code="SERVICE_INIT_ERROR",
            details={"service_name": service_name} if service_name else {}
        )