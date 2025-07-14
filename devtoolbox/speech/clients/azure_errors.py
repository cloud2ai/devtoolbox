class AzureError(Exception):
    """Base exception for Azure-related errors"""
    pass

class AzureRateLimitError(AzureError):
    """Raised when Azure rate limit is exceeded"""
    pass

class AzureConfigError(AzureError):
    """Raised when Azure configuration is invalid"""
    pass

class AzureSynthesisError(AzureError):
    """Raised when speech synthesis fails"""
    pass

class AzureRecognitionError(AzureError):
    """Raised when speech recognition fails"""
    pass