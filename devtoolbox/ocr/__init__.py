"""OCR module.

This module provides OCR (Optical Character Recognition) functionality.

Features:
- Azure Document Intelligence integration
- Automatic retry for temporary errors (429, 500, 502, 503, 504)
- Detailed error reporting and logging
- File size validation (max 500MB)
- Authentication error handling (401 errors fail fast)
- Comprehensive exception handling with custom error types
- Multi-provider support with provider-specific validation
- Image validation utilities

Supported Providers:
- Azure Document Intelligence (AzureOCRProvider)
- Google Cloud Vision (planned)
- Tesseract (planned)
"""

from devtoolbox.ocr.service import OCRService
from devtoolbox.ocr.provider import BaseOCRProvider, BaseOCRConfig
from devtoolbox.ocr.azure_provider import AzureOCRProvider, AzureOCRConfig
from devtoolbox.ocr.utils import (
    validate_image_for_ocr,
    list_supported_providers,
    get_provider_requirements,
    OCRRequirements
)

__all__ = [
    'OCRService',
    'BaseOCRProvider',
    'BaseOCRConfig',
    'AzureOCRProvider',
    'AzureOCRConfig',
    'validate_image_for_ocr',
    'list_supported_providers',
    'get_provider_requirements',
    'OCRRequirements'
]