"""OCR module.

This module provides OCR (Optical Character Recognition) functionality.

Features:
- Azure Document Intelligence integration
- Automatic retry for temporary errors (429, 500, 502, 503, 504)
- Detailed error reporting and logging
- File size validation (max 500MB)
- Authentication error handling (401 errors fail fast)
- Comprehensive exception handling with custom error types

Supported Providers:
- Azure Document Intelligence (AzureOCRProvider)
"""

from devtoolbox.ocr.service import OCRService
from devtoolbox.ocr.provider import BaseOCRProvider, BaseOCRConfig
from devtoolbox.ocr.azure_provider import AzureOCRProvider, AzureOCRConfig

__all__ = [
    'OCRService',
    'BaseOCRProvider',
    'BaseOCRConfig',
    'AzureOCRProvider',
    'AzureOCRConfig'
]