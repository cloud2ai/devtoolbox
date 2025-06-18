"""OCR module.

This module provides OCR (Optical Character Recognition) functionality.
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