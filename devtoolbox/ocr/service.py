"""OCR service implementation.

This module provides a high-level service interface for OCR operations,
managing different OCR providers and their configurations.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path

from devtoolbox.ocr.provider import BaseOCRProvider
from devtoolbox.ocr.azure_provider import AzureOCRProvider, AzureOCRConfig

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR operations"""

    def __init__(self, provider: Optional[BaseOCRProvider] = None):
        """
        Initialize OCR service

        Args:
            provider: OCR provider to use. If None, will use Azure provider
                    with default configuration
        """
        if provider is None:
            config = AzureOCRConfig()
            self.provider = AzureOCRProvider(config)
        else:
            self.provider = provider

    async def recognize(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> List[str]:
        """
        Recognize text from a file (image, PDF, etc.)

        Args:
            file_path: Path to the file
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines
        """
        return await self.provider.recognize(file_path, **kwargs)