"""OCR service implementation.

This module provides a high-level service interface for OCR operations,
managing different OCR providers and their configurations.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path

from devtoolbox.ocr.provider import BaseOCRProvider, BaseOCRConfig
from devtoolbox.ocr.azure_provider import AzureOCRProvider, AzureOCRConfig

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR operations"""

    def __init__(self, config: BaseOCRConfig):
        """
        Initialize OCR service

        Args:
            config: OCR provider config instance (e.g., AzureOCRConfig)
        """
        self.config = config
        self.provider = self._init_provider()

    def _init_provider(self):
        """
        Initialize provider based on config class.
        """
        config_class = self.config.__class__
        provider_name = config_class.__name__.replace(
            "Config", "Provider"
        )
        provider_module = config_class.__module__.replace(
            "config", "provider"
        )
        try:
            module = __import__(
                provider_module, fromlist=[provider_name]
            )
            provider_class = getattr(module, provider_name)
            return provider_class(self.config)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to initialize provider for config "
                f"{config_class}: {str(e)}"
            )

    def recognize(
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
        return self.provider.recognize(file_path, **kwargs)