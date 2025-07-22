"""Base OCR provider implementation.

This module provides the base classes for OCR providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class BaseOCRConfig:
    """Base configuration for OCR providers"""

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        pass

    def _validate_config(self):
        """Validate configuration."""
        pass


class BaseOCRProvider(ABC):
    """Base class for OCR providers"""

    def __init__(self, config: BaseOCRConfig):
        """Initialize OCR provider with configuration."""
        if not isinstance(config, BaseOCRConfig):
            raise ValueError("Config must be an instance of BaseOCRConfig")
        self.config = config

    def _recognize_raw(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Perform raw OCR recognition on a file

        Args:
            file_path: Path to the file
            **kwargs: Additional provider-specific parameters

        Returns:
            Raw result from the OCR provider
        """
        pass

    @abstractmethod
    def _convert_to_text(self, raw_result: Any) -> List[str]:
        """
        Convert provider-specific result to list of text lines

        Args:
            raw_result: Raw result from the OCR provider

        Returns:
            List of text lines
        """
        pass

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
        raw_result = self._recognize_raw(file_path, **kwargs)
        return self._convert_to_text(raw_result)

    def analyze_image(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> List[str]:
        """
        Analyze an image and extract text

        Args:
            image_path: Path to the image file
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines
        """
        raw_result = self._recognize_raw(image_path, **kwargs)
        return self._convert_to_text(raw_result)

    def analyze_document(
        self,
        document_path: Union[str, Path],
        **kwargs
    ) -> List[str]:
        """
        Analyze a document and extract text

        Args:
            document_path: Path to the document file
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines
        """
        raw_result = self._recognize_raw(document_path, **kwargs)
        return self._convert_to_text(raw_result)