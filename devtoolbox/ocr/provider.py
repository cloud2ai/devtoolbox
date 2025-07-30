"""Base OCR provider implementation.

This module provides the base classes for OCR providers.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    @abstractmethod
    def validate_image_compliance(
        self,
        image_path: Union[str, Path]
    ) -> Tuple[bool, str]:
        """
        Validate if image meets provider-specific requirements.

        This method must be implemented by each provider to implement
        their specific image validation requirements (file size, dimensions,
        format, etc.).

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_compliant, reason)
        """
        pass

    @abstractmethod
    def validate_document_compliance(
        self,
        document_path: Union[str, Path]
    ) -> Tuple[bool, str]:
        """
        Validate if document meets provider-specific requirements.

        This method must be implemented by each provider to implement
        their specific document validation requirements (file size, format,
        etc.).

        Args:
            document_path: Path to the document file

        Returns:
            Tuple of (is_compliant, reason)
        """
        pass

    @abstractmethod
    def recognize_image_raw(
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
        pass

    @abstractmethod
    def recognize_document_raw(
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
        pass
