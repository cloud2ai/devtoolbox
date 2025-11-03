"""OCR service implementation.

This module provides the service layer for OCR operations.
"""

import logging
from pathlib import Path
from typing import List, Union, Any

from devtoolbox.ocr.provider import BaseOCRConfig

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR operations with a single provider"""

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

    def _is_image_file(self, file_path: Union[str, Path]) -> bool:
        """
        Determine if a file is an image based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is an image, False otherwise
        """
        file_path = Path(file_path)
        image_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.gif', '.webp', '.svg', '.ico'
        }
        return file_path.suffix.lower() in image_extensions

    def _is_document_file(self, file_path: Union[str, Path]) -> bool:
        """
        Determine if a file is a document based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a document, False otherwise
        """
        file_path = Path(file_path)
        document_extensions = {
            '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'
        }
        return file_path.suffix.lower() in document_extensions

    def recognize(
        self,
        file_path: Union[str, Path],
        skip_invalid: bool = False,
        raw_response: bool = False,
        **kwargs
    ) -> Union[List[str], Any]:
        """
        Recognize text from a file.

        This method automatically detects whether the file is an image or
        document and calls the appropriate recognition method.

        Args:
            file_path: Path to the file
            skip_invalid: Whether to skip invalid files instead of raising errors
            raw_response: If True, returns raw provider response.
                         If False (default), returns list of strings only.
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of text lines (if raw_response=False)
            Any: Raw provider response (if raw_response=True)
                 For Azure: contains pages with lines, metadata, etc.

        Raises:
            ValueError: If file type is not supported or validation fails

        Example:
            # Simple usage - returns list of strings (default)
            lines = service.recognize(image_path)
            for line in lines:
                print(line)

            # With raw response - returns provider's original response
            result = service.recognize(image_path, raw_response=True)
            # Access provider-specific structure (e.g., Azure Document Intelligence)
            for page in result.pages:
                for line in page.lines:
                    print(line.content)
        """
        file_path = Path(file_path)

        if self._is_image_file(file_path):
            if not raw_response:
                is_compliant, reason = (
                    self.provider.validate_image_compliance(
                        file_path
                    )
                )
                if not is_compliant:
                    error_msg = (
                        f"Image validation failed: {reason}"
                    )
                    if skip_invalid:
                        logger.warning(error_msg)
                        return []
                    else:
                        raise ValueError(error_msg)

            return self.provider.recognize_image_raw(
                file_path,
                return_raw=raw_response,
                **kwargs
            )

        elif self._is_document_file(file_path):
            if not raw_response:
                is_compliant, reason = (
                    self.provider.validate_document_compliance(
                        file_path
                    )
                )
                if not is_compliant:
                    error_msg = (
                        f"Document validation failed: {reason}"
                    )
                    if skip_invalid:
                        logger.warning(error_msg)
                        return []
                    else:
                        raise ValueError(error_msg)

            return self.provider.recognize_document_raw(
                file_path,
                return_raw=raw_response,
                **kwargs
            )

        else:
            error_msg = (
                f"Unsupported file type: {file_path.suffix}"
            )
            if skip_invalid:
                logger.warning(error_msg)
                return [] if not raw_response else None
            else:
                raise ValueError(error_msg)