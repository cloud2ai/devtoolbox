"""OCR service implementation.

This module provides the service layer for OCR operations.
"""

import logging
from pathlib import Path
from typing import List, Union

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

    def _recognize_image(
        self,
        image_path: Union[str, Path],
        skip_invalid: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Recognize text from an image file using the configured provider.

        Args:
            image_path: Path to the image file
            skip_invalid: Whether to skip invalid files instead of raising errors
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines
        """
        image_path = Path(image_path)

        # Validate that the file is actually an image
        if not self._is_image_file(image_path):
            error_msg = f"File {image_path} is not recognized as an image"
            if skip_invalid:
                logger.warning(error_msg)
                return []
            else:
                raise ValueError(error_msg)

        # Validate image compliance using provider
        is_compliant, reason = self.provider.validate_image_compliance(
            image_path
        )

        if not is_compliant:
            if skip_invalid:
                logger.warning(
                    f"Image does not meet {self.provider.__class__.__name__} "
                    f"requirements: {reason}"
                )
                return []
            else:
                raise ValueError(f"Image validation failed: {reason}")

        # Perform recognition
        return self.provider.recognize_image_raw(image_path, **kwargs)

    def _recognize_document(
        self,
        document_path: Union[str, Path],
        skip_invalid: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Recognize text from a document file using the configured provider.

        Args:
            document_path: Path to the document file
            skip_invalid: Whether to skip invalid files instead of raising errors
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines
        """
        document_path = Path(document_path)

        # Validate that the file is actually a document
        if not self._is_document_file(document_path):
            error_msg = f"File {document_path} is not recognized as a document"
            if skip_invalid:
                logger.warning(error_msg)
                return []
            else:
                raise ValueError(error_msg)

        # Validate document compliance using provider
        is_compliant, reason = self.provider.validate_document_compliance(
            document_path
        )

        if not is_compliant:
            if skip_invalid:
                logger.warning(
                    f"Document does not meet {self.provider.__class__.__name__} "
                    f"requirements: {reason}"
                )
                return []
            else:
                raise ValueError(f"Document validation failed: {reason}")

        # Perform recognition
        return self.provider.recognize_document_raw(
            document_path, **kwargs)

    def recognize(
        self,
        file_path: Union[str, Path],
        skip_invalid: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Recognize text from a file, automatically determining the file type.

        This method automatically detects whether the file is an image or
        document and calls the appropriate recognition method.

        Args:
            file_path: Path to the file
            skip_invalid: Whether to skip invalid files instead of raising errors
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text lines

        Raises:
            ValueError: If file type is not supported or validation fails
        """
        file_path = Path(file_path)

        # Determine file type and call appropriate method
        if self._is_image_file(file_path):
            return self._recognize_image(
                file_path, skip_invalid=skip_invalid, **kwargs)
        elif self._is_document_file(file_path):
            return self._recognize_document(
                file_path, skip_invalid=skip_invalid, **kwargs)
        else:
            error_msg = f"Unsupported file type: {file_path.suffix}"
            if skip_invalid:
                logger.warning(error_msg)
                return []
            else:
                raise ValueError(error_msg)