"""OCR utilities for image validation and processing."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRRequirements:
    """OCR provider requirements configuration."""

    min_width: int
    min_height: int
    max_width: int
    max_height: int
    max_file_size: int
    supported_formats: list[str]
    provider_name: str


# Provider-specific requirements
AZURE_REQUIREMENTS = OCRRequirements(
    min_width=50,
    min_height=50,
    max_width=17000,
    max_height=17000,
    max_file_size=500 * 1024 * 1024,  # 500MB
    supported_formats=["JPEG", "PNG", "BMP", "TIFF"],
    provider_name="azure"
)

GOOGLE_REQUIREMENTS = OCRRequirements(
    min_width=100,
    min_height=100,
    max_width=10000,
    max_height=10000,
    max_file_size=10 * 1024 * 1024,  # 10MB
    supported_formats=["JPEG", "PNG", "GIF", "BMP", "WEBP"],
    provider_name="google"
)

TESSERACT_REQUIREMENTS = OCRRequirements(
    min_width=30,
    min_height=30,
    max_width=8000,
    max_height=8000,
    max_file_size=50 * 1024 * 1024,  # 50MB
    supported_formats=["JPEG", "PNG", "BMP", "TIFF", "GIF"],
    provider_name="tesseract"
)

# Provider requirements mapping
PROVIDER_REQUIREMENTS = {
    "azure": AZURE_REQUIREMENTS,
    "google": GOOGLE_REQUIREMENTS,
    "tesseract": TESSERACT_REQUIREMENTS,
}


def validate_document_for_ocr(
    document_path: Path,
    provider: str = "azure"
) -> Tuple[bool, str]:
    """Validate document against provider-specific requirements.

    Args:
        document_path: Path to the document file
        provider: OCR provider name (azure, google, tesseract)

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Get provider requirements
        requirements = PROVIDER_REQUIREMENTS.get(provider)
        if not requirements:
            logger.warning(
                f"Unknown provider '{provider}', skipping validation"
            )
            return True, "No validation for unknown provider"

        # Check file existence
        if not document_path.exists():
            return False, f"Document file not found: {document_path}"

        # Check file size
        file_size = document_path.stat().st_size
        if file_size > requirements.max_file_size:
            return False, (
                f"Document too large ({file_size} bytes). "
                f"{requirements.provider_name} limit is "
                f"{requirements.max_file_size // (1024*1024)}MB"
            )

        # Check file extension for supported formats
        # For documents, we check file extensions since we can't open all
        # formats
        file_extension = document_path.suffix.lower()

        # Document format mapping for different providers
        document_formats = {
            "azure": {
                '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                '.heif'
            },
            "google": {
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'
            },
            "tesseract": {
                '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                '.gif'
            }
        }

        supported_extensions = document_formats.get(provider, {'.pdf'})

        if file_extension not in supported_extensions:
            return False, (
                f"Unsupported document format: {file_extension}. "
                f"{requirements.provider_name} supports: "
                f"{', '.join(sorted(supported_extensions))}"
            )

        logger.info(
            f"Document validation passed for {requirements.provider_name}: "
            f"{file_extension}, {file_size} bytes"
        )
        return True, f"Document meets {requirements.provider_name} requirements"

    except Exception as e:
        logger.error(f"Failed to validate document {document_path}: {e}")
        return False, f"Failed to validate document: {e}"


def validate_image_for_ocr(
    image_path: Path,
    provider: str = "azure"
) -> Tuple[bool, str]:
    """Validate image against provider-specific requirements.

    Args:
        image_path: Path to the image file
        provider: OCR provider name (azure, google, tesseract)

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Get provider requirements
        requirements = PROVIDER_REQUIREMENTS.get(provider)
        if not requirements:
            logger.warning(
                f"Unknown provider '{provider}', skipping validation"
            )
            return True, "No validation for unknown provider"

        # Check file existence
        if not image_path.exists():
            return False, f"File not found: {image_path}"

        # Check file size
        file_size = image_path.stat().st_size
        if file_size > requirements.max_file_size:
            return False, (
                f"File too large ({file_size} bytes). "
                f"{requirements.provider_name} limit is "
                f"{requirements.max_file_size // (1024*1024)}MB"
            )

        # Open and analyze image
        with Image.open(image_path) as img:
            width, height = img.size

            # Check minimum dimensions
            if (width < requirements.min_width or
                    height < requirements.min_height):
                return False, (
                    f"Image too small ({width}x{height}). "
                    f"{requirements.provider_name} minimum is "
                    f"{requirements.min_width}x{requirements.min_height}"
                )

            # Check maximum dimensions
            if (width > requirements.max_width or
                    height > requirements.max_height):
                return False, (
                    f"Image too large ({width}x{height}). "
                    f"{requirements.provider_name} maximum is "
                    f"{requirements.max_width}x{requirements.max_height}"
                )

            # Check format support
            if img.format not in requirements.supported_formats:
                return False, (
                    f"Unsupported format: {img.format}. "
                    f"{requirements.provider_name} supports: "
                    f"{', '.join(requirements.supported_formats)}"
                )

            # Check aspect ratio (optional, for extreme cases)
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > 20:  # Very extreme aspect ratios
                return False, (
                    f"Image aspect ratio too extreme ({aspect_ratio:.2f}). "
                    f"This may cause recognition issues"
                )

            logger.info(
                f"Image validation passed for {requirements.provider_name}: "
                f"{width}x{height}, {img.format}, {file_size} bytes"
            )
            return True, f"Image meets {requirements.provider_name} requirements"

    except Exception as e:
        logger.error(f"Failed to validate image {image_path}: {e}")
        return False, f"Failed to validate image: {e}"


def get_provider_requirements(provider: str) -> OCRRequirements:
    """Get requirements for a specific provider.

    Args:
        provider: Provider name

    Returns:
        OCRRequirements for the provider

    Raises:
        ValueError: If provider is not supported
    """
    requirements = PROVIDER_REQUIREMENTS.get(provider)
    if not requirements:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {list(PROVIDER_REQUIREMENTS.keys())}"
        )
    return requirements


def list_supported_providers() -> list[str]:
    """Get list of supported OCR providers.

    Returns:
        List of supported provider names
    """
    return list(PROVIDER_REQUIREMENTS.keys())