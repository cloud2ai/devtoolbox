"""OCR command line tool.

This module provides command-line interface for OCR operations.
"""

import logging
from pathlib import Path
from typing import Optional

import typer

from devtoolbox.cli.utils import setup_logging
from devtoolbox.ocr import OCRService
from devtoolbox.ocr.azure_provider import AzureOCRConfig
from devtoolbox.ocr.utils import (
    get_provider_requirements,
    list_supported_providers
)

# Configure logging
logger = logging.getLogger("devtoolbox.ocr")
app = typer.Typer(help="OCR commands")


@app.callback()
def callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
):
    """
    OCR command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.ocr")


def get_service(
    provider: str = "azure",
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None
) -> OCRService:
    """Get OCR service with configuration.

    Args:
        provider: Provider name (azure, google, tesseract)
        api_key: Azure Document Intelligence API key
        endpoint: Azure Document Intelligence endpoint

    Returns:
        OCRService instance
    """
    if provider == "azure":
        if api_key or endpoint:
            config = AzureOCRConfig(
                api_key=api_key,
                endpoint=endpoint
            )
        else:
            config = AzureOCRConfig()
        return OCRService(config)
    else:
        raise NotImplementedError(
            f"Provider '{provider}' is not yet implemented. "
            f"Only 'azure' is currently supported."
        )


@app.command("recognize")
def recognize(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the file to recognize",
        exists=True,
    ),
    provider: str = typer.Option(
        "azure",
        "--provider",
        "-p",
        help="OCR provider to use (azure, google, tesseract)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="AZURE_DOCUMENT_INTELLIGENCE_KEY",
        help="Azure Document Intelligence API key",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        envvar="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        help="Azure Document Intelligence endpoint",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path (default: stdout)",
    ),
    skip_invalid: bool = typer.Option(
        True,
        "--skip-invalid",
        help="Skip invalid files instead of raising errors",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force processing even if file validation fails",
    ),
):
    """
    Recognize text from a file (image, PDF, etc.)

    Features:
    - Supports various image formats (JPEG, PNG, BMP, TIFF, etc.)
    - Automatic retry for temporary errors (429, 500, 502, 503, 504)
    - Detailed error reporting for debugging
    - File size validation (max 500MB)
    - Comprehensive logging for monitoring
    - Provider-specific file validation
    """
    try:
        # Override skip_invalid if force is True
        if force:
            skip_invalid = False

        # Initialize service with selected provider
        service = get_service(provider, api_key, endpoint)

        # Use service's recognize method which automatically determines file type
        lines = service.recognize(file_path, skip_invalid=skip_invalid)

        # Output results
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(f"{line}\n")
        else:
            for line in lines:
                typer.echo(line)

    except Exception as e:
        logger.error(
            "Failed to recognize text: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to recognize text: {str(e)}")
        raise typer.Exit(1)


@app.command("list-providers")
def list_providers():
    """List all supported OCR providers."""
    try:
        providers = list_supported_providers()
        typer.echo("Supported OCR providers:")
        for provider in providers:
            typer.echo(f"  - {provider}")
    except Exception as e:
        typer.echo(f"Failed to list providers: {str(e)}")
        raise typer.Exit(1)


@app.command("provider-info")
def provider_info(
    provider: str = typer.Argument(
        ...,
        help="Provider name to get information for",
    ),
):
    """Get detailed information about a specific provider."""
    try:
        requirements = get_provider_requirements(provider)

        typer.echo(f"Provider: {provider}")
        typer.echo(
            f"  Minimum dimensions: "
            f"{requirements.min_width}x{requirements.min_height}"
        )
        typer.echo(
            f"  Maximum dimensions: "
            f"{requirements.max_width}x{requirements.max_height}"
        )
        typer.echo(
            f"  Maximum file size: "
            f"{requirements.max_file_size // (1024*1024)}MB"
        )
        typer.echo(
            f"  Supported formats: "
            f"{', '.join(requirements.supported_formats)}"
        )

    except Exception as e:
        typer.echo(f"Failed to get provider info: {str(e)}")
        raise typer.Exit(1)