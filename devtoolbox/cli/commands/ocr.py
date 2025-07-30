"""OCR command line tool.

This module provides command-line interface for OCR operations.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional

import typer

from devtoolbox.ocr import OCRService
from devtoolbox.ocr.azure_provider import AzureOCRConfig, AzureOCRProvider
from devtoolbox.cli.utils import setup_logging

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
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None
) -> OCRService:
    """Get OCR service with configuration.

    Args:
        api_key: Azure Document Intelligence API key
        endpoint: Azure Document Intelligence endpoint

    Returns:
        OCRService instance
    """
    if api_key or endpoint:
        config = AzureOCRConfig(
            api_key=api_key,
            endpoint=endpoint
        )
        return OCRService(config)
    return OCRService(AzureOCRConfig())


@app.command("recognize")
def recognize(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the file to recognize",
        exists=True,
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
):
    """
    Recognize text from a file (image, PDF, etc.)

    Features:
    - Supports various image formats (JPEG, PNG, BMP, TIFF, etc.)
    - Automatic retry for temporary errors (429, 500, 502, 503, 504)
    - Detailed error reporting for debugging
    - File size validation (max 500MB)
    - Comprehensive logging for monitoring
    """
    try:
        # Initialize service
        service = get_service(api_key, endpoint)

        # Run recognition
        lines = service.recognize(file_path)

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