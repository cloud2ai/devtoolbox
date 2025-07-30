"""Azure Document Intelligence provider implementation.

This module provides Azure Document Intelligence service integration for OCR
operations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Union, Any
from pathlib import Path
import os

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.exceptions import HttpResponseError, ClientAuthenticationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    wait_random_exponential,
    retry_if_result
)

from devtoolbox.ocr.provider import BaseOCRConfig, BaseOCRProvider

logger = logging.getLogger(__name__)


def _should_retry_http_error(exception):
    """Check if HTTP error should be retried.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception should be retried
    """
    if not isinstance(exception, HttpResponseError):
        return False

    # Don't retry authentication errors (401)
    if exception.response.status_code == 401:
        return False

    # Retry on 429 (Too Many Requests), 500, 502, 503, 504
    return exception.response.status_code in [429, 500, 502, 503, 504]


class AzureOCRException(Exception):
    """Base exception for Azure OCR operations."""
    pass


class AzureOCRProcessingException(AzureOCRException):
    """Exception raised for processing errors."""
    pass


@dataclass
class AzureOCRConfig(BaseOCRConfig):
    """Azure Document Intelligence configuration settings."""

    # Azure specific settings
    api_key: str = field(
        default_factory=lambda: os.environ.get(
            'AZURE_DOCUMENT_INTELLIGENCE_KEY'
        )
    )
    endpoint: str = field(
        default_factory=lambda: os.environ.get(
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'
        )
    )

    def _log_config_loading(self):
        """Log configuration loading process."""
        if self.api_key:
            logger.info("Azure Document Intelligence API key loaded from "
                       "constructor")
        elif os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_KEY'):
            logger.info("Azure Document Intelligence API key loaded from "
                       "environment variable")
        else:
            logger.error("Azure Document Intelligence API key not found in "
                        "constructor or environment")

        if self.endpoint:
            logger.info("Azure Document Intelligence endpoint loaded from "
                       "constructor")
        elif os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'):
            logger.info("Azure Document Intelligence endpoint loaded from "
                       "environment variable")
        else:
            logger.error("Azure Document Intelligence endpoint not found in "
                        "constructor or environment")

    def _validate_config(self):
        """Validate Azure Document Intelligence configuration."""
        if not self.api_key:
            raise ValueError(
                "Azure Document Intelligence API key is required. Set it "
                "either in constructor or through "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variable"
            )
        if not self.endpoint:
            raise ValueError(
                "Azure Document Intelligence endpoint is required. Set it "
                "either in constructor or through "
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable"
            )


class AzureOCRProvider(BaseOCRProvider):
    """Azure Document Intelligence provider implementation."""

    def __init__(self, config: AzureOCRConfig):
        """Initialize Azure Document Intelligence provider."""
        if not isinstance(config, AzureOCRConfig):
            raise ValueError("Config must be an instance of AzureOCRConfig")

        super().__init__(config)

        try:
            self.client = DocumentIntelligenceClient(
                endpoint=config.endpoint,
                credential=AzureKeyCredential(config.api_key)
            )
            logger.info("Azure Document Intelligence client initialized "
                       "successfully")
        except ClientAuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise AzureOCRProcessingException(
                f"Failed to authenticate with Azure: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OCR provider: {e}")
            raise AzureOCRProcessingException(f"Initialization failed: {e}")

    @retry(
        retry=_should_retry_http_error,
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _recognize_raw(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Perform raw OCR recognition using Azure Document Intelligence

        Args:
            file_path: Path to the file
            **kwargs: Additional parameters for Azure Document Intelligence

        Returns:
            Raw result from Azure Document Intelligence

        Raises:
            AzureOCRProcessingException: If processing fails
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Basic file validation
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size (Azure has limits)
            file_size = file_path.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                raise AzureOCRProcessingException(
                    f"File too large ({file_size} bytes). "
                    "Azure limit is 500MB"
                )

            logger.info(f"Processing file: {file_path} ({file_size} bytes)")

            # Read and process file
            try:
                with open(file_path, "rb") as f:
                    poller = self.client.begin_analyze_document(
                        "prebuilt-read",
                        body=f,
                        **kwargs
                    )
            except (OSError, IOError) as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                raise AzureOCRProcessingException(
                    f"File read error: {e}"
                )

            logger.info("Document analysis started, waiting for completion")

            # Wait for completion with timeout handling
            try:
                result = poller.result()
                logger.info("Document analysis completed successfully")
                return result
            except HttpResponseError as e:
                # Handle specific error types
                status_code = e.response.status_code

                if status_code == 429:
                    logger.warning(
                        "Rate limit exceeded (429). Azure Document "
                        "Intelligence service is throttling requests. "
                        "Retrying with exponential backoff..."
                    )
                elif status_code in [500, 502, 503, 504]:
                    logger.warning(
                        f"Server error ({status_code}). "
                        "Retrying with exponential backoff..."
                    )
                elif status_code == 401:
                    logger.error(
                        "Authentication failed (401). Invalid API key or "
                        "endpoint. This error will not be retried."
                    )
                    # Don't retry authentication errors
                    raise AzureOCRProcessingException(
                        f"Authentication failed: {e}"
                    )
                else:
                    logger.error(f"Azure API error ({status_code}): {e}")

                # Re-raise HttpResponseError for retry decorator to handle
                raise
            except Exception as e:
                logger.error(f"Unexpected error during analysis: {e}")
                raise AzureOCRProcessingException(
                    f"Document analysis failed: {e}"
                )

        except FileNotFoundError:
            raise
        except AzureOCRProcessingException:
            raise
        except HttpResponseError as e:
            # Log detailed error information before re-raising
            status_code = e.response.status_code
            reason = getattr(e.response, 'reason', 'Unknown')
            error_message = str(e)

            # Extract additional error details if available
            error_details = ""
            if hasattr(e, 'error') and e.error:
                error_details = f" Error details: {e.error}"
            elif hasattr(e, 'model') and e.model:
                error_details = f" Model error: {e.model}"

            logger.error(
                f"Azure HTTP error: {status_code} {reason}. "
                f"Message: {error_message}{error_details}"
            )

            # Re-raise HttpResponseError for retry decorator to handle
            raise
        except Exception as e:
            logger.error(f"Unexpected error during document processing: {e}")
            raise AzureOCRProcessingException(
                f"Document processing failed: {e}"
            )

    def _convert_to_text(self, raw_result: Any) -> List[str]:
        """
        Convert Azure Document Intelligence result to list of text lines

        Args:
            raw_result: Raw result from Azure Document Intelligence

        Returns:
            List of text lines

        Raises:
            AzureOCRProcessingException: If conversion fails
        """
        try:
            if raw_result is None:
                logger.warning("Raw result is None, returning empty list")
                return []

            lines = []

            # Validate result structure
            if not hasattr(raw_result, 'pages'):
                logger.error("Result does not have 'pages' attribute")
                raise AzureOCRProcessingException(
                    "Invalid result structure: missing 'pages' attribute"
                )

            if not raw_result.pages:
                logger.warning("No pages found in document result")
                return lines

            # Process each page
            for page_num, page in enumerate(raw_result.pages):
                logger.debug(f"Processing page {page_num + 1}")

                # Validate page structure
                if not hasattr(page, 'lines'):
                    logger.warning(f"Page {page_num + 1} has no 'lines' "
                                 "attribute")
                    continue

                if not page.lines:
                    logger.warning(f"No lines found in page {page_num + 1}")
                    continue

                # Process each line in the page
                for line_num, line in enumerate(page.lines):
                    try:
                        if hasattr(line, 'content') and line.content:
                            lines.append(line.content)
                        else:
                            logger.debug(
                                f"Empty line {line_num + 1} in page "
                                f"{page_num + 1}"
                            )
                    except AttributeError as e:
                        logger.warning(
                            f"Invalid line structure in page {page_num + 1}, "
                            f"line {line_num + 1}: {e}"
                        )
                        continue

            logger.info(f"Successfully extracted {len(lines)} text lines")
            return lines

        except AttributeError as e:
            logger.error(f"Invalid result structure: {e}")
            raise AzureOCRProcessingException(
                f"Failed to parse Azure result structure: {e}"
            )
        except Exception as e:
            logger.error(f"Error converting result to text: {e}")
            raise AzureOCRProcessingException(
                f"Text conversion failed: {e}"
            )

    def recognize(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> List[str]:
        """
        Recognize text from document using Azure Document Intelligence

        Args:
            file_path: Path to the document file
            **kwargs: Additional parameters for Azure Document Intelligence

        Returns:
            List of recognized text lines

        Raises:
            AzureOCRProcessingException: If recognition fails
        """
        try:
            logger.info(f"Starting OCR recognition for: {file_path}")

            # Perform recognition
            raw_result = self._recognize_raw(file_path, **kwargs)

            # Convert to text
            text_lines = self._convert_to_text(raw_result)

            logger.info(f"OCR recognition completed successfully. "
                       f"Extracted {len(text_lines)} lines")

            return text_lines

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except AzureOCRProcessingException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OCR recognition: {e}")
            raise AzureOCRProcessingException(f"OCR recognition failed: {e}")