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
from azure.core.exceptions import HttpResponseError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from devtoolbox.ocr.provider import BaseOCRConfig, BaseOCRProvider

logger = logging.getLogger(__name__)


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

        self.client = DocumentIntelligenceClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key)
        )

    @retry(
        retry=retry_if_exception_type(HttpResponseError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _recognize_raw(
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
        """
        with open(file_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                "prebuilt-read",
                body=f,
                **kwargs
            )
            return poller.result()

    def _convert_to_text(self, raw_result: Any) -> List[str]:
        """
        Convert Azure Document Intelligence result to list of text lines

        Args:
            raw_result: Raw result from Azure Document Intelligence

        Returns:
            List of text lines
        """
        lines = []

        # Process each page
        for page in raw_result.pages:
            # Process each line in the page
            for line in page.lines:
                lines.append(line.content)

        return lines