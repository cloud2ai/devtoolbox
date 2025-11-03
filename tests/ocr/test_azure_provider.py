"""Unit tests for Azure OCR provider.

This module contains tests for the Azure Document Intelligence provider
implementation, focusing on testing provider functionality using mocks.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from devtoolbox.ocr.azure_provider import (
    AzureOCRConfig,
    AzureOCRProvider,
    AzureOCRException,
    AzureOCRProcessingException
)


class MockAnalyzeResult:
    """Mock Azure AnalyzeResult for testing."""

    def __init__(self, pages=None):
        self.pages = pages if pages is not None else [MockPage()]


class MockPage:
    """Mock page object."""

    def __init__(self, lines=None):
        self.page_number = 1
        self.lines = lines or [
            MockLine("Line 1"),
            MockLine("Line 2"),
            MockLine("Line 3")
        ]


class MockLine:
    """Mock line object."""

    def __init__(self, content):
        self.content = content
        self.confidence = 0.99


@pytest.fixture
def mock_document_intelligence_client():
    """Mock Azure Document Intelligence client."""
    with patch('devtoolbox.ocr.azure_provider.DocumentIntelligenceClient') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def azure_provider(mock_document_intelligence_client):
    """Create an AzureOCRProvider instance with mocked dependencies."""
    config = AzureOCRConfig(
        api_key="test-key",
        endpoint="https://test.endpoint.com"
    )
    return AzureOCRProvider(config)


class TestAzureOCRConfig:
    """Tests for AzureOCRConfig class."""

    def test_create_config_with_env_vars(self, monkeypatch):
        """Test creating config from environment variables."""
        monkeypatch.setenv('AZURE_DOCUMENT_INTELLIGENCE_KEY', 'env-key')
        monkeypatch.setenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT', 'https://env.endpoint.com')

        config = AzureOCRConfig()

        assert config.api_key == 'env-key'
        assert config.endpoint == 'https://env.endpoint.com'

    def test_config_validation_missing_api_key(self):
        """Test validation error when API key is missing."""
        with pytest.raises(ValueError, match="API key is required"):
            AzureOCRConfig(api_key="", endpoint="https://test.com")

    def test_config_validation_missing_endpoint(self):
        """Test validation error when endpoint is missing."""
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureOCRConfig(api_key="test-key", endpoint="")


class TestAzureOCRProvider:
    """Tests for AzureOCRProvider class."""

    def test_recognize_image_raw_returns_azure_result(
        self, azure_provider, mock_document_intelligence_client
    ):
        """Test recognize_image_raw returns raw Azure AnalyzeResult."""
        # Mock file operation
        mock_file_data = b"fake image data"

        # Mock the Azure client response
        mock_poller = MagicMock()
        mock_result = MockAnalyzeResult()
        mock_poller.result.return_value = mock_result
        mock_document_intelligence_client.begin_analyze_document.return_value = mock_poller

        # Mock file open
        with patch("builtins.open", unittest.mock.mock_open(read_data=mock_file_data)):
            # Call the method with return_raw=True
            result = azure_provider.recognize_image_raw("test.jpg", return_raw=True)

        # Verify Azure client was called
        mock_document_intelligence_client.begin_analyze_document.assert_called_once_with(
            "prebuilt-read",
            body=unittest.mock.ANY
        )

        # Verify raw Azure AnalyzeResult is returned (transparent pass-through)
        assert isinstance(result, MockAnalyzeResult)
        assert hasattr(result, 'pages')
        assert len(result.pages) == 1

    def test_recognize_document_raw_returns_azure_result(
        self, azure_provider, mock_document_intelligence_client
    ):
        """Test recognize_document_raw returns raw Azure AnalyzeResult."""
        # Mock file operation
        mock_file_data = b"fake document data"

        # Mock the Azure client response
        mock_poller = MagicMock()
        mock_result = MockAnalyzeResult()
        mock_poller.result.return_value = mock_result
        mock_document_intelligence_client.begin_analyze_document.return_value = mock_poller

        # Mock file open
        with patch("builtins.open", unittest.mock.mock_open(read_data=mock_file_data)):
            # Call the method with return_raw=True
            result = azure_provider.recognize_document_raw("test.pdf", return_raw=True)

        # Verify Azure client was called
        mock_document_intelligence_client.begin_analyze_document.assert_called_once()

        # Verify raw Azure AnalyzeResult is returned
        assert isinstance(result, MockAnalyzeResult)
        assert hasattr(result, 'pages')

    def test_convert_to_text_extracts_lines(self, azure_provider):
        """Test _convert_to_text extracts text lines from result."""
        mock_result = MockAnalyzeResult(pages=[
            MockPage(lines=[MockLine("Page 1 Line 1"), MockLine("Page 1 Line 2")]),
            MockPage(lines=[MockLine("Page 2 Line 1")])
        ])

        lines = azure_provider._convert_to_text(mock_result)

        assert len(lines) == 3
        assert lines == ["Page 1 Line 1", "Page 1 Line 2", "Page 2 Line 1"]

    def test_convert_to_text_empty_result(self, azure_provider):
        """Test _convert_to_text with empty result."""
        mock_result = MockAnalyzeResult(pages=[])

        lines = azure_provider._convert_to_text(mock_result)

        assert isinstance(lines, list)
        assert len(lines) == 0

    def test_convert_to_text_none_result(self, azure_provider):
        """Test _convert_to_text with None result."""
        lines = azure_provider._convert_to_text(None)

        assert lines == []

    def test_provider_passes_kwargs_to_azure(
        self, azure_provider, mock_document_intelligence_client
    ):
        """Test provider passes additional kwargs to Azure client."""
        # Mock file operation
        mock_file_data = b"fake image data"

        mock_poller = MagicMock()
        mock_poller.result.return_value = MockAnalyzeResult()
        mock_document_intelligence_client.begin_analyze_document.return_value = mock_poller

        # Mock file open
        with patch("builtins.open", unittest.mock.mock_open(read_data=mock_file_data)):
            # Call with additional kwargs
            azure_provider.recognize_image_raw(
                "test.jpg",
                custom_param="test_value"
            )

        # Verify kwargs were passed
        call_kwargs = mock_document_intelligence_client.begin_analyze_document.call_args[1]
        assert 'custom_param' in call_kwargs
        assert call_kwargs['custom_param'] == "test_value"


# Add this import at the top if not already present
import unittest.mock
