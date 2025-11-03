"""Unit tests for OCR service layer.

This module tests the OCR service layer logic:
- Input/output behavior
- raw_response parameter functionality
- File type detection
- Error handling

External dependencies (Azure API) are mocked.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from devtoolbox.ocr.service import OCRService
from devtoolbox.ocr.provider import BaseOCRConfig, BaseOCRProvider


class MockAnalyzeResult:
    """Mock Azure AnalyzeResult object - simulates external API response."""

    def __init__(self, page_count=1):
        self.pages = [MockPage() for _ in range(page_count)]
        self.model_id = "prebuilt-read"


class MockPage:
    """Mock page object from Azure response."""

    def __init__(self):
        self.page_number = 1
        self.lines = [
            MockLine("Line 1 content"),
            MockLine("Line 2 content"),
            MockLine("Line 3 content")
        ]


class MockLine:
    """Mock line object from Azure response."""

    def __init__(self, content):
        self.content = content
        self.confidence = 0.99
        self.polygon = [(0, 0), (100, 0), (100, 20), (0, 20)]


class MockConfig(BaseOCRConfig):
    """Mock configuration."""

    def __init__(self):
        self.api_key = "test-key"
        self.endpoint = "https://test.endpoint.com"

    def _log_config_loading(self):
        pass

    def _validate_config(self):
        pass


class MockProvider(BaseOCRProvider):
    """Mock provider - simulates Azure provider behavior."""

    def recognize_image_raw(self, image_path, return_raw=False, **kwargs):
        """
        Simulates Azure API behavior.

        Args:
            return_raw: If True, returns MockAnalyzeResult.
                       If False, returns list of text lines.
        """
        result = MockAnalyzeResult(page_count=1)
        if return_raw:
            return result

        # Extract text from result (provider's job)
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
        return lines

    def recognize_document_raw(self, document_path, return_raw=False, **kwargs):
        """
        Simulates Azure API behavior.

        Args:
            return_raw: If True, returns MockAnalyzeResult.
                       If False, returns list of text lines.
        """
        result = MockAnalyzeResult(page_count=3)
        if return_raw:
            return result

        # Extract text from result (provider's job)
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
        return lines

    def validate_image_compliance(self, image_path):
        """Mock validation - always passes."""
        return (True, "")

    def validate_document_compliance(self, document_path):
        """Mock validation - always passes."""
        return (True, "")


@pytest.fixture
def mock_provider():
    """Create a mock provider instance."""
    return MockProvider(MockConfig())


@pytest.fixture
def ocr_service(mock_provider):
    """Create an OCRService instance with mocked provider."""
    service = OCRService(MockConfig())
    service.provider = mock_provider
    return service


class TestOCRService:
    """Tests for OCRService class - testing our code logic."""

    def test_recognize_image_default_returns_list(self, ocr_service):
        """
        Test: recognize() with raw_response=False (default) returns list of strings.

        Input: image file path
        Expected: list of text lines (strings)
        Mock: provider.recognize_image_raw (external dependency)
        """
        # Our code should call provider and process the result
        result = ocr_service.recognize("test.jpg")

        # Verify output type and content
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)
        assert len(result) == 3
        assert result == ["Line 1 content", "Line 2 content", "Line 3 content"]

    def test_recognize_image_raw_returns_provider_response(self, ocr_service):
        """
        Test: recognize() with raw_response=True returns raw provider response.

        Input: image file path, raw_response=True
        Expected: raw MockAnalyzeResult object (transparent pass-through)
        Mock: provider.recognize_image_raw
        """
        result = ocr_service.recognize("test.jpg", raw_response=True)

        # Verify output is raw provider response
        assert isinstance(result, MockAnalyzeResult)
        assert hasattr(result, 'pages')
        assert len(result.pages) == 1

        # Verify all Azure metadata is preserved
        assert hasattr(result, 'model_id')
        assert result.model_id == "prebuilt-read"

    def test_recognize_document_default_returns_list(self, ocr_service):
        """
        Test: recognize() with document file returns list of strings.

        Input: PDF file path
        Expected: list of text lines
        """
        result = ocr_service.recognize("test.pdf")

        # Verify our code correctly processes document
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)
        assert len(result) == 9  # 3 pages × 3 lines

    def test_recognize_document_raw_returns_provider_response(self, ocr_service):
        """
        Test: recognize() with raw_response=True for document.

        Input: PDF file, raw_response=True
        Expected: raw provider response with multiple pages
        """
        result = ocr_service.recognize("test.pdf", raw_response=True)

        # Verify transparent pass-through
        assert isinstance(result, MockAnalyzeResult)
        assert hasattr(result, 'pages')
        assert len(result.pages) == 3  # Multi-page document

    def test_recognize_explicit_raw_response_false(self, ocr_service):
        """
        Test: explicit raw_response=False behaves same as default.

        Input: image file, raw_response=False (explicit)
        Expected: list of strings
        """
        result = ocr_service.recognize("test.png", raw_response=False)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_recognize_passes_kwargs_to_provider(self, ocr_service):
        """
        Test: additional kwargs are passed through to provider.

        Input: file path + raw_response=True + custom kwargs
        Expected: kwargs reach provider method
        Logic: verify our code doesn't drop kwargs
        """
        # Spy on provider calls
        original_method = ocr_service.provider.recognize_image_raw
        ocr_service.provider.recognize_image_raw = MagicMock(
            side_effect=original_method
        )

        # Call with custom parameters
        ocr_service.recognize(
            "test.jpg",
            raw_response=True,
            custom_param="test_value",
            another_param=123
        )

        # Verify our code passes kwargs through
        ocr_service.provider.recognize_image_raw.assert_called_once()
        call_kwargs = ocr_service.provider.recognize_image_raw.call_args[1]
        assert 'custom_param' in call_kwargs
        assert call_kwargs['custom_param'] == "test_value"
        assert call_kwargs['another_param'] == 123

    def test_recognize_unsupported_file_type_raises_error(self, ocr_service):
        """
        Test: unsupported file type raises ValueError.

        Input: .xyz file (not image or document)
        Expected: ValueError
        Logic: test our file type detection code
        """
        with pytest.raises(ValueError, match="Unsupported file type"):
            ocr_service.recognize("test.xyz")

    def test_recognize_unsupported_file_skip_invalid(self, ocr_service):
        """
        Test: skip_invalid=True returns empty list for unsupported file.

        Input: .xyz file, skip_invalid=True
        Expected: empty list []
        Logic: test error handling branch
        """
        result = ocr_service.recognize("test.xyz", skip_invalid=True)

        assert result == []
        assert isinstance(result, list)

    def test_recognize_unsupported_file_raw_skip_invalid(self, ocr_service):
        """
        Test: skip_invalid=True with raw_response=True returns None.

        Input: .xyz file, raw_response=True, skip_invalid=True
        Expected: None
        Logic: test error handling with raw mode
        """
        result = ocr_service.recognize(
            "test.xyz",
            raw_response=True,
            skip_invalid=True
        )

        assert result is None

    def test_file_type_detection_image_formats(self, ocr_service):
        """
        Test: our code correctly detects various image formats.

        Input: files with different image extensions
        Expected: all recognized as images
        Logic: test _is_image_file() method logic
        """
        # Spy on provider to verify correct method is called
        original_method = ocr_service.provider.recognize_image_raw
        ocr_service.provider.recognize_image_raw = MagicMock(
            side_effect=original_method
        )

        # Test various image formats
        image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        for ext in image_formats:
            # Should call recognize_image_raw for images
            result = ocr_service.recognize(f"test{ext}", raw_response=True)
            assert isinstance(result, MockAnalyzeResult)
            assert len(result.pages) == 1  # Single page for images

    def test_file_type_detection_document_format(self, ocr_service):
        """
        Test: our code correctly detects document format.

        Input: .pdf file
        Expected: recognized as document
        Logic: test _is_document_file() method
        """
        # Spy on provider
        original_method = ocr_service.provider.recognize_document_raw
        ocr_service.provider.recognize_document_raw = MagicMock(
            side_effect=original_method
        )

        # Should call recognize_document_raw for documents
        result = ocr_service.recognize("test.pdf", raw_response=True)
        assert isinstance(result, MockAnalyzeResult)
        assert len(result.pages) == 3  # Multi-page document

    def test_raw_mode_preserves_all_provider_metadata(self, ocr_service):
        """
        Test: raw_response=True preserves all provider metadata.

        Input: image file, raw_response=True
        Expected: all Azure metadata fields accessible
        Logic: verify transparent pass-through doesn't lose data
        """
        result = ocr_service.recognize("test.jpg", raw_response=True)

        # Verify top-level metadata
        assert hasattr(result, 'pages')
        assert hasattr(result, 'model_id')
        assert result.model_id == "prebuilt-read"

        # Verify page-level metadata
        page = result.pages[0]
        assert hasattr(page, 'page_number')
        assert hasattr(page, 'lines')
        assert page.page_number == 1

        # Verify line-level metadata (Azure-specific)
        line = page.lines[0]
        assert hasattr(line, 'content')
        assert hasattr(line, 'confidence')
        assert hasattr(line, 'polygon')
        assert line.content == "Line 1 content"
        assert line.confidence == 0.99
        assert len(line.polygon) == 4  # Bounding box coordinates

    def test_default_mode_extracts_text_only(self, ocr_service):
        """
        Test: default mode (raw_response=False) extracts only text content.

        Input: image file
        Expected: simple list of strings (no metadata)
        Logic: test our code's text extraction from provider response
        """
        result = ocr_service.recognize("test.jpg")

        # Verify simplified output
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)
        assert result == ["Line 1 content", "Line 2 content", "Line 3 content"]

        # Verify metadata is NOT present
        assert not hasattr(result, 'pages')
        assert not hasattr(result, 'model_id')

    def test_raw_mode_calls_provider_directly(self, ocr_service):
        """
        Test: raw_response=True calls provider.recognize_xxx_raw() directly.

        Logic: verify control flow - raw mode should skip processing
        """
        # Spy on provider
        spy = MagicMock(side_effect=ocr_service.provider.recognize_image_raw)
        ocr_service.provider.recognize_image_raw = spy

        result = ocr_service.recognize("test.jpg", raw_response=True)

        # Verify direct call to provider
        spy.assert_called_once()
        call_args = spy.call_args
        assert str(call_args[0][0]) == "test.jpg"

        # Verify result is untouched
        assert isinstance(result, MockAnalyzeResult)

    def test_default_mode_processes_provider_response(self, ocr_service):
        """
        Test: default mode processes provider response through internal methods.

        Logic: verify control flow - default should call _recognize_image
        """
        result = ocr_service.recognize("test.jpg", raw_response=False)

        # Verify text was extracted from provider response
        assert isinstance(result, list)
        assert len(result) == 3

        # Verify it's processed (strings only, no objects)
        for line in result:
            assert isinstance(line, str)
            assert not hasattr(line, 'confidence')
