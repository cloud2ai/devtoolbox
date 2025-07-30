#!/usr/bin/env python3
"""Test script to simulate 429 errors and test retry mechanism."""

import logging
import time
import tempfile
import os
from unittest.mock import Mock, patch
from azure.core.exceptions import HttpResponseError
from devtoolbox.ocr.azure_provider import AzureOCRProvider, AzureOCRConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_response(status_code):
    """Create a mock response with specific status code."""
    mock_response = Mock()
    mock_response.status_code = status_code
    return mock_response


def create_test_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b'fake image data')
        return f.name


def test_429_retry():
    """Test 429 retry mechanism."""
    logger.info("Testing 429 retry mechanism...")

    # Create test file
    test_file = create_test_file()

    try:
        # Create config
        config = AzureOCRConfig(
            api_key="test_key",
            endpoint="https://test.cognitiveservices.azure.com/"
        )

        # Create provider
        provider = AzureOCRProvider(config)

        # Mock the client to simulate 429 errors
        with patch.object(provider.client, 'begin_analyze_document') as mock_analyze:
            # First call returns 429, second call succeeds
            mock_poller = Mock()
            mock_poller.result.side_effect = [
                HttpResponseError(
                    response=create_mock_response(429),
                    message="Too Many Requests"
                ),
                Mock()  # Success response
            ]
            mock_analyze.return_value = mock_poller

            # Test the retry mechanism
            try:
                result = provider._recognize_raw(test_file)
                logger.info("✅ Retry mechanism worked - succeeded after 429")
            except Exception as e:
                logger.error(f"❌ Retry mechanism failed: {e}")

    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

    logger.info("Test completed.")


def test_multiple_429_errors():
    """Test multiple 429 errors."""
    logger.info("Testing multiple 429 errors...")

    # Create test file
    test_file = create_test_file()

    try:
        # Create config
        config = AzureOCRConfig(
            api_key="test_key",
            endpoint="https://test.cognitiveservices.azure.com/"
        )

        # Create provider
        provider = AzureOCRProvider(config)

        # Mock the client to simulate multiple 429 errors
        with patch.object(provider.client, 'begin_analyze_document') as mock_analyze:
            # Multiple 429 errors, then success
            mock_poller = Mock()
            mock_poller.result.side_effect = [
                HttpResponseError(
                    response=create_mock_response(429),
                    message="Too Many Requests"
                ),
                HttpResponseError(
                    response=create_mock_response(429),
                    message="Too Many Requests"
                ),
                HttpResponseError(
                    response=create_mock_response(429),
                    message="Too Many Requests"
                ),
                Mock()  # Success response
            ]
            mock_analyze.return_value = mock_poller

            # Test the retry mechanism
            try:
                result = provider._recognize_raw(test_file)
                logger.info("✅ Multiple 429 retry mechanism worked")
            except Exception as e:
                logger.error(f"❌ Multiple 429 retry mechanism failed: {e}")

    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

    logger.info("Test completed.")


def test_500_retry():
    """Test 500 retry mechanism."""
    logger.info("Testing 500 retry mechanism...")

    # Create test file
    test_file = create_test_file()

    try:
        # Create config
        config = AzureOCRConfig(
            api_key="test_key",
            endpoint="https://test.cognitiveservices.azure.com/"
        )

        # Create provider
        provider = AzureOCRProvider(config)

        # Mock the client to simulate 500 errors
        with patch.object(provider.client, 'begin_analyze_document') as mock_analyze:
            # First call returns 500, second call succeeds
            mock_poller = Mock()
            mock_poller.result.side_effect = [
                HttpResponseError(
                    response=create_mock_response(500),
                    message="Internal Server Error"
                ),
                Mock()  # Success response
            ]
            mock_analyze.return_value = mock_poller

            # Test the retry mechanism
            try:
                result = provider._recognize_raw(test_file)
                logger.info("✅ 500 retry mechanism worked")
            except Exception as e:
                logger.error(f"❌ 500 retry mechanism failed: {e}")

    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

    logger.info("Test completed.")


if __name__ == "__main__":
    logger.info("Starting retry mechanism tests...")

    test_429_retry()
    test_multiple_429_errors()
    test_500_retry()

    logger.info("All tests completed.")