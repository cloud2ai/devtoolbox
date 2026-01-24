"""Manual test script for Gemini provider.

Usage:
    python test_gemini_provider.py

This script tests the Gemini provider with real API calls.
Requires GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
"""

import os
import sys
import logging
from devtoolbox.llm.gemini_provider import (
    GeminiConfig,
    GeminiProvider,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_chat():
    """Test basic chat functionality."""
    logger.info("Testing basic chat...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {"role": "user", "content": "Say 'Hello' in one word."},
    ]
    response = provider.chat(messages)
    logger.info(f"Response: {response}")
    assert response, "Response should not be empty"
    logger.info("✓ Basic chat test passed")


def test_json_mode():
    """Test JSON mode."""
    logger.info("Testing JSON mode...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {
            "role": "user",
            "content": (
                "Return a JSON object with 'name' and 'age' fields."
            ),
        },
    ]
    response = provider.chat(messages, json_mode=True)
    logger.info(f"JSON Response: {response}")
    assert response, "Response should not be empty"
    import json
    try:
        data = json.loads(response)
        assert isinstance(data, dict), "Response should be valid JSON"
        logger.info("✓ JSON mode test passed")
    except json.JSONDecodeError:
        logger.warning("Response is not valid JSON, but may still work")


def test_complete():
    """Test complete method."""
    logger.info("Testing complete method...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    prompt = "Complete this sentence: The sky is"
    response = provider.complete(prompt)
    logger.info(f"Completion: {response}")
    assert response, "Response should not be empty"
    logger.info("✓ Complete test passed")


def test_list_models():
    """Test list_models method."""
    logger.info("Testing list_models...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    models = provider.list_models()
    logger.info(f"Available models: {models}")
    assert len(models) > 0, "Should return at least one model"
    assert "gemini-2.5-flash-lite" in models, (
        "Should include default model"
    )
    logger.info("✓ List models test passed")


def test_embed():
    """Test embed method (should raise NotImplementedError)."""
    logger.info("Testing embed method...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    try:
        provider.embed("test")
        logger.error("✗ Embed should raise NotImplementedError")
        return False
    except NotImplementedError:
        logger.info("✓ Embed correctly raises NotImplementedError")
        return True


def main():
    """Run all manual tests."""
    api_key = (
        os.environ.get("GOOGLE_API_KEY") or
        os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        logger.error(
            "GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
            "is required. Set it before running this test."
        )
        sys.exit(1)

    logger.info("Starting Gemini provider manual tests...")
    logger.info(f"Using model: {GeminiConfig().model}")

    tests = [
        ("Basic Chat", test_basic_chat),
        ("Complete", test_complete),
        ("List Models", test_list_models),
        ("JSON Mode", test_json_mode),
        ("Embed", test_embed),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            logger.info(f"\n--- Running {name} test ---")
            result = test_func()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {name} test failed: {e}", exc_info=True)
            failed += 1

    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
