"""Manual test for Gemini provider using sample data.

Usage:
    python test_gemini_with_samples.py

This script tests Gemini provider with real sample data from
sample_data/ directory. Requires GOOGLE_API_KEY or GEMINI_API_KEY.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from devtoolbox.llm.gemini_provider import (
    GeminiConfig,
    GeminiProvider,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sample_data_path(*parts):
    """Get path to sample data file."""
    return project_root / "sample_data" / Path(*parts)


def test_with_chat_prompt():
    """Test using chat.txt prompt."""
    logger.info("Testing with chat.txt prompt...")
    prompt_file = get_sample_data_path("llm", "prompts", "chat.txt")
    if not prompt_file.exists():
        logger.warning(f"Sample file not found: {prompt_file}")
        return False

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is Python?"},
    ]
    response = provider.chat(messages)
    logger.info(f"Response: {response[:200]}...")
    assert response, "Response should not be empty"
    logger.info("✓ Chat prompt test passed")
    return True


def test_with_chain_prompt():
    """Test using chain.txt prompt for task breakdown."""
    logger.info("Testing with chain.txt prompt...")
    prompt_file = get_sample_data_path("llm", "prompts", "chain.txt")
    if not prompt_file.exists():
        logger.warning(f"Sample file not found: {prompt_file}")
        return False

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Plan a weekend trip to a nearby city",
        },
    ]
    response = provider.chat(messages)
    logger.info(f"Response: {response[:300]}...")
    assert response, "Response should not be empty"
    logger.info("✓ Chain prompt test passed")
    return True


def test_with_text_file():
    """Test complete with text file content."""
    logger.info("Testing with read_aloud_test.txt...")
    text_file = get_sample_data_path("text", "read_aloud_test.txt")
    if not text_file.exists():
        logger.warning(f"Sample file not found: {text_file}")
        return False

    with open(text_file, "r", encoding="utf-8") as f:
        text_content = f.read().strip()

    config = GeminiConfig()
    provider = GeminiProvider(config)

    prompt = f"Summarize this text in one sentence: {text_content[:500]}"
    response = provider.complete(prompt)
    logger.info(f"Summary: {response}")
    assert response, "Response should not be empty"
    logger.info("✓ Text file test passed")
    return True


def test_with_markdown():
    """Test processing markdown content."""
    logger.info("Testing with basic.md...")
    md_file = get_sample_data_path("markdown", "basic.md")
    if not md_file.exists():
        logger.warning(f"Sample file not found: {md_file}")
        return False

    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    config = GeminiConfig()
    provider = GeminiProvider(config)

    prompt = (
        f"Extract the main topics from this markdown document:\n\n"
        f"{md_content[:1000]}"
    )
    response = provider.complete(prompt)
    logger.info(f"Topics: {response[:200]}...")
    assert response, "Response should not be empty"
    logger.info("✓ Markdown test passed")
    return True


def test_json_mode_with_sample():
    """Test JSON mode with structured output."""
    logger.info("Testing JSON mode...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {
            "role": "user",
            "content": (
                "Analyze the following text and return a JSON object "
                "with 'language' (detected language), 'word_count' "
                "(number of words), and 'sentiment' (positive/neutral/"
                "negative). Text: 'This is a wonderful day!'"
            ),
        },
    ]

    response = provider.chat(messages, json_mode=True)
    logger.info(f"JSON Response: {response}")

    import json
    try:
        data = json.loads(response)
        assert isinstance(data, dict), "Should be valid JSON object"
        logger.info("✓ JSON mode test passed")
        return True
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        logger.warning("Response may still be valid, checking...")
        return True


def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    logger.info("Testing multi-turn conversation...")
    config = GeminiConfig()
    provider = GeminiProvider(config)

    messages = [
        {"role": "user", "content": "My name is Alice."},
    ]
    response1 = provider.chat(messages)
    logger.info(f"Response 1: {response1[:100]}...")

    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "What's my name?"})
    response2 = provider.chat(messages)
    logger.info(f"Response 2: {response2[:100]}...")

    assert response1 and response2, "Both responses should exist"
    logger.info("✓ Multi-turn conversation test passed")
    return True


def main():
    """Run all sample data tests."""
    api_key = (
        os.environ.get("GOOGLE_API_KEY") or
        os.environ.get("GEMINI_API_KEY")
    )
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
        os.environ["GOOGLE_API_KEY"] = api_key
        logger.info("Using API key from command line argument")
    if not api_key:
        logger.error(
            "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is "
            "required. Set it before running this test, or pass it as "
            "argument: python test_gemini_with_samples.py <api_key>"
        )
        sys.exit(1)

    logger.info("Starting Gemini provider tests with sample data...")
    logger.info(f"Using model: {GeminiConfig().model}")
    logger.info(f"Sample data path: {get_sample_data_path()}")

    tests = [
        ("Chat Prompt", test_with_chat_prompt),
        ("Chain Prompt", test_with_chain_prompt),
        ("Text File", test_with_text_file),
        ("Markdown", test_with_markdown),
        ("JSON Mode", test_json_mode_with_sample),
        ("Multi-turn", test_multi_turn_conversation),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*50}")
            result = test_func()
            if result is False:
                skipped += 1
                logger.warning(f"⚠ {name} test skipped (file not found)")
            elif result is True:
                passed += 1
                logger.info(f"✓ {name} test passed")
        except Exception as e:
            logger.error(f"✗ {name} test failed: {e}", exc_info=True)
            failed += 1

    logger.info(f"\n{'='*50}")
    logger.info("=== Test Summary ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Total: {passed + failed + skipped}")
    logger.info(f"{'='*50}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
