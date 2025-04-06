"""Test Whisper speech recognition provider

This example demonstrates how to use Whisper for speech recognition with:
1. Different languages support
2. Different model sizes
3. Different output formats (txt, srt)
4. Cache functionality
5. Error handling
"""

import os
import logging
from pathlib import Path
import time
import tempfile
from typing import Optional

from devtoolbox.speech.whisper_provider import WhisperConfig
from devtoolbox.speech.provider import SpeechProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

TEST_AUDIO_DIR = SCRIPT_DIR / "test_speech_providers_audios"
TEST_OUTPUT_DIR = SCRIPT_DIR / "test_speech_providers_output"

SAMPLE_AUDIOS = {
    "en": "english_sample.mp3",
}


def setup_test_environment():
    """Setup test directories and environment"""
    logger.info(f"Creating temporary test output directory: {TEST_OUTPUT_DIR}")

    # Verify test audio directory exists
    if not TEST_AUDIO_DIR.exists():
        TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Created audio directory: {TEST_AUDIO_DIR}\n"
            "Please place your test audio files in this directory"
        )

    # Verify test audio files exist using absolute paths
    for lang, audio_file in SAMPLE_AUDIOS.items():
        audio_path = TEST_AUDIO_DIR / audio_file
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Test audio file not found: {audio_path}\n"
                f"Please ensure you have test audio files in {TEST_AUDIO_DIR}"
            )
        logger.info(f"Found test audio file: {audio_path}")


def cleanup_test_environment():
    """Clean up temporary test directories"""
    try:
        import shutil
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
            logger.info(f"Cleaned up temporary test directory: {TEST_OUTPUT_DIR}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {e}")


def test_basic_transcription(provider: SpeechProvider, lang: str):
    """Test basic transcription functionality"""
    logger.info(f"Testing basic transcription for language: {lang}")

    audio_file = SAMPLE_AUDIOS[lang]
    audio_path = Path(TEST_AUDIO_DIR) / audio_file

    # Convert audio_file to Path to use stem property
    audio_name = Path(audio_file).stem

    # Test txt output
    txt_output = Path(TEST_OUTPUT_DIR) / f"{audio_name}.txt"
    try:
        start_time = time.time()
        provider.speech_to_text(
            speech_path=str(audio_path),
            output_dir=str(txt_output.parent),
            output_format="txt"
        )
        duration = time.time() - start_time
        logger.info(f"Transcription completed in {duration:.2f} seconds")

        # Verify output
        if txt_output.exists() and txt_output.stat().st_size > 0:
            logger.info(f"Successfully generated text output: {txt_output}")
        else:
            logger.error("Failed to generate valid text output")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    # Test srt output
    srt_output = Path(TEST_OUTPUT_DIR) / f"{audio_name}.srt"
    try:
        provider.speech_to_text(
            speech_path=str(audio_path),
            output_dir=str(srt_output.parent),
            output_format="srt"
        )

        if srt_output.exists() and srt_output.stat().st_size > 0:
            logger.info(f"Successfully generated SRT output: {srt_output}")
        else:
            logger.error("Failed to generate valid SRT output")

    except Exception as e:
        logger.error(f"SRT generation failed: {e}")
        raise


def test_cache_functionality(provider: SpeechProvider, lang: str):
    """Test cache functionality"""
    logger.info(f"Testing cache functionality for language: {lang}")

    audio_file = SAMPLE_AUDIOS[lang]
    audio_path = Path(TEST_AUDIO_DIR) / audio_file
    output_dir = Path(TEST_OUTPUT_DIR) / "cache_test"
    output_dir.mkdir(exist_ok=True)

    # First transcription (no cache)
    start_time = time.time()
    provider.speech_to_text(
        speech_path=str(audio_path),
        output_dir=str(output_dir),
        use_cache=True
    )
    first_duration = time.time() - start_time
    logger.info(f"First transcription took {first_duration:.2f} seconds")

    # Second transcription (should use cache)
    start_time = time.time()
    provider.speech_to_text(
        speech_path=str(audio_path),
        output_dir=str(output_dir),
        use_cache=True
    )
    second_duration = time.time() - start_time
    logger.info(f"Second transcription took {second_duration:.2f} seconds")

    # Verify cache effectiveness
    if second_duration < first_duration * 0.5:
        logger.info("Cache is working effectively")
    else:
        logger.warning("Cache might not be working as expected")


def test_model_sizes():
    """Test different Whisper model sizes"""
    #test_sizes = ["tiny", "base", "small"]  # 可以根据需要测试更大的模型
    test_sizes = ["tiny"]  # 可以根据需要测试更大的模型

    for size in test_sizes:
        logger.info(f"Testing Whisper model size: {size}")

        config = WhisperConfig(
            model_size=size,
            device="cuda" if os.environ.get("USE_CUDA") else "cpu"
        )

        try:
            provider = SpeechProvider("whisper", config)
            # 测试英语识别
            test_basic_transcription(provider, "en")
            logger.info(f"Successfully tested {size} model")
        except Exception as e:
            logger.error(f"Failed to test {size} model: {e}")


def test_error_handling(provider: SpeechProvider):
    """Test error handling"""
    logger.info("Testing error handling")

    # Test nonexistent file
    try:
        provider.speech_to_text(
            speech_path="nonexistent.mp3",
            output_dir=TEST_OUTPUT_DIR
        )
        logger.error("Should have raised ValueError")
    except ValueError as e:
        logger.info(f"Successfully caught ValueError: {e}")


def main():
    """Main test function"""
    logger.info(f"Starting Whisper provider tests from {SCRIPT_DIR}")
    logger.info(f"Test audio directory: {TEST_AUDIO_DIR}")
    logger.info(f"Temporary test output directory: {TEST_OUTPUT_DIR}")

    try:
        setup_test_environment()

        # Create default provider with base model
        config = WhisperConfig(
            model_size="tiny",  # 使用最小的模型进行测试
            device="cuda" if os.environ.get("USE_CUDA") else "cpu"
        )
        provider = SpeechProvider("whisper", config)

        # Test different languages
        for lang in SAMPLE_AUDIOS.keys():
            test_basic_transcription(provider, lang)
            test_cache_functionality(provider, lang)

        # Test different model sizes
        test_model_sizes()

        # Test error handling
        test_error_handling(provider)

        logger.info("All tests completed successfully")

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    finally:
        # 清理临时目录
        #cleanup_test_environment()
        pass


if __name__ == "__main__":
    main()