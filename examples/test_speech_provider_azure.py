"""Test Azure Speech Service provider

This example demonstrates how to use Azure Speech Service for:
1. Different languages support
2. Text-to-speech functionality
3. Speech-to-text functionality
4. Different output formats (txt, srt)
5. Cache functionality
6. Error handling
"""

import os
import logging
from pathlib import Path
import time
import tempfile
from typing import Optional

from devtoolbox.speech.azure_provider import AzureConfig
from devtoolbox.speech.provider import SpeechProvider

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

TEST_AUDIO_DIR = SCRIPT_DIR / "test_speech_providers_audios"
TEST_OUTPUT_DIR = SCRIPT_DIR / "test_speech_providers_output"

# 测试音频文件
SAMPLE_AUDIOS = {
    #"en": "english_sample.mp3",
    "zh": "chinese_sample.mp3",
}

# 测试文本
TEST_TEXTS = {
    "en": "Hello, this is a test for Azure Speech Service.",
    "zh": "你好，这是 Azure 语音服务的测试。"
}

# 测试语音
TEST_VOICES = {
    "en": "en-US-JennyNeural",
    "zh": "zh-CN-XiaoxiaoNeural"
}


def setup_test_environment():
    """Setup test directories and environment"""
    logger.info(f"Creating temporary test output directory: {TEST_OUTPUT_DIR}")

    # 验证测试音频目录是否存在
    if not TEST_AUDIO_DIR.exists():
        TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Created audio directory: {TEST_AUDIO_DIR}\n"
            "Please place your test audio files in this directory"
        )

    # 验证测试音频文件是否存在
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


def test_text_to_speech(provider: SpeechProvider, lang: str):
    """Test text-to-speech functionality"""
    logger.info(f"Testing text-to-speech for language: {lang}")

    text = TEST_TEXTS[lang]
    voice = TEST_VOICES[lang]
    output_path = Path(TEST_OUTPUT_DIR) / f"tts_{lang}_output.wav"

    try:
        start_time = time.time()
        provider.text_to_speech(
            text=text,
            output_path=str(output_path),
            speaker=voice
        )
        duration = time.time() - start_time
        logger.info(f"TTS completed in {duration:.2f} seconds")

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Successfully generated audio output: {output_path}")
        else:
            logger.error("Failed to generate valid audio output")

    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise


def test_speech_to_text(provider: SpeechProvider, lang: str):
    """Test speech-to-text functionality"""
    logger.info(f"Testing speech-to-text for language: {lang}")

    audio_file = SAMPLE_AUDIOS[lang]
    audio_path = Path(TEST_AUDIO_DIR) / audio_file
    audio_name = Path(audio_file).stem

    # 测试 txt 输出
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

        if txt_output.exists() and txt_output.stat().st_size > 0:
            logger.info(f"Successfully generated text output: {txt_output}")
        else:
            logger.error("Failed to generate valid text output")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    # 测试 srt 输出
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

    # 第一次转录（无缓存）
    start_time = time.time()
    provider.speech_to_text(
        speech_path=str(audio_path),
        output_dir=str(output_dir),
        use_cache=True
    )
    first_duration = time.time() - start_time
    logger.info(f"First transcription took {first_duration:.2f} seconds")

    # 第二次转录（应使用缓存）
    start_time = time.time()
    provider.speech_to_text(
        speech_path=str(audio_path),
        output_dir=str(output_dir),
        use_cache=True
    )
    second_duration = time.time() - start_time
    logger.info(f"Second transcription took {second_duration:.2f} seconds")

    # 验证缓存效果
    if second_duration < first_duration * 0.5:
        logger.info("Cache is working effectively")
    else:
        logger.warning("Cache might not be working as expected")


def test_error_handling(provider: SpeechProvider):
    """Test error handling"""
    logger.info("Testing error handling")

    # 测试不存在的文件
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
    logger.info(f"Starting Azure provider tests from {SCRIPT_DIR}")
    logger.info(f"Test audio directory: {TEST_AUDIO_DIR}")
    logger.info(f"Test output directory: {TEST_OUTPUT_DIR}")

    try:
        setup_test_environment()

        # 创建 Azure provider
        config = AzureConfig.from_env()
        provider = SpeechProvider("azure", config)

        # 测试不同语言
        for lang in SAMPLE_AUDIOS.keys():
            # 测试文本转语音
            test_text_to_speech(provider, lang)
            # 测试语音转文本
            test_speech_to_text(provider, lang)
            # 测试缓存功能
            test_cache_functionality(provider, lang)

        # 测试错误处理
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