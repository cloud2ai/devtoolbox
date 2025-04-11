#!/usr/bin/env python3
"""
Basic example of using SpeechService with Azure provider.

This example demonstrates the basic usage of SpeechService for:
1. Text-to-speech (TTS)
2. Speech-to-text (STT)

Before running this example, please set the following environment variables:
- AZURE_SPEECH_KEY: Your Azure Speech Service subscription key
- AZURE_SPEECH_REGION: Your Azure service region (defaults to 'eastasia')
"""

import os
import logging
from pathlib import Path
import sys

# Add parent directory to system path to import devtoolbox
sys.path.append(str(Path(__file__).parent.parent))

from devtoolbox.speech.service import SpeechService
from devtoolbox.speech.azure_provider import AzureConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Run the basic speech service example."""
    print("STARTING SPEECH SERVICE EXAMPLE")
    print("=" * 50)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    try:
        # 1. Configure Azure Speech Service
        config = AzureConfig()

        # 2. Create speech service
        service = SpeechService(config)

        # 3. Text-to-speech example
        text = "你好，这是 Azure 语音服务的测试。"
        output_path = "outputs/tts_output.wav"

        print("\nConverting text to speech...")
        audio_path = service.text_to_speech(
            text=text,
            output_path=output_path,
            speaker="zh-CN-XiaoxiaoNeural"  # Use Chinese female voice
        )
        print(f"Text-to-speech output saved to: {audio_path}")

        # 4. Speech-to-text example
        # Note: You need to provide an audio file for this example
        # audio_path = "examples/outputs/tts_output.wav"  # Use generated audio
        # transcription_path = "examples/outputs/stt_output.txt"
        #
        # print("\nConverting speech to text...")
        # result_path = service.speech_to_text(
        #     speech_path=audio_path,
        #     output_path=transcription_path
        # )
        # print(f"Speech-to-text output saved to: {result_path}")

    except Exception as e:
        print(f"\nError during speech processing: {str(e)}")
        logging.exception(e)
        sys.exit(1)

    print("=" * 50)

    # Additional features documentation:
    """
    The SpeechService provides many advanced features:

    1. Text-to-Speech Features:
       - Caching: Automatically caches generated audio
       - Long text handling: Splits long text into segments
       - Multiple voices: Supports different speakers
       - Rate control: Adjust speech rate

    2. Speech-to-Text Features:
       - Multiple formats: Supports txt and srt output
       - Audio processing: Handles long audio files
       - Silence detection: Splits audio on silence
       - Caching: Caches transcription results

    3. Advanced Usage:
       # With custom settings
       service.text_to_speech(
           text="Hello world",
           output_path="output.mp3",
           speaker="en-US-JennyNeural",
           rate=10,  # Speed up by 10%
           use_cache=False
       )

       # With subtitle generation
       service.speech_to_text(
           speech_path="input.wav",
           output_path="output.srt",
           output_format="srt",
           min_silence_len=1000,
           silence_thresh=-40
       )
    """


if __name__ == "__main__":
    main()