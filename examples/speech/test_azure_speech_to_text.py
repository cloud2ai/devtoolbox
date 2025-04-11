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


def check_azure_credentials():
    """Check if required Azure credentials are set."""
    required_vars = {
        'AZURE_SPEECH_KEY': 'Azure Speech Service subscription key',
        'AZURE_SPEECH_REGION': 'Azure service region'
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        print("\nError: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables before running the example:")
        print("export AZURE_SPEECH_KEY='your-subscription-key'")
        print("export AZURE_SPEECH_REGION='your-region'")
        sys.exit(1)


def setup_directories():
    """Create necessary directories for the example."""
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_audio_file(service, audio_path, output_dir, language):
    """Process a single audio file and convert it to text.

    Args:
        service: SpeechService instance
        audio_path: Path to the audio file
        output_dir: Directory to save output
        language: Language of the audio (for output filename)
    """
    print(f"\nProcessing {language} audio file: {audio_path}")

    # Create output path
    output_path = output_dir / f"stt_output_{language}.txt"

    # Convert speech to text
    result_path = service.speech_to_text(
        speech_path=str(audio_path),
        output_path=str(output_path),
        output_format="srt",
    )
    print(f"Speech-to-text output saved to: {result_path}")

    # Print the transcription
    with open(result_path, 'r', encoding='utf-8') as f:
        print(f"\n{language.title()} Transcription:")
        print("-" * 50)
        print(f.read())
        print("-" * 50)


def main():
    """Run the basic speech service example."""
    print("STARTING SPEECH SERVICE EXAMPLE")
    print("=" * 50)

    # Check Azure credentials
    check_azure_credentials()

    # Setup directories
    output_dir = setup_directories()

    try:
        # 1. Configure Azure Speech Service
        config = AzureConfig(
            subscription_key=os.environ.get('AZURE_SPEECH_KEY'),
            service_region=os.environ.get('AZURE_SPEECH_REGION', 'eastasia')
        )

        # 2. Create speech service
        service = SpeechService(config)

        # 3. Process audio samples
        script_dir = Path(__file__).parent
        audio_dir = script_dir / "audio_samples"

        # Process Chinese sample
        chinese_audio = audio_dir / "chinese_sample.mp3"
        process_audio_file(service, chinese_audio, output_dir, "chinese")

        # Process English sample
        english_audio = audio_dir / "english_sample.mp3"
        process_audio_file(service, english_audio, output_dir, "english")

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