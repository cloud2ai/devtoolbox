#!/usr/bin/env python3
"""
Basic example of using SpeechService with Whisper provider.

This example demonstrates the basic usage of SpeechService for:
1. Speech-to-text (STT) conversion
2. Support for multiple languages
3. Advanced configuration options

Before running this example, please ensure:
1. Whisper is installed (pip install openai-whisper)
2. FFmpeg is installed for audio processing
3. Required environment variables are set (optional)
"""

import logging
import sys
from pathlib import Path

# Add parent directory to system path to import devtoolbox
sys.path.append(str(Path(__file__).parent.parent))

from devtoolbox.speech.service import SpeechService
from devtoolbox.speech.whisper_provider import WhisperConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories for the example.

    Returns:
        Path: Path to the output directory
    """
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created at: {output_dir}")
    return output_dir


def process_audio_file(service, audio_path, output_dir, language):
    """Process a single audio file and convert it to text.

    Args:
        service: SpeechService instance
        audio_path: Path to the audio file
        output_dir: Directory to save output
        language: Language of the audio (for output filename)

    Returns:
        Path: Path to the output file
    """
    logger.info(f"Processing {language} audio file: {audio_path}")

    # Create output path
    output_path = output_dir / f"stt_output_{language}.txt"

    try:
        # Convert speech to text using SpeechService
        result_path = service.speech_to_text(
            speech_path=str(audio_path),
            output_path=str(output_path),
            output_format="txt",
            use_cache=True  # Enable caching for better performance
        )

        logger.info(f"Speech-to-text output saved to: {result_path}")

        # Print the transcription
        with open(result_path, 'r', encoding='utf-8') as f:
            logger.info(f"\n{language.title()} Transcription:")
            logger.info("-" * 50)
            logger.info(f.read())
            logger.info("-" * 50)

        return result_path

    except Exception as e:
        logger.error(f"Error processing {language} audio file: {str(e)}")
        raise


def main():
    """Run the speech service example with Whisper provider."""
    logger.info("STARTING SPEECH SERVICE EXAMPLE (WHISPER)")
    logger.info("=" * 50)

    try:
        # Setup directories
        output_dir = setup_directories()

        # Configure Whisper Service with advanced options
        config = WhisperConfig(
            model_name="base",  # Use base model for good balance
            language="auto",    # Enable automatic language detection
            task="transcribe",  # Transcription task
            fp16=True,         # Use half-precision for better performance
            temperature=0.0,    # Deterministic output
            best_of=5,         # Number of candidates
            beam_size=5,       # Beam size for beam search
            patience=1.0,      # Beam search patience
            word_timestamps=True,  # Include word-level timestamps
            verbose=True       # Enable progress information
        )

        # Create speech service with Whisper provider
        service = SpeechService(config)

        # Process audio samples
        script_dir = Path(__file__).parent
        audio_dir = script_dir / "audio_samples"

        # Check if audio directory exists
        if not audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {audio_dir}"
            )

        # Process Chinese sample
        chinese_audio = audio_dir / "chinese_sample.mp3"
        if not chinese_audio.exists():
            raise FileNotFoundError(
                f"Chinese audio file not found: {chinese_audio}"
            )
        process_audio_file(service, chinese_audio, output_dir, "zh")

        # Process English sample
        english_audio = audio_dir / "english_sample.mp3"
        if not english_audio.exists():
            raise FileNotFoundError(
                f"English audio file not found: {english_audio}"
            )
        process_audio_file(service, english_audio, output_dir, "en")

    except Exception as e:
        logger.error(f"Error during speech processing: {str(e)}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Speech processing completed successfully")


if __name__ == "__main__":
    main()