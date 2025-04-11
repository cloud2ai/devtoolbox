"""Speech service layer implementation.

This module provides the SpeechService class which offers a high-level
interface for speech operations with advanced features like caching,
audio processing, and error handling.
"""

import os
import tempfile
import hashlib
import logging
from typing import List, Optional

from pydub import AudioSegment
from pydub.silence import split_on_silence
import pysubs2

from devtoolbox.text.splitter import TokenSplitter
from devtoolbox.speech.provider import BaseSpeechConfig

logger = logging.getLogger(__name__)


class SpeechService:
    """Speech service layer.

    This class provides a high-level interface for speech operations,
    handling business logic and advanced features like caching and
    audio processing.
    """

    # Audio format constants
    AUDIO_FORMAT = "mp3"
    AUDIO_BITRATE = "128k"

    # Audio processing constants
    DEFAULT_SILENCE_LEN = 1000  # ms
    DEFAULT_SILENCE_THRESH = -40  # dB
    DEFAULT_KEEP_SILENCE = 500  # ms
    DEFAULT_OUTPUT_FORMAT = "txt"

    # Supported output formats
    SUPPORTED_FORMATS = {
        "txt": "Plain text format",
        "srt": "SubRip subtitle format",
        "ass": "Advanced SubStation Alpha format",
        "vtt": "WebVTT format"
    }

    def __init__(self, config: BaseSpeechConfig):
        """Initialize speech service.

        Args:
            config: Provider configuration instance.
        """
        logger.debug("Initializing SpeechService with config: %s", config)
        self.config = config
        self.provider = self._init_provider()
        self.cache_dir = self._setup_cache_dir()

    def _init_provider(self):
        """Initialize provider based on config.

        This method dynamically loads the appropriate provider based
        on the config class.

        Returns:
            BaseSpeechProvider: Initialized provider instance.

        Raises:
            ValueError: If provider cannot be initialized.
        """
        # Get the provider module name from config class
        config_class = self.config.__class__
        provider_name = config_class.__name__.replace("Config", "Provider")
        provider_module = config_class.__module__.replace("config", "provider")

        logger.debug("Initializing provider: %s from module: %s",
                     provider_name, provider_module)

        try:
            # Dynamically import the provider
            module = __import__(provider_module, fromlist=[provider_name])
            provider_class = getattr(module, provider_name)
            logger.info("Provider initialized: %s", provider_class)
            return provider_class(self.config)
        except (ImportError, AttributeError) as e:
            error_msg = (
                f"Failed to initialize provider for config {config_class}: "
                f"{str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _setup_cache_dir(self) -> str:
        """Set up cache directory.

        Returns:
            str: Path to cache directory.
        """
        cache_dir = os.path.join(os.getcwd(), ".cache", "speech")
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug("Cache directory set up at: %s", cache_dir)
        return cache_dir

    def _get_cache_path(self, content_hash: str, extension: str) -> str:
        """Get path for cached content.

        Args:
            content_hash: Hash of the content.
            extension: File extension.

        Returns:
            str: Path to cached file.
        """
        cache_path = os.path.join(
            self.cache_dir, f"{content_hash}.{extension}")
        logger.debug("Cache path for content: %s", cache_path)
        return cache_path

    def _check_cache(self, cache_path: str, use_cache: bool) -> bool:
        """Check if content is cached.

        Args:
            cache_path: Path to cached file.
            use_cache: Whether to use cache.

        Returns:
            bool: True if cached content exists and should be used.
        """
        if use_cache and os.path.exists(cache_path):
            logger.info("Using cached content: %s", cache_path)
            return True
        logger.debug("Cache not used, path does not exist: %s", cache_path)
        return False

    def _setup_temp_dir(self) -> str:
        """Set up temporary directory.

        Returns:
            str: Path to temporary directory.
        """
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir

    def _cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory.

        Args:
            temp_dir: Path to temporary directory.
        """
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temporary directory: {temp_dir}")

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        use_cache: bool = True,
        speaker: Optional[str] = None,
        rate: int = 0
    ) -> str:
        """Convert text to speech with caching.

        Args:
            text: Text to convert
            output_path: Path to save the audio file
            use_cache: Whether to use cache
            speaker: Voice to use for synthesis
            rate: Speech rate adjustment

        Returns:
            str: Path to the generated audio file

        Raises:
            Exception: If text-to-speech fails
        """
        logger.debug("Converting text to speech: %s", text)
        segments = self._split_text(text)
        logger.debug("Text split into %d segments", len(segments))
        temp_dir = self._setup_temp_dir()

        try:
            # Process each segment
            audio_segments = []
            for i, segment in enumerate(segments):
                segment_hash = hashlib.md5(segment.encode()).hexdigest()
                cache_path = self._get_cache_path(
                    segment_hash, self.AUDIO_FORMAT)
                temp_path = os.path.join(
                    temp_dir, f"segment_{i}.{self.AUDIO_FORMAT}")

                # Try to use cached audio or generate new one
                if use_cache and os.path.exists(cache_path):
                    logger.debug("Using cached segment %d", i)
                    audio = AudioSegment.from_file(cache_path)
                else:
                    logger.debug("Processing segment %d: %s", i, segment)
                    self.provider.speak(
                        segment,
                        temp_path,
                        speaker=speaker,
                        rate=rate
                    )
                    audio = AudioSegment.from_file(temp_path)
                    if use_cache:
                        audio.export(
                            cache_path,
                            format=self.AUDIO_FORMAT,
                            bitrate=self.AUDIO_BITRATE
                        )
                        logger.debug("Cached segment %d at: %s", i, cache_path)

                audio_segments.append(audio)

            # Combine and save final audio
            combined = sum(audio_segments[1:], audio_segments[0])
            combined.export(
                output_path,
                format=self.AUDIO_FORMAT,
                bitrate=self.AUDIO_BITRATE
            )
            logger.info("Audio saved at: %s", output_path)

            return output_path

        except Exception as e:
            logger.error("Error in speech processing: %s", str(e))
            raise

        finally:
            try:
                self._cleanup_temp_dir(temp_dir)
            except Exception as e:
                logger.error(
                    "Error cleaning up temporary directory: %s",
                    str(e)
                )
                # Cleanup errors should not affect the main flow, so only log

    def _split_text(self, text: str) -> List[str]:
        """Split text into segments.

        Args:
            text: Text to split

        Returns:
            List[str]: List of text segments
        """
        splitter = TokenSplitter()
        paragraphs = splitter.split_text(text)
        logger.debug("Text split into %d paragraphs", len(paragraphs))
        return [p.content for p in paragraphs]

    def _format_as_text(
        self,
        texts: List[str],
        chunks: List[AudioSegment]
    ) -> str:
        """Format transcription as plain text.

        Args:
            texts: List of transcribed texts
            chunks: List of audio chunks

        Returns:
            str: Formatted text content
        """
        return "\n".join(texts)

    def _format_as_subtitle(
        self,
        texts: List[str],
        chunks: List[AudioSegment],
        format_type: str
    ) -> str:
        """Format transcription as subtitle.

        Args:
            texts: List of transcribed texts
            chunks: List of audio chunks
            format_type: Subtitle format type (srt, ass, vtt)

        Returns:
            str: Formatted subtitle content
        """
        subs = pysubs2.SSAFile()
        start_time = 0

        for i, text in enumerate(texts):
            duration = len(chunks[i])
            subs.append(pysubs2.SSAEvent(
                start=start_time,
                end=start_time + duration,
                text=text
            ))
            start_time += duration

        return subs.to_string(format_type)

    def _generate_content(
        self,
        texts: List[str],
        chunks: List[AudioSegment],
        output_format: str
    ) -> str:
        """Generate content in specified format.

        Args:
            texts: List of transcribed texts
            chunks: List of audio chunks
            output_format: Output format type

        Returns:
            str: Formatted content

        Raises:
            ValueError: If format type is not supported
        """
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format type: {output_format}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        if output_format == "txt":
            return self._format_as_text(texts, chunks)
        else:
            return self._format_as_subtitle(texts, chunks, output_format)

    def _process_audio_chunks(
        self,
        audio: AudioSegment,
        temp_dir: str,
        min_silence_len: int,
        silence_thresh: int,
        keep_silence: int,
        use_cache: bool = True
    ) -> List[str]:
        """Process audio chunks and transcribe them.

        Args:
            audio: Audio segment to process
            temp_dir: Temporary directory for processing
            min_silence_len: Minimum silence length for splitting
            silence_thresh: Silence threshold for splitting
            keep_silence: Amount of silence to keep
            use_cache: Whether to use cache for chunks

        Returns:
            List[str]: List of transcribed texts
        """
        # Split audio into chunks
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        logger.debug("Audio split into %d chunks", len(chunks))

        # Process each chunk
        texts = []
        for i, chunk in enumerate(chunks):
            # Generate chunk hash for caching
            chunk_hash = hashlib.md5(chunk.raw_data).hexdigest()
            cache_path = self._get_cache_path(chunk_hash, "txt")

            # Check cache first
            if use_cache and os.path.exists(cache_path):
                logger.debug("Using cached transcription for chunk %d", i)
                with open(cache_path, "r") as f:
                    texts.append(f.read().strip())
                continue

            # Process chunk if not cached
            temp_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            temp_output = os.path.join(temp_dir, f"chunk_{i}.txt")

            try:
                # Export chunk and transcribe
                chunk.export(temp_path, format="wav")
                logger.debug("Transcribing chunk %d", i)
                self.provider.transcribe(temp_path, temp_output)

                # Read transcription
                with open(temp_output, "r") as f:
                    text = f.read().strip()
                    texts.append(text)

                    # Cache the result
                    if use_cache:
                        with open(cache_path, "w") as f:
                            f.write(text)
                        logger.debug("Cached transcription for chunk %d", i)

            except Exception as e:
                logger.error("Error processing chunk %d: %s", i, str(e))
                raise

        return texts, chunks

    def speech_to_text(
        self,
        speech_path: str,
        output_path: str,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        use_cache: bool = True,
        min_silence_len: int = DEFAULT_SILENCE_LEN,
        silence_thresh: int = DEFAULT_SILENCE_THRESH,
        keep_silence: int = DEFAULT_KEEP_SILENCE
    ) -> str:
        """Convert speech to text with caching.

        Args:
            speech_path: Path to the audio file
            output_path: Path to save the transcription
            output_format: Output format (txt, srt, ass, vtt)
            use_cache: Whether to use cache
            min_silence_len: Minimum silence length for splitting
            silence_thresh: Silence threshold for splitting
            keep_silence: Amount of silence to keep

        Returns:
            str: Path to the transcription file

        Raises:
            Exception: If speech-to-text fails
        """
        logger.debug("Converting speech to text from: %s", speech_path)

        # Process audio
        audio = AudioSegment.from_file(speech_path)
        temp_dir = self._setup_temp_dir()

        try:
            # Process chunks and get transcriptions
            texts, chunks = self._process_audio_chunks(
                audio,
                temp_dir,
                min_silence_len,
                silence_thresh,
                keep_silence,
                use_cache
            )

            # Generate final content
            final_text = self._generate_content(texts, chunks, output_format)

            # Save result
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info("Transcription saved at: %s", output_path)

            return output_path

        finally:
            self._cleanup_temp_dir(temp_dir)

    def list_speakers(self) -> List[str]:
        """List available speakers/voices.

        Returns:
            List[str]: List of available speakers.
        """
        speakers = self.provider.list_speakers()
        logger.debug("Available speakers: %s", speakers)
        return speakers