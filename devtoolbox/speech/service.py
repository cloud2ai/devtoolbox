"""Speech service layer implementation.

This module provides the SpeechService class which offers a high-level
interface for speech operations with advanced features like caching,
audio processing, and error handling.
"""

import os
import tempfile
import hashlib
import logging
import shutil
import json
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import asdict

from pydub import AudioSegment
from pydub.silence import split_on_silence
import pysubs2
import wave
import webrtcvad

from devtoolbox.text_splitter.token_splitter import TokenSplitter
from devtoolbox.speech.provider import BaseSpeechConfig
from devtoolbox.speech.utils import (
    split_speech_chunks,
    convert_audio_ffmpeg,
    DEFAULT_MIN_CHUNK_DURATION,
    DEFAULT_MAX_CHUNK_DURATION,
    DEFAULT_VAD_AGGRESSIVENESS,
    DEFAULT_MAX_WAIT_FOR_SILENCE
)

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

    # Audio processing format constants
    PROCESSING_FORMAT = "wav"      # Format used for STT processing
    STORAGE_FORMAT = "mp3"         # Format used for storage
    STORAGE_BITRATE = "128k"       # Bitrate for storage format



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

    def _setup_cache_dir(self, base_path: str) -> str:
        """Set up cache directory based on input filename.

        Args:
            base_path: Path to the input file or output directory.

        Returns:
            str: Path to cache directory.
        """
        base_dir = os.path.dirname(os.path.abspath(base_path))
        filename = os.path.basename(base_path)
        cache_dir = os.path.join(base_dir, f"{filename}.chunk")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Cache directory created: %s", cache_dir)
        logger.debug("Cache directory set up at: %s", cache_dir)
        return cache_dir

    def _get_cache_path(self, cache_dir: str, content_hash: str,
                       extension: str) -> str:
        """Get path for cached content.

        Args:
            cache_dir: Cache directory path.
            content_hash: Hash of the content.
            extension: File extension.

        Returns:
            str: Path to cached file.
        """
        cache_path = os.path.join(cache_dir, f"{content_hash}.{extension}")
        logger.debug("Cache path generated: %s", cache_path)
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

    def _setup_temp_dir(self, base_path: str) -> str:
        """Set up temporary directory based on input filename.

        Args:
            base_path: Path to the input file or output directory.

        Returns:
            str: Path to temporary directory.
        """
        cache_dir = self._setup_cache_dir(base_path)
        logger.debug("Using cache directory as temp directory: %s", cache_dir)
        return cache_dir

    def _cleanup_cache_dir(self, cache_dir: str):
        """Clean up cache directory.

        Args:
            cache_dir: Path to cache directory.
        """
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info("Cache directory cleaned up: %s", cache_dir)
            else:
                logger.debug("Cache directory does not exist: %s", cache_dir)
        except Exception as e:
            logger.warning(
                "Failed to clean up cache directory %s: %s",
                cache_dir, str(e)
            )

    def _generate_output_filename(self, input_path: str,
                                output_format: str) -> str:
        """Generate output filename based on input filename.

        Args:
            input_path: Path to input file.
            output_format: Output format extension.

        Returns:
            str: Generated output filename.
        """
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        base_dir = os.path.dirname(os.path.abspath(input_path))
        output_filename = f"{base_name}.{output_format}"
        output_path = os.path.join(base_dir, output_filename)
        logger.debug("Generated output filename: %s", output_path)
        return output_path

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        use_cache: bool = True,
        speaker: Optional[str] = None,
        rate: int = 0,
        cleanup_cache: bool = False
    ) -> str:
        """Convert text to speech with caching.

        Args:
            text: Text to convert
            output_path: Path to save the audio file
            use_cache: Whether to use cache
            speaker: Voice to use for synthesis
            rate: Speech rate adjustment
            cleanup_cache: Whether to clean up cache after processing

        Returns:
            str: Path to the generated audio file

        Raises:
            Exception: If text-to-speech fails
        """
        logger.info("Starting text-to-speech conversion")
        logger.debug("Input text length: %d characters", len(text))
        logger.debug("Output path: %s", output_path)
        logger.debug("Use cache: %s, Speaker: %s, Rate: %d",
                    use_cache, speaker, rate)

        # Text splitting strategy:
        # - Split long text into manageable segments to avoid memory issues
        # - Each segment is processed independently and can be cached
        # - Segments are recombined into final audio file
        segments = self._split_text(text)
        logger.info("Text split into %d segments", len(segments))

        # Cache and temporary directory setup:
        # - Create cache directory based on output filename
        # - Use same directory for temporary files during processing
        # - Cache directory structure: <output_filename>.chunk/
        cache_dir = self._setup_cache_dir(output_path)
        temp_dir = self._setup_temp_dir(output_path)

        try:
            # Segment processing loop:
            # - Process each text segment individually
            # - Generate hash for caching based on segment content
            # - Check cache first, then generate new audio if needed
            # - Cache generated audio for future reuse
            audio_segments = []
            for i, segment in enumerate(segments):
                logger.debug("Processing segment %d/%d", i + 1, len(segments))

                # Cache key generation:
                # - Use MD5 hash of segment text as cache key
                # - Ensures identical text segments use same cache
                # - Cache file extension matches audio format
                segment_hash = hashlib.md5(segment.encode()).hexdigest()
                cache_path = self._get_cache_path(
                    cache_dir, segment_hash, self.AUDIO_FORMAT)
                temp_path = os.path.join(
                    temp_dir, f"segment_{i}.{self.AUDIO_FORMAT}")

                # Cache lookup and audio generation:
                # - Check if cached audio exists and cache is enabled
                # - Load cached audio if available
                # - Otherwise, generate new audio using TTS provider
                # - Save to cache for future use if caching enabled
                if use_cache and os.path.exists(cache_path):
                    logger.debug("Using cached segment %d", i)
                    audio = AudioSegment.from_file(cache_path)
                else:
                    logger.info("Generating new audio for segment %d", i)
                    segment_preview = (segment[:100] + "..."
                                     if len(segment) > 100 else segment)
                    logger.debug("Segment content: %s", segment_preview)

                    # TTS provider call:
                    # - Convert text segment to speech
                    # - Apply speaker voice and rate settings
                    # - Save temporary audio file
                    self.provider.speak(
                        segment,
                        temp_path,
                        speaker=speaker,
                        rate=rate
                    )
                    audio = AudioSegment.from_file(temp_path)

                    # Cache management:
                    # - Export audio with consistent format and bitrate
                    # - Save to cache directory for future reuse
                    # - Cache key ensures identical segments use same cache
                    if use_cache:
                        audio.export(
                            cache_path,
                            format=self.AUDIO_FORMAT,
                            bitrate=self.AUDIO_BITRATE
                        )
                        logger.debug("Cached segment %d at: %s", i, cache_path)

                audio_segments.append(audio)

            # Audio combination strategy:
            # - Combine all audio segments into single file
            # - Handle single segment case separately
            # - Concatenate segments in order for multi-segment case
            # - Export final combined audio with consistent settings
            logger.info("Combining %d audio segments", len(audio_segments))
            if len(audio_segments) == 1:
                combined = audio_segments[0]
            else:
                combined = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined += segment
            combined.export(
                output_path,
                format=self.AUDIO_FORMAT,
                bitrate=self.AUDIO_BITRATE
            )
            logger.info("Audio successfully saved at: %s", output_path)

            return output_path
        except Exception as e:
            logger.error("Error in speech processing: %s", str(e))
            raise
        finally:
            # Cleanup phase:
            # - Optionally clean up cache directory after processing
            # - Ensures temporary files are removed if requested
            # - Always executed regardless of success or failure
            if cleanup_cache:
                logger.info("Cleaning up cache directory")
                self._cleanup_cache_dir(cache_dir)

    def _split_text(self, text: str) -> List[str]:
        """Split text into segments.

        Args:
            text: Text to split

        Returns:
            List[str]: List of text segments
        """
        logger.debug("Splitting text into segments")
        splitter = TokenSplitter(text)
        paragraphs = splitter.split()
        logger.debug("Text split into %d paragraphs", len(paragraphs))
        return [p.text for p in paragraphs]

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
        logger.debug("Formatting as plain text")
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
        logger.debug("Formatting as subtitle: %s", format_type)
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

        logger.debug("Generated subtitle with %d entries", len(subs))
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
        logger.debug("Generating content in format: %s", output_format)
        if output_format not in self.SUPPORTED_FORMATS:
            error_msg = (
                f"Unsupported format type: {output_format}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if output_format == "txt":
            return self._format_as_text(texts, chunks)
        else:
            return self._format_as_subtitle(texts, chunks, output_format)

    def _process_audio_chunks(
        self,
        speech_path: str,
        temp_dir: str,
        cache_dir: str,
        use_cache: bool = True,
        min_chunk_duration: int = DEFAULT_MIN_CHUNK_DURATION,
        max_chunk_duration: int = DEFAULT_MAX_CHUNK_DURATION,
        vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS,
        max_wait_for_silence: int = DEFAULT_MAX_WAIT_FOR_SILENCE
    ) -> list:
        # Use split_speech_chunks to get List[ChunkMeta]
        from devtoolbox.speech.utils import ChunkMeta
        chunk_metas: list = split_speech_chunks(
            speech_path,
            temp_dir,
            min_chunk_duration,
            max_chunk_duration,
            vad_aggressiveness,
            max_wait_for_silence
        )
        logger.info(f"Audio split into {len(chunk_metas)} chunks")
        for chunk in chunk_metas:
            # Transcode wav to mp3
            mp3_path = os.path.splitext(chunk.wav_path)[0] + '.mp3'
            convert_audio_ffmpeg(
                chunk.wav_path,
                mp3_path,
                sample_rate=16000,
                channels=1,
                sample_fmt='s16'
            )
            chunk.mp3_path = mp3_path
            chunk.wav_size = self._get_file_size(chunk.wav_path)
            chunk.mp3_size = self._get_file_size(mp3_path)
            # Caching and transcription logic
            with open(chunk.wav_path, 'rb') as f:
                chunk_data = f.read()
            chunk_hash = hashlib.md5(chunk_data).hexdigest()
            cache_path = self._get_cache_path(cache_dir, chunk_hash, "txt")

            if use_cache and os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    chunk.transcript = f.read().strip()
                chunk.cached = True
                continue

            temp_output = os.path.join(temp_dir, f"chunk_{chunk.index}.txt")
            self.provider.transcribe(chunk.wav_path, temp_output)
            with open(temp_output, "r") as f:
                chunk.transcript = f.read().strip()
            chunk.cached = False
            if use_cache:
                with open(cache_path, "w") as f:
                    f.write(chunk.transcript)
        return chunk_metas

    def _generate_metadata(
        self,
        chunk_metas: list,
        audio: 'AudioSegment',
        output_path: str
    ) -> dict:
        logger.debug("Generating metadata for %d chunks", len(chunk_metas))
        total_wav_size = sum(chunk.wav_size or 0 for chunk in chunk_metas)
        total_mp3_size = sum(chunk.mp3_size or 0 for chunk in chunk_metas)
        overall_compression_ratio = (total_mp3_size / total_wav_size
                                   if total_wav_size > 0 else 0)
        if audio is not None:
            audio_info = {
                "total_duration": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "bit_depth": audio.sample_width * 8,
                "original_format": "wav"
            }
        else:
            audio_info = {
                "total_duration": None,
                "sample_rate": None,
                "channels": None,
                "bit_depth": None,
                "original_format": "wav"
            }
        metadata = {
            "audio_info": audio_info,
            "storage_info": {
                "processing_format": self.PROCESSING_FORMAT,
                "storage_format": self.STORAGE_FORMAT,
                "storage_bitrate": self.STORAGE_BITRATE,
                "total_wav_size": total_wav_size,
                "total_mp3_size": total_mp3_size,
                "overall_compression_ratio": overall_compression_ratio,
                "space_saved_bytes": total_wav_size - total_mp3_size,
                "space_saved_percent": (1 - overall_compression_ratio) * 100
            },
            "processing_info": {
                "total_chunks": len(chunk_metas),
                "total_text_length": sum(len(chunk.transcript or "") for chunk in chunk_metas),
                "output_file": output_path,
                "metadata_file": f"{output_path}.metadata.json",
            },
            "chunks": [asdict(chunk) for chunk in chunk_metas]
        }
        logger.debug("Metadata generated with %d chunk entries",
                    len(chunk_metas))
        logger.info("Storage optimization: %.1f%% space saved (%.1f MB -> %.1f MB)",
                   metadata["storage_info"]["space_saved_percent"],
                   total_wav_size / (1024 * 1024),
                   total_mp3_size / (1024 * 1024))
        return metadata

    def speech_to_text(
        self,
        speech_path: str,
        output_path: str,
        output_format: str = "txt",
        use_cache: bool = True,
        cleanup_cache: bool = False,
        min_chunk_duration: int = DEFAULT_MIN_CHUNK_DURATION,
        max_chunk_duration: int = DEFAULT_MAX_CHUNK_DURATION,
        vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS,
        max_wait_for_silence: int = DEFAULT_MAX_WAIT_FOR_SILENCE
    ) -> dict:
        logger.info(
            f"Starting speech-to-text conversion: input={speech_path}, "
            f"output={output_path}, format={output_format}, "
            f"use_cache={use_cache}, cleanup_cache={cleanup_cache}, "
            f"min_chunk_duration={min_chunk_duration}, "
            f"max_chunk_duration={max_chunk_duration}, "
            f"vad_aggressiveness={vad_aggressiveness}, "
            f"max_wait_for_silence={max_wait_for_silence}"
        )
        logger.debug(f"Input audio file: {speech_path}")
        logger.debug(
            f"Output file: {output_path}, Format: {output_format}"
        )
        logger.debug(
            f"Use cache: {use_cache}, Cleanup cache: {cleanup_cache}"
        )
        cache_dir = self._setup_cache_dir(speech_path)
        temp_dir = self._setup_temp_dir(speech_path)
        metadata_path = f"{output_path}.metadata.json"
        try:
            chunk_metas = self._process_audio_chunks(
                speech_path,
                temp_dir,
                cache_dir,
                use_cache,
                min_chunk_duration,
                max_chunk_duration,
                vad_aggressiveness,
                max_wait_for_silence
            )
            logger.info(
                f"Generating final content: output={output_path}, "
                f"format={output_format}, total_chunks={len(chunk_metas)}"
            )
            # Compose final text from all transcripts
            final_text = "\n".join(chunk.transcript or "" for chunk in chunk_metas)
            logger.debug(
                f"Final content length: {len(final_text)} characters, "
                f"output file: {output_path}"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info(
                f"Transcription successfully saved: {output_path} "
                f"(length: {len(final_text)} characters)"
            )
            metadata = self._generate_metadata(
                chunk_metas, None, output_path
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Metadata successfully saved: {metadata_path} "
                f"(size: {os.path.getsize(metadata_path)} bytes)"
            )
            return {
                "output_path": output_path,
                "metadata_path": metadata_path,
                "chunks_path": cache_dir
            }
        except Exception as e:
            logger.error(
                f"Error in speech-to-text processing: input={speech_path}, "
                f"output={output_path}, format={output_format}, error={str(e)}"
            )
            raise
        finally:
            if cleanup_cache:
                logger.info(
                    f"Cleaning up cache directory: {cache_dir}"
                )
                self._cleanup_cache_dir(cache_dir)

    def list_speakers(self) -> List[str]:
        """List available speakers/voices.

        Returns:
            List[str]: List of available speakers.
        """
        logger.debug("Listing available speakers")
        speakers = self.provider.list_speakers()
        logger.info("Found %d available speakers", len(speakers))
        logger.debug("Available speakers: %s", speakers)
        return speakers

    def _convert_audio_format(
        self,
        input_path: str,
        output_path: str,
        input_format: str = PROCESSING_FORMAT,
        output_format: str = STORAGE_FORMAT,
        bitrate: str = STORAGE_BITRATE
    ) -> Dict[str, Any]:
        """Convert audio file between different formats.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            input_format: Input audio format
            output_format: Output audio format
            bitrate: Output bitrate for compressed formats

        Returns:
            Dict[str, Any]: Conversion metadata including file sizes
        """
        logger.debug("Converting audio from %s to %s", input_format, output_format)

        try:
            # Load audio file
            audio = AudioSegment.from_file(input_path, format=input_format)

            # Get input file size
            input_size = os.path.getsize(input_path)

            # Export to new format
            audio.export(
                output_path,
                format=output_format,
                bitrate=bitrate if output_format == "mp3" else None
            )

            # Get output file size
            output_size = os.path.getsize(output_path)

            # Calculate compression ratio
            compression_ratio = output_size / input_size if input_size > 0 else 0

            conversion_info = {
                "input_format": input_format,
                "output_format": output_format,
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": compression_ratio,
                "bitrate": bitrate,
                "success": True
            }

            logger.debug("Audio conversion successful: %s -> %s (%.2f%% size)",
                        input_format, output_format, compression_ratio * 100)

            return conversion_info

        except Exception as e:
            logger.error("Audio conversion failed: %s", str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes, 0 if file doesn't exist
        """
        try:
            return os.path.getsize(file_path) if os.path.exists(file_path) else 0
        except Exception as e:
            logger.warning("Failed to get file size for %s: %s", file_path, str(e))
            return 0
