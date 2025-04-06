"""Base provider for multiple speech engines

This module provides the base infrastructure for implementing speech providers.
To add a new speech provider, follow these steps:

1. Create a new provider configuration class:
   Your provider should have its own configuration class that inherits from
   BaseSpeechConfig. This class should define all provider-specific settings.

   Example:
   ```python
   @dataclass
   class MyProviderConfig(BaseSpeechConfig):
       api_key: Optional[str] = None
       region: str = "us-east"

       @classmethod
       def from_env(cls) -> 'MyProviderConfig':
           return cls(
               api_key=os.environ.get('MY_PROVIDER_API_KEY'),
               region=os.environ.get('MY_PROVIDER_REGION', 'us-east')
           )

       def validate(self):
           if not self.api_key:
               raise ValueError(
                   "api_key is required. Set it either in constructor "
                   "or through MY_PROVIDER_API_KEY environment variable"
               )
   ```

2. Create a new provider class:
   Implement your provider by inheriting from BaseSpeechProvider. You must
   implement at least the `speak` and `transcribe` methods.

   Example:
   ```python
   @register_provider("my_provider")
   class MyProvider(BaseSpeechProvider):
       def __init__(self, config: MyProviderConfig):
           super().__init__(config)
           # Initialize your provider-specific client/SDK here
           self.client = MyProviderSDK(
               api_key=config.api_key,
               region=config.region
           )

       def speak(
           self,
           text: str,
           output_path: str,
           speaker: Optional[str] = None,
           rate: int = 0,
           *args,
           **kwargs
       ):
           # Implement text-to-speech logic
           self.client.synthesize(
               text=text,
               output=output_path,
               voice=speaker,
               speed=rate
           )

       def transcribe(
           self,
           speech_path: str,
           output_path: str,
           output_format: str = "txt"
       ) -> str:
           # Implement speech-to-text logic
           text = self.client.recognize(speech_path)
           with open(output_path, 'w') as f:
               f.write(text)
           return output_path
   ```

3. Usage:
   After implementing your provider, you can use it like this:
   ```python
   # Initialize with direct configuration
   config = MyProviderConfig(api_key="your-key", region="your-region")
   provider = SpeechProvider("my_provider", config)

   # Or initialize from environment variables
   config = MyProviderConfig.from_env()
   provider = SpeechProvider("my_provider", config)

   # Use the provider
   provider.text_to_speech("Hello world", "output.mp3")
   provider.speech_to_text("input.wav", "output_dir")
   ```

File Structure:
    provider.py
    ├── BaseSpeechConfig (abstract base class for configurations)
    ├── BaseSpeechProvider (abstract base class for providers)
    ├── SpeechProvider (main interface class)
    └── register_provider (decorator for provider registration)

Configuration:
    Each provider should handle its own configuration through environment
    variables or direct initialization. Common environment variables:
    - HTTP_PROXY: HTTP proxy server
    - HTTPS_PROXY: HTTPS proxy server

Error Handling:
    Providers should raise appropriate exceptions for:
    - Missing or invalid configuration
    - API errors
    - File I/O errors
    - Invalid input formats

Caching:
    The SpeechProvider class handles caching of generated speech files.
    Providers don't need to implement caching logic.

Audio Processing:
    The SpeechProvider class handles audio format conversion and combining.
    Providers should focus on their core TTS/STT functionality.
"""

import logging
import os
import pathlib
from typing import Dict, Type, List, Optional, Tuple
from dataclasses import dataclass
import json
import tempfile
import hashlib
from datetime import datetime

from pydub import AudioSegment
from devtoolbox.text.splitter import TokenSplitter
from pydub.silence import split_on_silence
import pysubs2

# Constants
BITRATE = "128k"
SPEECH_PROVIDERS: Dict[str, Type['BaseSpeechProvider']] = {}

# Cache related constants
CACHE_DIR = ".cache"
CACHE_INFO_FILENAME = "cache_info.json"


def register_provider(name: str):
    """
    Register speech provider

    Args:
        name: Provider name to register
    """
    def wrapper(provider_class):
        SPEECH_PROVIDERS[name] = provider_class
        return provider_class
    return wrapper


@dataclass
class BaseSpeechConfig:
    """Base configuration for speech providers"""
    def validate(self):
        """Validate configuration parameters"""
        raise NotImplementedError("validate() must be implemented")

    @classmethod
    def from_env(cls) -> 'BaseSpeechConfig':
        """Create configuration from environment variables"""
        raise NotImplementedError("from_env() must be implemented")


class BaseSpeechProvider:
    """Base provider for lower level of speech sdk as drivers"""

    def __init__(self, config: BaseSpeechConfig):
        """
        Initialize method for all providers

        Args:
            config: Provider configuration
        """
        config.validate()
        self.config = config
        self.proxies = self._set_proxies()

    def _set_proxies(self):
        """Return proxies dict with http proxy and https proxy"""
        return {
            "http_proxy": os.environ.get("HTTP_PROXY"),
            "https_proxy": os.environ.get("HTTPS_PROXY")
        }

    def speak(self, text, output_path, speaker=None, rate=0, *args, **kwargs):
        """
        Convert text to speech

        Args:
            text: Text to convert
            output_path: Path to save the audio file
            speaker: Speaker voice to use
            rate: Speech rate
        """
        raise NotImplementedError("speak() must be implemented")

    def transcribe(self, speech_path, output_path, output_format="txt"):
        """
        Convert speech to text

        Args:
            speech_path: Path to the audio file
            output_path: Path to save the text
            output_format: Output format for the text
        """
        raise NotImplementedError("transcribe() must be implemented")


class SpeechProvider:
    def __init__(self, provider_name: str, config: BaseSpeechConfig):
        """
        Initialize speech provider

        Args:
            provider_name: Name of the provider to use
            config: Provider specific configuration

        Raises:
            ValueError: If provider is not found or configuration is invalid
        """
        self.provider_name = provider_name

        if provider_name not in SPEECH_PROVIDERS:
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available providers: {list(SPEECH_PROVIDERS.keys())}"
            )

        # Create provider instance with config
        self.api = SPEECH_PROVIDERS[provider_name](config)

    def _ensure_dir(self, dir_path: str) -> str:
        """Ensure directory exists"""
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path

    def _setup_cache_dir(self, base_dir: str, name: str) -> Tuple[str, str]:
        """Setup cache directory and return paths"""
        cache_dir = os.path.join(base_dir, CACHE_DIR, name)
        self._ensure_dir(cache_dir)
        cache_info_path = os.path.join(cache_dir, CACHE_INFO_FILENAME)
        return cache_dir, cache_info_path

    def _load_cache(self, cache_path: str, use_cache: bool = True) -> dict:
        """Load cache from file"""
        if not use_cache or not os.path.exists(cache_path):
            return {}

        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            logging.info(f"Loaded cache with {len(cache)} entries")
            return cache
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return {}

    def _save_cache(
        self,
        cache_data: dict,
        cache_path: str,
        use_cache: bool = True
    ):
        """Save cache to file"""
        if not use_cache:
            return

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logging.info(f"Cache saved to {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate MD5 hash of content"""
        return hashlib.md5(content).hexdigest()

    def _split_text(self, text: str) -> List[str]:
        """Split text into sentences"""
        splitter = TokenSplitter()
        paragraphs = splitter.split_text(text)
        return [
            sentence
            for para in paragraphs
            for sentence in para.sentences
        ]

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        use_cache: bool = True,
        speaker: Optional[str] = None,
        rate: int = 0
    ) -> bool:
        """Generate speech from text"""
        if not text or not output_path:
            raise ValueError("Text and output path are required")

        try:
            # Setup paths
            output_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            name, _ = os.path.splitext(filename)

            logging.debug(f"Output directory: {output_dir}")
            logging.debug(f"Filename: {filename}")
            logging.debug(f"Base name: {name}")

            self._ensure_dir(output_dir)
            temp_dir = self._ensure_dir(
                os.path.join(output_dir, f"{name}_temp")
            )
            _, cache_path = self._setup_cache_dir(output_dir, name)

            # Process text
            segments = self._split_text(text)
            if not segments:
                raise ValueError("No segments found after splitting text")
            logging.debug(f"Split text into {len(segments)} segments")

            cache = self._load_cache(cache_path, use_cache)

            # Process segments
            output_audio = AudioSegment.empty()
            stats = {'total': len(segments), 'cached': 0, 'processed': 0}

            for idx, segment in enumerate(segments, 1):
                if not segment:
                    logging.debug(f"Skipping empty segment at index {idx}")
                    continue

                logging.debug(f"Processing segment {idx}/{len(segments)}")
                result = self._process_tts_segment(
                    segment, idx, temp_dir, cache,
                    use_cache, speaker, rate, stats
                )
                output_audio += result

            # Export final audio
            output_audio.export(output_path, format="mp3", bitrate=BITRATE)
            logging.info(f"Exported audio to {output_path}")

            # Update cache and cleanup
            if use_cache:
                self._save_cache(cache, cache_path)

            self._log_tts_stats(stats, output_path)
            return True

        except Exception as e:
            logging.error(f"Text to speech conversion failed: {e}")
            logging.debug("Error details:", exc_info=True)
            raise

    def _process_tts_segment(
        self,
        text: str,
        idx: int,
        temp_dir: str,
        cache: dict,
        use_cache: bool,
        speaker: Optional[str],
        rate: int,
        stats: dict
    ) -> AudioSegment:
        """Process single text segment for TTS"""
        text_hash = self._calculate_hash(text.encode())

        # Check cache
        if use_cache and text_hash in cache:
            stats['cached'] += 1
            logging.info(f"Using cached audio for segment {idx}")
            return AudioSegment.from_mp3(cache[text_hash]['audio_path'])

        # Generate new audio
        wav_path = os.path.join(temp_dir, f"segment_{idx}.wav")
        mp3_path = os.path.join(temp_dir, f"segment_{idx}.mp3")

        logging.debug(f"Speak insance {self.api.speak}")
        self.api.speak(text, wav_path, speaker=speaker, rate=rate)

        # Convert to MP3
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=BITRATE)

        # Update cache
        if use_cache:
            cache[text_hash] = {
                'text': text,
                'audio_path': mp3_path,
                'timestamp': datetime.now().isoformat()
            }

        stats['processed'] += 1
        logging.info(
            f"Processed segment {idx}/{stats['total']} "
            f"({stats['processed']/stats['total']*100:.1f}%)"
        )

        return audio

    def _log_tts_stats(self, stats: dict, output_path: str):
        """Log TTS statistics"""
        logging.info("Text to speech conversion completed:")
        logging.info(f"- Total segments: {stats['total']}")
        logging.info(f"- Cached segments: {stats['cached']}")
        logging.info(f"- Processed segments: {stats['processed']}")
        logging.info(f"- Output file: {output_path}")
        logging.info(f"- File size: {os.path.getsize(output_path)} bytes")

    def _preprocess_audio(self, speech_path: str, temp_dir: str) -> Tuple[str, bool]:
        """
        Preprocess audio file by converting to MP3 format if needed

        Args:
            speech_path: Path to the input audio file
            temp_dir: Directory for temporary files

        Returns:
            Tuple[str, bool]: (processed file path, whether conversion
            was performed)
        """
        supported_formats = {'.mp3'}
        file_ext = os.path.splitext(speech_path)[1].lower()

        if file_ext in supported_formats:
            return speech_path, False

        try:
            logging.info(f"Converting audio from {file_ext} to MP3 format")
            temp_mp3 = os.path.join(temp_dir, "converted_audio.mp3")

            audio = AudioSegment.from_file(speech_path)
            audio.export(temp_mp3, format="mp3", bitrate=BITRATE)

            logging.info("Audio conversion completed successfully")
            return temp_mp3, True
        except Exception as e:
            logging.error(f"Audio conversion failed: {str(e)}")
            error_msg = f"Failed to convert audio from {file_ext} to MP3: {str(e)}"
            raise ValueError(error_msg)

    def speech_to_text(
        self,
        speech_path: str,
        output_dir: str,
        output_format: str = "txt",
        use_cache: bool = True,
        min_silence_len: int = 1000,
        silence_thresh: int = -40,
        keep_silence: int = 500,
        chunk_size_limit: int = 10 * 1024 * 1024
    ) -> str:
        """Transcribe speech to text with automatic audio segmentation
        and caching.

        Args:
            speech_path: Path for speech to transcribe
            output_dir: Output directory for text
            output_format: Output format (default: txt)
            use_cache: Whether to use cache for processed chunks
                        (default: True)
            min_silence_len: Minimum length of silence for splitting (ms)
            silence_thresh: Silence threshold in dB
            keep_silence: Amount of silence to keep at chunk boundaries (ms)
            chunk_size_limit: Maximum size of each audio chunk in bytes

        Returns:
            str: Path to the output text file
        """
        self._validate_inputs(speech_path, output_dir)
        output_path, speech_extname, cache_info_path = self._setup_paths(
            speech_path, output_dir, output_format
        )
        cache_info = self._load_cache_info(use_cache, cache_info_path)

        temp_dir = tempfile.mkdtemp()
        try:
            processed_path, was_converted = self._preprocess_audio(
                speech_path, temp_dir
            )

            logging.info(f"Starting transcription of {speech_path}")
            audio = AudioSegment.from_file(processed_path)
            duration = len(audio) / 1000
            logging.info(f"Audio duration: {duration:.2f} seconds")

            chunks = self._split_audio_on_silence(
                audio, min_silence_len, silence_thresh, keep_silence
            )
            logging.info(f"Split audio into {len(chunks)} chunks")

            original_filename = os.path.basename(speech_path)

            # Always use WAV format for processing
            all_text, cached_chunks, processed_chunks = self._process_chunks(
                chunks=chunks,
                speech_extname='.wav',
                output_format=output_format,
                use_cache=use_cache,
                cache_info=cache_info,
                chunk_size_limit=chunk_size_limit,
                original_filename=original_filename
            )

            self._save_cache_info(use_cache, cache_info, cache_info_path)
            final_text = "\n".join(all_text)
            self._write_final_output(final_text, output_path)
            self._log_final_statistics(cached_chunks, processed_chunks,
                                       final_text, output_path)

            return output_path
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _validate_inputs(self, speech_path: str, output_dir: str):
        """Validate input paths."""
        if not os.path.exists(speech_path):
            raise ValueError(f"Speech file not found: {speech_path}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _setup_paths(self, speech_path: str, output_dir: str,
                     output_format: str):
        """Setup output paths and cache directory."""
        speech_file = os.path.basename(speech_path)
        speech_filename, speech_extname = os.path.splitext(speech_file)
        output_filename = f"{speech_filename}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        cache_base = os.path.join(output_dir, ".cache")
        cache_dir = os.path.join(cache_base, speech_filename)
        cache_info_path = os.path.join(cache_dir, "cache_info.json")
        os.makedirs(cache_dir, exist_ok=True)

        return output_path, speech_extname, cache_info_path

    def _load_cache_info(self, use_cache: bool, cache_info_path: str):
        """Load cache information if it exists."""
        cache_info = {}
        if use_cache and os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, 'r') as f:
                    cache_info = json.load(f)
                logging.info(f"Loaded cache info with {len(cache_info)} "
                             "entries")
            except Exception as e:
                logging.warning(f"Failed to load cache info: {e}")
                cache_info = {}
        return cache_info

    def _split_audio_on_silence(self, audio, min_silence_len,
                                 silence_thresh, keep_silence):
        """
        Split audio into chunks based on silence.

        This function analyzes the audio and divides it into smaller
        segments whenever a period of silence is detected. The parameters
        allow for customization of what constitutes silence and how much
        silence to retain in the resulting chunks.

        Args:
            audio: The audio segment to be split.
            min_silence_len: Minimum length of silence (in milliseconds)
                             that will be used to define a split point.
            silence_thresh: The silence threshold, below which the audio
                            is considered silent.
            keep_silence: Amount of silence to keep at the beginning and
                          end of each chunk.
        """
        return split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

    def _process_chunks(
        self,
        chunks,
        speech_extname,
        output_format,
        use_cache,
        cache_info,
        chunk_size_limit,
        original_filename: str
    ):
        """Process each audio chunk and return the transcribed text."""
        temp_dir = tempfile.mkdtemp()
        stats = self._initialize_processing_stats(len(chunks))
        safe_filename = self._get_safe_filename(original_filename)

        try:
            if output_format == 'srt':
                subs = pysubs2.SSAFile()
                all_text = self._process_chunks_to_srt(
                    chunks=chunks,
                    temp_dir=temp_dir,
                    safe_filename=safe_filename,
                    speech_extname=speech_extname,
                    use_cache=use_cache,
                    cache_info=cache_info,
                    chunk_size_limit=chunk_size_limit,
                    stats=stats,
                    subs=subs
                )
            else:
                all_text = self._process_chunks_to_text(
                    chunks=chunks,
                    temp_dir=temp_dir,
                    safe_filename=safe_filename,
                    speech_extname=speech_extname,
                    use_cache=use_cache,
                    cache_info=cache_info,
                    chunk_size_limit=chunk_size_limit,
                    stats=stats
                )

            return all_text, stats['cached'], stats['processed']
        finally:
            self._cleanup_temp_dir(temp_dir)

    def _process_chunks_to_srt(
        self,
        chunks,
        temp_dir: str,
        safe_filename: str,
        speech_extname: str,
        use_cache: bool,
        cache_info: dict,
        chunk_size_limit: int,
        stats: dict,
        subs: pysubs2.SSAFile
    ) -> List[str]:
        """Process chunks and convert to SRT format."""
        current_time = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_paths = self._setup_chunk_paths(
                temp_dir, safe_filename, i,
                speech_extname, "txt"
            )

            chunk_duration = len(chunk)
            chunk.export(chunk_paths['audio'], format=speech_extname[1:])
            chunk_hash = self._calculate_chunk_hash(chunk_paths['audio'])

            text = self._get_chunk_text(
                chunk_hash=chunk_hash,
                chunk_paths=chunk_paths,
                chunk_index=i,
                chunk_duration=chunk_duration,
                use_cache=use_cache,
                cache_info=cache_info,
                chunk_size_limit=chunk_size_limit,
                stats=stats
            )

            if text:
                subs.append(pysubs2.SSAEvent(
                    start=current_time,
                    end=current_time + chunk_duration,
                    text=text
                ))

            current_time += chunk_duration
            self._cleanup_chunk_files(chunk_paths)

        return [subs.to_string('srt')]

    def _process_chunks_to_text(
        self,
        chunks,
        temp_dir: str,
        safe_filename: str,
        speech_extname: str,
        use_cache: bool,
        cache_info: dict,
        chunk_size_limit: int,
        stats: dict
    ) -> List[str]:
        """Process chunks and convert to plain text."""
        all_text = []

        for i, chunk in enumerate(chunks, 1):
            chunk_paths = self._setup_chunk_paths(
                temp_dir, safe_filename, i,
                speech_extname, "txt"
            )

            chunk_duration = len(chunk)
            chunk.export(chunk_paths['audio'], format=speech_extname[1:])
            chunk_hash = self._calculate_chunk_hash(chunk_paths['audio'])

            text = self._get_chunk_text(
                chunk_hash=chunk_hash,
                chunk_paths=chunk_paths,
                chunk_index=i,
                chunk_duration=chunk_duration,
                use_cache=use_cache,
                cache_info=cache_info,
                chunk_size_limit=chunk_size_limit,
                stats=stats
            )

            if text:
                all_text.append(text)

            self._cleanup_chunk_files(chunk_paths)

        return all_text

    def _get_chunk_text(
        self,
        chunk_hash: str,
        chunk_paths: dict,
        chunk_index: int,
        chunk_duration: int,
        use_cache: bool,
        cache_info: dict,
        chunk_size_limit: int,
        stats: dict
    ) -> Optional[str]:
        """Get text from cache or process new chunk."""
        if use_cache and chunk_hash in cache_info:
            cached_text = cache_info[chunk_hash].get('text')
            if cached_text:
                stats['cached'] += 1
                logging.info(
                    f"Using cached text for chunk {chunk_index}/{stats['total']}"
                )
                return cached_text

        self._log_chunk_processing(chunk_index, chunk_duration, stats['total'])
        self._check_chunk_size(chunk_paths['audio'], chunk_size_limit)

        processed_text = self._transcribe_chunk(
            chunk_paths['audio'],
            chunk_paths['text'],
            use_cache,
            chunk_hash
        )

        if processed_text:
            stats['processed'] += 1
            if use_cache:
                self._update_cache_entry(
                    cache_info, chunk_hash, processed_text,
                    chunk_duration
                )

        return processed_text

    def _write_final_output(self, final_text: str, output_path: str):
        """Write the final transcribed text to the output file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.srt'):
                f.write(final_text)
            else:
                f.write(final_text.replace('\n\n', ' '))

    def _log_final_statistics(self, cached_chunks: int,
                              processed_chunks: int, final_text: str,
                              output_path: str):
        """Log the final statistics of the transcription process."""
        logging.info("Transcription completed successfully:")
        logging.info(f"- Total chunks: {len(final_text)}")
        logging.info(f"- Cached chunks used: {cached_chunks}")
        logging.info(f"- Newly processed chunks: {processed_chunks}")
        logging.info(f"- Total text length: {len(final_text)} characters")
        logging.info(f"- Output file: {output_path}")

    def _initialize_processing_stats(self, total_chunks: int) -> dict:
        """Initialize statistics for chunk processing."""
        return {
            'total': total_chunks,
            'cached': 0,
            'processed': 0,
            'current_time': 0
        }

    def _get_safe_filename(self, original_filename: str) -> str:
        """Convert filename to safe format."""
        base_filename = os.path.splitext(original_filename)[0]
        return "".join(c if c.isalnum() else '_' for c in base_filename)

    def _setup_chunk_paths(
        self,
        temp_dir: str,
        safe_filename: str,
        chunk_index: int,
        speech_extname: str,
        output_format: str
    ) -> dict:
        """Setup paths for chunk processing."""
        chunk_name = f"{safe_filename}_chunk_{chunk_index}"
        return {
            'audio': os.path.join(temp_dir, f"{chunk_name}{speech_extname}"),
            'text': os.path.join(temp_dir, f"{chunk_name}.{output_format}")
        }

    def _update_cache_entry(
        self,
        cache_info: dict,
        chunk_hash: str,
        text: str,
        duration: int
    ):
        """Update cache with new entry."""
        cache_info[chunk_hash] = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'duration': duration / 1000
        }

    def _cleanup_chunk_files(self, chunk_paths: dict):
        """Clean up temporary chunk files."""
        for path in chunk_paths.values():
            if os.path.exists(path):
                os.remove(path)

    def _cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _calculate_chunk_hash(self, chunk_path: str):
        """Calculate the MD5 hash of the audio chunk."""
        with open(chunk_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _transcribe_chunk(self, chunk_path: str, temp_output: str,
                          use_cache: bool, chunk_hash: str):
        """Transcribe a single audio chunk."""
        try:
            self.api.transcribe(chunk_path, temp_output)
            with open(temp_output, 'r', encoding='utf-8') as f:
                chunk_text = f.read().strip()
                return chunk_text
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            return None

    def _log_chunk_processing(
        self,
        chunk_index: int,
        chunk_duration: int,
        total_chunks: int
    ):
        """Log chunk processing progress."""
        logging.info(
            f"Processing chunk {chunk_index}/{total_chunks} "
            f"(Duration: {chunk_duration/1000:.2f}s)"
        )

    def _check_chunk_size(self, chunk_path: str, size_limit: int):
        """Check if chunk size exceeds limit."""
        if os.path.getsize(chunk_path) > size_limit:
            logging.warning(
                f"Chunk exceeds size limit of "
                f"{size_limit/1024/1024:.1f}MB"
            )

    def _save_cache_info(self, use_cache: bool, cache_info: dict,
                         cache_info_path: str):
        """Save updated cache information."""
        if use_cache:
            try:
                with open(cache_info_path, 'w') as f:
                    json.dump(cache_info, f, indent=2)
                logging.info(f"Updated cache info saved to {cache_info_path}")
            except Exception as e:
                logging.warning(f"Failed to save cache info: {e}")
