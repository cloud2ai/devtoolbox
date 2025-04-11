"""
Whisper Speech Recognition Provider

This module implements the Whisper speech recognition model as a provider.
Whisper is a general-purpose speech recognition model developed by OpenAI,
trained on a large dataset of diverse audio. It supports:
- Multilingual speech recognition
- Speech translation
- Language identification

Note: Whisper only supports speech-to-text (transcription),
      text-to-speech is not supported.
"""

import logging
import os
from typing import Optional, List, Any
from dataclasses import dataclass, field
import whisper

from .provider import (
    BaseSpeechProvider,
    BaseSpeechConfig,
    register_provider,
    register_config
)

logger = logging.getLogger(__name__)


@register_config("whisper")
@dataclass
class WhisperConfig(BaseSpeechConfig):
    """Whisper Speech Service configuration.

    This class automatically loads configuration from environment variables
    if not provided during initialization.

    Attributes:
        model_name: Model size to use (tiny, base, small, medium, large)
        language: Language code for transcription (e.g., 'zh', 'en')
        task: Task type: 'transcribe' for transcription, 'translate' for
            translation
        fp16: Whether to use half-precision (must be False on CPU)
        temperature: Sampling temperature (0.0 for deterministic output)
        best_of: Number of candidates when temperature is 0.0
        beam_size: Number of beams in beam search
        patience: Beam search patience factor
        suppress_tokens: List of token IDs to suppress during generation
        initial_prompt: Initial text prompt to guide transcription
        condition_on_previous_text: Whether to condition on previous text
            context
        word_timestamps: Whether to include word-level timestamps
        verbose: Whether to print progress information
        supported_languages: List of supported language codes

    Environment variables:
        WHISPER_MODEL: Model size (default: base)
        WHISPER_LANGUAGE: Language code (default: en)
        WHISPER_TASK: Task type (default: transcribe)
        WHISPER_FP16: Use half-precision (default: False)
        WHISPER_TEMPERATURE: Sampling temperature (default: 0.0)
        WHISPER_BEST_OF: Number of candidates (default: 5)
        WHISPER_BEAM_SIZE: Beam size (default: 5)
        WHISPER_PATIENCE: Beam search patience (default: 1.0)
        WHISPER_WORD_TIMESTAMPS: Include word timestamps (default: False)
        WHISPER_VERBOSE: Print progress info (default: True)
    """
    model_name: str = field(
        default_factory=lambda: os.environ.get('WHISPER_MODEL', 'base')
    )
    language: str = field(
        default_factory=lambda: os.environ.get('WHISPER_LANGUAGE', 'en')
    )
    task: str = field(
        default_factory=lambda: os.environ.get('WHISPER_TASK', 'transcribe')
    )
    fp16: bool = field(
        default_factory=lambda: os.environ.get(
            'WHISPER_FP16', 'False'
        ).lower() == 'true'
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('WHISPER_TEMPERATURE', '0.0')
        )
    )
    best_of: int = field(
        default_factory=lambda: int(
            os.environ.get('WHISPER_BEST_OF', '5')
        )
    )
    beam_size: int = field(
        default_factory=lambda: int(
            os.environ.get('WHISPER_BEAM_SIZE', '5')
        )
    )
    patience: float = field(
        default_factory=lambda: float(
            os.environ.get('WHISPER_PATIENCE', '1.0')
        )
    )
    suppress_tokens: List[int] = field(default_factory=list)
    initial_prompt: str = field(default="")
    condition_on_previous_text: bool = field(default=True)
    word_timestamps: bool = field(
        default_factory=lambda: os.environ.get(
            'WHISPER_WORD_TIMESTAMPS', 'False'
        ).lower() == 'true'
    )
    verbose: bool = field(
        default_factory=lambda: os.environ.get(
            'WHISPER_VERBOSE', 'True'
        ).lower() == 'true'
    )
    supported_languages: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        logger.info(f"Whisper model: {self.model_name}")
        logger.info(f"Language: {self.language}")
        logger.info(f"Task: {self.task}")
        logger.info(f"FP16: {self.fp16}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Best of: {self.best_of}")
        logger.info(f"Beam size: {self.beam_size}")
        logger.info(f"Patience: {self.patience}")
        logger.info(f"Word timestamps: {self.word_timestamps}")
        logger.info(f"Verbose: {self.verbose}")

    def _validate_config(self):
        """Validate Whisper configuration."""
        # Validate model name
        available_models = ["tiny", "base", "small", "medium", "large"]
        if self.model_name not in available_models:
            msg = f"Invalid model name. Must be one of: {available_models}"
            raise ValueError(msg)

        # Validate task type
        if self.task not in ["transcribe", "translate"]:
            raise ValueError("Task must be either 'transcribe' or 'translate'")

        # Validate temperature
        if not isinstance(self.temperature, (float, list)):
            raise ValueError("Temperature must be float or list of floats")

        # Validate beam size
        if self.beam_size < 1:
            raise ValueError("Beam size must be positive")

        # Validate best_of
        if self.best_of < 1:
            raise ValueError("best_of must be positive")

        # Validate patience
        if self.patience <= 0:
            raise ValueError("Patience must be positive")

    @classmethod
    def from_env(cls) -> 'WhisperConfig':
        """Create Whisper configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


class WhisperError(Exception):
    """Base exception for Whisper-related errors"""
    pass


class WhisperModelError(WhisperError):
    """Raised when there is an error with the Whisper model"""
    pass


class WhisperTranscriptionError(WhisperError):
    """Raised when there is an error during transcription"""
    pass


@register_provider("whisper")
class WhisperProvider(BaseSpeechProvider):
    """Whisper speech provider implementation."""

    def __init__(self, config: WhisperConfig):
        """Initialize Whisper provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self._model = None

    @property
    def model(self) -> Any:
        """Get the whisper model.

        Returns:
            The loaded Whisper model
        """
        if self._model is None:
            self._model = whisper.load_model(self.config.model_name)
        return self._model

    def transcribe(self, speech_path: str, output_path: str) -> str:
        """Transcribe speech to text.

        Args:
            speech_path: Path to the speech file
            output_path: Path to save the transcribed text

        Returns:
            Path to the transcribed text file
        """
        result = self.model.transcribe(
            speech_path,
            language=self.config.language,
            task=self.config.task
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        return output_path

    def speak(
        self,
        text: str,
        output_path: str,
        speaker: Optional[str] = None,
        rate: float = 1.0
    ) -> str:
        """Text to speech is not supported by Whisper.

        Args:
            text: Text to convert to speech
            output_path: Path to save the speech file
            speaker: Voice to use (not supported)
            rate: Speech rate (not supported)

        Raises:
            NotImplementedError: Always raised as Whisper does not support TTS
        """
        raise NotImplementedError("Whisper does not support text-to-speech")

    def list_speakers(self) -> List[str]:
        """List available speakers.

        Returns:
            Empty list as Whisper does not support text-to-speech
        """
        return []
