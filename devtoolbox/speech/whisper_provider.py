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
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from faster_whisper import WhisperModel
import torch

from devtoolbox.speech.provider import (
    BaseSpeechProvider,
    register_provider,
    BaseSpeechConfig
)

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig(BaseSpeechConfig):
    """
    Whisper configuration settings

    Attributes:
        model_size: Size of the Whisper model
        device: Computing device for inference
        compute_type: Computation type for model
        language: Target language for transcription
        batch_size: Batch size for processing
        beam_size: Beam size for search
        temperature: Temperature for sampling

    Environment variables:
        WHISPER_MODEL_SIZE: Model size (tiny, base, small, medium, large)
        WHISPER_DEVICE: Computing device (cpu, cuda)
        WHISPER_COMPUTE_TYPE: Compute type (int8, float16, float32)
        WHISPER_LANGUAGE: Target language code
        WHISPER_BATCH_SIZE: Batch size for processing
        WHISPER_BEAM_SIZE: Beam size for search
        WHISPER_TEMPERATURE: Temperature for sampling
    """
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    batch_size: int = 16
    beam_size: int = 5
    temperature: float = 0.0

    VALID_SIZES = ["tiny", "base", "small", "medium", "large"]
    VALID_DEVICES = ["cpu", "cuda"]
    VALID_COMPUTE_TYPES = ["int8", "float16", "float32"]

    @classmethod
    def from_env(cls) -> 'WhisperConfig':
        """Create Whisper configuration from environment variables"""
        return cls(
            model_size=os.environ.get('WHISPER_MODEL_SIZE', 'base'),
            device=os.environ.get('WHISPER_DEVICE', 'cpu'),
            compute_type=os.environ.get('WHISPER_COMPUTE_TYPE', 'int8'),
            language=os.environ.get('WHISPER_LANGUAGE'),
            batch_size=int(os.environ.get('WHISPER_BATCH_SIZE', '16')),
            beam_size=int(os.environ.get('WHISPER_BEAM_SIZE', '5')),
            temperature=float(os.environ.get('WHISPER_TEMPERATURE', '0.0'))
        )

    def validate(self):
        """
        Validate Whisper configuration

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.model_size not in self.VALID_SIZES:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                f"Must be one of: {', '.join(self.VALID_SIZES)}"
            )

        if self.device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device: {self.device}. "
                f"Must be one of: {', '.join(self.VALID_DEVICES)}"
            )

        if self.compute_type not in self.VALID_COMPUTE_TYPES:
            raise ValueError(
                f"Invalid compute_type: {self.compute_type}. "
                f"Must be one of: {', '.join(self.VALID_COMPUTE_TYPES)}"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but torch.cuda is not available"
            )


@register_provider("whisper")
class WhisperSpeechProvider(BaseSpeechProvider):
    """
    Whisper speech recognition provider

    This provider implements speech-to-text functionality using the
    Whisper model. Text-to-speech is not supported.
    """

    def __init__(self, config: WhisperConfig):
        """
        Initialize Whisper provider

        Args:
            config: Whisper configuration settings
        """
        super().__init__(config)
        self.config = config
        self._model = None

        logger.info(
            f"Initializing Whisper provider (model: {config.model_size}, "
            f"device: {config.device}, compute_type: {config.compute_type})"
        )

    @property
    def model(self) -> WhisperModel:
        """
        Lazy loading of Whisper model

        Returns:
            Loaded Whisper model instance
        """
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            self._model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
        return self._model

    def speak(self, *args, **kwargs):
        """Text-to-speech is not supported by Whisper"""
        raise NotImplementedError(
            "Text-to-speech functionality is not supported by Whisper"
        )

    def transcribe(
        self,
        speech_path: str,
        output_path: str,
        output_format: str = "txt"
    ) -> str:
        """
        Transcribe audio using Whisper

        Args:
            speech_path: Path to the audio file
            output_path: Path to save the transcribed text
            output_format: Output format (handled by parent SpeechProvider)

        Returns:
            Path to the output file

        Raises:
            FileNotFoundError: If speech file doesn't exist
        """
        speech_path = Path(speech_path)
        if not speech_path.exists():
            raise FileNotFoundError(f"Audio file not found: {speech_path}")

        logger.info(f"Transcribing audio: {speech_path}")

        segments, info = self.model.transcribe(
            str(speech_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            temperature=self.config.temperature
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding='utf-8')

        logger.info(f"Transcription saved to: {output_path}")
        return str(output_path)
