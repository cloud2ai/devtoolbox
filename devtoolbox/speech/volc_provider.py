"""Volc speech provider implementation.

This module provides the Volc speech provider implementation using
Volcengine's speech service.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
import os
import time
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from devtoolbox.speech.provider import (
    BaseSpeechConfig,
    BaseSpeechProvider,
    register_provider,
    register_config,
)

logger = logging.getLogger(__name__)


@register_config('volc')
@dataclass
class VolcConfig(BaseSpeechConfig):
    """Volcengine Speech Service configuration.

    This class automatically loads configuration from environment variables
    if not provided during initialization.

    Attributes:
        access_key: Volcengine access key
        secret_key: Volcengine secret key
        app_id: Volcengine app ID
        voice_id: Voice ID for text-to-speech
        language: Speech language (default: zh-CN)
        rate: Speech rate (default: 0)
        supported_languages: List of supported language codes

    Environment variables:
        VOLC_ACCESS_KEY: Volcengine access key
        VOLC_SECRET_KEY: Volcengine secret key
        VOLC_APP_ID: Volcengine app ID
        VOLC_VOICE_ID: Voice ID for text-to-speech
        VOLC_LANGUAGE: Speech language (default: zh-CN)
        VOLC_RATE: Speech rate (default: 0)
    """
    access_key: Optional[str] = field(
        default_factory=lambda: os.environ.get('VOLC_ACCESS_KEY')
    )
    secret_key: Optional[str] = field(
        default_factory=lambda: os.environ.get('VOLC_SECRET_KEY')
    )
    app_id: Optional[str] = field(
        default_factory=lambda: os.environ.get('VOLC_APP_ID')
    )
    voice_id: Optional[str] = field(
        default_factory=lambda: os.environ.get('VOLC_VOICE_ID')
    )
    language: str = field(
        default_factory=lambda: os.environ.get('VOLC_LANGUAGE', 'zh-CN')
    )
    rate: float = field(
        default_factory=lambda: float(
            os.environ.get('VOLC_RATE', '0')
        )
    )
    supported_languages: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        logger.info(f"Volc app_id: {self.app_id}")
        logger.info(f"Language: {self.language}")
        logger.info(f"Voice ID: {self.voice_id}")
        logger.info(f"Rate: {self.rate}")

    def _validate_config(self):
        """Validate Volc configuration."""
        # Validate required credentials
        if not self.access_key:
            raise ValueError(
                "access_key is required. Set it either in constructor "
                "or through VOLC_ACCESS_KEY environment variable"
            )
        if not self.secret_key:
            raise ValueError(
                "secret_key is required. Set it either in constructor "
                "or through VOLC_SECRET_KEY environment variable"
            )
        if not self.app_id:
            raise ValueError(
                "app_id is required. Set it either in constructor "
                "or through VOLC_APP_ID environment variable"
            )

        # Validate language
        if not self.language:
            raise ValueError("Language is required")

        # Validate rate
        if not isinstance(self.rate, (int, float)):
            raise ValueError("Rate must be a number")
        if self.rate < -1 or self.rate > 1:
            raise ValueError("Rate must be between -1 and 1")

    @classmethod
    def from_env(cls) -> 'VolcConfig':
        """Create Volc configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


class VolcError(Exception):
    """Base exception for Volc-related errors"""
    pass


class VolcRateLimitError(VolcError):
    """Raised when Volc rate limit is exceeded"""
    pass


@register_provider('VolcProvider')
class VolcProvider(BaseSpeechProvider):
    """
    Volcengine speech provider implementation

    Supports text-to-speech functionality using Volcengine's speech service.
    """

    def __init__(self, config: VolcConfig):
        """
        Initialize Volc provider

        Args:
            config: Volc configuration settings
        """
        super().__init__(config)
        self.config = config
        self._base_url = "https://open.volcengineapi.com/v2/tts"
        self._token = None
        self._token_expiry = 0

        logger.info(
            f"Initializing Volc provider (app_id: {config.app_id}, "
            f"language: {config.language})"
        )

    def _get_token(self) -> str:
        """
        Get or refresh authentication token

        Returns:
            str: Authentication token

        Raises:
            VolcError: If token retrieval fails
        """
        if self._token and time.time() < self._token_expiry:
            return self._token

        try:
            response = requests.post(
                "https://open.volcengineapi.com/v2/auth",
                json={
                    "access_key": self.config.access_key,
                    "secret_key": self.config.secret_key
                }
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["token"]
            self._token_expiry = time.time() + data["expires_in"] - 60
            return self._token
        except requests.exceptions.RequestException as e:
            raise VolcError(f"Failed to get token: {str(e)}")

    @retry(
        retry=retry_if_exception_type(VolcRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def speak(
        self,
        text: str,
        save_path: str,
        speaker: Optional[str] = None,
        rate: int = 0,
        **kwargs
    ) -> str:
        """
        Convert text to speech using Volcengine Speech Service

        Args:
            text: Text to convert
            save_path: Path to save the audio file
            speaker: Voice to use (default: config voice_id)
            rate: Speech rate adjustment (default: 0)
            **kwargs: Additional arguments

        Returns:
            str: Path to the saved audio file

        Raises:
            VolcError: If synthesis fails
        """
        if not text or not save_path:
            raise ValueError("Text and save_path are required")

        # Use configured voice if not specified
        voice_id = speaker or self.config.voice_id
        if not voice_id:
            raise ValueError("Voice ID is required")

        try:
            # Get authentication token
            token = self._get_token()

            # Prepare request
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {
                "app_id": self.config.app_id,
                "text": text,
                "voice_id": voice_id,
                "language": self.config.language,
                "rate": rate,
                "format": "mp3"
            }

            # Make request
            response = requests.post(
                self._base_url,
                headers=headers,
                json=data
            )

            # Handle response
            if response.status_code == 429:
                raise VolcRateLimitError("Rate limit exceeded")
            response.raise_for_status()

            # Save audio
            with open(save_path, 'wb') as f:
                f.write(response.content)

            return save_path

        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                raise VolcRateLimitError("Rate limit exceeded")
            raise VolcError(f"Synthesis failed: {str(e)}")

    def transcribe(
        self,
        audio_path: str,
        save_path: str,
        output_format: str = "txt",
        **kwargs
    ) -> str:
        """
        Convert speech to text using Volcengine Speech Service

        Note: This method is not implemented as Volcengine's speech service
        currently only supports text-to-speech functionality.

        Args:
            audio_path: Path to the audio file
            save_path: Path to save the transcription
            output_format: Output format (txt or srt)
            **kwargs: Additional arguments

        Returns:
            str: Path to the saved transcription file

        Raises:
            NotImplementedError: As this functionality is not supported
        """
        raise NotImplementedError(
            "Volcengine's speech service currently only supports "
            "text-to-speech functionality"
        )

    def list_speakers(self) -> List[str]:
        """
        List available speakers/voices

        Returns:
            List[str]: List of available voices
        """
        # Return a list of commonly used voices
        return [
            "zh-CN-Xiaoxiao",
            "zh-CN-Yunxi",
            "en-US-Jenny",
            "en-US-Guy",
            "es-ES-Alvaro",
            "es-ES-Elvira"
        ]