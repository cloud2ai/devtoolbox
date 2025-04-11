"""Base provider for speech services.

This module provides the base infrastructure for implementing speech providers.
It defines the base classes for configuration and providers.
"""

import os
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Provider registry
_registered_providers: Dict[str, type] = {}
_registered_configs: Dict[str, type] = {}


def register_provider(name: str):
    """Decorator to register a speech provider.

    Args:
        name: The name of the provider to register.
    """
    def decorator(cls):
        _registered_providers[name] = cls
        logger.debug(f"Registered provider: {name}")
        return cls
    return decorator


def register_config(name: str):
    """Decorator to register a speech config.

    Args:
        name: The name of the config to register.
    """
    def decorator(cls):
        _registered_configs[name] = cls
        logger.debug(f"Registered config: {name}")
        return cls
    return decorator


@dataclass
class BaseSpeechConfig:
    """Base configuration for speech providers.

    This class should be inherited by all provider-specific configurations.
    Each provider should implement its own configuration class with
    necessary settings.
    """

    def validate(self) -> bool:
        """Validate configuration parameters.

        This method should be implemented by each provider's config class
        to validate its specific settings.

        Raises:
            ValueError: If configuration is invalid.
        """
        raise NotImplementedError("validate() must be implemented")

    @classmethod
    def from_env(cls) -> 'BaseSpeechConfig':
        """Create configuration from environment variables.

        This method should be implemented by each provider's config class
        to load settings from environment variables.

        Returns:
            BaseSpeechConfig: Configuration instance.
        """
        raise NotImplementedError("from_env() must be implemented")


class BaseSpeechProvider(ABC):
    """Base class for speech providers.

    This class defines the interface that all speech providers must implement.
    It provides basic functionality for text-to-speech and speech-to-text
    operations.
    """

    def __init__(self, config: BaseSpeechConfig):
        """Initialize the provider.

        Args:
            config: Provider configuration.
        """
        logger.info(f"Initializing {self.__class__.__name__}")
        self.config = config
        self.proxies = self._get_proxies()
        logger.debug(f"Using proxies: {self.proxies}")

    def _get_proxies(self) -> Dict[str, Optional[str]]:
        """Get proxy settings from environment.

        Returns:
            Dict[str, Optional[str]]: Proxy settings.
        """
        proxies = {
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY")
        }
        logger.debug(f"Proxy settings: {proxies}")
        return proxies

    @abstractmethod
    def speak(
        self,
        text: str,
        save_path: str,
        speaker: Optional[str] = None,
        rate: int = 0,
        **kwargs
    ) -> str:
        """Convert text to speech.

        Args:
            text: Text to convert.
            save_path: Path to save the audio file.
            speaker: Voice to use for synthesis.
            rate: Speech rate adjustment.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: Path to the saved audio file.

        Raises:
            NotImplementedError: If not implemented by provider.
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        save_path: str,
        output_format: str = "txt",
        **kwargs
    ) -> str:
        """Convert speech to text.

        Args:
            audio_path: Path to the audio file.
            save_path: Path to save the transcription.
            output_format: Output format (txt or srt).
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: Path to the transcription file.

        Raises:
            NotImplementedError: If not implemented by provider.
        """
        pass

    @abstractmethod
    def list_speakers(self) -> List[str]:
        """List available speakers/voices.

        Returns:
            List[str]: List of available speakers.

        Raises:
            NotImplementedError: If not implemented by provider.
        """
        pass
