"""Azure speech provider implementation.

This module provides the Azure speech provider implementation using
Azure Cognitive Services Speech SDK.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
import os

import azure.cognitiveservices.speech as speechsdk
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


@register_config('azure')
@dataclass
class AzureConfig(BaseSpeechConfig):
    """Azure Speech Service configuration.

    This class automatically loads configuration from environment variables
    if not provided during initialization.

    Attributes:
        subscription_key: Azure subscription key
        service_region: Azure service region
        voice_name: Azure voice name
        language: Speech language (default: auto for auto-detection)
        rate: Speech rate (default: 0)
        supported_languages: List of supported languages for auto-detection
                           (default: Simplified Chinese and US English)
        ssml_template: SSML template for text-to-speech synthesis

    Supported Languages (by usage population):
        - "ar-AE": Arabic (UAE)
        - "de-DE": German (Germany)
        - "en-AU": English (Australia)
        - "en-CA": English (Canada)
        - "en-GB": English (UK)
        - "en-HK": English (Hong Kong)
        - "en-IE": English (Ireland)
        - "en-IN": English (India)
        - "en-US": English (US)
        - "es-ES": Spanish (Spain)
        - "es-MX": Spanish (Mexico)
        - "fr-CA": French (Canada)
        - "fr-FR": French (France)
        - "hi-IN": Hindi (India)
        - "it-IT": Italian (Italy)
        - "ja-JP": Japanese (Japan)
        - "ko-KR": Korean (Korea)
        - "nl-NL": Dutch (Netherlands)
        - "pl-PL": Polish (Poland)
        - "pt-BR": Portuguese (Brazil)
        - "pt-PT": Portuguese (Portugal)
        - "ru-RU": Russian (Russia)
        - "sv-SE": Swedish (Sweden)
        - "th-TH": Thai (Thailand)
        - "tr-TR": Turkish (Turkey)
        - "vi-VN": Vietnamese (Vietnam)
        - "zh-CN": Chinese (Mainland)
        - "zh-HK": Chinese (Hong Kong)
        - "zh-TW": Chinese (Taiwan)

    Environment variables:
        AZURE_SPEECH_KEY: Azure subscription key
        AZURE_SPEECH_REGION: Azure service region
        AZURE_SPEECH_VOICE: Azure voice name
        AZURE_SPEECH_LANGUAGE: Speech language (default: auto)
        AZURE_SPEECH_RATE: Speech rate (default: 0)
    """
    subscription_key: str = field(
        default_factory=lambda: os.environ.get('AZURE_SPEECH_KEY')
    )
    service_region: str = field(
        default_factory=lambda: os.environ.get('AZURE_SPEECH_REGION')
    )
    voice_name: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            'AZURE_SPEECH_VOICE', 'zh-CN-YunjianNeural'
        )
    )
    language: str = field(
        default_factory=lambda: os.environ.get('AZURE_SPEECH_LANGUAGE', 'auto')
    )
    rate: float = field(
        default_factory=lambda: float(
            os.environ.get('AZURE_SPEECH_RATE', '0.0')
        )
    )
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "zh-CN",  # Chinese (Mainland)
            "en-US",  # English (US)
        ]
    )
    ssml_template: str = field(
        default="""
<speak xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xmlns:emo="http://www.w3.org/2009/10/emotionml"
       version="1.0" xml:lang="zh-CN">
    <voice name="{speaker}">
        <prosody rate="{rate}%">{text}</prosody>
    </voice>
</speak>
"""
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        if self.subscription_key:
            logger.info("Azure Speech subscription key loaded from constructor")
        elif os.environ.get('AZURE_SPEECH_KEY'):
            logger.info(
                "Azure Speech subscription key loaded from environment variable"
            )
        else:
            logger.error(
                "Azure Speech subscription key not found in constructor "
                "or environment"
            )

        if self.service_region:
            logger.info("Azure Speech region loaded from constructor")
        elif os.environ.get('AZURE_SPEECH_REGION'):
            logger.info(
                "Azure Speech region loaded from environment variable"
            )
        else:
            logger.error(
                "Azure Speech region not found in constructor or environment"
            )

        logger.info(f"Voice name: {self.voice_name or 'Not set'}")
        logger.info(f"Language: {self.language}")
        logger.info(f"Speech rate: {self.rate}")
        logger.info(
            f"Active languages (max 4): {', '.join(self.supported_languages)}"
        )

    def _validate_config(self):
        """Validate Azure configuration."""
        if not self.subscription_key:
            raise ValueError(
                "subscription_key is required. Set it either in constructor "
                "or through AZURE_SPEECH_KEY environment variable"
            )
        if not self.service_region:
            raise ValueError(
                "service_region is required. Set it either in constructor "
                "or through AZURE_SPEECH_REGION environment variable"
            )

    @classmethod
    def from_env(cls) -> 'AzureConfig':
        """Create Azure configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


class AzureError(Exception):
    """Base exception for Azure-related errors"""
    pass


class AzureRateLimitError(AzureError):
    """Raised when Azure rate limit is exceeded"""
    pass


class AzureConfigError(AzureError):
    """Raised when Azure configuration is invalid"""
    pass


class AzureSynthesisError(AzureError):
    """Raised when speech synthesis fails"""
    pass


class AzureRecognitionError(AzureError):
    """Raised when speech recognition fails"""
    pass


@register_provider('AzureProvider')
class AzureProvider(BaseSpeechProvider):
    """
    Azure speech provider implementation

    Supports both text-to-speech and speech-to-text functionality using
    Azure Cognitive Services Speech SDK with auto language detection.
    """

    def __init__(self, config: AzureConfig):
        """
        Initialize Azure provider

        Args:
            config: Azure configuration settings
        """
        super().__init__(config)
        self.config = config
        self._speech_config = None
        self._speech_recognizer = None

        logger.info(
            "Initializing Azure provider "
            f"(region: {config.service_region}, "
            f"language: {config.language})"
        )

    @property
    def speech_config(self) -> speechsdk.SpeechConfig:
        """
        Lazy loading of Azure speech config

        Returns:
            Configured Azure speech config instance

        Raises:
            AzureConfigError: If speech config creation fails
        """
        if self._speech_config is None:
            try:
                self._speech_config = speechsdk.SpeechConfig(
                    subscription=self.config.subscription_key,
                    region=self.config.service_region
                )
                logger.debug("Azure speech config created successfully")
            except Exception as e:
                raise AzureConfigError(f"Failed to create speech config: {str(e)}")
        return self._speech_config

    def _handle_synthesis_result(
        self,
        result: speechsdk.SpeechSynthesisResult,
        save_path: str
    ) -> str:
        """
        Handle speech synthesis result

        Args:
            result: Speech synthesis result
            save_path: Path to save the audio file

        Returns:
            str: Path to the saved audio file

        Raises:
            AzureSynthesisError: If synthesis failed
        """
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            try:
                with open(save_path, 'wb') as f:
                    f.write(result.audio_data)
                logger.info(f"Audio saved successfully to {save_path}")
                return save_path
            except Exception as e:
                raise AzureSynthesisError(f"Failed to save audio: {str(e)}")

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            if (details.reason == speechsdk.CancellationReason.Error
                    and "429" in details.error_details):
                raise AzureRateLimitError("Rate limit exceeded")
            raise AzureSynthesisError(
                f"Synthesis canceled: {details.reason}. "
                f"Error details: {details.error_details}"
            )

        raise AzureSynthesisError(f"Synthesis failed: {result.reason}")

    @retry(
        retry=retry_if_exception_type(AzureRateLimitError),
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
        Convert text to speech using Azure Speech Service

        Args:
            text: Text to convert
            save_path: Path to save the audio file
            speaker: Voice to use (default: config voice_name)
            rate: Speech rate adjustment (default: 0)
            **kwargs: Additional arguments

        Returns:
            str: Path to the saved audio file

        Raises:
            AzureSynthesisError: If synthesis fails
        """
        if not text or not save_path:
            raise ValueError("Text and save_path are required")

        # Use configured voice if not specified
        voice_name = speaker or self.config.voice_name
        if not voice_name:
            raise ValueError(
                "Voice name is required. Set it either in constructor "
                "or through AZURE_SPEECH_VOICE environment variable"
            )

        try:
            # Prepare SSML text
            ssml_text = self.config.ssml_template.format(
                speaker=voice_name,
                rate=rate,
                text=text
            )

            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=save_path
            )

            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )

            # Synthesize speech
            logger.info(f"Synthesizing speech with voice: {voice_name}")
            result = synthesizer.speak_ssml_async(ssml_text).get()

            return self._handle_synthesis_result(result, save_path)

        except Exception as e:
            raise AzureSynthesisError(f"Synthesis failed: {str(e)}")

    @retry(
        retry=retry_if_exception_type(AzureRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def transcribe(
        self,
        audio_path: str,
        save_path: str,
        output_format: str = "txt",
        **kwargs
    ) -> str:
        """
        Convert speech to text using Azure Speech Service

        Args:
            audio_path: Path to the audio file
            save_path: Path to save the transcription
            output_format: Output format (txt or srt)
            **kwargs: Additional arguments

        Returns:
            str: Path to the saved transcription file

        Raises:
            AzureRecognitionError: If recognition fails
        """
        if not audio_path or not save_path:
            raise ValueError("Audio path and save path are required")

        try:
            # Configure audio input
            audio_config = speechsdk.audio.AudioConfig(
                filename=audio_path
            )

            # Configure speech recognition
            if self.config.language == "auto":
                auto_detect_config = (
                    speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                        languages=self.config.supported_languages
                    )
                )
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    auto_detect_source_language_config=auto_detect_config,
                    audio_config=audio_config
                )
            else:
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config
                )

            # Recognize speech
            logger.info(
                "Recognizing speech from "
                f"{audio_path} "
                f"(language: {self.config.language})"
            )
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(result.text)
                    logger.info(f"Transcription saved to {save_path}")
                    return save_path
                except Exception as e:
                    raise AzureRecognitionError(f"Failed to save transcription: {str(e)}")

            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                if (details.reason == speechsdk.CancellationReason.Error
                        and "429" in details.error_details):
                    raise AzureRateLimitError("Rate limit exceeded")
                raise AzureRecognitionError(
                    f"Recognition canceled: {details.reason}. "
                    f"Error details: {details.error_details}"
                )

            raise AzureRecognitionError(f"Recognition failed: {result.reason}")

        except Exception as e:
            raise AzureRecognitionError(f"Recognition failed: {str(e)}")

    def list_speakers(self) -> List[str]:
        """
        List available speakers/voices

        Returns:
            List[str]: List of available voices
        """
        # Azure doesn't provide a direct API to list voices
        # Return a list of commonly used voices
        return [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "es-ES-AlvaroNeural",
            "es-ES-ElviraNeural"
        ]