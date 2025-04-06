import logging
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

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
    register_provider
)

logger = logging.getLogger(__name__)

# SSML template for text-to-speech synthesis
SSML_TEXT = """
<speak xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xmlns:emo="http://www.w3.org/2009/10/emotionml"
       version="1.0" xml:lang="zh-CN">
    <voice name="{speaker}">
        <prosody rate="{rate}%">{text}</prosody>
    </voice>
</speak>
"""


@dataclass
class AzureConfig(BaseSpeechConfig):
    """
    Azure Speech Service configuration

    Attributes:
        subscription_key: Azure subscription key
        service_region: Azure service region
        voice_name: Azure voice name
        language: Speech language (default: auto for auto-detection)
        rate: Speech rate (default: 0)
        supported_languages: List of supported languages for auto-detection
                           (maximum 4 languages for auto-detection)

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
        AZURE_SUBSCRIPTION_KEY: Azure subscription key
        AZURE_SERVICE_REGION: Azure service region
        AZURE_VOICE_NAME: Azure voice name
        AZURE_LANGUAGE: Speech language (default: auto)
        AZURE_SPEECH_RATE: Speech rate (default: 0)
    """
    subscription_key: Optional[str] = None
    service_region: Optional[str] = None
    voice_name: Optional[str] = None
    language: str = "auto"
    rate: int = 0
    supported_languages: list = None

    def __post_init__(self):
        """
        Initialize default supported languages list with the most commonly used languages:
        Chinese, English, and Spanish
        """
        if self.supported_languages is None:
            # Default to 3 most commonly used languages globally
            self.supported_languages = [
                "zh-CN",     # Chinese (Simplified)
                "en-US",     # English (US)
                "es-ES",     # Spanish (Spain)
            ]
        elif len(self.supported_languages) > 4:
            logger.warning(
                "Azure auto language detection supports maximum 4 languages. "
                f"Truncating from {len(self.supported_languages)} to first 4."
            )
            self.supported_languages = self.supported_languages[:4]

    @classmethod
    def from_env(cls) -> 'AzureConfig':
        """
        Create Azure configuration from environment variables

        Returns:
            AzureConfig: Configuration instance with values from environment
        """
        return cls(
            subscription_key=os.environ.get('AZURE_SUBSCRIPTION_KEY'),
            service_region=os.environ.get('AZURE_SERVICE_REGION'),
            voice_name=os.environ.get('AZURE_VOICE_NAME'),
            language=os.environ.get('AZURE_LANGUAGE', 'auto'),
            rate=int(os.environ.get('AZURE_SPEECH_RATE', '0'))
        )

    def validate(self):
        """
        Validate Azure specific configuration

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.subscription_key:
            raise ValueError(
                "subscription_key is required. Set it either in constructor "
                "or through AZURE_SUBSCRIPTION_KEY environment variable"
            )
        if not self.service_region:
            raise ValueError(
                "service_region is required. Set it either in constructor "
                "or through AZURE_SERVICE_REGION environment variable"
            )


class AzureError(Exception):
    """Base exception for Azure-related errors"""
    pass


class AzureRateLimitError(AzureError):
    """Raised when Azure rate limit is exceeded"""
    pass


@register_provider("azure")
class AzureSpeechProvider(BaseSpeechProvider):
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
            f"Initializing Azure provider (region: {config.service_region}, "
            f"language: {config.language})"
        )

    @property
    def speech_config(self) -> speechsdk.SpeechConfig:
        """
        Lazy loading of Azure speech config

        Returns:
            Configured Azure speech config instance
        """
        if self._speech_config is None:
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.config.subscription_key,
                region=self.config.service_region
            )
        return self._speech_config

    def _handle_synthesis_result(
        self,
        result: speechsdk.SpeechSynthesisResult,
        save_path: str
    ) -> str:
        """
        Handle the result of speech synthesis

        Args:
            result: Speech synthesis result from Azure
            save_path: Path to save the audio file

        Returns:
            str: Path to the saved audio file

        Raises:
            AzureRateLimitError: If rate limit is exceeded
            AzureError: If synthesis fails for other reasons
        """
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            stream = speechsdk.AudioDataStream(result)
            stream.save_to_wav_file(save_path)
            logger.info(f"Generated audio saved to: {save_path}")
            return save_path

        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            error_code = getattr(details, 'error_code', None)

            if error_code == 429:
                raise AzureRateLimitError(
                    "Rate limit exceeded. Please retry after some time."
                )

            error_msg = f"Speech synthesis canceled: {details.reason}"
            if details.reason == speechsdk.CancellationReason.Error:
                error_msg = f"{error_msg}. Details: {details.error_details}"

            raise AzureError(error_msg)

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
        *args,
        **kwargs
    ) -> str:
        """
        Convert text to speech using Azure TTS with automatic retry

        Args:
            text: Text to convert to speech
            save_path: Path to save the audio file
            speaker: Voice name to use (optional)
            rate: Speech rate adjustment (optional)

        Returns:
            str: Path to the saved audio file

        Raises:
            ValueError: If configuration is invalid
            AzureError: If synthesis fails
        """
        self.config.validate()

        voice_name = speaker or self.config.voice_name
        if not voice_name:
            raise ValueError("Voice name is required")

        self.speech_config.speech_synthesis_voice_name = voice_name
        output_format = (
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        self.speech_config.set_speech_synthesis_output_format(output_format)

        try:
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )

            ssml_text = SSML_TEXT.format(
                speaker=voice_name,
                rate=rate,
                text=text
            )

            result = synthesizer.speak_ssml_async(ssml_text).get()
            return self._handle_synthesis_result(result, save_path)

        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception_type(AzureRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def transcribe(
        self,
        speech_path: str,
        output_path: str,
        output_format: str = "txt"
    ) -> str:
        """
        Transcribe audio using Azure STT with automatic retry and language detection

        Args:
            speech_path: Path to the audio file
            output_path: Path to save the transcription
            output_format: Output format (default: txt)

        Returns:
            str: Path to the saved transcription file

        Raises:
            FileNotFoundError: If audio file doesn't exist
            AzureError: If transcription fails
        """
        speech_path = Path(speech_path)
        if not speech_path.exists():
            raise FileNotFoundError(f"Audio file not found: {speech_path}")

        logger.info(f"Transcribing audio: {speech_path}")

        try:
            audio_config = speechsdk.audio.AudioConfig(
                filename=str(speech_path)
            )

            if self.config.language == "auto":
                # Use auto language detection
                auto_detect_source_language_config = \
                    speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                        languages=self.config.supported_languages
                    )

                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config,
                    auto_detect_source_language_config=\
                        auto_detect_source_language_config
                )
            else:
                # Use specified language
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config,
                    language=self.config.language
                )

            logger.debug("Starting speech recognition...")
            result = recognizer.recognize_once()
            logger.debug(f"Recognition result: {result.reason}")

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(result.text, encoding='utf-8')
                logger.info(f"Transcription saved to: {output_path}")
                return str(output_path)

            elif result.reason == speechsdk.ResultReason.NoMatch:
                no_match_details = result.no_match_details
                logger.warning(
                    f"No speech recognized. Reason: {no_match_details.reason}"
                )
                # Create empty output file
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("", encoding='utf-8')
                return str(output_path)

            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details

                # Log detailed cancellation information
                logger.error(
                    f"Recognition canceled. Reason: {details.reason}, "
                    f"Error details: {details.error_details}"
                )

                if details.reason == speechsdk.CancellationReason.Error:
                    # Check for specific error conditions in error_details
                    if "401" in details.error_details:
                        raise AzureError("Authentication failed")
                    elif "429" in details.error_details:
                        raise AzureRateLimitError(
                            "Rate limit exceeded during transcription"
                        )
                    else:
                        raise AzureError(
                            f"Transcription error: {details.error_details}"
                        )

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise