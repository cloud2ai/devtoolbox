"""Azure speech provider implementation.

This module provides the Azure speech provider implementation using
Azure Cognitive Services Speech SDK.
"""

import logging
from dataclasses import dataclass, field
import json
from typing import Optional, List
import os
import time
import uuid

import azure.cognitiveservices.speech as speechsdk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from devtoolbox.speech.clients.azure_client import AzureClient
from devtoolbox.speech.clients.azure_errors import (
    AzureRateLimitError,
    AzureConfigError,
    AzureSynthesisError,
    AzureRecognitionError
)
from devtoolbox.speech.provider import (
    BaseSpeechConfig,
    BaseSpeechProvider,
    register_provider,
    register_config,
)
from devtoolbox.speech.utils import get_audio_duration

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
    subscription_key: Optional[str] = field(
        default_factory=lambda: os.environ.get('AZURE_SPEECH_KEY')
    )
    service_region: Optional[str] = field(
        default_factory=lambda: os.environ.get('AZURE_SPEECH_REGION')
    )
    voice_name: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            'AZURE_SPEECH_VOICE', 'zh-CN-YunjianNeural'
        )
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
    storage_account: Optional[str] = field(
        default_factory=lambda: os.environ.get('AZURE_STORAGE_ACCOUNT')
    )
    storage_key: Optional[str] = field(
        default_factory=lambda: os.environ.get('AZURE_STORAGE_KEY')
    )
    container_name: str = field(
        default_factory=lambda: os.environ.get('AZURE_CONTAINER_NAME', 'speech-audio')
    )

    # Retry configuration for upload operations only
    upload_retry_attempts: int = field(
        default_factory=lambda: int(
            os.environ.get('AZURE_UPLOAD_RETRY_ATTEMPTS', '3')
        )
    )
    upload_retry_min_wait: int = field(
        default_factory=lambda: int(
            os.environ.get('AZURE_UPLOAD_RETRY_MIN_WAIT', '4')
        )
    )
    upload_retry_max_wait: int = field(
        default_factory=lambda: int(
            os.environ.get('AZURE_UPLOAD_RETRY_MAX_WAIT', '10')
        )
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()
        # Validate storage config
        if not self.storage_account:
            raise ValueError(
                "storage_account is required. Set it either in constructor "
                "or through AZURE_STORAGE_ACCOUNT environment variable"
            )
        if not self.storage_key:
            raise ValueError(
                "storage_key is required. Set it either in constructor "
                "or through AZURE_STORAGE_KEY environment variable"
            )
        if not self.container_name:
            raise ValueError(
                "container_name is required. Set it either in constructor "
                "or through AZURE_CONTAINER_NAME environment variable"
            )

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

    @property
    def locale(self) -> str:
        """
        Return the first language in supported_languages as the preferred
        locale.
        """
        return (self.supported_languages[0]
                if self.supported_languages else "zh-CN")


@register_provider('AzureProvider')
class AzureProvider(BaseSpeechProvider):
    """
    Azure speech provider implementation

    Supports both text-to-speech and speech-to-text functionality using
    Azure Cognitive Services Speech SDK with auto language detection.
    """

    # Maximum duration (seconds) for real-time recognition
    MAX_REALTIME_DURATION = 30.0

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
        self._azure_client = None  # Lazy initialization

        logger.info(
            "Initializing Azure provider "
            f"(region: {config.service_region}, "
            f"language: {config.supported_languages})"
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

    @property
    def azure_client(self):
        """
        Lazy initialization of AzureClient.
        Only create when needed (e.g., batch transcription).
        """
        if self._azure_client is None:
            self._azure_client = AzureClient(self.config)
        return self._azure_client

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

    def transcribe(
        self,
        audio_path: str,
        save_path: str,
        force_batch: bool = False,
        **kwargs
    ) -> str:
        """
        Unified entry for speech recognition.
        Write result to save_path as txt.
        """
        duration = get_audio_duration(audio_path)
        if force_batch or duration > self.MAX_REALTIME_DURATION:
            text = self._batch_transcribe(audio_path, **kwargs)
        else:
            text = self._real_time_transcribe(audio_path, **kwargs)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        return save_path

    def _real_time_transcribe(
        self,
        audio_path: str,
        **kwargs
    ) -> str:
        """
        Real-time recognition, return recognized text.
        """
        if not audio_path:
            raise ValueError("Audio path is required")
        try:
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
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
            logger.info(
                "Recognizing speech from "
                f"{audio_path} "
                f"(language: {self.config.supported_languages})"
            )
            result = recognizer.recognize_once_async().get()
            logger.debug(f"Azure recognition result: {result}")
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
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

    def _generate_random_blob_name(self, original_path: str) -> str:
        """
        Generate a random blob name based on the original file extension.

        Args:
            original_path (str): Path to the original file.

        Returns:
            str: Randomized blob name with original extension.
        """
        ext = os.path.splitext(original_path)[1]
        return f"{uuid.uuid4().hex}{ext}"

    def _batch_transcribe(self, audio_path: str, **kwargs) -> str:
        """
        Batch recognition logic (Azure Batch Transcription).

        1. Upload audio to Azure Blob Storage (random name).
        2. Submit Azure Batch Transcription job with the blob URL.
        3. Poll for job completion and download the result.
        4. Delete the blob after processing.

        Returns:
            str: Recognized text result.
        """
        properties = {
            "languageIdentification": True,
            "languageIdentificationLanguages": (
                self.config.supported_languages
            )
        }
        locale = self.config.locale
        blob_name, sas_url = self.azure_client.upload_blob(audio_path)
        logger.info(
            f"[BLOB_UPLOAD] name={blob_name} "
            f"sas_url={sas_url}"
        )
        try:
            transcription_id = self.azure_client.submit_batch_transcription(
                sas_url,
                properties=properties
            )
            logger.info(
                f"[BATCH_SUBMIT] id={transcription_id} "
                f"blob_name={blob_name}"
            )
            result_json = self._poll_transcription_result(
                transcription_id,
                blob_name=blob_name
            )
            data = json.loads(result_json)
            phrases = data.get("combinedRecognizedPhrases", [])
            final_text = "\n".join(
                p.get("display", "")
                for p in phrases
                if "display" in p
            )
            return final_text
        finally:
            self.azure_client.delete_blob(blob_name)
            logger.info(
                f"[BLOB_DELETE] name={blob_name}"
            )

    def _poll_transcription_result(
        self, transcription_id: str, blob_name: str = "", timeout: int = 3600
    ) -> str:
        """
        Poll the batch transcription result until completion or timeout.
        """
        status_data = self._wait_for_transcription_succeeded(
            transcription_id, blob_name, timeout
        )
        files_data = self.azure_client.get_transcription_files(transcription_id)
        logger.info(
            f"[BATCH_FILES] id={transcription_id} files={files_data}"
        )
        result_url = self._get_transcription_result_url(
            files_data, transcription_id, blob_name
        )
        return self._download_transcription_result_text(result_url)

    def _wait_for_transcription_succeeded(
        self, transcription_id: str, blob_name: str, timeout: int
    ) -> dict:
        """
        Poll the transcription status until succeeded, failed, or timeout.
        Return the final status data.
        """
        import time
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise AzureRecognitionError(
                    f"Batch transcription timeout: id={transcription_id} "
                    f"blob_name={blob_name}"
                )
            status_data = self.azure_client.get_transcription_status(
                transcription_id
            )
            status = status_data.get("status")
            if status == "Succeeded":
                return status_data
            elif status in ("Failed", "FailedWithPartialResults"):
                logger.error(
                    f"[BATCH_FAIL] id={transcription_id} blob_name="
                    f"{blob_name} data={status_data}"
                )
                raise AzureRecognitionError(
                    f"Transcription failed: id={transcription_id} "
                    f"blob_name={blob_name} data={status_data}"
                )
            else:
                logger.info(
                    f"[BATCH_STATUS] id={transcription_id} "
                    f"blob_name={blob_name} status={status}"
                )
                time.sleep(10)

    def _get_transcription_result_url(
        self, files_data: dict, transcription_id: str, blob_name: str
    ) -> str:
        """
        Extract the result_url from files_data, retry up to 3 times if not found.
        """
        max_retries = 3
        retry_interval = 10
        retry_count = 0
        while retry_count <= max_retries:
            for file_data in files_data.get("values", []):
                kind = file_data.get("kind", "")
                if kind in ("Transcription", "TranscriptionFile", "Results"):
                    result_url = file_data.get("links", {}).get("contentUrl")
                    if result_url:
                        return result_url
            if retry_count < max_retries:
                logger.warning(
                    f"No transcription result URL found, retrying... "
                    f"(attempt {retry_count+1}/{max_retries})"
                )
                retry_count += 1
                time.sleep(retry_interval)
                files_data = self.azure_client.get_transcription_files(
                    transcription_id
                )
            else:
                break
        raise AzureRecognitionError(
            f"No transcription result URL found after {max_retries} retries: "
            f"id={transcription_id} blob_name={blob_name}"
        )

    def _download_transcription_result_text(self, result_url: str) -> str:
        """
        Download and return the transcription result text from the result_url.
        """
        return self.azure_client.download_transcription_result(result_url)