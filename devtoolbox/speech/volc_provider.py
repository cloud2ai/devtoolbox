"""
Volc speech engine providers

Currently the best voice clone is volc, so I import this providers for
voice clone

Note: This implementation is TODO and not tested yet due to missing test keys
"""

import base64
import json
import logging
import uuid
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

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
    register_provider
)

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # API endpoints
    "host": "openspeech.bytedance.com",
    "tts_path": "/api/v1/tts",
    "stt_path": "/api/v1/auc",

    # Audio settings
    "audio_encoding": "wav",
    "audio_rate": 24000,
    "audio_bits": 16,
    "audio_channel": 1,
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0,

    # Request settings
    "text_type": "plain",
    "operation": "query",
    "with_frontend": 1,
    "frontend_type": "unitTson",
    "split_sentence": 0,

    # Retry settings
    "max_retries": 5,
    "retry_min_wait": 4,
    "retry_max_wait": 30,
    "task_timeout": 300,  # 5 minutes
    "poll_interval": 2,   # 2 seconds
}


class VolcError(Exception):
    """Base exception for Volc-related errors"""
    pass


class VolcRateLimitError(VolcError):
    """Raised when Volc rate limit is exceeded"""
    pass


@dataclass
class VolcConfig(BaseSpeechConfig):
    """
    Volc Speech Service configuration

    Attributes:
        app_id: Volc application ID
        app_secret: Volc application secret
        voice_type: Voice type to use
        language: Speech language (default: zh)
        rate: Speech rate (default: 0)
        cluster: Volc cluster name
        uid: User ID for Volc service
        settings: Additional configuration settings

    Environment variables:
        VOLC_APP_ID: Volc application ID
        VOLC_APP_SECRET: Volc application secret
        VOLC_VOICE_TYPE: Voice type to use
        VOLC_LANGUAGE: Speech language
        VOLC_CLUSTER: Volc cluster name
        VOLC_UID: User ID for Volc service
        VOLC_HOST: API host (optional)
        VOLC_TASK_TIMEOUT: Task timeout in seconds (optional)
    """
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    voice_type: Optional[str] = None
    language: str = "zh"
    rate: int = 0
    cluster: Optional[str] = None
    uid: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    @classmethod
    def from_env(cls) -> 'VolcConfig':
        """Create Volc configuration from environment variables"""
        settings = DEFAULT_CONFIG.copy()

        # Override settings from environment variables
        if os.environ.get('VOLC_HOST'):
            settings['host'] = os.environ.get('VOLC_HOST')
        if os.environ.get('VOLC_TASK_TIMEOUT'):
            settings['task_timeout'] = int(os.environ.get('VOLC_TASK_TIMEOUT'))

        return cls(
            app_id=os.environ.get('VOLC_APP_ID'),
            app_secret=os.environ.get('VOLC_APP_SECRET'),
            voice_type=os.environ.get('VOLC_VOICE_TYPE'),
            language=os.environ.get('VOLC_LANGUAGE', 'zh'),
            rate=int(os.environ.get('VOLC_SPEECH_RATE', '0')),
            cluster=os.environ.get('VOLC_CLUSTER'),
            uid=os.environ.get('VOLC_UID'),
            settings=settings
        )

    @property
    def api_urls(self) -> Dict[str, str]:
        """Get API URLs based on current configuration"""
        host = self.settings['host']
        return {
            'tts': f"https://{host}{self.settings['tts_path']}",
            'stt': f"https://{host}{self.settings['stt_path']}"
        }

    def validate(self):
        """
        Validate Volc specific configuration

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.app_id:
            raise ValueError(
                "app_id is required. Set it either in constructor "
                "or through VOLC_APP_ID environment variable"
            )
        if not self.app_secret:
            raise ValueError(
                "app_secret is required. Set it either in constructor "
                "or through VOLC_APP_SECRET environment variable"
            )
        if not self.cluster:
            raise ValueError(
                "cluster is required. Set it either in constructor "
                "or through VOLC_CLUSTER environment variable"
            )
        if not self.uid:
            raise ValueError(
                "uid is required. Set it either in constructor "
                "or through VOLC_UID environment variable"
            )


@register_provider("volc")
class VolcSpeechProvider(BaseSpeechProvider):
    """
    Volc speech provider implementation

    Supports both text-to-speech and speech-to-text functionality.
    Note: This implementation is TODO and not tested yet.
    """

    def __init__(self, config: VolcConfig):
        """
        Initialize Volc provider

        Args:
            config: Volc configuration settings
        """
        super().__init__(config)
        self.config = config
        self.headers = {
            "Authorization": f"Bearer;{self.config.app_secret}"
        }

        logger.info(
            f"Initializing Volc provider (app_id: {config.app_id}, "
            f"language: {config.language})"
        )

    @retry(
        retry=retry_if_exception_type(VolcRateLimitError),
        stop=stop_after_attempt(DEFAULT_CONFIG['max_retries']),
        wait=wait_exponential(
            multiplier=1,
            min=DEFAULT_CONFIG['retry_min_wait'],
            max=DEFAULT_CONFIG['retry_max_wait']
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def speak(
        self,
        text: str,
        save_path: str,
        speaker: Optional[str] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Convert text to speech using Volc TTS

        Args:
            text: Text to convert to speech
            save_path: Path to save the audio file
            speaker: Voice type to use (optional)

        Returns:
            str: Path to the saved audio file

        Raises:
            ValueError: If configuration is invalid
            VolcError: If synthesis fails
        """
        self.config.validate()

        voice_type = speaker or self.config.voice_type
        if not voice_type:
            raise ValueError("Voice type is required")

        request_payload = {
            "app": {
                "appid": self.config.app_id,
                "token": self.config.app_secret,
                "cluster": self.config.cluster
            },
            "user": {
                "uid": self.config.uid
            },
            "audio": {
                "voice_type": voice_type,
                "encoding": self.config.settings["audio_encoding"],
                "rate": self.config.settings["audio_rate"],
                "bits": self.config.settings["audio_bits"],
                "channel": self.config.settings["audio_channel"],
                "speed_ratio": self.config.settings["speed_ratio"],
                "volume_ratio": self.config.settings["volume_ratio"],
                "pitch_ratio": self.config.settings["pitch_ratio"],
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": self.config.settings["text_type"],
                "operation": self.config.settings["operation"],
                "with_frontend": self.config.settings["with_frontend"],
                "frontend_type": self.config.settings["frontend_type"],
                "split_sentence": self.config.settings["split_sentence"]
            }
        }

        try:
            logger.debug(f"Request payload: {request_payload}")
            response = requests.post(
                self.config.api_urls['tts'],
                json.dumps(request_payload),
                headers=self.headers
            )
            response_data = response.json()

            if "data" in response_data:
                with open(save_path, "wb") as fhd:
                    fhd.write(base64.b64decode(response_data["data"]))
                logger.info(f"Generated audio saved to: {save_path}")
                return save_path
            else:
                raise VolcError(f"Volc response failed: {response.text}")

        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            raise

    def _submit_transcription_task(self, audio_path: str) -> str:
        """Submit transcription task to Volc service"""
        request = {
            "app": {
                "appid": self.config.app_id,
                "token": self.config.app_secret,
                "cluster": self.config.cluster
            },
            "user": {
                "uid": self.config.uid
            },
            "audio": {
                "format": "wav",
                "url": audio_path
            },
            "additions": {
                "with_speaker_info": "False",
            }
        }

        response = requests.post(
            f"{self.config.api_urls['stt']}/submit",
            data=json.dumps(request),
            headers=self.headers
        )
        response_data = response.json()

        if "resp" in response_data and "id" in response_data["resp"]:
            return response_data["resp"]["id"]
        else:
            raise VolcError(f"Failed to submit task: {response.text}")

    def _query_task_status(self, task_id: str) -> dict:
        """Query transcription task status"""
        query_data = {
            "appid": self.config.app_id,
            "token": self.config.app_secret,
            "id": task_id,
            "cluster": self.config.cluster
        }

        response = requests.post(
            f"{self.config.api_urls['stt']}/query",
            data=json.dumps(query_data),
            headers=self.headers
        )
        return response.json()

    @retry(
        retry=retry_if_exception_type(VolcRateLimitError),
        stop=stop_after_attempt(DEFAULT_CONFIG['max_retries']),
        wait=wait_exponential(
            multiplier=1,
            min=DEFAULT_CONFIG['retry_min_wait'],
            max=DEFAULT_CONFIG['retry_max_wait']
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def transcribe(
        self,
        speech_path: str,
        output_path: str,
        output_format: str = "txt"
    ) -> str:
        """
        Transcribe audio using Volc STT

        Args:
            speech_path: Path to the audio file
            output_path: Path to save the transcription
            output_format: Output format (default: txt)

        Returns:
            str: Path to the saved transcription file

        Raises:
            FileNotFoundError: If audio file doesn't exist
            VolcError: If transcription fails
            TimeoutError: If task takes too long
        """
        speech_path = Path(speech_path)
        if not speech_path.exists():
            raise FileNotFoundError(f"Audio file not found: {speech_path}")

        logger.info(f"Transcribing audio: {speech_path}")

        try:
            # Submit transcription task
            task_id = self._submit_transcription_task(str(speech_path))
            logger.info(f"Submitted transcription task: {task_id}")

            # Poll for results
            start_time = time.time()
            while True:
                time.sleep(self.config.settings["poll_interval"])
                response = self._query_task_status(task_id)

                if response["resp"]["code"] == 1000:  # Task finished
                    # TODO: Extract and save transcription text
                    # This part needs to be implemented once we have test access
                    logger.info("Transcription completed successfully")
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    # output_path.write_text(response["resp"]["text"])
                    return str(output_path)

                elif response["resp"]["code"] < 2000:  # Task failed
                    raise VolcError(
                        f"Transcription failed: {response['resp']}"
                    )

                if time.time() - start_time > self.config.settings["task_timeout"]:  # Task timeout
                    raise TimeoutError(
                        f"Transcription task timeout after {self.config.settings['task_timeout']} seconds"
                    )

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise