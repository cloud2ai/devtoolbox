#!/usr/bin/env python3
"""
Unit tests for Azure speech provider.

This module contains comprehensive tests for the Azure speech provider,
focusing on testing the logic of each method using mocks.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import pytest
from azure.cognitiveservices.speech import ResultReason, CancellationReason

from devtoolbox.speech.azure_provider import (
    AzureConfig,
    AzureProvider,
)
from devtoolbox.speech.clients.azure_errors import (
    AzureError,
    AzureSynthesisError,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_speech_config():
    """Mock SpeechConfig."""
    with patch('azure.cognitiveservices.speech.SpeechConfig') as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_audio_output_config():
    """Mock AudioOutputConfig."""
    with patch(
        'azure.cognitiveservices.speech.audio.AudioOutputConfig'
    ) as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_audio_config():
    """Mock AudioConfig."""
    with patch('azure.cognitiveservices.speech.audio.AudioConfig') as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_synthesizer():
    """Mock SpeechSynthesizer."""
    with patch('azure.cognitiveservices.speech.SpeechSynthesizer') as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_recognizer():
    """Mock SpeechRecognizer."""
    with patch('azure.cognitiveservices.speech.SpeechRecognizer') as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def azure_config():
    """Create AzureConfig instance."""
    return AzureConfig(
        subscription_key='test-key',
        service_region='eastus',
        voice_name='en-US-JennyNeural',
        rate=1.0
    )


@pytest.fixture
def azure_provider(azure_config):
    """Create AzureProvider instance."""
    return AzureProvider(azure_config)


class TestAzureConfig:
    """Tests for AzureConfig class."""

    def test_create_config_with_defaults(self, monkeypatch):
        """Test creating config with default values from environment."""
        monkeypatch.setenv('AZURE_SPEECH_KEY', 'test-key')
        monkeypatch.setenv('AZURE_SPEECH_REGION', 'eastus')
        monkeypatch.setenv('AZURE_SPEECH_VOICE', 'en-US-JennyNeural')

        monkeypatch.setenv('AZURE_SPEECH_RATE', '1.0')

        config = AzureConfig()

        assert config.subscription_key == 'test-key'
        assert config.service_region == 'eastus'
        assert config.voice_name == 'en-US-JennyNeural'
        assert 'en-US' in config.supported_languages
        assert config.rate == 1.0

    def test_config_validation_error(self):
        """Test validation error when subscription key is missing."""
        with pytest.raises(ValueError, match="subscription_key is required"):
            AzureConfig(subscription_key="")._validate_config()

    def test_deprecated_from_env_warning(self, caplog):
        """Test from_env() deprecation warning."""
        AzureConfig.from_env()
        assert "from_env() is deprecated" in caplog.text


class TestAzureProvider:
    """Tests for AzureProvider class."""

    def test_speak_success(
        self, azure_provider, mock_synthesizer, mock_audio_output_config,
        mock_speech_config, temp_dir
    ):
        """Test successful text-to-speech conversion."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.reason = ResultReason.SynthesizingAudioCompleted
        mock_result.audio_data = b'test audio data'

        # Setup mock chain
        synth = mock_synthesizer.return_value
        ssml = synth.speak_ssml_async.return_value
        ssml.get.return_value = mock_result

        # Test speak
        output_path = os.path.join(temp_dir, 'output.wav')
        result = azure_provider.speak('Hello world', output_path)

        # Verify result
        assert result == output_path
        mock_speech_config.assert_called_once()
        mock_audio_output_config.assert_called_once()
        mock_synthesizer.assert_called_once()

        # Verify file was written
        with open(output_path, 'rb') as f:
            assert f.read() == b'test audio data'

    def test_speak_rate_limit(
        self, azure_provider, mock_synthesizer, mock_audio_output_config,
        mock_speech_config, temp_dir
    ):
        """Test rate limit error handling."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.reason = ResultReason.Canceled
        mock_result.cancellation_details = MagicMock()
        mock_result.cancellation_details.reason = CancellationReason.Error
        mock_result.cancellation_details.error_details = "Rate limit exceeded"

        # Setup mock chain
        synth = mock_synthesizer.return_value
        ssml = synth.speak_ssml_async.return_value
        ssml.get.return_value = mock_result

        # Test speak with rate limit
        output_path = os.path.join(temp_dir, 'output.wav')
        with pytest.raises(AzureSynthesisError):
            azure_provider.speak('Hello world', output_path)

    def test_transcribe_success(
        self, azure_provider, mock_recognizer, mock_audio_config,
        mock_speech_config, temp_dir
    ):
        """Test successful speech-to-text conversion."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.reason = ResultReason.RecognizedSpeech
        mock_result.text = 'Hello world'

        # Setup mock chain
        recognizer = mock_recognizer.return_value
        recognize = recognizer.recognize_once_async.return_value
        recognize.get.return_value = mock_result

        # Test transcribe
        input_path = os.path.join(temp_dir, 'input.wav')
        output_path = os.path.join(temp_dir, 'output.txt')

        # Create test audio file
        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        # Configure AudioConfig
        mock_audio_config_instance = MagicMock()
        mock_audio_config.return_value = mock_audio_config_instance

        result = azure_provider.transcribe(input_path, output_path)

        # Verify result
        assert result == output_path
        mock_speech_config.assert_called_once()
        mock_audio_config.assert_called_once_with(filename=input_path)

        # Verify SpeechRecognizer call
        mock_recognizer.assert_called_once()
        call_args = mock_recognizer.call_args
        assert call_args.kwargs['speech_config'] == mock_speech_config.return_value
        assert call_args.kwargs['audio_config'] == mock_audio_config_instance

        # Verify file was written
        with open(output_path, 'r') as f:
            assert f.read() == 'Hello world'

    def test_transcribe_error(
        self, azure_provider, mock_recognizer, mock_audio_config,
        mock_speech_config, temp_dir
    ):
        """Test error handling in speech-to-text conversion."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.reason = ResultReason.Canceled
        mock_result.cancellation_details = MagicMock()
        mock_result.cancellation_details.reason = CancellationReason.Error
        mock_result.cancellation_details.error_details = "Service error"

        # Setup mock chain
        recognizer = mock_recognizer.return_value
        recognize = recognizer.recognize_once_async.return_value
        recognize.get.return_value = mock_result

        # Test transcribe with error
        input_path = os.path.join(temp_dir, 'input.wav')
        output_path = os.path.join(temp_dir, 'output.txt')

        # Create test audio file
        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        with pytest.raises(AzureError):
            azure_provider.transcribe(input_path, output_path)

    def test_list_speakers(self, azure_provider):
        """Test listing available speakers."""
        speakers = azure_provider.list_speakers()
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert 'en-US-JennyNeural' in speakers
        assert 'zh-CN-XiaoxiaoNeural' in speakers
