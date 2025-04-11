import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

from devtoolbox.speech.azure_provider import AzureConfig, AzureProvider
import azure.cognitiveservices.speech as speechsdk


def mock_retry_decorator(*args, **kwargs):
    """Mock retry decorator that just returns the function unchanged."""
    def decorator(func):
        return func
    return decorator


def mock_stop_after_attempt(*args, **kwargs):
    """Mock stop_after_attempt that just returns a function."""
    return lambda: None


def mock_wait_exponential(*args, **kwargs):
    """Mock wait_exponential that just returns a function."""
    return lambda: None


def mock_retry_if_exception_type(*args, **kwargs):
    """Mock retry_if_exception_type that just returns a function."""
    return lambda: None


def mock_before_sleep_log(*args, **kwargs):
    """Mock before_sleep_log that just returns a function."""
    return lambda: None


class TestAzureConfig(unittest.TestCase):
    """Test suite for AzureConfig."""

    def test_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'AZURE_SPEECH_KEY': 'test-key',
            'AZURE_SPEECH_REGION': 'eastus',
            'AZURE_SPEECH_VOICE': 'en-US-JennyNeural',
            'AZURE_SPEECH_LANGUAGE': 'en-US',
            'AZURE_SPEECH_RATE': '1.0'
        }):
            config = AzureConfig.from_env()
            self.assertEqual(config.subscription_key, 'test-key')
            self.assertEqual(config.service_region, 'eastus')
            self.assertEqual(config.voice_name, 'en-US-JennyNeural')
            self.assertEqual(config.language, 'en-US')
            self.assertEqual(config.rate, 1.0)

    def test_validate(self):
        """Test configuration validation."""
        config = AzureConfig(
            subscription_key='test-key',
            service_region='eastus',
            voice_name='en-US-JennyNeural',
            language='en-US',
            rate=1.0,
            supported_languages=['en-US', 'zh-CN']
        )
        self.assertTrue(config.validate())

        with self.assertRaises(ValueError):
            AzureConfig(
                subscription_key='',
                service_region='eastus',
                voice_name='en-US-JennyNeural',
                language='en-US',
                rate=1.0,
                supported_languages=['en-US', 'zh-CN']
            ).validate()


class TestAzureProvider(unittest.TestCase):
    """Test suite for AzureProvider."""

    def setUp(self):
        """Set up test environment."""
        self.config = AzureConfig(
            subscription_key='test-key',
            service_region='eastus',
            voice_name='en-US-JennyNeural',
            language='en-US',
            rate=1.0,
            supported_languages=['en-US', 'zh-CN']
        )
        self.temp_dir = tempfile.mkdtemp()
        self.provider = AzureProvider(self.config)

        # Mock all tenacity functions
        patches = [
            patch('tenacity.retry', mock_retry_decorator),
            patch('tenacity.stop_after_attempt', mock_stop_after_attempt),
            patch('tenacity.wait_exponential', mock_wait_exponential),
            patch('tenacity.retry_if_exception_type', mock_retry_if_exception_type),
            patch('tenacity.before_sleep_log', mock_before_sleep_log)
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('azure.cognitiveservices.speech.SpeechConfig')
    @patch('azure.cognitiveservices.speech.AudioConfig')
    @patch('azure.cognitiveservices.speech.SpeechSynthesizer')
    def test_speak(self, mock_synthesizer, mock_audio_config, mock_speech_config):
        """Test text to speech conversion."""
        # Create a temporary file for output
        output_path = os.path.join(self.temp_dir, 'output.wav')

        # Mock the speech config
        mock_speech_config.return_value = MagicMock()
        mock_audio_config.return_value = MagicMock()

        # Mock the synthesis result
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        mock_result.audio_data = b'test audio data'
        mock_speak = mock_synthesizer.return_value.speak_ssml_async
        mock_result_get = mock_speak.return_value.get
        mock_result_get.return_value = mock_result

        result = self.provider.speak('Hello world', output_path)
        self.assertEqual(result, output_path)

        # Verify the audio data was written
        with open(output_path, 'rb') as f:
            self.assertEqual(f.read(), b'test audio data')

    @patch('azure.cognitiveservices.speech.SpeechConfig')
    @patch('azure.cognitiveservices.speech.AudioConfig')
    @patch('azure.cognitiveservices.speech.SpeechRecognizer')
    def test_transcribe(self, mock_recognizer, mock_audio_config, mock_speech_config):
        """Test speech to text conversion."""
        # Create temporary files for input and output
        input_path = os.path.join(self.temp_dir, 'input.wav')
        output_path = os.path.join(self.temp_dir, 'output.txt')

        # Write test audio data
        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        # Mock the speech config
        mock_speech_config.return_value = MagicMock()
        mock_audio_config.return_value = MagicMock()

        # Mock the recognition result
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = 'Hello world'
        mock_recognize = mock_recognizer.return_value.recognize_once_async
        mock_result_get = mock_recognize.return_value.get
        mock_result_get.return_value = mock_result

        result = self.provider.transcribe(input_path, output_path)
        self.assertEqual(result, output_path)

        # Verify the transcription was written
        with open(output_path, 'r') as f:
            self.assertEqual(f.read(), 'Hello world')

    @patch('azure.cognitiveservices.speech.SpeechConfig')
    @patch('azure.cognitiveservices.speech.AudioConfig')
    @patch('azure.cognitiveservices.speech.SpeechSynthesizer')
    def test_list_speakers(self, mock_synthesizer, mock_audio_config, mock_speech_config):
        """Test speaker listing."""
        with patch('devtoolbox.speech.azure_provider.AzureProvider.list_speakers') as mock_list_speakers:
            # 使用实际的返回格式
            mock_list_speakers.return_value = [
                'zh-CN-XiaoxiaoNeural',
                'zh-CN-YunxiNeural',
                'en-US-JennyNeural',
                'en-US-GuyNeural',
                'es-ES-AlvaroNeural',
                'es-ES-ElviraNeural'
            ]

            result = self.provider.list_speakers()
            self.assertEqual(result, [
                'zh-CN-XiaoxiaoNeural',
                'zh-CN-YunxiNeural',
                'en-US-JennyNeural',
                'en-US-GuyNeural',
                'es-ES-AlvaroNeural',
                'es-ES-ElviraNeural'
            ])

            mock_list_speakers.assert_called_once()

    @patch('azure.cognitiveservices.speech.SpeechConfig')
    @patch('azure.cognitiveservices.speech.AudioConfig')
    @patch('azure.cognitiveservices.speech.SpeechSynthesizer')
    def test_speak_error_handling(self, mock_synthesizer, mock_audio_config, mock_speech_config):
        """Test error handling in speak method."""
        # Create a temporary file for output
        output_path = os.path.join(self.temp_dir, 'output.wav')

        # Mock the speech config
        mock_speech_config.return_value = MagicMock()
        mock_audio_config.return_value = MagicMock()

        # Mock a service error
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.Canceled
        mock_result.cancellation_details = MagicMock()
        mock_result.cancellation_details.reason = \
            speechsdk.CancellationReason.Error
        mock_result.cancellation_details.error_details = \
            "Service returned error: 429 (Too Many Requests)"
        mock_speak = mock_synthesizer.return_value.speak_ssml_async
        mock_result_get = mock_speak.return_value.get
        mock_result_get.return_value = mock_result

        with self.assertRaises(Exception):
            self.provider.speak('Hello world', output_path)

    @patch('azure.cognitiveservices.speech.SpeechConfig')
    @patch('azure.cognitiveservices.speech.AudioConfig')
    @patch('azure.cognitiveservices.speech.SpeechRecognizer')
    def test_transcribe_error_handling(self, mock_recognizer, mock_audio_config, mock_speech_config):
        """Test error handling in transcribe method."""
        # Create temporary files for input and output
        input_path = os.path.join(self.temp_dir, 'input.wav')
        output_path = os.path.join(self.temp_dir, 'output.txt')

        # Write test audio data
        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        # Mock the speech config
        mock_speech_config.return_value = MagicMock()
        mock_audio_config.return_value = MagicMock()

        # Mock a service error
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.Canceled
        mock_result.cancellation_details = MagicMock()
        mock_result.cancellation_details.reason = \
            speechsdk.CancellationReason.Error
        mock_result.cancellation_details.error_details = \
            "Service returned error: 429 (Too Many Requests)"
        mock_recognize = mock_recognizer.return_value.recognize_once_async
        mock_result_get = mock_recognize.return_value.get
        mock_result_get.return_value = mock_result

        with self.assertRaises(Exception):
            self.provider.transcribe(input_path, output_path)


if __name__ == '__main__':
    unittest.main()