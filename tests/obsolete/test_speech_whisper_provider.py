import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil

from devtoolbox.speech.whisper_provider import WhisperConfig, WhisperProvider


class TestWhisperConfig(unittest.TestCase):
    """Test suite for WhisperConfig."""

    def test_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'WHISPER_MODEL': 'base',
            'WHISPER_LANGUAGE': 'en',
            'WHISPER_TASK': 'transcribe'
        }):
            config = WhisperConfig.from_env()
            self.assertEqual(config.model_name, 'base')
            self.assertEqual(config.language, 'en')
            self.assertEqual(config.task, 'transcribe')

    def test_validate(self):
        """Test configuration validation."""
        config = WhisperConfig(
            model_name='base',
            language='en',
            task='transcribe',
            supported_languages=['en', 'zh']
        )
        self.assertTrue(config.validate())

        with self.assertRaises(ValueError):
            WhisperConfig(
                model_name='',
                language='en',
                task='transcribe',
                supported_languages=['en', 'zh']
            ).validate()


class TestWhisperProvider(unittest.TestCase):
    """Test suite for WhisperProvider."""

    def setUp(self):
        """Set up test environment."""
        self.config = WhisperConfig(
            model_name='base',
            language='en',
            task='transcribe',
            supported_languages=['en', 'zh']
        )
        self.provider = WhisperProvider(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('devtoolbox.speech.whisper_provider.whisper')
    def test_transcribe(self, mock_whisper):
        """Test speech to text conversion."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Hello world'
        }
        mock_whisper.load_model.return_value = mock_model

        input_path = os.path.join(self.temp_dir, 'input.wav')
        output_path = os.path.join(self.temp_dir, 'output.txt')

        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        result = self.provider.transcribe(input_path, output_path)
        self.assertEqual(result, output_path)

        mock_whisper.load_model.assert_called_once_with('base')
        mock_model.transcribe.assert_called_once_with(
            input_path,
            language='en',
            task='transcribe'
        )

    @patch('devtoolbox.speech.whisper_provider.whisper')
    def test_transcribe_error_handling(self, mock_whisper):
        """Test error handling in transcribe method."""
        mock_whisper.load_model.side_effect = Exception('Model Error')

        input_path = os.path.join(self.temp_dir, 'input.wav')
        output_path = os.path.join(self.temp_dir, 'output.txt')

        with open(input_path, 'wb') as f:
            f.write(b'test audio data')

        with self.assertRaises(Exception):
            self.provider.transcribe(input_path, output_path)

    def test_speak_not_implemented(self):
        """Test speak method raises NotImplementedError."""
        output_path = os.path.join(self.temp_dir, 'test.mp3')
        with self.assertRaises(NotImplementedError):
            self.provider.speak('Hello world', output_path)

    def test_list_speakers(self):
        """Test list_speakers method returns empty list."""
        speakers = self.provider.list_speakers()
        self.assertEqual(speakers, [])


if __name__ == '__main__':
    unittest.main()