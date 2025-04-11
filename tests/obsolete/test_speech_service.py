import unittest
from unittest.mock import patch, Mock, MagicMock, call
import hashlib

from devtoolbox.speech.service import SpeechService
from devtoolbox.speech.azure_provider import AzureConfig


class TestSpeechService(unittest.TestCase):
    """Test suite for SpeechService."""

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
        with patch('devtoolbox.speech.service.AudioSegment'):
            self.service = SpeechService(self.config)
            self.service.cache_dir = '/mock/cache/dir'

    def test_init_with_config(self):
        """Test service initialization with config."""
        self.assertIsNotNone(self.service.provider)
        self.assertEqual(
            self.service.provider.config.subscription_key,
            'test-key'
        )

    def test_list_speakers(self):
        """Test speaker listing."""
        with patch('devtoolbox.speech.azure_provider.AzureProvider.list_speakers') as mock_list_speakers:
            mock_list_speakers.return_value = [
                'en-US-JennyNeural',
                'zh-CN-XiaoxiaoNeural'
            ]

            result = self.service.list_speakers()
            self.assertEqual(result, [
                'en-US-JennyNeural',
                'zh-CN-XiaoxiaoNeural'
            ])

            mock_list_speakers.assert_called_once()

    def test_speech_to_text(self):
        """Test speech to text conversion."""
        with patch('devtoolbox.speech.service.AudioSegment') as mock_audio_segment, \
             patch('devtoolbox.speech.azure_provider.AzureProvider.transcribe') as mock_transcribe, \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open, \
             patch('devtoolbox.speech.service.split_on_silence') as mock_split_on_silence:

            # Mock file does not exist to force provider call
            mock_exists.return_value = False

            # Mock transcribe to write text content
            def mock_transcribe_impl(input_path, output_path):
                # 写入文本文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("transcribed text")
                return output_path
            mock_transcribe.side_effect = mock_transcribe_impl

            # Mock file reading for hash calculation
            mock_file_hash = Mock()
            mock_file_hash.__enter__ = Mock(return_value=mock_file_hash)
            mock_file_hash.__exit__ = Mock(return_value=None)
            mock_file_hash.read = Mock(return_value=b'test audio data')

            # Mock file reading for transcribed text
            mock_file_text = Mock()
            mock_file_text.__enter__ = Mock(return_value=mock_file_text)
            mock_file_text.__exit__ = Mock(return_value=None)
            mock_file_text.read = Mock(return_value="transcribed text")

            # Mock open to return different file objects based on mode
            def mock_open_impl(file_path, mode='r', encoding=None):
                if mode == 'rb':  # 哈希计算时的二进制读取
                    return mock_file_hash
                else:  # 文本读取
                    return mock_file_text
            mock_open.side_effect = mock_open_impl

            # Mock AudioSegment
            mock_audio = MagicMock()
            mock_audio_segment.from_file.return_value = mock_audio
            mock_chunks = [MagicMock(), MagicMock()]  # Two mock audio chunks
            mock_split_on_silence.return_value = mock_chunks

            # Mock temporary file operations
            with patch('tempfile.mkdtemp') as mock_mkdtemp, \
                 patch('shutil.rmtree') as mock_rmtree:
                mock_mkdtemp.return_value = '/mock/temp/dir'

                result = self.service.speech_to_text(
                    '/mock/input.wav',
                    '/mock/output.txt',
                    use_cache=False  # 禁用缓存
                )

            self.assertEqual(result, '/mock/output.txt')

            # Verify all mocks were called correctly
            mock_open.assert_any_call('/mock/input.wav', 'rb')
            mock_audio_segment.from_file.assert_called_once_with('/mock/input.wav')
            mock_split_on_silence.assert_called_once_with(
                mock_audio,
                min_silence_len=1000,
                silence_thresh=-40,
                keep_silence=500
            )

            # 验证 transcribe 被调用了两次，每个音频块一次
            self.assertEqual(mock_transcribe.call_count, 2)
            mock_transcribe.assert_has_calls([
                call('/mock/temp/dir/chunk_0.wav', '/mock/temp/dir/chunk_0.txt'),
                call('/mock/temp/dir/chunk_1.wav', '/mock/temp/dir/chunk_1.txt')
            ])

            mock_rmtree.assert_called_once_with('/mock/temp/dir', ignore_errors=True)

    def test_text_to_speech(self):
        """Test text to speech conversion."""
        with patch('devtoolbox.speech.service.AudioSegment') as mock_audio_segment, \
             patch('devtoolbox.speech.azure_provider.AzureProvider.speak') as mock_speak, \
             patch('os.path.exists') as mock_exists, \
             patch('devtoolbox.speech.service.TokenSplitter') as mock_token_splitter_class:

            # Mock file does not exist to force provider call
            mock_exists.return_value = False

            # Mock TokenSplitter
            mock_token_splitter = Mock()
            mock_token_splitter_class.return_value = mock_token_splitter

            # Create a mock Paragraph object
            mock_paragraph = Mock()
            mock_paragraph.content = 'Hello world'
            mock_token_splitter.split_text.return_value = [mock_paragraph]

            # Mock AudioSegment
            mock_audio = MagicMock()
            mock_audio_segment.from_mp3.return_value = mock_audio
            mock_audio_segment.empty.return_value = mock_audio
            mock_audio.__add__.return_value = mock_audio

            # Mock temporary file operations
            with patch('tempfile.mkdtemp') as mock_mkdtemp, \
                 patch('shutil.rmtree') as mock_rmtree, \
                 patch('os.path.join') as mock_join:
                mock_mkdtemp.return_value = '/mock/temp/dir'
                mock_join.side_effect = lambda *args: '/mock/temp/dir/segment_0.mp3'

                result = self.service.text_to_speech(
                    'Hello world',
                    '/mock/output.mp3',
                    use_cache=False,  # 禁用缓存以确保使用指定的输出路径
                    speaker='en-US-JennyNeural',  # 显式指定 speaker
                    rate=1.0  # 显式指定 rate
                )

            self.assertEqual(result, '/mock/output.mp3')

            # Verify all mocks were called correctly
            mock_token_splitter.split_text.assert_called_once_with('Hello world')
            mock_speak.assert_called_once_with(
                'Hello world',
                '/mock/temp/dir/segment_0.mp3',
                speaker='en-US-JennyNeural',
                rate=1.0
            )
            mock_audio_segment.from_mp3.assert_called_once()
            mock_audio.export.assert_called_once()
            mock_rmtree.assert_called_once_with('/mock/temp/dir', ignore_errors=True)

    def test_text_to_speech_with_cache(self):
        """Test text to speech with caching."""
        with patch('devtoolbox.speech.service.AudioSegment') as mock_audio_segment, \
             patch('devtoolbox.speech.azure_provider.AzureProvider.speak') as mock_speak, \
             patch('os.path.exists') as mock_exists:

            cache_path = '/mock/cache/dir/3e25960a79dbc69b674cd4ec67a72c62.mp3'

            # First call: cache miss
            mock_exists.return_value = False

            # Mock AudioSegment
            mock_audio = MagicMock()
            mock_audio_segment.from_mp3.return_value = mock_audio
            mock_audio_segment.empty.return_value = mock_audio
            mock_audio.__add__.return_value = mock_audio

            # Mock temporary file operations
            with patch('tempfile.mkdtemp') as mock_mkdtemp, \
                 patch('shutil.rmtree') as mock_rmtree:
                mock_mkdtemp.return_value = '/mock/temp/dir'

                result1 = self.service.text_to_speech(
                    'Hello world',
                    '/mock/output.mp3',
                    use_cache=True
                )

            self.assertEqual(result1, cache_path)
            mock_speak.assert_called_once()

            # Second call: cache hit
            mock_exists.return_value = True
            mock_speak.reset_mock()

            result2 = self.service.text_to_speech(
                'Hello world',
                '/mock/output.mp3',
                use_cache=True
            )
            self.assertEqual(result2, cache_path)
            mock_speak.assert_not_called()

    def test_text_to_speech_error_handling(self):
        """Test error handling in text to speech."""
        with patch('devtoolbox.speech.service.AudioSegment') as mock_audio_segment, \
             patch('devtoolbox.speech.azure_provider.AzureProvider.speak') as mock_speak, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = False
            mock_speak.side_effect = Exception('API Error')

            with self.assertRaises(Exception):
                self.service.text_to_speech(
                    'Hello world',
                    '/mock/output.mp3'
                )

    def test_speech_to_text_error_handling(self):
        """Test error handling in speech to text."""
        with patch('devtoolbox.speech.service.AudioSegment') as mock_audio_segment, \
             patch('devtoolbox.speech.azure_provider.AzureProvider.transcribe') as mock_transcribe, \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:

            # Mock file reading for hash calculation
            mock_file = Mock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.read = Mock(return_value=b'test audio data')
            mock_open.return_value = mock_file

            mock_exists.return_value = False
            mock_transcribe.side_effect = Exception('API Error')

            with self.assertRaises(Exception):
                self.service.speech_to_text(
                    '/mock/input.wav',
                    '/mock/output.txt'
                )


if __name__ == '__main__':
    unittest.main()