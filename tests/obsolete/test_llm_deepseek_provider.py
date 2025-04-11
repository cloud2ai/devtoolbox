import unittest
from unittest.mock import patch, MagicMock

from devtoolbox.llm.deepseek_provider import DeepSeekConfig, DeepSeekProvider


class TestDeepSeekConfig(unittest.TestCase):
    """Test suite for DeepSeekConfig."""

    def test_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'DEEPSEEK_API_KEY': 'test-key',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_TEMPERATURE': '0.8',
            'DEEPSEEK_MAX_TOKENS': '1000',
            'DEEPSEEK_TOP_P': '0.9',
            'DEEPSEEK_FREQUENCY_PENALTY': '0.5',
            'DEEPSEEK_PRESENCE_PENALTY': '0.5'
        }):
            config = DeepSeekConfig.from_env()
            self.assertEqual(config.api_key, 'test-key')
            self.assertEqual(config.model, 'deepseek-chat')
            self.assertEqual(config.temperature, 0.8)
            self.assertEqual(config.max_tokens, 1000)
            self.assertEqual(config.top_p, 0.9)
            self.assertEqual(config.frequency_penalty, 0.5)
            self.assertEqual(config.presence_penalty, 0.5)

    def test_validate(self):
        """Test configuration validation."""
        config = DeepSeekConfig(
            api_key='test-key',
            model='deepseek-chat',
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.assertTrue(config.validate())

        with self.assertRaises(ValueError):
            DeepSeekConfig(
                api_key='',
                model='deepseek-chat',
                temperature=0.7,
                max_tokens=2000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ).validate()


class TestDeepSeekProvider(unittest.TestCase):
    """Test suite for DeepSeekProvider."""

    def setUp(self):
        """Set up test environment."""
        self.config = DeepSeekConfig(
            api_key='test-key',
            model='deepseek-chat',
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.provider = DeepSeekProvider(self.config)

    @patch('requests.post')
    def test_chat(self, mock_post):
        """Test chat completion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = self.provider.chat(messages)
        self.assertEqual(result, 'Test response')

        mock_post.assert_called_once_with(
            'https://api.deepseek.com/v1/chat/completions',
            headers={
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        )

    @patch('requests.post')
    def test_complete(self, mock_post):
        """Test text completion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'text': 'Test completion'}]
        }
        mock_post.return_value = mock_response

        result = self.provider.complete('Test prompt')
        self.assertEqual(result, 'Test completion')

        mock_post.assert_called_once_with(
            'https://api.deepseek.com/v1/completions',
            headers={
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'prompt': 'Test prompt',
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        )

    @patch('requests.post')
    def test_embed(self, mock_post):
        """Test embedding generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        result = self.provider.embed('Test text')
        self.assertEqual(result, [0.1, 0.2, 0.3])

        mock_post.assert_called_once_with(
            'https://api.deepseek.com/v1/embeddings',
            headers={
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-embedding',
                'input': 'Test text'
            }
        )

    @patch('requests.get')
    def test_list_models(self, mock_get):
        """Test model listing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [
                {'id': 'deepseek-chat'},
                {'id': 'deepseek-embedding'}
            ]
        }
        mock_get.return_value = mock_response

        result = self.provider.list_models()
        self.assertEqual(result, ['deepseek-chat', 'deepseek-embedding'])

        mock_get.assert_called_once_with(
            'https://api.deepseek.com/v1/models',
            headers={
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json'
            }
        )

    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling."""
        mock_post.side_effect = Exception('API Error')
        with self.assertRaises(Exception):
            self.provider.chat([{"role": "user", "content": "Hello"}])


if __name__ == '__main__':
    unittest.main()