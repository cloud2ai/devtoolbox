import unittest
from devtoolbox.text.splitter import TokenSplitter, Paragraph, SEPARATORS
from tests.utils.test_logging import setup_test_logging


# Initialize logging
logger = setup_test_logging()


class TestTokenSplitter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
        self.splitter = TokenSplitter(
            max_tokens=100,
            overlap_tokens=10,
            model_name="gpt-4"
        )

    def test_detect_language(self):
        """Test language detection functionality."""
        logger.info("Testing language detection")
        
        # Test English detection
        logger.debug("Testing English text detection")
        result = self.splitter._detect_language("This is an English text.")
        self.assertEqual(result, 'en')
        logger.debug(f"English detection result: {result}")

        # Test Chinese detection
        logger.debug("Testing Chinese text detection")
        result = self.splitter._detect_language("这是一个中文文本。")
        self.assertEqual(result, 'zh')
        logger.debug(f"Chinese detection result: {result}")

        # Test mixed language text
        logger.debug("Testing mixed language text detection")
        result = self.splitter._detect_language("This is English. 这是中文。")
        self.assertIn(result, ['en', 'zh'])
        logger.debug(f"Mixed language detection result: {result}")

    def test_count_tokens(self):
        """Test token counting functionality."""
        logger.info("Testing token counting")
        
        # Test English text
        logger.debug("Testing English text token counting")
        text = "This is a test sentence."
        result = self.splitter._count_tokens(text)
        self.assertGreater(result, 0)
        logger.debug(f"English text token count: {result}")

        # Test Chinese text
        logger.debug("Testing Chinese text token counting")
        text = "这是一个测试句子。"
        result = self.splitter._count_tokens(text)
        self.assertGreater(result, 0)
        logger.debug(f"Chinese text token count: {result}")

        # Test mixed language text
        logger.debug("Testing mixed language text token counting")
        text = "This is English. 这是中文。"
        result = self.splitter._count_tokens(text)
        self.assertGreater(result, 0)
        logger.debug(f"Mixed language text token count: {result}")

    def test_preprocess_text(self):
        """Test text preprocessing functionality."""
        logger.info("Testing text preprocessing")
        
        # Test basic paragraph merging
        logger.debug("Testing basic paragraph merging")
        text = "Line 1\nLine 2\n\nLine 3"
        result = self.splitter._preprocess_text(text)
        self.assertEqual(result, "Line 1 Line 2\n\nLine 3")
        logger.debug(f"Preprocessed text: {result}")

        # Test sentence ending handling
        logger.debug("Testing sentence ending handling")
        text = "Sentence 1.\nSentence 2!"
        result = self.splitter._preprocess_text(text)
        self.assertEqual(result, "Sentence 1.\nSentence 2!")
        logger.debug(f"Preprocessed text: {result}")

        # Test Chinese text preprocessing
        logger.debug("Testing Chinese text preprocessing")
        text = "第一行\n第二行\n\n第三行"
        result = self.splitter._preprocess_text(text)
        self.assertEqual(result, "第一行 第二行\n\n第三行")
        logger.debug(f"Preprocessed Chinese text: {result}")

    def test_load_language_model(self):
        """Test language model loading."""
        logger.info("Testing language model loading")
        
        # Test English model loading
        logger.debug("Testing English model loading")
        result = self.splitter._load_language_model('en')
        self.assertIsNotNone(result)
        logger.debug("English model loaded successfully")

        # Test Chinese model loading
        logger.debug("Testing Chinese model loading")
        result = self.splitter._load_language_model('zh')
        self.assertIsNotNone(result)
        logger.debug("Chinese model loaded successfully")

    def test_split_chinese_text(self):
        """Test Chinese text splitting."""
        logger.info("Testing Chinese text splitting")
        
        # Test basic Chinese text
        logger.debug("Testing basic Chinese text splitting")
        text = "这是一个测试。这是第二个句子！这是第三个句子？"
        result = self.splitter._split_chinese_text(text)
        expected = [
            "这是一个测试。",
            "这是第二个句子！",
            "这是第三个句子？"
        ]
        self.assertEqual(result, expected)
        logger.debug(f"Split Chinese text into {len(result)} sentences")

        # Test Chinese text with mixed punctuation
        logger.debug("Testing Chinese text with mixed punctuation")
        text = "第一句。第二句！第三句？第四句；第五句，"
        result = self.splitter._split_chinese_text(text)
        self.assertEqual(len(result), 5)
        logger.debug(
            f"Split Chinese text with mixed punctuation into {len(result)} "
            "sentences"
        )

    def test_create_splitter(self):
        """Test splitter creation with different languages."""
        logger.info("Testing splitter creation")
        
        # Test Chinese splitter
        logger.debug("Testing Chinese splitter creation")
        splitter = self.splitter._create_splitter('zh')
        self.assertEqual(splitter._separators, SEPARATORS['zh'])
        logger.debug("Chinese splitter created successfully")

        # Test default splitter
        logger.debug("Testing default splitter creation")
        splitter = self.splitter._create_splitter('en')
        self.assertEqual(splitter._separators, SEPARATORS['default'])
        logger.debug("Default splitter created successfully")

    def test_split_text(self):
        """Test main text splitting functionality."""
        logger.info("Testing main text splitting")
        
        # Test English text that fits in one chunk
        logger.debug("Testing English text that fits in one chunk")
        text = "This is a test sentence."
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Paragraph)
        self.assertEqual(result[0].content, text)
        logger.debug(f"Split English text into {len(result)} paragraphs")

        # Test Chinese text that fits in one chunk
        logger.debug("Testing Chinese text that fits in one chunk")
        text = "这是一个测试句子。"
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Paragraph)
        logger.debug(f"Split Chinese text into {len(result)} paragraphs")

        # Test long text that needs splitting
        logger.debug("Testing long text splitting")
        text = "This is a long text. " * 50
        result = self.splitter.split_text(text)
        self.assertGreater(len(result), 1)
        logger.debug(f"Split long text into {len(result)} paragraphs")

        # Test mixed language text
        logger.debug("Testing mixed language text splitting")
        text = "This is English. 这是中文。" * 30
        result = self.splitter.split_text(text)
        self.assertGreater(len(result), 1)
        logger.debug(
            f"Split mixed language text into {len(result)} paragraphs"
        )

    def test_split_text_formats(self):
        """Test text splitting with different text formats."""
        logger.info("Testing text splitting with different formats")
        
        # Test single line text
        logger.debug("Testing single line text")
        text = "This is a single line of text."
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, text)
        logger.debug("Single line text test passed")

        # Test multi-line text with empty lines
        logger.debug("Testing multi-line text with empty lines")
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = self.splitter.split_text(text)
        # Check that the text is properly preprocessed
        self.assertEqual(len(result), 1)
        self.assertIn("First paragraph", result[0].content)
        self.assertIn("Second paragraph", result[0].content)
        self.assertIn("Third paragraph", result[0].content)
        logger.debug("Multi-line text test passed")

        # Test text with multiple sentences per line
        logger.debug("Testing text with multiple sentences per line")
        text = "First sentence. Second sentence. Third sentence."
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].sentences), 3)
        logger.debug(
            f"Split text with multiple sentences into "
            f"{len(result[0].sentences)} sentences"
        )

        # Test Chinese text with multiple paragraphs
        logger.debug("Testing Chinese text with multiple paragraphs")
        text = "第一段。\n\n第二段。\n\n第三段。"
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIn("第一段", result[0].content)
        self.assertIn("第二段", result[0].content)
        self.assertIn("第三段", result[0].content)
        logger.debug("Chinese multi-paragraph text test passed")

        # Test text with special characters
        logger.debug("Testing text with special characters")
        text = "Line 1: !@#$%^&*()\nLine 2: []{}<>"
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIn("!@#$%^&*()", result[0].content)
        self.assertIn("[]{}<>", result[0].content)
        logger.debug("Text with special characters test passed")

        # Test text with different spacing
        logger.debug("Testing text with different spacing")
        text = "Line 1    Line 2\nLine 3\tLine 4"
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIn("Line 1", result[0].content)
        self.assertIn("Line 4", result[0].content)
        logger.debug("Text with different spacing test passed")

        # Test text with URLs and email addresses
        logger.debug("Testing text with URLs and email addresses")
        text = "Visit https://example.com or email test@example.com"
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIn("https://example.com", result[0].content)
        self.assertIn("test@example.com", result[0].content)
        logger.debug("Text with URLs and email addresses test passed")

    def test_paragraph_class(self):
        """Test Paragraph class functionality."""
        logger.info("Testing Paragraph class")
        
        # Test basic paragraph
        logger.debug("Testing basic paragraph")
        content = "Test content"
        token_count = 10
        sentences = ["Test content"]
        
        paragraph = Paragraph(content, token_count, sentences)
        self.assertEqual(paragraph.content, content)
        self.assertEqual(paragraph.token_count, token_count)
        self.assertEqual(paragraph.sentences, sentences)
        logger.debug("Basic paragraph test passed")

        # Test Chinese paragraph
        logger.debug("Testing Chinese paragraph")
        content = "这是一个测试"
        token_count = 5
        sentences = ["这是一个测试"]
        
        paragraph = Paragraph(content, token_count, sentences)
        self.assertEqual(paragraph.content, content)
        self.assertEqual(paragraph.token_count, token_count)
        self.assertEqual(paragraph.sentences, sentences)
        logger.debug("Chinese paragraph test passed")


if __name__ == '__main__':
    unittest.main() 