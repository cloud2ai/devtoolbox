import logging
from typing import List, Dict
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect, LangDetectException
import spacy
import tiktoken
from transformers import AutoTokenizer

# Define language-specific separators as constants
SEPARATORS: Dict[str, List[str]] = {
    'zh': [
        "\n\n",     # Paragraph break for Chinese
        "\n",       # Line break for Chinese
        "。",       # Period for Chinese
        "！",       # Exclamation mark for Chinese
        "？",       # Question mark for Chinese
        "；",       # Semicolon for Chinese
        "，",       # Comma for Chinese
        " ",        # Space for Chinese
        ""          # Character for Chinese
    ],
    'default': [
        "\n\n",     # Paragraph break for default languages
        "\n",       # Line break for default languages
        ". ",       # Period for default languages
        "! ",       # Exclamation mark for default languages
        "? ",       # Question mark for default languages
        "; ",       # Semicolon for default languages
        ", ",       # Comma for default languages
        " ",        # Space for default languages
        ""          # Character for default languages
    ]
}

# Regular expression for Chinese sentence ending punctuation marks
CHINESE_SENTENCE_ENDINGS = r'([。！？；\n])'

# Sentence endings for all languages, used for pre-processing text
SENTENCE_ENDINGS = ['.', '!', '?', '。', '！', '？']

logger = logging.getLogger(__name__)

class Paragraph:
    """Represents a paragraph of text after splitting."""
    def __init__(self, content: str, token_count: int):
        # The actual content of the paragraph
        self.content = content
        # Number of tokens in the paragraph
        self.token_count = token_count

    def __repr__(self):
        return (
            f"Paragraph(content={self.content[:50]}..., "
            f"token_count={self.token_count})"
        )


class TokenSplitter:
    """Splits long text into chunks, ensuring sentence integrity and
    considering language."""

    def __init__(
        self,
        max_tokens: int = 2000,
        overlap_tokens: int = 0,
        model_name: str = "gpt-4"
    ):
        """Initialize the TokenSplitter.

        Args:
            max_tokens (int): The maximum number of tokens per chunk.
            overlap_tokens (int): The number of tokens to overlap between
                chunks.
            model_name (str): The name of the model to use for tokenization.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model_name = model_name
        self.nlp = None
        self.language = None
        logger.info(
            f"Initialized TokenSplitter with max_tokens={max_tokens}, "
            f"overlap_tokens={overlap_tokens}, model_name={model_name}"
        )

    def _create_splitter(self, language: str):
        """Create a text splitter with language-specific separators."""
        separators = SEPARATORS.get(language, SEPARATORS['default'])
        logger.debug(f"Creating splitter for language: {language}")
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=4000,
            chunk_overlap=200,
            length_function=self._count_tokens,
            is_separator_regex=False
        )

    def _split_chinese_text(self, text: str) -> List[str]:
        """Split Chinese text into sentences using regex patterns."""
        logger.debug("Splitting Chinese text into sentences")
        sentences = []
        segments = re.split(CHINESE_SENTENCE_ENDINGS, text)

        current_sentence = ""
        for segment in segments:
            current_sentence += segment
            if re.match(CHINESE_SENTENCE_ENDINGS, segment):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        logger.debug(f"Split Chinese text into {len(sentences)} sentences")
        return sentences

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text.

        Args:
            text (str): The text to analyze.

        Returns:
            str: Two-letter language code (e.g., 'en', 'zh', 'es').
        """
        try:
            # Use a sample of the text for faster detection
            # Use at least 100 characters, but not more than the first 1000
            sample_text = text[:1000].strip()
            if len(sample_text) < 100:
                sample_text = text.strip()

            # Detect language
            lang_code = detect(sample_text)
            logger.info(f"Detected language: {lang_code}")

            # Map some language codes to their spaCy equivalents if needed
            lang_map = {
                'zh-cn': 'zh',
                'zh-tw': 'zh',
                'jap': 'ja',
                'kor': 'ko'
            }

            mapped_code = lang_map.get(lang_code, lang_code)
            logger.debug(f"Mapped language code {lang_code} to {mapped_code}")
            return mapped_code

        except ImportError:
            logger.warning(
                "langdetect library not found. "
                "Install with: pip install langdetect"
            )
            return 'en'
        except LangDetectException as e:
            logger.warning(
                f"Language detection failed: {str(e)}. "
                "Defaulting to English."
            )
            return 'en'
        except Exception as e:
            logger.warning(
                f"Unexpected error in language detection: {str(e)}. "
                "Defaulting to English."
            )
            return 'en'

    def _load_language_model(self, language_code: str):
        """Load the appropriate spaCy language model.

        Args:
            language_code (str): Two-letter language code.

        Returns:
            The loaded spaCy model.
        """
        try:
            # Try to load the model directly
            model_name = f"{language_code}_core_web_sm"
            logger.debug(f"Attempting to load language model: {model_name}")
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded language model: {model_name}")
        except OSError:
            # If model isn't available, fall back to English
            logger.warning(
                f"Model for language {language_code} not available. "
                "Falling back to English model."
            )
            self.nlp = spacy.load("en_core_web_sm")

        return self.nlp

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text using the appropriate
        tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        try:
            # Try using OpenAI compatible tiktoken
            logger.debug("Attempting to use tiktoken "
                         f"for model {self.model_name}")
            enc = tiktoken.encoding_for_model(self.model_name)
            token_count = len(enc.encode(text))
            logger.debug(f"Token count using tiktoken: {token_count}")
            return token_count
        except (KeyError, ImportError, ValueError):
            # Not an OpenAI model, try Hugging Face model
            logger.debug("Tiktoken failed, trying Hugging Face tokenizer")
            pass

        try:
            # Try using Hugging Face tokenizer
            logger.debug("Loading Hugging Face tokenizer "
                         f"for {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            token_count = len(tokenizer(text)["input_ids"])
            logger.debug(f"Token count using Hugging Face: {token_count}")
            return token_count
        except (ImportError, ValueError, OSError):
            # If all else fails, use a rough character-based estimate
            logger.warning(
                f"Could not find a tokenizer for {self.model_name}. "
                "Using approximate character count instead."
            )
            # Rough approximation: 1 token ≈ 4 chars
            token_count = len(text) // 4
            logger.debug(f"Approximate token count: {token_count}")
            return token_count

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by merging lines that belong to the same
        paragraph.

        This method handles cases where a single sentence or paragraph is
        split across multiple lines, merging them into a single line while
        preserving actual paragraph breaks (double newlines).

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text with proper paragraph formatting.
        """
        logger.debug("Starting text preprocessing")
        # Split text into lines
        lines = text.split('\n')
        processed_lines = []
        current_paragraph = []

        for line in lines:
            line = line.strip()
            if not line:  # Empty line indicates paragraph break
                if current_paragraph:
                    # Join accumulated lines with space and add to processed
                    processed_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                processed_lines.append('')  # Preserve paragraph break
            else:
                # Check if line ends with sentence-ending punctuation
                ends_with_punct = any(
                    line.endswith(p) for p in SENTENCE_ENDINGS
                )
                if ends_with_punct:
                    # Add to current paragraph and start new one
                    current_paragraph.append(line)
                    processed_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                else:
                    # Line doesn't end with punctuation, add to current
                    current_paragraph.append(line)

        # Handle any remaining lines in the current paragraph
        if current_paragraph:
            processed_lines.append(' '.join(current_paragraph))

        # Join all processed lines with newlines
        result = '\n'.join(processed_lines)
        logger.debug(f"Preprocessing complete. Output length: {len(result)}")
        return result

    def split_text(self, text: str) -> List[Paragraph]:
        """Split the input text into paragraphs while maintaining sentence
        integrity.
        """
        try:
            logger.info("Starting text splitting process")
            # Preprocess the text first
            text = self._preprocess_text(text)

            # Detect language and load appropriate model
            language = self._detect_language(text)

            # Check total tokens
            total_tokens = self._count_tokens(text)
            logger.info(f"Input text has {total_tokens} tokens total")

            if total_tokens <= self.max_tokens:
                logger.info(
                    "Text fits within max token limit, no splitting needed"
                )
                return [Paragraph(text, total_tokens)]

            # Create splitter with appropriate separators
            splitter = self._create_splitter(language)
            chunks = splitter.create_documents([text])
            logger.info(f"Initial split created {len(chunks)} chunks")

            # Process chunks into paragraphs
            paragraphs = []
            current_chunk = []
            current_count = 0

            for chunk in chunks:
                chunk_text = chunk.page_content.strip()
                if not chunk_text:
                    continue

                chunk_tokens = self._count_tokens(chunk_text)
                logger.debug(f"Processing chunk with {chunk_tokens} tokens")

                # If chunk is within token limit, add it directly
                if chunk_tokens <= self.max_tokens:
                    paragraphs.append(Paragraph(chunk_text, chunk_tokens))
                    continue

                # For chunks that exceed max_tokens, split further if needed
                if language == 'zh':
                    sentences = self._split_chinese_text(chunk_text)
                    sentence_joiner = ""
                else:
                    if self.nlp is None:
                        self._load_language_model(language)
                    doc = self.nlp(chunk_text)
                    sentences = [sent.text.strip() for sent in doc.sents]
                    sentence_joiner = " "

                logger.debug(f"Split chunk into {len(sentences)} sentences")

                # Process sentences in the chunk
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)

                    if current_count + sentence_tokens <= self.max_tokens:
                        current_chunk.append(sentence)
                        current_count += sentence_tokens
                    else:
                        # Save current chunk if it exists
                        if current_chunk:
                            combined = sentence_joiner.join(current_chunk)
                            paragraphs.append(
                                Paragraph(combined, current_count)
                            )
                        # Start new chunk
                        current_chunk = [sentence]
                        current_count = sentence_tokens

                # Add remaining sentences from this chunk
                if current_chunk:
                    combined = sentence_joiner.join(current_chunk)
                    paragraphs.append(Paragraph(combined, current_count))
                    current_chunk = []
                    current_count = 0

            logger.info("Text splitting complete. "
                        f"Created {len(paragraphs)} paragraphs")
            return paragraphs

        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise
