import re
import logging
from typing import Tuple, List
import langid
import spacy
from collections import Counter
import tiktoken
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Patterns that should not be treated as sentence boundaries
PROTECTED_PATTERNS = [
    r'\d+\.\d+',           # Decimal numbers (e.g., 3.5)
    r'\d+:\d+',            # Time (e.g., 12:30)
    r'\d+\.\d+\.\d+',      # Version numbers (e.g., 1.2.3)
    r'[A-Z]\.\s*[A-Z]\.',  # Abbreviations with spaces (e.g., U.S.)
    r'[A-Z]\.\s*[a-z]',    # Abbreviations followed by lowercase (e.g., Dr. Smith)
    r'https?://\S+',       # URLs
    r'[\w\.-]+@[\w\.-]+',  # Email addresses
    r'["\'].*?["\']',      # Quoted text
    r'\(.*?\)',            # Parenthesized text
    r'\$\d+\.\d+',         # Currency in dollars
    r'¥\d+\.\d+',          # Currency in yen
]

# Sentence endings for all languages, used for pre-processing text
SENTENCE_ENDINGS = ['.', '!', '?', '。', '！', '？']

# Default sentence delimiters for multiple languages
DEFAULT_DELIMITERS = r'[.!?。！？]+'

# Chinese and Chinese-like sentence endings, example: Japanese, Korean, etc.
ZH_SENTENCE_ENDINGS = ['。', '！', '？', '…', '～', '；']

# English and English-like sentence endings, example: French, German, etc.
EN_SENTENCE_ENDINGS = ['.', '!', '?', '...', ':', ';']

# Common ignored POS tags across languages
IGNORED_POS = {
    'PUNCT',    # Punctuation
    'NUM',      # Number
    'SYM',      # Symbol
    'SPACE',    # Space
    'DET',      # Determiner
    'AUX',      # Auxiliary
    'PART',     # Particle
    'INTJ',     # Interjection
    'CCONJ',    # Coordinating conjunction
    'SCONJ',    # Subordinating conjunction
    'ADP',      # Adposition
}

# Chinese specific ignored POS tags
CHINESE_IGNORED_POS = {
    'x',  # String
    'w',  # Punctuation
    'u',  # Particle
    'r',  # Pronoun
    'e',  # Interjection
    'y',  # Modal particle
    'o',  # Onomatopoeia
    'p',  # Preposition
    'd',  # Adverb
    'm',  # Numeral
    'q',  # Classifier
}

# Global variable to store loaded spaCy models
_nlp_models = {}


def get_word_pos_pairs(
    text: str,
    nlp: spacy.Language
) -> List[Tuple[str, str]]:
    """Get word and part-of-speech pairs from text using spaCy.

    Args:
        text: Input text to process
        nlp: Loaded spaCy language model

    Returns:
        List of (word, pos) tuples
    """
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def load_spacy_model(language: str) -> spacy.Language:
    """Load spaCy model for the specified language.

    Args:
        language: Language code (e.g., 'zh', 'en')

    Returns:
        Loaded spaCy language model
    """
    try:
        # Map language codes to spaCy model names
        model_map = {
            'zh': 'zh_core_web_sm',
            'en': 'en_core_web_sm',
            'ja': 'ja_core_news_sm',
            'ko': 'ko_core_news_sm'
        }

        model_name = model_map.get(language, 'en_core_web_sm')
        logger.debug(f"Loading spaCy model: {model_name}")
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            f"Model for language {language} not available. "
            "Falling back to English model."
        )
        return spacy.load("en_core_web_sm")


def extract_keywords(
    text: str,
    language: str,
    top_k: int = 20,
    min_length: int = 2
) -> List[str]:
    """Extract top keywords by frequency using spaCy.

    Args:
        text: Input text to extract keywords from
        language: Language code (e.g., 'zh', 'en')
        top_k: Maximum number of keywords to return
        min_length: Minimum length for keywords

    Returns:
        List[str]: List of top keywords sorted by frequency
    """
    # Load spaCy model
    nlp = load_spacy_model(language)

    # Get word-POS pairs
    doc = nlp(text)
    word_counts = Counter()

    for token in doc:
        word = token.text
        pos = token.pos_

        # Skip if word is too short
        if len(word) < min_length:
            continue

        # Skip if pure number
        if word.isdigit():
            continue

        # Skip common ignored POS tags
        if (pos in IGNORED_POS or
                pos in CHINESE_IGNORED_POS):
            continue

        word_counts[word.lower()] += 1

    # Get top k keywords by frequency
    return [word for word, _ in word_counts.most_common(top_k)]


def preprocess_text(text: str, language: str = 'en') -> str:
    """Preprocess text by merging lines that belong to the same paragraph.

    This method handles cases where a single sentence or paragraph is
    split across multiple lines, merging them into a single line while
    preserving actual paragraph breaks (double newlines).

    For Chinese text, it preserves the original spacing between characters.
    For English and other languages, it adds appropriate spaces between words.

    Args:
        text: Input text to preprocess
        language: Language code (e.g., 'en', 'zh'). Defaults to 'en'.

    Returns:
        str: Preprocessed text with proper paragraph formatting
    """
    logger.debug("Starting text preprocessing")
    logger.debug(f"Processing text in language: {language}")

    # Split text into lines
    lines = text.split('\n')
    processed_lines = []
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if not line:  # Empty line indicates paragraph break
            if current_paragraph:
                # Join accumulated lines with appropriate separator
                separator = ' ' if language != 'zh' else ''
                processed_lines.append(separator.join(current_paragraph))
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
                separator = ' ' if language != 'zh' else ''
                processed_lines.append(separator.join(current_paragraph))
                current_paragraph = []
            else:
                # Line doesn't end with punctuation, add to current
                current_paragraph.append(line)

    # Handle any remaining lines in the current paragraph
    if current_paragraph:
        separator = ' ' if language != 'zh' else ''
        processed_lines.append(separator.join(current_paragraph))

    # Join all processed lines with newlines
    result = '\n'.join(processed_lines)
    logger.debug(f"Preprocessing complete. Output length: {len(result)}")
    return result


def protect_special_patterns(text: str) -> Tuple[str, dict]:
    """Protect special patterns from being split during sentence detection.

    This method replaces special patterns (like numbers, URLs, etc.) with
    unique placeholders to prevent them from being incorrectly split.

    Args:
        text: Input text containing special patterns

    Returns:
        Tuple[str, dict]: Modified text with placeholders and mapping of
                        placeholders to original patterns
    """
    protected = {}
    modified_text = text

    for i, pattern in enumerate(PROTECTED_PATTERNS):
        for match in re.finditer(pattern, text):
            placeholder = f'__PROTECTED_{i}_{len(protected)}__'
            protected[placeholder] = match.group()
            modified_text = modified_text.replace(
                match.group(),
                placeholder
            )

    return modified_text, protected


def restore_protected_patterns(text: str, protected: dict) -> str:
    """Restore protected patterns in text after sentence splitting.

    This method replaces placeholders with their original patterns.

    Args:
        text: Text containing placeholders
        protected: Mapping of placeholders to original patterns

    Returns:
        str: Text with restored patterns
    """
    result = text
    for placeholder, original in protected.items():
        result = result.replace(placeholder, original)
    return result


def detect_language(text: str) -> str:
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

        # Detect language using langid
        lang_code, confidence = langid.classify(sample_text)
        logger.info(
            f"Detected language: {lang_code} "
            f"(confidence: {confidence:.2f})"
        )

        # Map some language codes to their spaCy equivalents if needed
        lang_map = {
            'zh-cn': 'zh',
            'zh-tw': 'zh',
            'jap': 'ja',
            'kor': 'ko'
        }

        mapped_code = lang_map.get(lang_code, lang_code)
        logger.debug(
            f"Mapped language code {lang_code} to {mapped_code}"
        )
        return mapped_code

    except ImportError:
        logger.warning(
            "langid library not found. "
            "Install with: pip install langid"
        )
        return 'en'
    except Exception as e:
        logger.warning(
            f"Language detection failed: {str(e)}. "
            "Defaulting to English."
        )
        return 'en'

def _create_sentence_pattern(endings: List[str]) -> str:
    """Create regex pattern from sentence endings.

    Args:
        endings: List of sentence ending characters

    Returns:
        str: Compiled regex pattern
    """
    return r'([' + ''.join(re.escape(p) for p in endings) + r']+)'

def _process_sentences(
    text: str,
    endings: List[str],
    protected: dict
) -> List[str]:
    """Process text into sentences using given endings.

    Args:
        text: Input text to process
        endings: List of sentence ending characters
        protected: Dictionary of protected patterns

    Returns:
        List[str]: List of processed sentences
    """
    logger.debug(f"Processing text with endings: {endings}")
    logger.debug(f"Original text: {text}")
    logger.debug(f"Protected patterns: {protected}")

    # Create regex pattern from endings
    pattern = r'([' + ''.join(re.escape(p) for p in endings) + r']+)'
    logger.debug(f"Created regex pattern: {pattern}")

    raw_sentences = []
    segments = re.split(pattern, text)
    logger.debug(f"Split into {len(segments)} segments")
    logger.debug(f"Segments: {segments}")

    current_sentence = ""
    for i, segment in enumerate(segments):
        logger.debug(f"Processing segment {i}: {segment}")

        # Skip empty segments
        if not segment.strip():
            continue

        # If segment is a sentence ending
        if re.match(pattern, segment):
            # If current sentence is empty and segment is just punctuation,
            # skip it (this handles cases like consecutive punctuation)
            if not current_sentence.strip() and segment.strip() in endings:
                continue

            # Add the punctuation to current sentence
            current_sentence += segment
            if current_sentence.strip():
                logger.debug(f"Adding sentence: {current_sentence.strip()}")
                raw_sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += segment

    # Handle the last sentence if it exists
    if current_sentence.strip():
        # If the last sentence doesn't end with punctuation,
        # add the default ending (usually period)
        if not any(current_sentence.strip().endswith(end) for end in endings):
            current_sentence += endings[0]  # Use the first ending as default
        logger.debug(f"Adding final sentence: {current_sentence.strip()}")
        raw_sentences.append(current_sentence.strip())

    logger.debug(f"Found {len(raw_sentences)} raw sentences")

    # Restore protected patterns and filter empty sentences
    sentences = [
        restore_protected_patterns(s.strip(), protected)
        for s in raw_sentences
        if s.strip()
    ]

    logger.debug(f"Final processed sentences: {sentences}")
    return sentences

def split_sentences_zh(text: str) -> List[str]:
    """Split text into sentences using Chinese sentence endings.

    This function splits Chinese text into sentences using Chinese-specific
    sentence endings and handles special patterns that should not be split.

    Args:
        text: Input text to split

    Returns:
        List[str]: List of sentences
    """
    logger.debug("Starting Chinese sentence splitting")
    modified_text, protected = protect_special_patterns(text)
    logger.debug(f"Protected patterns: {protected}")
    return _process_sentences(modified_text, ZH_SENTENCE_ENDINGS, protected)

def split_sentences_en(text: str) -> List[str]:
    """Split text into sentences using English sentence endings.

    This function splits English text into sentences using English-specific
    sentence endings and handles special patterns that should not be split.

    Args:
        text: Input text to split

    Returns:
        List[str]: List of sentences
    """
    logger.debug("Starting English sentence splitting")
    modified_text, protected = protect_special_patterns(text)
    logger.debug(f"Protected patterns: {protected}")
    return _process_sentences(modified_text, EN_SENTENCE_ENDINGS, protected)

def split_sentences(
    text: str,
    language: str = 'en',
) -> List[str]:
    """Split text into sentences using language-specific methods.
    """
    if language in ['zh', 'ja', 'ko']:
        return split_sentences_zh(text)
    else:
        return split_sentences_en(text)

def get_tokenizer(model_name: str):
    """Get the appropriate tokenizer based on model name.

    Args:
        model_name: Name of the model to use for tokenization

    Returns:
        Tokenizer instance (either tiktoken or HuggingFace)
    """
    if model_name.startswith("gpt"):
        return tiktoken.encoding_for_model(model_name)
    else:
        return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in the text using the appropriate tokenizer.

    Args:
        text: Input text to count tokens for
        model_name: Name of the model to use for tokenization

    Returns:
        Number of tokens in the text
    """
    tokenizer = get_tokenizer(model_name)
    if model_name.startswith("gpt"):
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.tokenize(text))

def get_unique_words(
    text: str,
    language: str,
    min_length: int = 2
) -> List[str]:
    """Get unique words from text using spaCy.

    Args:
        text: Input text to process
        language: Language code (e.g., 'zh', 'en')
        min_length: Minimum length for words

    Returns:
        List[str]: List of unique words
    """
    # Load spaCy model
    nlp = load_spacy_model(language)

    # Process text
    doc = nlp(text)
    unique_words = set()

    for token in doc:
        word = token.text
        pos = token.pos_

        # Skip if word is too short
        if len(word) < min_length:
            continue

        # Skip if pure number
        if word.isdigit():
            continue

        # Skip common ignored POS tags
        if (pos in IGNORED_POS or
                pos in CHINESE_IGNORED_POS):
            continue

        unique_words.add(word.lower())

    return list(unique_words)