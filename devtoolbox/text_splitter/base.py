from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from .utils import (
    extract_keywords,
    detect_language,
    preprocess_text,
    get_unique_words
)

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    """A paragraph containing text content.

    A simple data structure to hold paragraph information.

    Attributes:
        text: The complete text content of the paragraph
        sentences: List of sentences in the paragraph
        index: The index of this paragraph within the document
        length: Pre-calculated length of the paragraph
        metadata: Optional metadata about the paragraph
    """
    text: str
    sentences: List[str]
    index: int
    length: int
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        text: str,
        sentences: List[str],
        index: int,
        length: int,
        metadata: Optional[dict] = None
    ):
        """Initialize a paragraph.

        Args:
            text: The complete text content
            sentences: List of sentences in the paragraph
            index: The paragraph index
            length: The paragraph length
            metadata: Optional metadata
        """
        self.text = text
        self.sentences = sentences
        self.index = index
        self.length = length
        self.metadata = metadata or {}


class BaseSplitter(ABC):
    """Base class for text splitters.

    This class provides the foundation for text splitting functionality.
    Subclasses must implement specific language processing methods.
    """

    def __init__(
        self,
        text: str,
        length_func: Optional[Callable[[str], int]] = None,
        paragraph_class=Paragraph,
        preprocess: bool = True
    ):
        """Initialize the splitter.

        Args:
            text: Input text to process
            length_func: Function to calculate text length
            paragraph_class: Class to use for paragraphs
            preprocess: Whether to preprocess the text during initialization
        """
        self.length_func = length_func or len
        self.paragraph_class = paragraph_class

        # Detect language
        self.language = detect_language(text)
        logger.info(f"Detected language: {self.language}")

        # Preprocess text if requested
        if preprocess:
            self._text = preprocess_text(text, self.language)
        else:
            self._text = text

        self._keywords = None

    @property
    def keywords(self) -> List[tuple[str, int]]:
        """Get the keywords in the text.

        Returns:
            List[tuple[str, int]]: List of (keyword, frequency) pairs
        """
        if not self._keywords:
            self._keywords = self.get_keywords()
        return self._keywords

    @property
    def text(self) -> str:
        """Get the processed text.

        Returns:
            The processed text.
        """
        return self._text

    @property
    def length(self) -> int:
        """Get the length of the processed text.

        Returns:
            The length of the processed text.
        """
        return self.length_func(self._text)

    @property
    def unique_words(self) -> List[str]:
        """Get the list of unique words in the text.

        Returns:
            List[str]: List of unique words
        """
        return get_unique_words(
            self._text,
            language=self.language,
            min_length=2
        )

    def get_keywords(
        self,
        top_k: int = 20,
        min_length: int = 2
    ) -> List[tuple[str, int]]:
        """Extract keywords with their frequencies.

        Args:
            top_k: Maximum number of keywords to return
            min_length: Minimum length for keywords

        Returns:
            List[tuple[str, int]]: List of (keyword, frequency) pairs
        """
        return extract_keywords(
            self._text,
            language=self.language,
            top_k=top_k,
            min_length=min_length
        )

    def __call__(self) -> List[Paragraph]:
        """Split the text into paragraphs.

        This method allows the splitter to be called like a function.

        Returns:
            List[Paragraph]: List of Paragraph objects
        """
        return self.split()

    @abstractmethod
    def split(self) -> List[Paragraph]:
        """Split the text into paragraphs.

        This method must be implemented by subclasses to provide specific
        text splitting functionality.

        Returns:
            List[Paragraph]: List of Paragraph objects
        """
        pass