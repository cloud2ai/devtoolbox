from __future__ import annotations
import logging
from typing import List, Optional


from .base import BaseSplitter, Paragraph
from .utils import (
    split_sentences,
    count_tokens
)

# Define language-specific separators as constants
SEPARATORS: dict[str, list[str]] = {
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

logger = logging.getLogger(__name__)


class TokenSplitter(BaseSplitter):
    """A text splitter that splits text based on token count.

    This splitter uses either tiktoken (for GPT models) or HuggingFace
    tokenizers to split text into chunks based on token count limits.
    """

    def __init__(
        self,
        text: str,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_func: Optional[callable] = None,
        paragraph_class=Paragraph,
        preprocess: bool = True
    ):
        """Initialize the token splitter.

        Args:
            text: Input text to process
            model_name: Name of the model to use for tokenization
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            length_func: Function to calculate text length
            paragraph_class: Class to use for paragraphs
            preprocess: Whether to preprocess the text before splitting
        """
        super().__init__(text, length_func, paragraph_class, preprocess)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized TokenSplitter with model_name={model_name}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def _create_splitter(
        self,
        language: str
    ) -> RecursiveCharacterTextSplitter:
        """Create a text splitter with language-specific separators.

        Args:
            language: Language code (e.g., 'zh', 'en')

        Returns:
            RecursiveCharacterTextSplitter instance
        """
        separators = SEPARATORS.get(language, SEPARATORS['default'])
        logger.debug(f"Creating splitter for language: {language}")
        # Lazy import to speed up CLI startup
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda x: count_tokens(x, self.model_name),
            is_separator_regex=False
        )

    def split(self) -> List[Paragraph]:
        """Split the text into paragraphs based on token count.

        Returns:
            List[Paragraph]: List of Paragraph objects
        """
        logger.info("Starting text splitting process")

        # Create appropriate splitter based on language
        splitter = self._create_splitter(self.language)

        # Split text into chunks
        chunks = splitter.split_text(self.text)
        logger.info(f"Split text into {len(chunks)} chunks")

        # Create Paragraph objects
        paragraphs = []
        for i, chunk in enumerate(chunks):
            # Split chunk into sentences
            sentences = split_sentences(chunk, language=self.language)

            # Create paragraph
            paragraph = self.paragraph_class(
                text=chunk,
                sentences=sentences,
                index=i,
                length=count_tokens(chunk, self.model_name)
            )
            paragraphs.append(paragraph)

        logger.info(f"Created {len(paragraphs)} paragraphs")
        return paragraphs
