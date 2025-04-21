import re
import logging
from typing import List

from .base import BaseSplitter, Paragraph
from .utils import split_sentences

logger = logging.getLogger(__name__)


class ParagraphSplitter(BaseSplitter):
    """A text splitter that splits text by natural paragraphs using regex.
    
    This splitter uses regex to identify natural paragraphs and splits
    sentences using the split_sentences method.
    """

    def split(self) -> List[Paragraph]:
        """Split text into paragraphs.
        
        Returns:
            List[Paragraph]: List of Paragraph objects
        """
        logger.debug("Starting paragraph split")
        
        # Split text into paragraphs using regex
        # Match one or more newlines as paragraph separator
        paragraphs_text = re.split(r'\n\s*\n+', self.text)
        logger.debug(f"Split into {len(paragraphs_text)} paragraphs")
        
        paragraphs = []
        for i, para_text in enumerate(paragraphs_text):
            # Skip empty paragraphs
            if not para_text.strip():
                continue
                
            # Split paragraph into sentences
            sentences = split_sentences(para_text, language=self.language)
            logger.debug(f"Paragraph {i} has {len(sentences)} sentences")
            
            # Calculate paragraph length
            length = sum(self.length_func(s) for s in sentences)
            
            # Create paragraph object
            paragraph = self.paragraph_class(
                text=para_text,
                sentences=sentences,
                index=i,
                length=length
            )
            paragraphs.append(paragraph)
            
        logger.debug(f"Split complete. Found {len(paragraphs)} paragraphs")
        return paragraphs