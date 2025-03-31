import logging
import os

import pypandoc

from devtoolbox.markdown.base import MarkdownBase


class MarkdownConverter(MarkdownBase):
    """A class to handle markdown file operations and conversions.

    This class provides functionality to handle markdown files, including
    loading content from files or strings and converting markdown to other
    formats like docx.
    """
    def to_docx(self, output_path):
        """Convert markdown to docx format.

        Converts the markdown content to a Word document using pandoc.
        Changes working directory during conversion to handle relative paths.

        Args:
            output_path (str): Path where the docx file should be saved.

        Raises:
            ValueError: If output path is not provided.
            RuntimeError: If pandoc conversion fails.
        """
        if not output_path:
            raise ValueError("Output path must be provided")

        # Save current working directory
        original_cwd = os.getcwd()

        try:
            # Change to markdown file's directory before conversion
            # This ensures relative paths (e.g. for images) work correctly
            os.chdir(os.path.dirname(self.path))

            # Convert markdown to docx using pandoc
            logging.info(
                f"Converting markdown file to Word document {output_path}"
            )
            pypandoc.convert_file(
                self.path,
                "docx",
                outputfile=output_path
            )
            logging.info("Successfully converted markdown to Word document")

        finally:
            # Restore original working directory
            os.chdir(original_cwd)
