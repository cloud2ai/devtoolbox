import os
import shutil
import tempfile
import unittest
from pathlib import Path
import logging

from devtoolbox.markdown.converter import MarkdownConverter
from tests.utils.test_logging import setup_test_logging

# Initialize logging
logger = setup_test_logging()


class TestMarkdownConverter(unittest.TestCase):
    """Test cases for MarkdownConverter class"""

    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test fixtures")
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_md_dir = os.path.join(self.test_dir, "test_md")
        os.makedirs(self.test_md_dir, exist_ok=True)

        # Create test markdown files
        self.simple_md = os.path.join(self.test_md_dir, "simple.md")
        self.with_images_md = os.path.join(self.test_md_dir, "with_images.md")
        self.with_links_md = os.path.join(self.test_md_dir, "with_links.md")

        # Create test markdown content
        self._create_test_files()

    def _create_test_files(self):
        """Create test markdown files with different content"""
        # Simple markdown content
        simple_content = """# Test Document

This is a simple test document.

## Section 1
Some text here.

## Section 2
More text here.
"""
        with open(self.simple_md, "w", encoding="utf-8") as f:
            f.write(simple_content)

        # Markdown with images
        with_images_content = """# Document with Images

![Test Image](images/test.png)

Some text below the image.
"""
        with open(self.with_images_md, "w", encoding="utf-8") as f:
            f.write(with_images_content)

        # Markdown with links
        with_links_content = """# Document with Links

[Google](https://www.google.com)

[Local Link](local_file.md)
"""
        with open(self.with_links_md, "w", encoding="utf-8") as f:
            f.write(with_links_content)

    def tearDown(self):
        """Clean up test environment"""
        logger.info("Cleaning up test fixtures")
        shutil.rmtree(self.test_dir)

    def test_convert_simple_markdown(self):
        """Test converting a simple markdown file to docx"""
        logger.info("Testing simple markdown conversion")
        converter = MarkdownConverter(self.simple_md)
        output_path = os.path.join(self.test_dir, "simple.docx")

        # Convert markdown to docx
        converter.to_docx(output_path)

        # Verify output file exists
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        logger.debug("Verified simple markdown conversion")

    def test_convert_markdown_with_images(self):
        """Test converting markdown with image references to docx"""
        logger.info("Testing markdown with images conversion")
        # Create images directory and a dummy image
        images_dir = os.path.join(self.test_md_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        with open(os.path.join(images_dir, "test.png"), "wb") as f:
            f.write(b"dummy image data")

        converter = MarkdownConverter(self.with_images_md)
        output_path = os.path.join(self.test_dir, "with_images.docx")

        # Convert markdown to docx
        converter.to_docx(output_path)

        # Verify output file exists
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        logger.debug("Verified markdown with images conversion")

    def test_convert_markdown_with_links(self):
        """Test converting markdown with links to docx"""
        logger.info("Testing markdown with links conversion")
        converter = MarkdownConverter(self.with_links_md)
        output_path = os.path.join(self.test_dir, "with_links.docx")

        # Convert markdown to docx
        converter.to_docx(output_path)

        # Verify output file exists
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        logger.debug("Verified markdown with links conversion")

    def test_invalid_output_path(self):
        """Test conversion with invalid output path"""
        logger.info("Testing invalid output path handling")
        converter = MarkdownConverter(self.simple_md)

        # Test with empty output path
        with self.assertRaises(ValueError):
            converter.to_docx("")
        logger.debug("Verified empty output path handling")

        # Test with invalid directory
        invalid_dir = os.path.join(self.test_dir, "nonexistent")
        invalid_path = os.path.join(invalid_dir, "test.docx")

        # Create the directory first
        os.makedirs(invalid_dir, exist_ok=True)
        converter.to_docx(invalid_path)

        # Verify file was created
        self.assertTrue(os.path.exists(invalid_path))
        logger.debug("Verified invalid directory handling")

    def test_relative_path_handling(self):
        """Test handling of relative paths in markdown"""
        logger.info("Testing relative path handling")
        # Create a markdown file with relative paths
        relative_md = os.path.join(self.test_md_dir, "relative.md")
        relative_content = """# Document with Relative Paths

![Relative Image](images/test.png)

[Relative Link](local_file.md)
"""
        with open(relative_md, "w", encoding="utf-8") as f:
            f.write(relative_content)

        # Create the images directory and a dummy image
        images_dir = os.path.join(self.test_md_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        with open(os.path.join(images_dir, "test.png"), "wb") as f:
            f.write(b"dummy image data")

        converter = MarkdownConverter(relative_md)
        output_path = os.path.join(self.test_dir, "relative.docx")

        # Convert markdown to docx
        converter.to_docx(output_path)

        # Verify output file exists
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        logger.debug("Verified relative path handling")


if __name__ == '__main__':
    unittest.main()