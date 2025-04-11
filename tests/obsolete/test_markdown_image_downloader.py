import unittest
import os
import tempfile
import shutil
from devtoolbox.markdown.image_downloader import MarkdownImageDownloader
from tests.utils.test_logging import setup_test_logging


# Initialize logging
logger = setup_test_logging()


class TestMarkdownImageDownloader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Create a test markdown file with different image URLs
        self.test_md_path = os.path.join(self.test_dir, "test.md")
        self.test_content = """# Test Markdown

This is a test markdown file with images.

![Test Image 1](https://picsum.photos/200/300?random=1)
![Test Image 2](https://picsum.photos/200/300?random=2)

Some more text here.

![Test Image 3](https://picsum.photos/200/300?random=3)
"""
        with open(self.test_md_path, "w", encoding="utf-8") as f:
            f.write(self.test_content)

        # Initialize the downloader
        self.downloader = MarkdownImageDownloader(self.test_md_path)

    def tearDown(self):
        """Clean up test fixtures."""
        logger.info("Cleaning up test fixtures")
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)

    def test_download_images(self):
        """Test downloading images from markdown content."""
        logger.info("Testing image download functionality")
        
        # Download images
        updated_content = self.downloader.download_images("images")
        
        # Check if images directory exists and contains files
        self.assertTrue(os.path.exists(self.image_dir))
        image_files = os.listdir(self.image_dir)
        self.assertGreater(len(image_files), 0)
        logger.debug(f"Found {len(image_files)} downloaded images")

        # Check if all images are downloaded
        expected_image_count = self.test_content.count("![")
        self.assertEqual(len(image_files), expected_image_count)
        logger.debug(
            f"Expected {expected_image_count} images, "
            f"found {len(image_files)}"
        )

        # Check if image files are valid
        for image_file in image_files:
            image_path = os.path.join(self.image_dir, image_file)
            self.assertTrue(os.path.exists(image_path))
            self.assertGreater(os.path.getsize(image_path), 0)
            logger.debug(f"Verified image file: {image_file}")

        # Check if markdown content is updated correctly
        self.assertNotEqual(updated_content, self.test_content)
        for image_file in image_files:
            self.assertIn(image_file, updated_content)
            self.assertIn("./images/", updated_content)
        logger.debug("Markdown content updated correctly")

    def test_image_naming(self):
        """Test image file naming convention."""
        logger.info("Testing image naming convention")
        
        # Download images
        self.downloader.download_images("images")
        
        # Get list of downloaded images
        image_files = os.listdir(self.image_dir)
        
        # Check naming pattern
        for image_file in image_files:
            # Check if filename follows the pattern: test-{number}.{extension}
            self.assertTrue(image_file.startswith("test-"))
            self.assertTrue(image_file.endswith((".jpg", ".png")))
            logger.debug(f"Verified image naming: {image_file}")

    def test_image_conversion(self):
        """Test image format conversion."""
        logger.info("Testing image format conversion")
        
        # Download images
        self.downloader.download_images("images")
        
        # Get list of downloaded images
        image_files = os.listdir(self.image_dir)
        
        # Check if images are converted to PNG
        png_count = sum(1 for f in image_files if f.endswith(".png"))
        self.assertGreater(png_count, 0)
        logger.debug(f"Found {png_count} PNG images")

    def test_markdown_content_update(self):
        """Test markdown content update after image download."""
        logger.info("Testing markdown content update")
        
        # Download images
        updated_content = self.downloader.download_images("images")
        
        # Check if content is properly updated
        self.assertNotEqual(updated_content, self.test_content)
        
        # Check if image references are updated correctly
        image_files = os.listdir(self.image_dir)
        for image_file in image_files:
            expected_line = f"![{image_file}](./images/{image_file})"
            self.assertIn(expected_line, updated_content)
            logger.debug(f"Verified image reference: {expected_line}")

        # Check if non-image content is preserved
        original_lines = self.test_content.split("\n")
        for line in original_lines:
            if not line.startswith("!["):
                self.assertIn(line, updated_content)
                logger.debug(f"Verified preserved content: {line}")

    def test_duplicate_images(self):
        """Test handling of duplicate image downloads."""
        logger.info("Testing duplicate image handling")
        
        # Create markdown with duplicate image URLs
        duplicate_content = """# Test Duplicate Images

![Same Image](https://picsum.photos/200/300?random=1)
![Same Image](https://picsum.photos/200/300?random=1)
"""
        duplicate_md_path = os.path.join(self.test_dir, "duplicate.md")
        with open(duplicate_md_path, "w", encoding="utf-8") as f:
            f.write(duplicate_content)

        # Initialize new downloader with duplicate content
        duplicate_downloader = MarkdownImageDownloader(duplicate_md_path)
        
        # Download images
        updated_content = duplicate_downloader.download_images("images")
        
        # Check if only one image is downloaded
        image_files = os.listdir(self.image_dir)
        self.assertEqual(len(image_files), 1)
        logger.debug("Verified duplicate image handling")

        # Check if both references point to the same image
        image_file = image_files[0]
        self.assertEqual(updated_content.count(f"![{image_file}"), 2)
        logger.debug("Verified duplicate image references")


if __name__ == '__main__':
    unittest.main() 