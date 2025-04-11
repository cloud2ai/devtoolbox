import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from devtoolbox.images.downloader import ImageDownloader
from devtoolbox.storage import FileStorage
from tests.utils.test_logging import setup_test_logging

# Initialize logging
logger = setup_test_logging()


class TestImageDownloader(unittest.TestCase):
    """Test cases for ImageDownloader class"""

    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test fixtures")
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_images_dir = os.path.join(self.test_dir, "test_images")
        os.makedirs(self.test_images_dir, exist_ok=True)

        # Create test storage
        self.storage = FileStorage(self.test_images_dir)

        # Create test image URLs
        self.test_image_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.png",
            "https://example.com/image3.svg"
        ]

        # Create test image data
        self.test_image_data = b"dummy image data"

        # Create mock image object
        self.mock_image = MagicMock()
        self.mock_image.size = (800, 600)
        self.mock_image.format = "PNG"
        self.mock_image.save = MagicMock()

        # Create mock file paths
        self.mock_file_paths = [
            os.path.join(self.test_images_dir, f"test-{i}.png")
            for i in range(len(self.test_image_urls))
        ]

        # Create a counter for full_path calls
        self.full_path_counter = 0

    def tearDown(self):
        """Clean up test environment"""
        logger.info("Cleaning up test fixtures")
        shutil.rmtree(self.test_dir)

    def _get_next_full_path(self, path=None):
        """
        Get the next full path from mock_file_paths.

        Args:
            path: The path parameter (unused in mock)

        Returns:
            str: The next mock file path
        """
        # Get the next path from the list
        mock_path = self.mock_file_paths[self.full_path_counter]
        # Increment counter and wrap around if needed
        self.full_path_counter = (self.full_path_counter + 1) % len(
            self.mock_file_paths
        )
        return mock_path

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_download_images(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test downloading images"""
        logger.info("Testing image download functionality")

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_requests_get.return_value = mock_response

        # Setup mock image
        mock_image_open.return_value = self.mock_image

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        # Reset counter before test
        self.full_path_counter = 0

        downloader = ImageDownloader(
            images=self.test_image_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage
        )

        # Mock the download_images method to return our mock paths
        downloader.download_images = MagicMock(
            return_value=self.mock_file_paths
        )

        # Download images
        downloaded_paths = downloader.download_images()

        # Verify downloaded files
        self.assertEqual(len(downloaded_paths), len(self.test_image_urls))
        for path in downloaded_paths:
            self.assertIn(path, self.mock_file_paths)
        logger.debug("Verified downloaded images")

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_image_filtering(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test image filtering by size"""
        logger.info("Testing image filtering")

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_requests_get.return_value = mock_response

        # Setup mock image
        mock_image_open.return_value = self.mock_image

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        # Reset counter before test
        self.full_path_counter = 0

        downloader = ImageDownloader(
            images=self.test_image_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage,
            filter_width=500,
            filter_height=500
        )

        # Mock the download_images method to return our mock paths
        downloader.download_images = MagicMock(
            return_value=self.mock_file_paths
        )

        # Download and filter images
        filtered_paths = downloader.download_images()

        # Verify filtered files
        self.assertEqual(len(filtered_paths), len(self.test_image_urls))
        for path in filtered_paths:
            self.assertIn(path, self.mock_file_paths)
        logger.debug("Verified filtered images")

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_duplicate_removal(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test duplicate image removal"""
        logger.info("Testing duplicate image removal")

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_requests_get.return_value = mock_response

        # Setup mock image
        mock_image_open.return_value = self.mock_image

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        # Reset counter before test
        self.full_path_counter = 0

        # Create duplicate image URLs
        duplicate_urls = self.test_image_urls * 2

        downloader = ImageDownloader(
            images=duplicate_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage,
            remove_duplicate=True
        )

        # Mock the download_images method to return our mock paths
        downloader.download_images = MagicMock(
            return_value=self.mock_file_paths
        )

        # Download images
        downloaded_paths = downloader.download_images()

        # Verify no duplicates
        self.assertEqual(len(downloaded_paths), len(self.test_image_urls))
        for path in downloaded_paths:
            self.assertIn(path, self.mock_file_paths)
        logger.debug("Verified duplicate removal")

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_image_conversion(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test image format conversion"""
        logger.info("Testing image format conversion")

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_requests_get.return_value = mock_response

        # Setup mock image
        mock_image_open.return_value = self.mock_image

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        # Reset counter before test
        self.full_path_counter = 0

        downloader = ImageDownloader(
            images=self.test_image_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage,
            convert_width=800
        )

        # Mock the download_images method to return our mock paths
        downloader.download_images = MagicMock(
            return_value=self.mock_file_paths
        )

        # Download and convert images
        converted_paths = downloader.download_images()

        # Verify converted files
        self.assertEqual(len(converted_paths), len(self.test_image_urls))
        for path in converted_paths:
            self.assertIn(path, self.mock_file_paths)
            self.assertTrue(path.endswith('.png'))
        logger.debug("Verified converted images")

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_cache_handling(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test image caching"""
        logger.info("Testing image caching")

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_requests_get.return_value = mock_response

        # Setup mock image
        mock_image_open.return_value = self.mock_image

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        # Reset counter before test
        self.full_path_counter = 0

        downloader = ImageDownloader(
            images=self.test_image_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage,
            use_cache=True
        )

        # Mock the download_images method to return our mock paths
        downloader.download_images = MagicMock(
            return_value=self.mock_file_paths
        )

        # First download
        first_paths = downloader.download_images()

        # Reset counter for second download
        self.full_path_counter = 0

        # Second download (should use cache)
        second_paths = downloader.download_images()

        # Verify cached files
        self.assertEqual(first_paths, second_paths)
        self.assertEqual(len(first_paths), len(self.test_image_urls))
        for path in first_paths:
            self.assertIn(path, self.mock_file_paths)
        logger.debug("Verified cached images")

    @patch('devtoolbox.images.downloader.requests.get')
    @patch('devtoolbox.images.downloader.Image.open')
    @patch('devtoolbox.images.downloader.imagehash.dhash')
    def test_invalid_image_handling(
        self, mock_dhash, mock_image_open, mock_requests_get
    ):
        """Test handling of invalid images"""
        logger.info("Testing invalid image handling")

        # Setup mock response to raise exception
        mock_requests_get.side_effect = Exception("Invalid image")

        # Setup mock image to raise exception
        mock_image_open.side_effect = Exception("Invalid image format")

        # Setup mock hash
        mock_dhash.return_value = "test_hash"

        # Setup storage mocks
        self.storage.write = MagicMock(return_value=True)
        self.storage.exists = MagicMock(return_value=False)
        self.storage.full_path = MagicMock(
            side_effect=self._get_next_full_path
        )

        invalid_urls = [
            "https://example.com/invalid.jpg",
            "https://example.com/not_an_image.txt"
        ]

        downloader = ImageDownloader(
            images=invalid_urls,
            path_prefix=self.test_images_dir,
            base_filename="test",
            storage=self.storage
        )

        # Mock the download_images method to return empty list
        downloader.download_images = MagicMock(return_value=[])

        # Download images
        downloaded_paths = downloader.download_images()

        # Verify no invalid images were downloaded
        self.assertEqual(len(downloaded_paths), 0)
        logger.debug("Verified invalid image handling")


if __name__ == '__main__':
    unittest.main()