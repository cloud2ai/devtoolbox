import unittest
from unittest.mock import patch, MagicMock
from minio.error import S3Error

from devtoolbox.storage import ObjectStorage
from tests.utils.test_logging import setup_test_logging

# Initialize logging
logger = setup_test_logging()


class TestObjectStorage(unittest.TestCase):
    """Test cases for ObjectStorage class"""

    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test fixtures")
        self.bucket = "test-bucket"
        self.endpoint = "test.endpoint.com"
        self.access_key = "test-access-key"
        self.secret_key = "test-secret-key"
        self.region = "test-region"

        # Create mock Minio client
        self.mock_minio = MagicMock()
        self.mock_minio.get_object.return_value = MagicMock(
            data=b"test content",
            close=MagicMock(),
            release_conn=MagicMock()
        )
        self.mock_minio.put_object.return_value = MagicMock(
            object_name="test/path"
        )
        self.mock_minio.stat_object.return_value = MagicMock()
        self.mock_minio.fput_object.return_value = MagicMock()
        self.mock_minio.list_objects.return_value = [
            MagicMock(object_name="test/file1.txt"),
            MagicMock(object_name="test/file2.txt")
        ]

        # Patch Minio class
        self.patcher = patch('devtoolbox.storage.Minio')
        self.mock_minio_class = self.patcher.start()
        self.mock_minio_class.return_value = self.mock_minio

        # Initialize storage
        self.storage = ObjectStorage(
            self.bucket,
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region
        )

    def tearDown(self):
        """Clean up test environment"""
        logger.info("Cleaning up test fixtures")
        self.patcher.stop()

    def test_initialization(self):
        """Test ObjectStorage initialization"""
        logger.info("Testing ObjectStorage initialization")
        self.mock_minio_class.assert_called_once_with(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region
        )
        self.assertEqual(self.storage.bucket, self.bucket)

    def test_read(self):
        """Test read method"""
        logger.info("Testing read method")
        path = "test/path"
        content = self.storage.read(path)

        self.mock_minio.get_object.assert_called_once_with(
            self.bucket, path
        )
        self.assertEqual(content, "test content")

    def test_write(self):
        """Test write method"""
        logger.info("Testing write method")
        path = "test/path"
        content = b"test content"
        content_type = "text/plain"

        result = self.storage.write(path, content, content_type)
        self.mock_minio.put_object.assert_called_once()
        self.assertEqual(result.object_name, "test/path")

    def test_exists(self):
        """Test exists method"""
        logger.info("Testing exists method")
        path = "test/path"

        # Test existing object
        exists = self.storage.exists(path)
        self.mock_minio.stat_object.assert_called_once_with(
            self.bucket, path
        )
        self.assertTrue(exists)

        # Test non-existing object
        self.mock_minio.stat_object.side_effect = S3Error(
            "NoSuchKey", "test-bucket", "test/path", "No such object",
            "test-host-id", "test-response"
        )
        exists = self.storage.exists(path)
        self.assertFalse(exists)

    def test_full_path(self):
        """Test full_path method"""
        logger.info("Testing full_path method")
        path = "test/path"

        # Test permanent URL
        url = self.storage.full_path(path, permanent=True)
        self.assertTrue(url.startswith("https://"))
        self.assertIn(self.bucket, url)
        self.assertIn(path, url)

        # Test presigned URL
        url = self.storage.full_path(path, permanent=False)
        self.mock_minio.get_presigned_url.assert_called_once_with(
            "GET", self.bucket, path
        )

    def test_cp_from_path(self):
        """Test cp_from_path method"""
        logger.info("Testing cp_from_path method")
        src_path = "/local/path"
        dest_path = "test/path"
        content_type = "text/plain"

        self.storage.cp_from_path(src_path, dest_path, content_type)
        self.mock_minio.fput_object.assert_called_once_with(
            self.bucket, dest_path, src_path,
            content_type=content_type
        )

    def test_ls(self):
        """Test ls method"""
        logger.info("Testing ls method")
        path = "test"
        pattern = "*.txt"

        files = self.storage.ls(path, pattern)
        self.mock_minio.list_objects.assert_called_once_with(
            self.bucket, prefix=path, recursive=True
        )
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], "test/file1.txt")

    def test_error_handling(self):
        """Test error handling"""
        logger.info("Testing error handling")
        path = "test/path"

        # Test read error
        self.mock_minio.get_object.side_effect = Exception("Read Error")
        with self.assertRaises(Exception):
            self.storage.read(path)

        # Test write error
        self.mock_minio.put_object.side_effect = Exception("Write Error")
        with self.assertRaises(Exception):
            self.storage.write(path, b"content")

        # Test copy error
        self.mock_minio.fput_object.side_effect = Exception("Copy Error")
        with self.assertRaises(Exception):
            self.storage.cp_from_path("src", "dest")


if __name__ == '__main__':
    unittest.main()