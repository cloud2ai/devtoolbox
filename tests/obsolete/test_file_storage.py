import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from devtoolbox.storage import FileStorage
from tests.utils.test_logging import setup_test_logging

# Initialize logging
logger = setup_test_logging()


class TestFileStorage(unittest.TestCase):
    """Test cases for FileStorage class"""

    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test fixtures")
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorage(self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        logger.info("Cleaning up test fixtures")
        shutil.rmtree(self.temp_dir)

    def test_read(self):
        """Test read method"""
        logger.info("Testing read method")
        path = "test.txt"
        content = "test content"
        file_path = os.path.join(self.temp_dir, path)

        # Create test file
        with open(file_path, "w") as f:
            f.write(content)

        # Test read
        result = self.storage.read(path)
        self.assertEqual(result, bytes(content, 'utf-8'))

    def test_write(self):
        """Test write method"""
        logger.info("Testing write method")
        path = "test.txt"
        content = b"test content"
        content_type = "text/plain"

        # Test write
        self.storage.write(path, content, content_type)

        # Verify file was created
        file_path = os.path.join(self.temp_dir, path)
        self.assertTrue(os.path.exists(file_path))

        # Verify content
        with open(file_path, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_exists(self):
        """Test exists method"""
        logger.info("Testing exists method")
        path = "test.txt"
        file_path = os.path.join(self.temp_dir, path)

        # Test non-existent file
        self.assertFalse(self.storage.exists(path))

        # Create file
        with open(file_path, "w") as f:
            f.write("test")

        # Test existing file
        self.assertTrue(self.storage.exists(path))

    def test_full_path(self):
        """Test full_path method"""
        logger.info("Testing full_path method")
        path = "test.txt"
        expected_path = os.path.join(self.temp_dir, path)

        result = self.storage.full_path(path)
        self.assertEqual(result, expected_path)

    def test_cp_from_path(self):
        """Test cp_from_path method"""
        logger.info("Testing cp_from_path method")
        src_path = os.path.join(self.temp_dir, "src.txt")
        dest_path = "dest.txt"
        content = "test content"

        # Create source file
        with open(src_path, "w") as f:
            f.write(content)

        # Test copy
        self.storage.cp_from_path(src_path, dest_path)

        # Verify destination file
        dest_file_path = os.path.join(self.temp_dir, dest_path)
        self.assertTrue(os.path.exists(dest_file_path))
        with open(dest_file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_ls(self):
        """Test ls method"""
        logger.info("Testing ls method")
        # Create test files
        files = ["test1.txt", "test2.txt", "other.dat"]
        for file in files:
            with open(os.path.join(self.temp_dir, file), "w") as f:
                f.write("test")

        # Test listing all files
        result = self.storage.ls("")
        self.assertEqual(len(result), 3)
        self.assertIn("test1.txt", result)
        self.assertIn("test2.txt", result)
        self.assertIn("other.dat", result)

        # Test listing with pattern
        result = self.storage.ls("", "*.txt")
        self.assertEqual(len(result), 2)
        self.assertIn("test1.txt", result)
        self.assertIn("test2.txt", result)

    def test_binary_content(self):
        """Test handling of binary content"""
        logger.info("Testing binary content handling")
        path = "test.bin"
        content = b"\x00\x01\x02\x03"

        # Test write
        self.storage.write(path, content)

        # Test read
        result = self.storage.read(path)
        self.assertEqual(result, content)

    def test_directory_creation(self):
        """Test automatic directory creation"""
        logger.info("Testing directory creation")
        path = "subdir/test.txt"
        content = b"test content"

        # Test write to non-existent directory
        self.storage.write(path, content)

        # Verify directory and file were created
        dir_path = os.path.join(self.temp_dir, "subdir")
        file_path = os.path.join(dir_path, "test.txt")
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.exists(file_path))

        # Verify content
        with open(file_path, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_relative_path_handling(self):
        """Test handling of relative paths"""
        logger.info("Testing relative path handling")
        # Create a file in a subdirectory
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)
        file_path = os.path.join(subdir, "test.txt")
        content = b"test content"

        with open(file_path, "wb") as f:
            f.write(content)

        # Test reading with relative path
        result = self.storage.read("subdir/test.txt")
        self.assertEqual(result, content)

        # Test writing with relative path
        new_content = b"new content"
        self.storage.write("subdir/new.txt", new_content)
        new_file_path = os.path.join(subdir, "new.txt")
        self.assertTrue(os.path.exists(new_file_path))
        with open(new_file_path, "rb") as f:
            self.assertEqual(f.read(), new_content)


if __name__ == '__main__':
    unittest.main()