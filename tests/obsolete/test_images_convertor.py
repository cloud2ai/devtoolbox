import unittest
import os
import tempfile
import shutil
from PIL import Image
from devtoolbox.images.convertor import ImageConverter
from tests.utils.test_logging import setup_test_logging
from io import BytesIO


# Initialize logging
logger = setup_test_logging()


class TestImageConverter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create test images
        self.create_test_images()

    def tearDown(self):
        """Clean up test fixtures."""
        logger.info("Cleaning up test fixtures")
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)

    def create_test_images(self):
        """Create test images in different formats."""
        # Create a test PNG image
        self.png_path = os.path.join(self.test_dir, "test.png")
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.png_path)

        # Create a test JPG image
        self.jpg_path = os.path.join(self.test_dir, "test.jpg")
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(self.jpg_path)

        # Create a test RGBA image
        self.rgba_path = os.path.join(self.test_dir, "test_rgba.png")
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(self.rgba_path)

    def test_convert_to_png_from_png(self):
        """Test converting PNG to PNG (should return original)."""
        logger.info("Testing PNG to PNG conversion")

        converter = ImageConverter(self.png_path)
        result = converter.convert_to_png()

        self.assertEqual(result, self.png_path)
        logger.debug("Verified PNG to PNG conversion")

    def test_convert_to_png_from_jpg(self):
        """Test converting JPG to PNG."""
        logger.info("Testing JPG to PNG conversion")

        converter = ImageConverter(self.jpg_path)
        result = converter.convert_to_png()

        # Check if file was created
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith('.png'))

        # Check if original was removed
        self.assertFalse(os.path.exists(self.jpg_path))
        logger.debug("Verified JPG to PNG conversion")

    def test_convert_to_png_from_rgba(self):
        """Test converting RGBA image to PNG."""
        logger.info("Testing RGBA to PNG conversion")
        
        converter = ImageConverter(self.rgba_path)
        result = converter.convert_to_png()
        
        # Check if file was created
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith('.png'))
        
        # Check if image mode is preserved (PNG supports RGBA)
        with Image.open(result) as img:
            self.assertEqual(img.mode, 'RGBA')
        logger.debug("Verified RGBA to PNG conversion")

    def test_convert_to_png_with_output_dir(self):
        """Test converting to PNG with custom output directory."""
        logger.info("Testing PNG conversion with custom output directory")

        output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        converter = ImageConverter(self.jpg_path)
        result = converter.convert_to_png(output_dir=output_dir)

        # Check if file was created in correct directory
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.startswith(output_dir))
        logger.debug("Verified PNG conversion with custom output directory")

    def test_convert_to_png_keep_original(self):
        """Test converting to PNG while keeping original file."""
        logger.info("Testing PNG conversion with keep original")

        converter = ImageConverter(self.jpg_path)
        result = converter.convert_to_png(remove_original=False)

        # Check if both files exist
        self.assertTrue(os.path.exists(result))
        self.assertTrue(os.path.exists(self.jpg_path))
        logger.debug("Verified PNG conversion with keep original")

    def test_resize_with_width(self):
        """Test resizing image with width constraint."""
        logger.info("Testing image resize with width constraint")

        converter = ImageConverter(self.png_path)
        result = converter.resize(width=50)

        # Check if file was created
        self.assertTrue(os.path.exists(result))

        # Check dimensions
        with Image.open(result) as img:
            self.assertEqual(img.size[0], 50)
            self.assertEqual(img.size[1], 50)  # Aspect ratio maintained
        logger.debug("Verified image resize with width constraint")

    def test_resize_with_height(self):
        """Test resizing image with height constraint."""
        logger.info("Testing image resize with height constraint")

        converter = ImageConverter(self.png_path)
        result = converter.resize(height=75)

        # Check if file was created
        self.assertTrue(os.path.exists(result))

        # Check dimensions
        with Image.open(result) as img:
            self.assertEqual(img.size[0], 75)  # Aspect ratio maintained
            self.assertEqual(img.size[1], 75)
        logger.debug("Verified image resize with height constraint")

    def test_resize_with_both_dimensions(self):
        """Test resizing image with both width and height constraints."""
        logger.info("Testing image resize with both dimensions")

        converter = ImageConverter(self.png_path)
        result = converter.resize(width=60, height=40)

        # Check if file was created
        self.assertTrue(os.path.exists(result))

        # Check dimensions
        with Image.open(result) as img:
            self.assertEqual(img.size[0], 40)  # Aspect ratio maintained
            self.assertEqual(img.size[1], 40)
        logger.debug("Verified image resize with both dimensions")

    def test_resize_without_aspect_ratio(self):
        """Test resizing image without maintaining aspect ratio."""
        logger.info("Testing image resize without aspect ratio")

        converter = ImageConverter(self.png_path)
        result = converter.resize(width=60, height=40, maintain_aspect=False)

        # Check if file was created
        self.assertTrue(os.path.exists(result))

        # Check dimensions
        with Image.open(result) as img:
            self.assertEqual(img.size[0], 60)
            self.assertEqual(img.size[1], 40)
        logger.debug("Verified image resize without aspect ratio")

    def test_resize_with_bytes_source(self):
        """Test resizing image with bytes source."""
        logger.info("Testing image resize with bytes source")
        
        with open(self.png_path, 'rb') as f:
            image_bytes = f.read()
        
        converter = ImageConverter(image_bytes)
        result = converter.resize(width=50)
        
        # Check if result is bytes
        self.assertIsInstance(result, bytes)
        
        # Check dimensions
        with Image.open(BytesIO(result)) as img:
            self.assertEqual(img.size[0], 50)
            self.assertEqual(img.size[1], 50)
            self.assertEqual(img.mode, 'RGB')  # Ensure output is RGB
        logger.debug("Verified image resize with bytes source")

    def test_invalid_source(self):
        """Test handling of invalid image source."""
        logger.info("Testing invalid image source handling")

        with self.assertRaises(ValueError):
            ImageConverter("nonexistent.jpg")
        logger.debug("Verified invalid source handling")

    def test_invalid_resize_parameters(self):
        """Test handling of invalid resize parameters."""
        logger.info("Testing invalid resize parameters")

        converter = ImageConverter(self.png_path)
        with self.assertRaises(ValueError):
            converter.resize()  # No width or height provided
        logger.debug("Verified invalid resize parameters")


if __name__ == '__main__':
    unittest.main()