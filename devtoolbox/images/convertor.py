import os
import logging
from io import BytesIO

import cairosvg
from PIL import Image

logger = logging.getLogger(__name__)


class ImageConverter:
    """A unified class for image conversion and manipulation.

    This class combines the functionality of file-based image conversion
    and raw image data processing, providing methods to convert images to PNG
    format and resize images while maintaining aspect ratio.
    """

    # Default values
    DEFAULT_OUTPUT_FORMAT = "png"
    DEFAULT_COMPRESSION_QUALITY = 85
    # White background for RGBA conversion
    DEFAULT_BACKGROUND_COLOR = (255, 255, 255)

    # File extensions
    PNG_EXTENSION = ".png"
    SVG_EXTENSION = ".svg"

    # Image modes
    RGB_MODE = "RGB"
    RGBA_MODE = "RGBA"
    LA_MODE = "LA"

    # Resize parameters
    DEFAULT_MAINTAIN_ASPECT = True
    DEFAULT_COMPRESS = True
    DEFAULT_MOBILE_WIDTH = 1080  # Common width for mobile displays

    def __init__(self, source=None, output_format=DEFAULT_OUTPUT_FORMAT):
        """Initialize the ImageConverter with a source and output format.

        Args:
            source (str or bytes, optional): Either a file path to an image or
                                            raw image data as bytes.
            output_format (str, optional): The desired output format.
                                          Defaults to "png".
        """
        self.source = source
        self.output_format = output_format.lower()
        self.image_obj = None

        # Initialize based on source type
        if source is not None:
            if isinstance(source, str) and os.path.isfile(source):
                self.source_type = "file"
                logger.debug(
                    f"Initialized ImageConverter with file source: "
                    f"{self.source}"
                )
                # Don't load the image yet, we'll do it when needed
            elif isinstance(source, bytes):
                self.source_type = "bytes"
                self.image_obj = Image.open(BytesIO(source))
                logger.debug("Initialized ImageConverter with byte source.")
            else:
                raise ValueError(
                    "Source must be either a file path or bytes object"
                )

    def convert_to_png(self, output_dir=None, remove_original=True):
        """Convert an image file to PNG format.

        Args:
            output_dir (str, optional): Directory to save the converted PNG
                                       image. Defaults to the same directory
                                       as the input image.
            remove_original (bool, optional): Whether to remove the original
                                             file after conversion.
                                             Defaults to True.

        Returns:
            str: Path to the converted PNG image if source is a file,
                 or None if source is bytes.

        Raises:
            ValueError: If the source is not a file path.
        """
        if self.source_type != "file":
            raise ValueError("This method requires a file path source")

        try:
            # Determine the output directory
            if output_dir is None:
                output_dir = os.path.dirname(self.source)
                logger.debug(f"Output directory set to: {output_dir}")

            # Get the file extension
            _, extension = os.path.splitext(self.source)
            extension = extension.lower()
            logger.debug(f"File extension detected: {extension}")

            # Handle SVG files separately
            if extension == self.SVG_EXTENSION:
                png_image_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(self.source))[0] +
                    self.PNG_EXTENSION
                )
                logger.info(
                    f"Converting SVG to PNG: {self.source} -> "
                    f"{png_image_path}"
                )
                cairosvg.svg2png(url=self.source, write_to=png_image_path)
                if remove_original:
                    os.remove(self.source)
                    logger.info(
                        f"Removed original SVG file: {self.source}"
                    )
                logger.debug(
                    f"Converted SVG {self.source} to {png_image_path}"
                )
                return png_image_path

            # Check if the image is already in PNG format
            if extension == self.PNG_EXTENSION:
                logger.debug(
                    f"Image {self.source} is already in PNG format."
                )
                return self.source

            # Convert other image formats to PNG
            with Image.open(self.source) as image:
                png_image_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(self.source))[0] +
                    self.PNG_EXTENSION
                )
                logger.info(
                    f"Converting image to PNG: {self.source} -> "
                    f"{png_image_path}"
                )

                # Save as PNG (if image is in RGBA mode, convert to RGB first)
                if image.mode in (self.RGBA_MODE, self.LA_MODE):
                    background = Image.new(
                        self.RGB_MODE,
                        image.size,
                        self.DEFAULT_BACKGROUND_COLOR
                    )
                    background.paste(image, mask=image.split()[-1])
                    background.save(png_image_path, 'PNG')
                    logger.debug(
                        f"Saved RGBA image as PNG: {png_image_path}"
                    )
                else:
                    image.convert(self.RGB_MODE).save(png_image_path, 'PNG')
                    logger.debug(
                        f"Saved image as PNG: {png_image_path}"
                    )

            if remove_original:
                os.remove(self.source)
                logger.info(
                    f"Removed original image file: {self.source}"
                )
            logger.debug(
                f"Converted {self.source} to {png_image_path}"
            )
            return png_image_path

        except Exception as e:
            logger.error(
                f"Failed to convert image {self.source}: {str(e)}"
            )
            return self.source

    def compress_image(self, image_obj, output_path=None, quality=9):
        """Compress an image with specified quality.

        Args:
            image_obj (PIL.Image): Image object to compress.
            output_path (str, optional): Path to save the compressed image.
            quality (int, optional): PNG compression level (0-9).
                Defaults to 9 (maximum compression).

        Returns:
            bytes: Compressed image data.
        """
        # Log original image information
        logging.info(
            "Starting image compression. Mode: %s, Size: %s, Format: %s" % (
                image_obj.mode,
                image_obj.size,
                image_obj.format
            )
        )

        # Convert RGBA or LA to RGB if needed
        if image_obj.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image_obj.size, (255, 255, 255))
            background.paste(image_obj, mask=image_obj.split()[-1])
            image_obj = background
        elif image_obj.mode not in ('RGB', 'L'):
            image_obj = image_obj.convert('RGB')

        # Save image with compression
        output = BytesIO()
        image_obj.save(
            output,
            format='PNG',
            optimize=True,
            compress_level=quality
        )
        compressed_data = output.getvalue()

        # Log compression results
        logging.info("Image compressed to %d bytes" % len(compressed_data))
        logging.info(
            "Compressed image size: %.2f MB" % (
                len(compressed_data) / (1024 * 1024)
            )
        )

        if output_path:
            logging.info(
                "Writing content type image to storage: %s..." % output_path
            )
            with open(output_path, 'wb') as f:
                f.write(compressed_data)

        return compressed_data

    def resize(
        self,
        width=None,
        height=None,
        maintain_aspect=DEFAULT_MAINTAIN_ASPECT
    ):
        """Resize an image while maintaining aspect ratio.

        Args:
            width (int, optional): Target width. Only resize if image is wider.
            height (int, optional): Target height. Only resize if image is taller.
            maintain_aspect (bool, optional): Whether to maintain aspect ratio.
                Defaults to True.

        Returns:
            bytes: Resized image data.
        """
        # Load image if not already loaded
        if self.image_obj is None:
            if self.source_type == "file":
                logger.info(f"Loading image from file: {self.source}")
                self.image_obj = Image.open(self.source)
                logger.debug(
                    f"Image loaded. Mode: {self.image_obj.mode}, "
                    f"Size: {self.image_obj.size}, "
                    f"Format: {self.image_obj.format}"
                )
            else:
                raise ValueError("Image source not properly initialized")

        # Log original dimensions
        orig_width, orig_height = self.image_obj.size
        logger.info(
            f"Original image dimensions: {orig_width}x{orig_height} pixels"
        )

        # If no dimensions provided, return original image
        if width is None and height is None:
            logger.info("No dimensions provided, returning original image")
            if self.source_type == "bytes":
                output = BytesIO()
                self.image_obj.save(output, format=self.output_format)
                return output.getvalue()
            else:
                return self.source

        # Only resize if image is larger than target dimensions
        if width and orig_width > width:
            # Calculate new dimensions
            if maintain_aspect:
                ratio = width / orig_width
                new_width = width
                new_height = int(orig_height * ratio)
                logger.info(
                    f"Resizing image to {new_width}x{new_height} pixels"
                )
                resized_image = self.image_obj.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
            else:
                resized_image = self.image_obj.resize(
                    (width, height),
                    Image.Resampling.LANCZOS
                )
        else:
            # Keep original size if image is smaller than target
            logger.info("Image is smaller than target width, keeping original size")
            resized_image = self.image_obj

        # Handle RGBA images
        if (resized_image.mode in (self.RGBA_MODE, self.LA_MODE) and
                self.output_format.lower() != 'png'):
            logger.debug(
                f"Converting RGBA image to RGB. "
                f"Original mode: {resized_image.mode}"
            )
            background = Image.new(
                self.RGB_MODE,
                resized_image.size,
                self.DEFAULT_BACKGROUND_COLOR
            )
            background.paste(
                resized_image,
                mask=resized_image.split()[-1]
            )
            processed_image = background
            logger.debug("RGBA to RGB conversion completed")
        else:
            processed_image = resized_image

        # Handle output based on source type
        if self.source_type == "bytes":
            logger.debug("Processing image as bytes")
            output = BytesIO()
            processed_image.save(output, format=self.output_format)
            return output.getvalue()
        else:
            # Save to file
            file_dir = os.path.dirname(self.source)
            file_name = os.path.splitext(
                os.path.basename(self.source))[0]
            output_path = os.path.join(
                file_dir, f"{file_name}_resized.{self.output_format}"
            )
            logger.info(f"Saving image to: {output_path}")

            processed_image.save(output_path)
            logger.info(
                f"Image saved successfully. "
                f"Size: {os.path.getsize(output_path)} bytes"
            )
            return output_path
