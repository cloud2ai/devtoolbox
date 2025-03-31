import os
import logging
from io import BytesIO

import cairosvg
from PIL import Image


class ImageConverter:
    """A unified class for image conversion and manipulation.

    This class combines the functionality of file-based image conversion
    and raw image data processing, providing methods to convert images to PNG
    format and resize images while maintaining aspect ratio.
    """

    def __init__(self, source=None, output_format="png"):
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
                logging.debug(f"Initialized ImageConverter with file source: "
                              f"{self.source}")
                # Don't load the image yet, we'll do it when needed
            elif isinstance(source, bytes):
                self.source_type = "bytes"
                self.image_obj = Image.open(BytesIO(source))
                logging.debug("Initialized ImageConverter with byte source.")
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
                logging.debug(f"Output directory set to: {output_dir}")

            # Get the file extension
            _, extension = os.path.splitext(self.source)
            extension = extension.lower()
            logging.debug(f"File extension detected: {extension}")

            # Handle SVG files separately
            if extension == '.svg':
                png_image_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(self.source))[0] + \
                    '.png'
                )
                logging.info(f"Converting SVG to PNG: {self.source} -> "
                             f"{png_image_path}")
                cairosvg.svg2png(url=self.source, write_to=png_image_path)
                if remove_original:
                    os.remove(self.source)
                    logging.info(f"Removed original SVG file: {self.source}")
                logging.debug(f"Converted SVG {self.source} to {png_image_path}")
                return png_image_path

            # Check if the image is already in PNG format
            if extension == '.png':
                logging.debug(f"Image {self.source} is already in PNG format.")
                return self.source

            # Convert other image formats to PNG
            with Image.open(self.source) as image:
                png_image_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(self.source))[0] + \
                    '.png'
                )
                logging.info(f"Converting image to PNG: {self.source} -> "
                             f"{png_image_path}")

                # Save as PNG (if image is in RGBA mode, convert to RGB first)
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    background.save(png_image_path, 'PNG')
                    logging.debug(f"Saved RGBA image as PNG: {png_image_path}")
                else:
                    image.convert('RGB').save(png_image_path, 'PNG')
                    logging.debug(f"Saved image as PNG: {png_image_path}")

            if remove_original:
                os.remove(self.source)
                logging.info(f"Removed original image file: {self.source}")
            logging.debug(f"Converted {self.source} to {png_image_path}")
            return png_image_path

        except Exception as e:
            logging.error(f"Failed to convert image {self.source}: {str(e)}")
            return self.source

    def resize(self, width=None, height=None, maintain_aspect=True):
        """Resize an image while maintaining aspect ratio.

        Args:
            width (int, optional): The target width. If None, will be
                                  calculated based on height and aspect ratio.
            height (int, optional): The target height. If None, will be
                                   calculated based on width and aspect ratio.
            maintain_aspect (bool, optional): Whether to maintain the aspect
                                            ratio. Defaults to True.

        Returns:
            bytes: The resized image data as bytes if source is bytes.
            str: Path to the resized image if source is a file.

        Raises:
            ValueError: If neither width nor height is provided.
        """
        if width is None and height is None:
            raise ValueError("Either width or height must be provided")

        # Load image if not already loaded
        if self.image_obj is None:
            if self.source_type == "file":
                self.image_obj = Image.open(self.source)
                logging.debug(f"Loaded image from file: {self.source}")
            else:
                raise ValueError("Image source not properly initialized")
        orig_width, orig_height = self.image_obj.size
        logging.debug(f"Original image dimensions: width={orig_width}, "
                      f"height={orig_height}")

        # Calculate new dimensions
        if maintain_aspect:
            if width is not None and height is None:
                # Calculate height based on width
                ratio = width / orig_width
                new_width = width
                new_height = int(orig_height * ratio)
            elif height is not None and width is None:
                # Calculate width based on height
                ratio = height / orig_height
                new_height = height
                new_width = int(orig_width * ratio)
            elif width is not None and height is not None:
                # Use the smaller ratio to ensure image fits within bounds
                width_ratio = width / orig_width
                height_ratio = height / orig_height
                ratio = min(width_ratio, height_ratio)
                new_width = int(orig_width * ratio)
                new_height = int(orig_height * ratio)
        else:
            # Use provided dimensions without maintaining aspect ratio
            new_width = (width if width is not None
                          else orig_width)
            new_height = (height if height is not None
                           else orig_height)

        logging.info(f"Resizing image to width={new_width}, "
                     f"height={new_height}")
        resized_image = self.image_obj.resize((new_width, new_height))
        logging.debug("Image resized successfully.")

        # Handle output based on source type
        if self.source_type == "bytes":
            # Return bytes
            output = BytesIO()
            resized_image.save(output, format=self.output_format)
            logging.debug("Returning resized image as bytes.")
            return output.getvalue()
        else:
            # Save to file
            file_dir = os.path.dirname(self.source)
            file_name = os.path.splitext(
                os.path.basename(self.source))[0]
            output_path = os.path.join(
                file_dir, f"{file_name}_resized.{self.output_format}"
            )
            logging.debug(f"Saving resized image to file: {output_path}")

            if (resized_image.mode in ('RGBA', 'LA') and
                    self.output_format.lower() != 'png'):
                background = Image.new('RGB', resized_image.size,
                                        (255, 255, 255))
                background.paste(resized_image,
                                 mask=resized_image.split()[-1])
                background.save(output_path)
                logging.debug(f"Saved resized RGBA image to: {output_path}")
            else:
                resized_image.save(output_path)
                logging.debug(f"Saved resized image to: {output_path}")

            logging.info(f"Resized image saved successfully: {output_path}")
            return output_path
