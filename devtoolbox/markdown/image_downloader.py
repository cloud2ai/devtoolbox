import logging
import os
import re

import requests
from pypinyin import lazy_pinyin
from retry import retry

from devtoolbox.images.convertor import convert_to_png
from devtoolbox.markdown.base import MarkdownBase


class MarkdownImageDownloader(MarkdownBase):
    """A class to handle markdown file operations and conversions.

    This class provides functionality to handle markdown files, including
    loading content from files or strings and converting markdown to other
    formats like docx.
    """

    @retry(
        exceptions=requests.RequestException,
        tries=3,
        delay=2,
        backoff=2,
        jitter=(0, 1)
    )
    def _download_image(self, image_url, save_path):
        """Download an image and save it to the specified path.

        Args:
            image_url (str): URL of the image to download.
            save_path (str): Path where the image will be saved.

        Returns:
            bool: True if download was successful.

        Raises:
            requests.RequestException: When the request fails.
        """
        logging.info(f"Downloading image {image_url} to {save_path}...")

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()  # Raises exception for non-200 status

        with open(save_path, "wb") as file:
            file.write(response.content)

        logging.info(f"Successfully downloaded image: {image_url}")
        return True

    def _convert_image(self, save_path):
        """Try to convert the image to PNG format.

        Args:
            save_path (str): Path to the image file.

        Returns:
            str: Path to the converted image or original if conversion failed.
        """
        try:
            converted_path = convert_to_png(save_path)
            if converted_path != save_path:
                logging.info(f"Converted image to PNG: {converted_path}")
                return converted_path
        except Exception as e:
            logging.error(f"Failed to convert image {save_path}: {str(e)}")

        return save_path

    def download_images(self, image_download_dir="images"):
        """Download images from markdown file.

        Downloads images from the markdown file and saves them to the
        images directory. Replace the original image lines with the new
        image lines.

        Args:
            image_download_dir (str): Directory to save downloaded images.
                Defaults to "images".

        Returns:
            str: Updated markdown content with local image references.
        """
        base_file_path = os.path.dirname(self.path)
        image_full_path = os.path.join(base_file_path, image_download_dir)
        image_relative_path = f"./{image_download_dir}"

        # Ensure images path exists
        if not os.path.exists(image_full_path):
            os.makedirs(image_full_path)
            logging.info(f"Created directory for images: {image_full_path}")

        # Split content into lines for processing, keep line endings
        lines = self.content.splitlines(True)
        updated_lines = []

        # Extract all image URLs first to count them
        image_urls = []
        for line in lines:
            if re.match(r"!\[(.*?)\]", line):
                find_pattern = re.compile(r'(https?://[^\s\)]+)')
                match = find_pattern.search(line)
                if match:
                    image_urls.append(match.group(1))

        # Process each line with natural ordering for images
        image_counter = 0
        for line in lines:
            # Match image URLs from markdown
            if re.match(r"!\[(.*?)\]", line):
                logging.debug(f"Found image line: {line}")
                find_pattern = re.compile(
                    r'(https?://[^\s\)]+)'
                )
                match = find_pattern.search(line)
                if not match:
                    logging.debug("No valid image URL found, skipping line.")
                    updated_lines.append(line)
                    continue

                image_url = match.group(1)
                md_basename = os.path.splitext(
                    os.path.basename(self.path)
                )[0]

                # Convert Chinese to pinyin and keep only alphanumeric chars
                md_basename = ''.join(lazy_pinyin(md_basename))
                md_basename = re.sub(r'[^a-zA-Z0-9]', '', md_basename)

                image_extname = os.path.splitext(
                    os.path.basename(image_url)
                )[1]

                # If no extension found, use .jpg as default
                if not image_extname:
                    image_extname = ".jpg"

                # Use natural ordering for image name (1, 2, 3, etc.)
                image_counter += 1
                # Format number with leading zeros based on total count
                padding = len(str(len(image_urls)))
                image_number = str(image_counter).zfill(padding)
                image_name = f"{md_basename}-{image_number}{image_extname}"
                save_path = os.path.join(image_full_path, image_name)

                if os.path.exists(save_path):
                    logging.warning(
                        f"Skip downloading image from {image_url} "
                        f"because it already exists at {save_path}"
                    )
                    updated_lines.append(line)
                    continue

                try:
                    # Download image with retry mechanism
                    download_success = self._download_image(
                        image_url, save_path
                    )

                    if download_success:
                        # Convert image format
                        converted_path = self._convert_image(save_path)
                        if converted_path != save_path:
                            # Update image name if conversion was successful
                            image_name = os.path.basename(converted_path)
                            save_path = converted_path

                        # Replace image reference
                        replace_image_line = (
                            f"![{image_name}]({image_relative_path}/"
                            f"{image_name})\n\n"
                        )
                        logging.debug(f"Old image line: {line}")
                        logging.debug(f"New image line: {replace_image_line}")
                        updated_lines.append(replace_image_line)
                    else:
                        # If download failed, keep original line
                        updated_lines.append(line)

                except Exception as e:
                    logging.error(
                        f"Failed to process image {image_url}: {str(e)}"
                    )
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Update the content in the object
        updated_content = ''.join(updated_lines)
        self.content = updated_content

        # Write updated content back to file
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
            logging.info(f"Updated markdown content written to: {self.path}")

        return updated_content
