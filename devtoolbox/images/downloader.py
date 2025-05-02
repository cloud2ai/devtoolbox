import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from io import BytesIO
import os
import tempfile

import cairosvg
import imagehash
import requests
from PIL import Image, UnidentifiedImageError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from devtoolbox.storage import FileStorage
from devtoolbox.search_engine.duckduckgo import DuckDuckGoImageSearch
from devtoolbox.images.convertor import ImageConverter

# Constants for image downloader settings

# Minimum width for images to be considered valid
MIN_IMAGE_WIDTH = 500

# Minimum height for images to be considered valid
MIN_IMAGE_HEIGHT = 500

# Allowed aspect ratio range (width/height)
ASPECT_RATIO_RANGE = (0.5, 2.0)

# Maximum number of images to download
MAX_DOWNLOAD_IMAGE_NUM = 5

# Timeout duration for image download requests
IMAGE_DOWNLOAD_TIMEOUT = 10

# Maximum width for images after conversion
CONVERT_IMAGE_MAX_WIDTH = 1280

# Maximum number of trending images to retrieve
MAX_TRENDINGS_NUM = 25

# Number of worker threads for collecting images
COLLECTOR_WORKERS = 2


class ImageDownloader:
    """Image downloader for HTTP images"""

    def __init__(
        self,
        images,
        path_prefix,
        base_filename,
        storage=None,
        max_download_num=MAX_DOWNLOAD_IMAGE_NUM,
        filter_width=MIN_IMAGE_WIDTH,
        filter_height=MIN_IMAGE_HEIGHT,
        convert_width=CONVERT_IMAGE_MAX_WIDTH,
        top_image=None,
        use_cache=True,
        remove_duplicate=True,
        enable_search_download=False,
        search_keywords=None,
        compress=True
    ):
        """Initialize the ImageDownloader with the specified parameters.

        This class handles downloading images from URLs, filtering them based on
        size and aspect ratio criteria, removing duplicates, and optionally
        searching for additional images when needed. All images are converted to
        PNG format during the process.

        The parameters are organized in several logical groups:
        - Core parameters: images, path_prefix, base_filename
        - Storage parameter: extra storage for downloaded images (optional)
        - Image filtering parameters: filter_width, filter_height, etc.
        - Image processing parameters: convert_width, use_cache, etc.
        - Search-related parameters: enable_search_download, search_keywords

        Args:
            images (list): List of image URLs to download.
            path_prefix (str): Directory path where images will be saved.
            base_filename (str): Base name for saved image files. Each downloaded
                image will be named as "{base_filename}-{index}.png".
            storage (Storage, optional): Storage object used to save downloaded images.
                If None, a default file system storage will be used or the class will
                operate in memory-only mode. Defaults to None.
            max_download_num (int, optional): Maximum number of images to download.
                This limits the total number of images processed.
                Defaults to MAX_DOWNLOAD_IMAGE_NUM (5).
            filter_width (int, optional): Minimum width for images to be considered valid.
                Images smaller than this will be filtered out unless they are the top image.
                Defaults to MIN_IMAGE_WIDTH (500).
            filter_height (int, optional): Minimum height for images to be considered valid.
                Images smaller than this will be filtered out unless they are the top image.
                Defaults to MIN_IMAGE_HEIGHT (500).
            convert_width (int, optional): Maximum width for images after conversion.
                Images wider than this will be resized while maintaining aspect ratio.
                Defaults to CONVERT_IMAGE_MAX_WIDTH (1280).
            top_image (str, optional): URL of high priority image that should be included
                even if it doesn't meet filter criteria. This ensures that at least
                one important image is always included. Defaults to None.
            use_cache (bool, optional): Whether to use cached images if available.
                If True, existing images with the same path won't be re-downloaded.
                Defaults to True.
            remove_duplicate (bool, optional): Whether to remove duplicate images.
                If True, images with the same perceptual hash will be considered duplicates.
                Defaults to True.
            enable_search_download (bool, optional): Whether to download additional
                images from search engine if max_download_num is not reached.
                Defaults to False.
            search_keywords (str, optional): Keywords to use for image search if
                enable_search_download is True. These keywords help find relevant images.
                Defaults to None.
            compress (bool, optional): Whether to compress images after resizing.
                Defaults to True.

        Note:
            The class uses image hashing to detect and filter duplicate images,
            even if they have different sizes or slight variations.

        Example:
            downloader = ImageDownloader(
                images=["http://example.com/image1.jpg", "http://example.com/image2.png"],
                path_prefix="downloads/article123",
                base_filename="article-img",
                max_download_num=3
            )
            image_paths = downloader.download_images()
        """
        self.images = images
        self.path_prefix = path_prefix
        self.base_filename = base_filename

        if storage is None:
            self.path_prefix = tempfile.mkdtemp()
            # Initialize FileStorage with the temporary directory
            self.storage = FileStorage(self.path_prefix)
        else:
            self.storage = storage

        self.max_download_num = max_download_num
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.convert_width = convert_width
        self.top_image = top_image
        self.use_cache = use_cache
        self.remove_duplicate = remove_duplicate
        self.compress = compress

        self.enable_search_download = enable_search_download
        self.search_keywords = search_keywords

        self._validate_configuration()

    def _validate_configuration(self):
        """Validate the configuration and set up any defaults needed.

        This method performs several validation tasks:
        1. Validates that there are images to download
        2. Ensures the target directory exists

        If validation issues are found, appropriate warnings are logged but
        operation continues when possible.
        """
        if not self.images:
            logging.warning("No images provided for downloading.")

        if not os.path.exists(self.path_prefix):
            logging.info(f"Creating directory: {self.path_prefix}")
            os.makedirs(self.path_prefix, exist_ok=True)

    def _parallel_download_images(self, all_images):
        """Download image in parallel model

        Return a dict with image_url => image_hash
        """
        results = {}
        with ThreadPoolExecutor(max_workers=COLLECTOR_WORKERS) as executor:
            futures = []
            article_images = all_images
            for idx, url in enumerate(article_images):
                future = executor.submit(self._download_image, idx, url)
                futures.append(future)

        filter_images_hashes = []
        for future in as_completed(futures):
            result = future.result()

            # Skip to empty image
            if not result["content"]:
                continue

            image_url = result["image_url"]
            image_hash = result["hash"]

            # NOTE(Ray): Use image hash to remove duplicate images, dhash
            # can remove duplicate images with different size and colors
            if image_hash in filter_images_hashes:
                logging.warn("Duplicate image found, skip to "
                             "download image from %s" % image_url)
                continue
            else:
                logging.debug("Add image hash %s" % image_hash)
                filter_images_hashes.append(image_hash)

            results[image_url] = result

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            UnidentifiedImageError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException
        )),
        reraise=True
    )
    def _download_image(self, idx, image_url):
        """Download a single image with automatic retry mechanism.

        Args:
            idx (int): Index of the image in the list.
            image_url (str): URL of the image to download.

        Returns:
            dict: Image information including content and hash.
        """
        # Initial a dict to save image content and hash
        image_infos = {
            "index": idx,
            "image_url": image_url,
            "content": None,
            "hash": None
        }

        try:
            logging.info(f"Downloading image from {image_url}")

            save_content = None
            if image_url.startswith('data:image/'):
                image_data = image_url.split(',')[-1]
                save_content = base64.b64decode(image_data)
            else:
                response = requests.get(
                    image_url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    logging.warning(
                        f"Invalid content type for {image_url}: {content_type}"
                    )
                    return image_infos
                save_content = response.content

            if image_url.endswith('.svg'):
                save_content = cairosvg.svg2png(bytestring=save_content)

            # Verify image content
            try:
                image_obj = Image.open(BytesIO(save_content))
                # Try to load the image to verify it's valid
                image_obj.load()
            except Exception as e:
                logging.warning(
                    f"Invalid image content from {image_url}: {str(e)}"
                )
                return image_infos

            width, height = image_obj.size
            aspect_ratio = width / height

            # Output original image information
            logging.info(
                "Original image size: %dx%d (%.2f MB)" % (
                    width,
                    height,
                    len(save_content) / (1024 * 1024)
                )
            )

            image_infos["hash"] = imagehash.dhash(image_obj)

            # if image is top image, we still put it in filter images
            # to ensure we can have a image
            if image_url == self.top_image:
                logging.info("Filtered top image by default %s" % image_url)
                image_infos["content"] = save_content
                return image_infos

            is_size_valid = (
                width >= self.filter_width and
                height >= self.filter_height
            )
            is_aspect_ratio_valid = (
                aspect_ratio > ASPECT_RATIO_RANGE[0] and
                aspect_ratio < ASPECT_RATIO_RANGE[1]
            )

            if is_size_valid and is_aspect_ratio_valid:
                logging.info("Success to filter image %s" % image_url)
                image_infos["content"] = save_content

                # Output resized image information
                if self.convert_width and width > self.convert_width:
                    new_height = int(height * (self.convert_width / width))
                    logging.info(
                        "Image will be resized to: %dx%d" % (
                            self.convert_width,
                            new_height
                        )
                    )
            else:
                logging.warning(
                    f"Ignore image {image_url} due to valid size: "
                    f"{is_size_valid}, invalid aspect ratio: "
                    f"{not is_aspect_ratio_valid}"
                )

        except UnidentifiedImageError:
            logging.warning(
                f"Failed to download image from {image_url}, "
                f"due to invalid image format"
            )
            raise
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout downloading image from {image_url}")
            raise
        except requests.exceptions.RequestException as e:
            logging.warning(
                f"Network error downloading image from {image_url}: "
                f"{str(e)}"
            )
            raise
        except Exception as e:
            logging.warning(
                f"Failed to download image from {image_url}, "
                f"due to: {str(e)}"
            )
            raise

        return image_infos

    def download_images(self):
        logging.debug("Download images in parallel model...")

        # NOTE(Ray): If search images is enabled, added search images into
        # download image list
        all_images = list(self.images)
        if self.enable_search_download:
            search_engine = DuckDuckGoImageSearch(
                self.search_keywords, max_results=MAX_DOWNLOAD_IMAGE_NUM)
            search_images = search_engine.search_image_urls()
            all_images.extend(search_images)

        logging.info(f"Trying to filter ({len(all_images)}) images")
        logging.debug(f"All Images: {all_images}")
        filter_images = self._parallel_download_images(all_images)
        logging.info(f"Get ({len(filter_images)}) filtered images")

        sorted_images = sorted(
            filter_images.values(), key=lambda x: x["index"])

        # Get specific numbers of images
        filtered_images = sorted_images[0:self.max_download_num]

        collect_images = []
        save_index = 0
        for idx, collect_image in enumerate(filtered_images):
            image_filename = "%s-%s.png" % (
                self.base_filename, save_index)
            image_path = os.path.join(self.path_prefix, image_filename)

            image_url = collect_image["image_url"]
            save_content = collect_image["content"]
            logging.info("Downloading image from %s to %s" % (
                image_url, image_path))

            if self.storage.exists(image_path) and self.use_cache:
                logging.info("Image already exists in %s" % image_path)
            else:
                logging.info("Writing image to path %s" % image_path)

                image_converter = ImageConverter(save_content)
                try:
                    # First resize the image
                    save_content = image_converter.resize(self.convert_width)
                    # Then compress the resized image if enabled
                    if self.compress:
                        save_content = image_converter.compress_image(
                            Image.open(BytesIO(save_content)),
                            image_path
                        )
                        # Output compressed image information
                        logging.info(
                            "Compressed image size: %.2f MB" % (
                                len(save_content) / (1024 * 1024)
                            )
                        )
                except Exception as e:
                    logging.warn(f"Ignore to save image "
                               f"{image_url} due to: {e}")
                    continue

                self.storage.write(
                    image_path, save_content, content_type="image")

            collect_images.append(self.storage.full_path(image_path))

            save_index += 1

        return collect_images

    def upload_images(self, storage, image_paths):
        """Upload downloaded images to extra stroage"""
        logging.info("Uploading images %s to storage..." % image_paths)
        upload_images = []
        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            dest_image_path = os.path.join(
                self.path_prefix, image_filename)
            storage.cp_from_path(image_path, dest_image_path)
            upload_path = storage.full_path(dest_image_path)
            logging.info("Success to upload image to %s" % upload_path)
            upload_images.append(storage.full_path(dest_image_path))

        return upload_images

    def serial_download_images(self):
        """Download images one by one"""
        # TODO(Ray): This method is not implemented enable_search_download
        article_images = self.images
        filter_images = []
        filter_images_hashes = []
        for idx, image_url in enumerate(article_images):
            image_filename = "%s-%s.png" % (self.base_filename, idx)
            image_path = os.path.join(self.path_prefix, image_filename)

            logging.info("Filtering image from %s..." % image_url)
            try:
                save_content = None
                if image_url.startswith('data:image/'):
                    image_data = image_url.split(',')[-1]
                    save_content = base64.b64decode(image_data)
                else:
                    response = requests.get(
                        image_url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
                    save_content = response.content

                if image_url.endswith('.svg'):
                    save_content = cairosvg.svg2png(bytestring=save_content)

                image_obj = Image.open(BytesIO(save_content))
                width, height = image_obj.size

                logging.debug("Image info: %s width = %s, height = %s" % (
                    image_url, width, height))

                # NOTE(Ray): Use image hash to remove duplicate images, dhash
                # can remove duplicate images with different size and colors
                image_hash = imagehash.dhash(image_obj)
                if image_hash in filter_images_hashes:
                    logging.warn("Duplicate image found, skip to "
                                 "download image from %s" % image_url)
                    continue
                else:
                    logging.debug("Add image hash %s" % image_hash)
                    filter_images_hashes.append(image_hash)

                # if image is top image, we still put it in filter images
                # to ensure we can have a image
                if image_url == self.top_image:
                    logging.info("Filtered top image "
                                 "by default %s" % image_url)
                    filter_images.append(save_content)
                    continue

                if width >= self.filter_width and height >= self.filter_height:
                    logging.info("Filtered image %s" % image_url)
                    filter_images.append(save_content)
                else:
                    logging.warn("Ignore image %s due to "
                                 "image is too small." % image_url)
            except UnidentifiedImageError:
                logging.warn("Failed to download image from %s to %s, "
                             "due to download content is not valid "
                             "image format" % (image_url, image_path))
            except requests.exceptions.Timeout:
                logging.warn("Skip to download image from %s to %s "
                             "due to requests timeout" % (
                                 image_url, image_path))
            except Exception as e:
                logging.warn("Failed to download image from %s to %s, "
                             "due to: %s" % (image_url, image_path, e))
                # logging.exception(e)

        # logging.debug("Filtered images: %s" % filter_images)
        # Get specific numbers of images
        filtered_images = filter_images[0:self.max_download_num]

        collect_images = []
        save_index = 0
        for idx, collect_image in enumerate(filtered_images):
            image_filename = "%s-%s.png" % (
                self.base_filename, save_index)
            image_path = os.path.join(self.path_prefix, image_filename)

            logging.info("Downloading image from %s to %s" % (
                image_url, image_path))
            save_content = collect_image

            if self.storage.exists(image_path) and self.use_cache:
                logging.info("Image already exists in %s" % image_path)
            else:
                logging.info("Writting image to path %s" % image_path)

                image_converter = ImageConverter(save_content)
                save_content = image_converter.resize(self.convert_width)

                self.storage.write(
                    image_path, save_content, content_type="image")

            collect_images.append(self.storage.full_path(image_path))

            save_index += 1

        return collect_images