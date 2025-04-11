#!/usr/bin/env python3
"""
Example script demonstrating the basic usage of ImageDownloader.

This script shows the most common use case of the ImageDownloader class,
with additional features documented in comments.
"""

import os
import logging
from pathlib import Path
import sys

# Add parent directory to system path to import devtoolbox
sys.path.append(str(Path(__file__).parent.parent))

from devtoolbox.images.downloader import ImageDownloader
from devtoolbox.storage import FileStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Run the basic image downloader example."""
    print("STARTING IMAGE DOWNLOADER EXAMPLE")
    print("=" * 50)

    # Get the absolute path of the script directory
    script_dir = Path(__file__).parent.absolute()
    download_dir = script_dir / "download_images"

    # Create downloads directory
    os.makedirs(download_dir, exist_ok=True)

    # Initialize storage with our specified directory
    storage = FileStorage(str(download_dir))

    # Basic example: Download images from URLs
    image_urls = [
        "https://picsum.photos/800/600",
        "https://picsum.photos/900/600",
        "https://picsum.photos/800/700"
    ]

    # Initialize the downloader with basic settings
    downloader = ImageDownloader(
        images=image_urls,
        path_prefix=str(download_dir),
        base_filename="sample",
        max_download_num=3,
        storage=storage  # Use our specified storage
    )

    try:
        # Download images and get their paths
        downloaded_images = downloader.download_images()
        print(f"Successfully downloaded images: {downloaded_images}")
    except Exception as e:
        print(f"Error during download: {str(e)}")
        logging.exception(e)

    print("=" * 50)

    # Additional features documentation:
    """
    The ImageDownloader class supports many advanced features:

    1. Image Filtering:
       - filter_width: Minimum width for images (default: 500)
       - filter_height: Minimum height for images (default: 500)
       - remove_duplicate: Remove duplicate images using perceptual hashing

    2. Image Processing:
       - convert_width: Resize images to specified width while maintaining
                        aspect ratio
       - use_cache: Use cached images if available

    3. Priority Images:
       - top_image: Specify a high-priority image URL that will be included
                    even if it doesn't meet filter criteria

    4. Search Integration:
       - enable_search_download: Enable downloading additional images from
                                search
       - search_keywords: Keywords to use for image search

    5. Storage Options:
       - storage: Custom storage backend for saving images
       - upload_images(): Method to upload downloaded images to additional
                         storage

    Example usage of advanced features:

    # With filtering and resizing
    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="downloads/filtered",
        base_filename="filtered",
        filter_width=800,
        filter_height=600,
        convert_width=1024,
        remove_duplicate=True
    )

    # With priority image
    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="downloads/priority",
        base_filename="priority",
        top_image=image_urls[0]
    )

    # With search integration
    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="downloads/search",
        base_filename="search",
        enable_search_download=True,
        search_keywords="nature landscape"
    )
    """

if __name__ == "__main__":
    main()