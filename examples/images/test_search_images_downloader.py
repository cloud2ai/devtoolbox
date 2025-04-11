#!/usr/bin/env python3
"""
Example script demonstrating how to use ImageDownloader with search functionality.

This script shows how to download images from both provided URLs and search results.
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
    """Run the image downloader example with search functionality."""
    print("STARTING IMAGE DOWNLOADER WITH SEARCH EXAMPLE")
    print("=" * 50)

    # Get the absolute path of the script directory
    script_dir = Path(__file__).parent.absolute()
    download_dir = script_dir / "download_images"

    # Create downloads directory
    os.makedirs(download_dir, exist_ok=True)

    # Initialize storage with our specified directory
    storage = FileStorage(str(download_dir))

    # Initial image URLs
    image_urls = [
        "https://picsum.photos/800/600",
        "https://picsum.photos/900/600"
    ]

    # Initialize the downloader with search settings
    downloader = ImageDownloader(
        images=image_urls,
        path_prefix=str(download_dir),
        base_filename="search_sample",
        max_download_num=5,  # Total number of images to download
        enable_search_download=True,  # Enable search functionality
        search_keywords="beautiful landscape",  # Keywords for image search
        filter_width=800,  # Minimum width for images
        filter_height=600,  # Minimum height for images
        remove_duplicate=True,  # Remove duplicate images
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


if __name__ == "__main__":
    main()