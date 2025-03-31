#!/usr/bin/env python3
"""
Example script demonstrating the usage of ImageDownloader.

This script shows different ways to use the ImageDownloader class,
including basic usage, advanced filtering, and search functionality.
"""

import os
import logging
from pathlib import Path
import sys

# Add parent directory to system path to import devtoolbox
sys.path.append(str(Path(__file__).parent.parent))

from devtoolbox.images.downloader import ImageDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_basic_download():
    """Demonstrate basic image downloading functionality."""
    print("\n=== Testing Basic Image Download ===")

    # Sample image URLs (using placeholder URLs)
    image_urls = [
        "https://picsum.photos/800/600",
        "https://picsum.photos/900/600",
        "https://picsum.photos/800/700"
    ]

    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="examples/downloads/basic",
        base_filename="sample",
        max_download_num=3
    )

    try:
        downloaded_images = downloader.download_images()
        print(f"Successfully downloaded images: {downloaded_images}")
    except Exception as e:
        print(f"Error during basic download: {str(e)}")
        logging.exception(e)

def test_filtered_download():
    """Demonstrate image downloading with size filtering."""
    print("\n=== Testing Filtered Image Download ===")

    image_urls = [
        "https://picsum.photos/300/200",  # Should be filtered out (too small)
        "https://picsum.photos/1200/800",  # Should be included
        "https://picsum.photos/200/1000"   # Should be filtered out (wrong ratio)
    ]

    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="examples/downloads/filtered",
        base_filename="filtered",
        filter_width=800,
        filter_height=600,
        convert_width=1024,
        remove_duplicate=True
    )

    try:
        downloaded_images = downloader.download_images()
        print(f"Successfully downloaded filtered images: {downloaded_images}")
    except Exception as e:
        print(f"Error during filtered download: {str(e)}")

def test_priority_download():
    """Demonstrate downloading with a priority (top) image."""
    print("\n=== Testing Priority Image Download ===")

    image_urls = [
        "https://picsum.photos/300/200",
        "https://picsum.photos/400/300",
        "https://picsum.photos/350/250"
    ]

    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="examples/downloads/priority",
        base_filename="priority",
        filter_width=800,  # Higher than actual image sizes
        filter_height=600,
        top_image=image_urls[0],  # First image will be included despite size
        max_download_num=2
    )

    try:
        downloaded_images = downloader.download_images()
        print(f"Successfully downloaded priority images: {downloaded_images}")
    except Exception as e:
        print(f"Error during priority download: {str(e)}")

def test_search_download():
    """Demonstrate image downloading with search functionality."""
    print("\n=== Testing Search and Download ===")

    image_urls = [
        "https://picsum.photos/800/600"
    ]

    downloader = ImageDownloader(
        images=image_urls,
        path_prefix="examples/downloads/search",
        base_filename="search",
        max_download_num=3,
        enable_search_download=True,
        search_keywords="nature landscape"
    )

    try:
        downloaded_images = downloader.download_images()
        print(f"Successfully downloaded images with search: {downloaded_images}")
    except Exception as e:
        print(f"Error during search download: {str(e)}")

def cleanup_example_files():
    """Clean up downloaded files after tests."""
    try:
        import shutil
        downloads_dir = Path("examples/downloads")
        if downloads_dir.exists():
            shutil.rmtree(downloads_dir)
            print("\nCleaned up example files.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def main():
    """Run all example scenarios."""
    print("STARTING IMAGE DOWNLOADER EXAMPLES")
    print("=" * 50)

    # Create downloads directory
    os.makedirs("examples/downloads", exist_ok=True)

    try:
        # Run all test scenarios
        test_basic_download()
        test_filtered_download()
        test_priority_download()
        test_search_download()

        # Clean up after tests
        cleanup_example_files()

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during examples: {str(e)}")
        logging.exception(e)

    print("=" * 50)

if __name__ == "__main__":
    main()