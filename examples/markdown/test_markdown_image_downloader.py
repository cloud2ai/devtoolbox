import os
import sys
import logging
from pathlib import Path

# Add parent directory to system path to import devtoolbox
sys.path.append(str(Path(__file__).parent.parent))

from devtoolbox.markdown.image_downloader import MarkdownImageDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_markdown():
    """Create a sample markdown file with image links."""

    # Sample markdown content with network image links
    markdown_content = """# Markdown Image Downloader Test

## Sample Images

Here are some images from the web:

![Sample Image 1](https://images.unsplash.com/photo-1501854140801-50d01698950b?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=60)

![Sample Image 2](https://images.unsplash.com/photo-1426604966848-d7adac402bff?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=60)

## Text Content

This is some regular text content without images.

## Another Image

![Sample Image 3](https://images.unsplash.com/photo-1472214103451-9374bd1c798e?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=60)

## Testing Retry Mechanism

Below is an image link that might need retries (unstable link):

![Unstable Image](https://httpbin.org/image/jpeg)

"""

    # Create sample file
    example_dir = Path(__file__).parent
    example_file = example_dir / "sample_markdown.md"

    with open(example_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Created sample markdown file: {example_file}")
    return example_file


def test_markdown_image_downloader():
    """Test the MarkdownImageDownloader class functionality."""

    # Create sample markdown file
    markdown_file = create_sample_markdown()

    try:
        # Initialize MarkdownImageDownloader
        print("\nStarting to process markdown file...")
        downloader = MarkdownImageDownloader(path=str(markdown_file))

        # Download images and update markdown content
        print("Starting to download images (with retry mechanism)...")
        custom_image_dir = "downloaded_images"
        downloader.download_images(image_download_dir=custom_image_dir)

        # Show image download location
        image_dir = Path(markdown_file).parent / custom_image_dir
        print(f"\nImages downloaded to: {image_dir}")

        # List downloaded images
        if image_dir.exists():
            print("\nList of downloaded images:")
            for img_file in image_dir.glob("*"):
                print(f"- {img_file.name}")

        print(f"\nMarkdown file has been updated: {markdown_file}")
        print("You can open this file to see if image references "
              "have been updated to local paths")

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    test_markdown_image_downloader()
