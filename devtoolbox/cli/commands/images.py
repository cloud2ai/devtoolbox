import logging
import typer
from pathlib import Path
from typing import Optional

from devtoolbox.images.downloader import ImageDownloader
from devtoolbox.storage import FileStorage

# Configure logging
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(help="Image processing commands")


@app.command("download")
def download_images(
    urls: list[str] = typer.Argument(
        ...,
        help="List of image URLs to download",
    ),
    output_dir: str = typer.Option(
        "downloads",
        "-o", "--output-dir",
        help="Output directory for downloaded images",
    ),
    base_filename: str = typer.Option(
        "image",
        "-f", "--filename",
        help="Base filename for downloaded images",
    ),
    max_download: int = typer.Option(
        5,
        "-m", "--max-download",
        help="Maximum number of images to download",
    ),
    min_width: int = typer.Option(
        500,
        "-w", "--min-width",
        help="Minimum width for images",
    ),
    min_height: int = typer.Option(
        500,
        "-h", "--min-height",
        help="Minimum height for images",
    ),
    max_width: int = typer.Option(
        1280,
        "-W", "--max-width",
        help="Maximum width for images after conversion",
    ),
    enable_search: bool = typer.Option(
        False,
        "-s", "--enable-search",
        help="Enable downloading additional images from search",
    ),
    search_keywords: Optional[str] = typer.Option(
        None,
        "-k", "--search-keywords",
        help="Keywords for image search",
    ),
    debug: bool = typer.Option(
        False,
        "-d", "--debug",
        help="Enable debug logging",
    ),
):
    """
    Download and process images from URLs
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize storage
    storage = FileStorage(str(output_path))

    logger.debug(
        "Downloading images with settings: urls=%s, output_dir=%s, "
        "base_filename=%s, max_download=%s, min_width=%s, min_height=%s, "
        "max_width=%s, enable_search=%s, search_keywords=%s",
        urls, output_dir, base_filename, max_download, min_width, min_height,
        max_width, enable_search, search_keywords
    )

    try:
        # Initialize downloader
        downloader = ImageDownloader(
            images=urls,
            path_prefix=str(output_path),
            base_filename=base_filename,
            max_download_num=max_download,
            filter_width=min_width,
            filter_height=min_height,
            convert_width=max_width,
            enable_search_download=enable_search,
            search_keywords=search_keywords,
            storage=storage
        )

        # Download images
        downloaded_images = downloader.download_images()

        typer.echo(f"Successfully downloaded {len(downloaded_images)} images:")
        for image_path in downloaded_images:
            typer.echo(f"- {image_path}")

    except Exception as e:
        logger.error(
            "Failed to download images: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to download images: {str(e)}")
        raise typer.Exit(1)