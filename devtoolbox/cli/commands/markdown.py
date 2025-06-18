import logging
import typer
from pathlib import Path
from typing import Optional

from devtoolbox.markdown.image_downloader import MarkdownImageDownloader
from devtoolbox.markdown.converter import MarkdownConverter
from devtoolbox.cli.utils import setup_logging

# Configure logging
logger = logging.getLogger("devtoolbox.markdown")
app = typer.Typer(help="Markdown processing commands")


@app.callback()
def callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
):
    """
    Markdown command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.markdown")


@app.command("download-images")
def download_images(
    markdown_file: str = typer.Argument(
        ...,
        help="Path to the markdown file",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "-o", "--output-dir",
        help="Output directory for downloaded images. If not specified, "
             "images will be saved in the same directory as the markdown file",
    ),
):
    """
    Download images from markdown file and replace remote URLs with local paths

    This command will:
    1. Download all images referenced in the markdown file
    2. Save them to the specified output directory (default: 'images'
       subdirectory)
    3. Replace remote image URLs in the markdown content with local
       relative paths
    4. Update the markdown file with the new local image references
    """
    # Convert to Path object
    md_path = Path(markdown_file)
    if not md_path.exists():
        typer.echo(f"Error: Markdown file not found: {markdown_file}")
        raise typer.Exit(1)

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = md_path.parent

    logger.debug(
        "Processing markdown file: %s (output_dir=%s)",
        markdown_file, output_path
    )

    try:
        # Initialize markdown processor
        processor = MarkdownImageDownloader(str(md_path))

        # Download images and update markdown content
        processor.download_images()

        typer.echo(f"Successfully processed markdown file: {markdown_file}")
        typer.echo("Images have been downloaded and markdown content updated")
        typer.echo(f"Images saved to: {output_path}")

    except Exception as e:
        logger.error(
            "Failed to process markdown file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to process markdown file: {str(e)}")
        raise typer.Exit(1)


@app.command("convert")
def convert_markdown(
    markdown_file: str = typer.Argument(
        ...,
        help="Path to the markdown file",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path. If not specified, will use the same name as "
             "the markdown file with .docx extension",
    ),
    download_images: bool = typer.Option(
        True,
        "-i", "--download-images",
        help="Whether to download images before conversion",
    ),
):
    """
    Convert markdown file to Word document (docx)
    """
    # Convert to Path object
    md_path = Path(markdown_file)
    if not md_path.exists():
        typer.echo(f"Error: Markdown file not found: {markdown_file}")
        raise typer.Exit(1)

    # Set output file path
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = md_path.with_suffix(".docx")

    logger.debug(
        "Converting markdown file: %s (output=%s, download_images=%s)",
        markdown_file, output_path, download_images
    )

    try:
        # Initialize converter with absolute path
        converter = MarkdownConverter(str(md_path.absolute()))

        # Download images if needed
        if download_images:
            image_downloader = MarkdownImageDownloader(str(md_path.absolute()))
            image_downloader.download_images()

        # Convert markdown to docx
        converter.to_docx(str(output_path.absolute()))

        typer.echo("Successfully converted markdown file to Word document")
        typer.echo(f"Output file: {output_path}")

    except Exception as e:
        logger.error(
            "Failed to convert markdown file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to convert markdown file: {str(e)}")
        raise typer.Exit(1)