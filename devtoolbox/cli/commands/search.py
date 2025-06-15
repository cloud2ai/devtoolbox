import logging
import typer

from devtoolbox.search_engine.duckduckgo import DuckDuckGoImageSearch

# Configure logging
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(help="Search engine commands")


@app.command("images")
def search_images(
    keywords: str = typer.Argument(
        ...,
        help="Search keywords",
    ),
    region: str = typer.Option(
        "us-en",
        "-r", "--region",
        help="Search region (e.g. us-en, cn-zh)",
    ),
    safesearch: str = typer.Option(
        "moderate",
        "-s", "--safesearch",
        help="Safe search level (off, moderate, strict)",
    ),
    size: str = typer.Option(
        "Large",
        "-z", "--size",
        help="Image size (Large, Medium, Small)",
    ),
    max_results: int = typer.Option(
        5,
        "-m", "--max-results",
        help="Maximum number of results to return",
    ),
    debug: bool = typer.Option(
        False,
        "-d", "--debug",
        help="Enable debug logging",
    ),
):
    """
    Search for images using DuckDuckGo
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug(
        "Searching images with keywords: %s (region=%s, safesearch=%s, "
        "size=%s, max_results=%s)",
        keywords, region, safesearch, size, max_results
    )

    try:
        search_engine = DuckDuckGoImageSearch(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            size=size,
            max_results=max_results
        )
        image_urls = search_engine.search_image_urls()

        typer.echo(f"Found {len(image_urls)} images:")
        for i, url in enumerate(image_urls, 1):
            typer.echo(f"{i}. {url}")

    except Exception as e:
        logger.error(
            "Failed to search images: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to search images: {str(e)}")
        raise typer.Exit(1)