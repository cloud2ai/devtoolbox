import typer
import logging
from devtoolbox.cli.utils import setup_logging

app = typer.Typer(help="Whisper model management commands")

@app.command()
def download(
    model_size: str = typer.Option(
        "base",
        "--model-size",
        "-m",
        help="Whisper model size (tiny, base, small, medium, large)",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logging"
    ),
):
    """Download Whisper models for offline use"""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    try:
        import whisper  # Lazy import to speed up CLI startup
        logger.info(f"Downloading Whisper {model_size} model...")
        whisper.load_model(model_size)
        logger.info(f"Successfully downloaded Whisper {model_size} model")
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {str(e)}")
        raise typer.Exit(1)