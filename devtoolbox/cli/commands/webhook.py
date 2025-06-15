"""
Webhook related commands
"""
import typer
from pathlib import Path
import base64
import hashlib
import logging
from typing import Optional, List
from devtoolbox.webhook import Webhook
from devtoolbox.cli.utils import setup_logging


# Configure logging
logger = logging.getLogger("devtoolbox.webhook")
app = typer.Typer(help="Webhook related commands")


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
    Webhook command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.webhook")


@app.command("text")
def send_text(
    url: str = typer.Option(
        ...,
        "-u", "--url",
        help="Webhook URL to send message to",
    ),
    content: str = typer.Option(
        ...,
        "-c", "--content",
        help="Text content to send",
    ),
    mention: Optional[List[str]] = typer.Option(
        None,
        "-m", "--mention",
        help="User IDs to mention (can be used multiple times)",
    ),
    mention_mobile: Optional[List[str]] = typer.Option(
        None,
        "-M", "--mention-mobile",
        help="Mobile numbers to mention (can be used multiple times)",
    ),
):
    """
    Send a text message via webhook
    """
    logger.debug(
        "Sending text message to %s: %s (mentions: %s, mobile: %s)",
        url, content, mention, mention_mobile
    )
    webhook = Webhook(url)
    try:
        webhook.send_text_message(
            content,
            mentioned_list=mention,
            mentioned_mobile_list=mention_mobile
        )
        typer.echo("Text message sent successfully")
    except Exception as e:
        logger.error(
            "Failed to send text message: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to send text message: {str(e)}")
        raise typer.Exit(1)


@app.command("markdown")
def send_markdown(
    url: str = typer.Option(
        ...,
        "-u", "--url",
        help="Webhook URL to send message to",
    ),
    content: str = typer.Option(
        ...,
        "-c", "--content",
        help="Markdown content to send",
    ),
):
    """
    Send a markdown message via webhook
    """
    logger.debug(
        "Sending markdown message to %s: %s",
        url, content
    )
    webhook = Webhook(url)
    try:
        webhook.send_markdown_message(content)
        typer.echo("Markdown message sent successfully")
    except Exception as e:
        logger.error(
            "Failed to send markdown message: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to send markdown message: {str(e)}")
        raise typer.Exit(1)


@app.command("image")
def send_image(
    url: str = typer.Option(
        ...,
        "-u", "--url",
        help="Webhook URL to send message to",
    ),
    file: Path = typer.Option(
        ...,
        "-f", "--file",
        help="Path to the image file to send",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """
    Send an image message via webhook
    """
    logger.debug(
        "Sending image message to %s from file: %s",
        url, file
    )
    webhook = Webhook(url)
    try:
        with open(file, "rb") as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            md5_hash = hashlib.md5(image_data).hexdigest()
            logger.debug(
                "Image processed: size=%d, md5=%s",
                len(image_data), md5_hash
            )

        webhook.send_image_message(base64_data, md5_hash)
        typer.echo("Image message sent successfully")
    except Exception as e:
        logger.error(
            "Failed to send image message: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to send image message: {str(e)}")
        raise typer.Exit(1)


@app.command("file")
def send_file(
    url: str = typer.Option(
        ...,
        "-u", "--url",
        help="Webhook URL to send message to",
    ),
    media_id: str = typer.Option(
        ...,
        "-i", "--media-id",
        help="Media ID of the file to send",
    ),
):
    """
    Send a file message via webhook
    """
    logger.debug(
        "Sending file message to %s with media_id: %s",
        url, media_id
    )
    webhook = Webhook(url)
    try:
        webhook.send_file_message(media_id)
        typer.echo("File message sent successfully")
    except Exception as e:
        logger.error(
            "Failed to send file message: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to send file message: {str(e)}")
        raise typer.Exit(1)