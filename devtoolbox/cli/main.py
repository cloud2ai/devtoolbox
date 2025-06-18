"""
Main CLI entry point for devtoolbox
"""
import typer

# Import and register subcommands
from devtoolbox.cli.commands import (
    webhook, storage, jira, speech, whisper, search, images, markdown, llm,
    ocr
)

app = typer.Typer(
    name="devtoolbox",
    help="A collection of development tools and utilities",
    add_completion=False,
)

# Register webhook commands
app.add_typer(webhook.app, name="webhook", help="Webhook related commands")

# Register storage commands
app.add_typer(storage.app, name="storage", help="Storage related commands")

# Register jira commands
app.add_typer(jira.app, name="jira", help="JIRA related commands")

# Register speech commands
app.add_typer(speech.app, name="speech", help="Speech related commands")

# Register whisper commands
app.add_typer(
    whisper.app,
    name="whisper",
    help="Whisper model management commands"
)

# Register search commands
app.add_typer(
    search.app,
    name="search",
    help="Search engine related commands"
)

# Register images commands
app.add_typer(
    images.app,
    name="images",
    help="Image processing commands"
)

# Register markdown commands
app.add_typer(
    markdown.app,
    name="markdown",
    help="Markdown processing commands"
)

# Register llm commands
app.add_typer(llm.app, name="llm")

# Register ocr commands
app.add_typer(ocr.app, name="ocr", help="OCR related commands")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()