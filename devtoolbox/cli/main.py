"""
Main CLI entry point for devtoolbox
"""
import typer

# Import and register subcommands
from devtoolbox.cli.commands import webhook, storage, jira, speech, whisper

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
app.add_typer(whisper.app, name="whisper", help="Whisper model management commands")


def main():
    """Main entry point for the CLI"""
    app()

if __name__ == "__main__":
    main()