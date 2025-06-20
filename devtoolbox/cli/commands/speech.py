"""
Speech related commands
"""
import typer
import logging
from pathlib import Path
from devtoolbox.speech.service import SpeechService
from devtoolbox.speech.whisper_provider import WhisperConfig
from devtoolbox.speech.azure_provider import AzureConfig
from devtoolbox.speech.volc_provider import VolcConfig
from devtoolbox.cli.utils import setup_logging


# Configure logging
logger = logging.getLogger("devtoolbox.speech")
app = typer.Typer(help="Speech related commands")


# Provider config mapping
PROVIDER_CONFIGS = {
    "whisper": WhisperConfig,
    "azure": AzureConfig,
    "volc": VolcConfig,
}


# Common parameters
def get_provider(
    provider: str = typer.Option(
        ...,
        "-p", "--provider",
        help=(
            "Provider type (whisper: STT only, azure: TTS & STT, "
            "volc: TTS only)"
        ),
    ),
) -> str:
    """
    Get provider from command line
    """
    if provider.lower() not in PROVIDER_CONFIGS:
        raise typer.BadParameter(
            f"Provider must be one of: {', '.join(PROVIDER_CONFIGS.keys())}"
        )
    return provider.lower()


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
    Speech command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.speech")


@app.command("text-to-speech")
def text_to_speech(
    text_file: Path = typer.Option(
        ...,
        "-t", "--text",
        help="Text file to convert to speech",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_file: Path = typer.Option(
        ...,
        "-o", "--output",
        help="Output audio file path",
        file_okay=True,
        dir_okay=False,
    ),
    provider: str = get_provider,
    speaker: str = typer.Option(
        None,
        "-s", "--speaker",
        help="Voice to use for synthesis",
    ),
    rate: int = typer.Option(
        0,
        "-r", "--rate",
        help="Speech rate adjustment",
    ),
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Whether to use cache",
    ),
):
    """
    Convert text to speech
    """
    logger.debug(
        "Converting text to speech: %s -> %s (provider=%s, speaker=%s, "
        "rate=%s, use_cache=%s)",
        text_file, output_file, provider, speaker, rate, use_cache
    )

    try:
        # Initialize service with provider config
        config_class = PROVIDER_CONFIGS[provider]
        service = SpeechService(config_class())

        # Read text file
        with open(text_file) as f:
            text = f.read()

        # Convert text to speech
        service.text_to_speech(
            text,
            str(output_file),
            use_cache=use_cache,
            speaker=speaker,
            rate=rate
        )

        typer.echo(f"Successfully converted text to speech: {output_file}")
    except Exception as e:
        logger.error(
            "Failed to convert text to speech: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to convert text to speech: {str(e)}")
        raise typer.Exit(1)


@app.command("speech-to-text")
def speech_to_text(
    audio_file: Path = typer.Option(
        ...,
        "-a", "--audio",
        help="Audio file to convert to text",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_file: Path = typer.Option(
        ...,
        "-o", "--output",
        help="Output text file path",
        file_okay=True,
        dir_okay=False,
    ),
    provider: str = get_provider,
    output_format: str = typer.Option(
        "txt",
        "-f", "--format",
        help="Output format (txt, srt, ass, vtt)",
    ),
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Whether to use cache",
    ),
    min_silence_len: int = typer.Option(
        1000,
        "--min-silence",
        help="Minimum silence length in milliseconds",
    ),
    silence_thresh: int = typer.Option(
        -40,
        "--silence-thresh",
        help="Silence threshold in dB",
    ),
    keep_silence: int = typer.Option(
        500,
        "--keep-silence",
        help="Silence to keep in milliseconds",
    ),
):
    """
    Convert speech to text
    """
    logger.debug(
        "Converting speech to text: %s -> %s (provider=%s, format=%s, "
        "use_cache=%s)",
        audio_file, output_file, provider, output_format, use_cache
    )

    try:
        # Initialize service with provider config
        config_class = PROVIDER_CONFIGS[provider]
        service = SpeechService(config_class())

        # Convert speech to text
        service.speech_to_text(
            str(audio_file),
            str(output_file),
            output_format=output_format,
            use_cache=use_cache,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

        typer.echo(f"Successfully converted speech to text: {output_file}")
    except Exception as e:
        logger.error(
            "Failed to convert speech to text: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to convert speech to text: {str(e)}")
        raise typer.Exit(1)


@app.command()
def setup_llm(
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
    """Pre-download Whisper models for offline use"""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Pre-downloading Whisper {model_size} model...")
        service = SpeechService(provider="whisper")
        service._whisper_provider._load_model(model_size)
        logger.info(f"Successfully downloaded Whisper {model_size} model")
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {str(e)}")
        raise typer.Exit(1)