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


# Common provider option
PROVIDER_OPTION = typer.Option(
    ...,
    "-p", "--provider",
    help=(
        "Provider type (whisper: STT only, azure: TTS & STT, "
        "volc: TTS only)"
    ),
)


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
    provider: str = PROVIDER_OPTION,
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
        # Validate provider
        if provider.lower() not in PROVIDER_CONFIGS:
            raise typer.BadParameter(
                f"Provider must be one of: {', '.join(PROVIDER_CONFIGS.keys())}"
            )

        # Initialize service with provider config
        config_class = PROVIDER_CONFIGS[provider.lower()]
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
    provider: str = PROVIDER_OPTION,
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
    min_chunk_duration: int = typer.Option(
        300000,
        "--min-chunk-duration",
        help="Minimum chunk duration in milliseconds (default: 300000, 5min)",
    ),
    max_chunk_duration: int = typer.Option(
        600000,
        "--max-chunk-duration",
        help="Maximum chunk duration in milliseconds (default: 600000, 10min)",
    ),
    vad_aggressiveness: int = typer.Option(
        2,
        "--vad-aggressiveness",
        help="VAD aggressiveness (0-3, default: 2)",
    ),
    max_wait_for_silence: int = typer.Option(
        120000,
        "--max-wait-for-silence",
        help="Max wait for silence after max chunk duration (ms, default: 120000)",
    ),
):
    """
    Convert speech to text
    """
    logger.debug(
        f"Converting speech to text: {audio_file} -> {output_file} "
        f"(provider={provider}, format={output_format}, use_cache={use_cache}, "
        f"min_chunk_duration={min_chunk_duration}, max_chunk_duration={max_chunk_duration}, "
        f"vad_aggressiveness={vad_aggressiveness}, max_wait_for_silence={max_wait_for_silence})"
    )

    try:
        # Validate provider
        if provider.lower() not in PROVIDER_CONFIGS:
            raise typer.BadParameter(
                f"Provider must be one of: {', '.join(PROVIDER_CONFIGS.keys())}"
            )

        # Initialize service with provider config
        config_class = PROVIDER_CONFIGS[provider.lower()]
        service = SpeechService(config_class())

        # Convert speech to text
        result = service.speech_to_text(
            str(audio_file),
            str(output_file),
            output_format=output_format,
            use_cache=use_cache,
            min_chunk_duration=min_chunk_duration,
            max_chunk_duration=max_chunk_duration,
            vad_aggressiveness=vad_aggressiveness,
            max_wait_for_silence=max_wait_for_silence
        )

        typer.echo(f"Successfully converted speech to text: {result['output_path']}")
        typer.echo(f"Metadata file: {result['metadata_path']}")
        typer.echo(f"Chunks directory: {result['chunks_path']}")
    except Exception as e:
        logger.error(
            f"Failed to convert speech to text: input={audio_file}, "
            f"output={output_file}, provider={provider}, format={output_format}, "
            f"error={str(e)}",
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
    """Pre-download Whisper models for offline use

    Note: This command is deprecated. Use 'devtoolbox whisper download' instead.
    """
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    try:
        logger.warning(
            "This command is deprecated. Use 'devtoolbox whisper download' "
            "instead for better Whisper model management."
        )

        # Use the existing whisper download functionality
        import whisper
        logger.info(f"Pre-downloading Whisper {model_size} model...")
        whisper.load_model(model_size)
        logger.info(f"Successfully downloaded Whisper {model_size} model")
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {str(e)}")
        raise typer.Exit(1)