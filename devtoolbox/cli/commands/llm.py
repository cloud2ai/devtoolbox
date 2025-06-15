import logging
import typer
from pathlib import Path
from typing import Optional, Dict, Type, List

from devtoolbox.llm.service import LLMService
from devtoolbox.llm.openai_provider import OpenAIConfig
from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.deepseek_provider import DeepSeekConfig
from devtoolbox.llm.provider import BaseLLMConfig
from devtoolbox.cli.utils import setup_logging

# Configure logging
logger = logging.getLogger("devtoolbox.llm")
app = typer.Typer(help="LLM commands")

# Provider configuration mapping
PROVIDER_CONFIGS: Dict[str, Type[BaseLLMConfig]] = {
    "openai": OpenAIConfig,
    "azure": AzureOpenAIConfig,
    "deepseek": DeepSeekConfig,
}

# Default prompt files
DEFAULT_PROMPTS = {
    "chat": "sample_data/llm/prompts/chat.txt",
    "chain": "sample_data/llm/prompts/chain.txt",
}


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
    LLM command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.llm")


def get_config(provider: str) -> BaseLLMConfig:
    """Get configuration for the specified provider.

    Args:
        provider: Provider name (openai, azure, deepseek)

    Returns:
        Configuration object for the provider

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(PROVIDER_CONFIGS.keys())}"
        )
    return PROVIDER_CONFIGS[provider]()


def load_prompt(
    prompt_file: Optional[str],
    prompt_type: str
) -> str:
    """Load prompt from file or use default.

    Args:
        prompt_file: Path to custom prompt file
        prompt_type: Type of prompt (chat or chain)

    Returns:
        str: Prompt text

    Raises:
        typer.Exit: If prompt file not found
    """
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            typer.echo(f"Error: Prompt file not found: {prompt_file}")
            raise typer.Exit(1)
    else:
        prompt_path = Path(DEFAULT_PROMPTS[prompt_type])
        if not prompt_path.exists():
            typer.echo(f"Error: Default {prompt_type} prompt file not found")
            raise typer.Exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def process_llm_request(
    provider: str,
    prompt_file: Optional[str],
    input_text: str,
    prompt_type: str,
    method: str
) -> None:
    """Process LLM request with common logic.

    Args:
        provider: Provider name
        prompt_file: Path to custom prompt file
        input_text: Input text to process
        prompt_type: Type of prompt (chat or chain)
        method: Method to call (chat or chain_prompts)

    Raises:
        typer.Exit: If request fails
    """
    try:
        # Get configuration and initialize service
        config = get_config(provider)
        service = LLMService(config)

        # Load prompt
        prompt = load_prompt(prompt_file, prompt_type)

        logger.debug(
            "Processing %s request (provider=%s, prompt_file=%s, input=%s)",
            prompt_type,
            provider,
            prompt_file,
            input_text
        )

        # Prepare messages based on method
        if method == "chat":
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ]
            response = service.chat(messages)
        else:
            # For chain_prompts, we still use the old format
            response = getattr(service, method)(prompt, input_text)

        typer.echo(response)

    except Exception as e:
        logger.error(
            "Failed to process %s request: %s",
            prompt_type,
            str(e),
            exc_info=True
        )
        typer.echo(
            f"Failed to process {prompt_type} request: {str(e)}"
        )
        raise typer.Exit(1)


@app.command("chat")
def chat(
    prompt_file: Optional[str] = typer.Option(
        None,
        "-p", "--prompt",
        help="Path to the prompt file. If not specified, will use default "
             "chat prompt",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        help=f"LLM provider to use ({', '.join(PROVIDER_CONFIGS.keys())})",
    ),
    input_text: str = typer.Argument(
        ...,
        help="Input text to process",
    ),
):
    """
    Chat with LLM using specified or default prompt
    """
    process_llm_request(provider, prompt_file, input_text, "chat", "chat")


@app.command("chain")
def chain_prompts(
    prompt_file: Optional[str] = typer.Option(
        None,
        "-p", "--prompt",
        help="Path to the prompt file. If not specified, will use default "
             "chain prompt",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        help=f"LLM provider to use ({', '.join(PROVIDER_CONFIGS.keys())})",
    ),
    input_text: str = typer.Argument(
        ...,
        help="Input text to process",
    ),
):
    """
    Process input text through a chain of prompts
    """
    process_llm_request(provider, prompt_file, input_text, "chain", "chain_prompts")