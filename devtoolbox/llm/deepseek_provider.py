import logging
from dataclasses import dataclass, field
import os

from devtoolbox.llm.openai_provider import OpenAIProvider, OpenAIConfig
from devtoolbox.llm.provider import register_provider, register_config

logger = logging.getLogger(__name__)


# Default API endpoints for Deepseek
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_VERSION = "v1"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


@register_config('deepseek')
@dataclass
class DeepSeekConfig(OpenAIConfig):
    """DeepSeek configuration settings.

    Inherits from OpenAIConfig since DeepSeek API is compatible with OpenAI.
    Supports both cloud and self-hosted deployments via API base config.
    """

    # Override default values for DeepSeek
    api_key: str = field(
        default_factory=lambda: os.environ.get('DEEPSEEK_API_KEY')
    )
    api_base: str = field(
        default_factory=lambda: (
            f"{os.environ.get('DEEPSEEK_API_BASE', DEEPSEEK_API_BASE)}/"
            f"{os.environ.get('DEEPSEEK_API_VERSION', DEEPSEEK_API_VERSION)}"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            'DEEPSEEK_MODEL',
            DEEPSEEK_DEFAULT_MODEL
        )
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('DEEPSEEK_TEMPERATURE', '0.7')
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get('DEEPSEEK_MAX_TOKENS', '2000')
        )
    )
    top_p: float = field(
        default_factory=lambda: float(
            os.environ.get('DEEPSEEK_TOP_P', '1.0')
        )
    )
    frequency_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('DEEPSEEK_FREQUENCY_PENALTY', '0.0')
        )
    )
    presence_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('DEEPSEEK_PRESENCE_PENALTY', '0.0')
        )
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        if self.api_key:
            logger.info("DeepSeek API key loaded from constructor")
        elif os.environ.get('DEEPSEEK_API_KEY'):
            logger.info("DeepSeek API key loaded from environment variable")
        else:
            logger.error(
                "DeepSeek API key not found in constructor or environment"
            )

        if self.api_base:
            logger.info("DeepSeek API base loaded from constructor")
        elif os.environ.get('DEEPSEEK_API_BASE'):
            logger.info("DeepSeek API base loaded from environment variable")
        else:
            logger.info("DeepSeek API base not set, using default")

        logger.info(f"Model: {self.model}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Max tokens: {self.max_tokens}")
        logger.info(f"Top P: {self.top_p}")
        logger.info(f"Frequency penalty: {self.frequency_penalty}")
        logger.info(f"Presence penalty: {self.presence_penalty}")

    def _validate_config(self):
        """Validate DeepSeek configuration."""
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Set it either in constructor "
                "or through DEEPSEEK_API_KEY environment variable"
            )

    @classmethod
    def from_env(cls) -> 'DeepSeekConfig':
        """Create DeepSeek configuration from environment variables.
        
        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


@register_provider('DeepSeekProvider')
class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider implementation.

    Inherits from OpenAIProvider since DeepSeek API is compatible with OpenAI.
    All the core functionality is inherited from OpenAIProvider.
    """

    def __init__(self, config: DeepSeekConfig):
        """Initialize the DeepSeek provider.

        Args:
            config: DeepSeek configuration
        """
        # Validate config type
        if not isinstance(config, DeepSeekConfig):
            raise ValueError(
                "Config must be an instance of DeepSeekConfig"
            )

        super().__init__(config)
