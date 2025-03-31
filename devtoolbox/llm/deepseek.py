import logging
import os

from devtoolbox.llm.openai import OpenAIClient

# Default temperature controls the randomness of responses; a value of 0
# makes outputs more deterministic.
DEFAULT_TEMPERATURE = 0

# Default model name used for LLM interactions.
DEFAULT_MODEL = "gpt-4o-mini"

# Default token limit for LLM responses; setting it to None allows the
# model to use a dynamic length for responses.
DEFAULT_MAX_TOKENS = None

# Default timeout duration for LLM API calls (in seconds).
DEFAULT_TIMEOUT = 60

# Default number of retries on request failure.
DEFAULT_MAX_RETRIES = 3


class DeepseekClient(OpenAIClient):
    """
    A client class for interacting with Deepseek's API service using
    LangChain. Inherits from OpenAIClient to utilize common functionality.
    """

    # Default Deepseek Base URL
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    
    # Default Deepseek model
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key=None,
        base_url=DEEPSEEK_BASE_URL,
        model=DEFAULT_MODEL,
        *args,
        **kwargs
    ):
        """
        Initialize the Deepseek client with model configuration and API key.

        Parameters:
        - api_key (str): API key for accessing Deepseek API.
        - base_url (str): The base URL for Deepseek API.
        - model (str): The model name to use.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_API_ENDPOINT")
        self.model = model

        if not self.api_key:
            raise Exception(
                "API key is required for DeepseekClient. You can set it "
                "using the environment variable 'DEEPSEEK_API_KEY'."
            )

        logging.debug(
            f"Initializing DeepseekClient with model: {self.model}"
        )

        # Initialize the LLM with default parameters
        super().__init__(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES)
        )
        logging.info("DeepseekClient initialized successfully.")
