import logging
import os

from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

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


class OpenAIClient:
    """
    A client class for interacting with a language model (LLM) via
    LangChain's ChatOpenAI. Provides methods to configure the LLM and
    query it with system and human prompts.
    """

    # Default OpenAI Base URL.
    OPENAI_BASE_URL = "https://api.openai.com"

    def __init__(self, api_key, base_url=OPENAI_BASE_URL,
                 model=DEFAULT_MODEL, *args, **kwargs):
        """
        Initialize the LLM client with model configuration and API key.

        Parameters:
        - model (str): The name of the model to use (default is
          "gpt-4o-mini").
        - api_key (str): API key for accessing the OpenAI API (defaults to
          environment variable if not provided).
        - *args, **kwargs: Additional arguments for customizing the LLM
          initialization.
        """
        if not api_key:
            raise Exception("API key is required for OpenAIClient.")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        logging.debug(f"Initializing OpenAIClient with model: "
                      f"{self.model}")

        # Initialize the LLM with default parameters; these can be
        # customized as needed.
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
            api_key=self.api_key,
            base_url=self.base_url
        )
        logging.info("OpenAIClient initialized successfully.")

    def ask(self, system_prompt, human_prompt, *args, **kwargs):
        """
        Sends a prompt to the LLM and returns the generated response.

        Parameters:
        - system_prompt (str): The initial context or setup message for the
          conversation.
        - human_prompt (str): The prompt or question from the user to the
          LLM.
        - *args, **kwargs: Additional arguments for response customization.

        Returns:
        - response.content (str): The text content of the LLM's response.
        """
        messages = [
            ("system", str(system_prompt),),
            ("human", str(human_prompt))
        ]
        logging.debug(f"Sending messages to LLM: {messages}")
        response = self.llm.invoke(messages)
        logging.debug(f"LLM Response: {response.content}")
        return response.content


class AzureOpenAIClient(OpenAIClient):
    """
    A client class for interacting with Azure's OpenAI service using
    LangChain. Inherits from OpenAIClient to utilize common functionality.
    """

    def __init__(self, api_key=None, deployment_name=None, base_url=None,
                 model=None, *args, **kwargs):
        """
        Initialize the Azure OpenAI client with model configuration and
        API key.

        Parameters:
        - api_key (str): API key for accessing Azure OpenAI.
        - deployment_name (str): The name of the model deployment.
        - base_url (str): The base URL for Azure OpenAI (default is set).
        - model_version (str): The version of the Azure OpenAI API to use.
        - temperature (float): Controls the randomness of responses.
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT"
        )
        self.base_url = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model = model or os.getenv("AZURE_OPENAI_API_VERSION")

        if not self.api_key:
            raise Exception(
                "API key is required for AzureOpenAIClient. You can set it "
                "using the environment variable 'AZURE_OPENAI_API_KEY'."
            )
        if not self.deployment_name:
            raise Exception(
                "Deployment name is required for AzureOpenAIClient. You can "
                "set it using the environment variable 'AZURE_OPENAI_DEPLOYMENT'."
            )
        if not self.base_url:
            raise Exception(
                "Base URL is required for AzureOpenAIClient. You can set it "
                "using the environment variable 'AZURE_OPENAI_ENDPOINT'."
            )
        if not self.model:
            raise Exception(
                "Model version is required for AzureOpenAIClient. You can set "
                "it using the environment variable 'AZURE_OPENAI_API_VERSION'."
            )

        logging.debug(f"Initializing AzureOpenAIClient with deployment: "
                      f"{self.deployment_name}")

        # Initialize the LLM with default parameters; these can be
        # customized as needed.
        self.llm = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            api_version=self.model,
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
            api_key=self.api_key,
            azure_endpoint=self.base_url
        )
        logging.info("AzureOpenAIClient initialized successfully.")
