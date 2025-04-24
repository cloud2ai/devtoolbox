"""LLM service layer implementation.

This module provides the LLMService class which offers a high-level
interface for LLM operations with advanced features like context
management and fallback handling.
"""

from typing import List, Dict, Any
import logging
import importlib
from langchain.prompts import PromptTemplate

from devtoolbox.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)

NAMESPACE = 'devtoolbox.llm'
PROVIDER_MODULE_NAME = 'provider'
PROVIDER_CLASS_NAME = 'Provider'

# NOTE(Ray): Special case mappings for provider module names
# Use this mapping when the provider name needs a specific format
# Example:
# - AzureOpenAI -> azure_openai_provider.py (need mapping)
# - OpenAI -> openai_provider.py (no mapping needed, use default lowercase)
# - Whisper -> whisper_provider.py (no mapping needed, use default lowercase)
PROVIDER_MODULE_NAMES = {
    'AzureOpenAI': 'azure_openai',
}


class LLMService:
    """LLM service layer.

    This class provides a high-level interface for LLM operations,
    handling business logic and advanced features.
    """

    def __init__(self, config: Any):
        """Initialize LLM service.

        Args:
            config: Provider configuration instance.
        """
        self.config = config
        self.provider = self._init_provider()

    def _init_provider(self) -> BaseLLMProvider:
        """Initialize provider based on config.

        This method dynamically loads the appropriate provider based
        on the config class.

        Returns:
            BaseLLMProvider: Initialized provider instance.

        Raises:
            ValueError: If provider cannot be initialized.
        """
        # Get provider name from config class
        config_class_name = self.config.__class__.__name__.replace(
            'Config',
            ''
        )

        # Use mapping for special cases, otherwise just lowercase
        module_name = PROVIDER_MODULE_NAMES.get(
            config_class_name,
            config_class_name.lower()
        )

        # Import provider module
        try:
            module = importlib.import_module(
                f'{NAMESPACE}.{module_name}_{PROVIDER_MODULE_NAME}'
            )
            provider_class = getattr(
                module,
                f'{config_class_name}{PROVIDER_CLASS_NAME}'
            )
            return provider_class(self.config)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to initialize provider for config {self.config}: {e}"
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Chat with LLM.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the chat

        Returns:
            str: Chat response

        Raises:
            Exception: If chat fails
        """
        return self.provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def chat_with_context(
        self,
        messages: List[Dict[str, str]],
        context: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Chat with context.

        This method combines current messages with context for more
        coherent conversations.

        Args:
            messages: Current messages
            context: Conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the chat

        Returns:
            str: Chat response
        """
        full_messages = context + messages
        return self.chat(
            full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        fallback_messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Chat with fallback.

        This method tries the main messages first, and falls back to
        alternative messages if the first attempt fails.

        Args:
            messages: Main messages
            fallback_messages: Fallback messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the chat

        Returns:
            str: Chat response
        """
        try:
            return self.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Main chat failed: {str(e)}, trying fallback")
            return self.chat(
                fallback_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

    def complete(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the completion

        Returns:
            str: The completed text

        Raises:
            Exception: If completion fails
        """
        return self.provider.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def embed(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings.

        Args:
            text: The text to generate embeddings for
            **kwargs: Additional arguments for the embedding

        Returns:
            List[float]: The generated embeddings

        Raises:
            Exception: If embedding generation fails
        """
        return self.provider.embed(text, **kwargs)

    def chain_prompts(
        self,
        prompts: List[Dict[str, str]],
        variables: Dict[str, str] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, str]:
        """Execute a chain of prompts sequentially.

        Args:
            prompts: List of dictionaries containing 'name' and 'template' keys
            variables: Initial variables for the first prompt
            verbose: Whether to print verbose output
            **kwargs: Additional arguments for each prompt

        Returns:
            Dict[str, str]: Dictionary containing all variable values

        Raises:
            Exception: If any prompt execution fails

        Example:
            ```python
            # Create LLM service instance
            llm = LLMService(config)

            # Define prompt chain
            prompts = [
                {
                    'name': 'title',
                    'template': 'Generate an attractive title for "{topic}"'
                },
                {
                    'name': 'content',
                    'template': 'Write a paragraph based on this title: '
                               '{title}'
                },
                {
                    'name': 'summary',
                    'template': 'Summarize this content: {content}'
                }
            ]

            # Initial variables
            variables = {
                'topic': 'The Future of Green Energy'
            }

            # Execute the chain
            result = llm.chain_prompts(
                prompts=prompts,
                variables=variables,
                verbose=True
            )

            # Access results
            print("Title:", result['title'])
            print("Content:", result['content'])
            print("Summary:", result['summary'])
            ```
        """
        if variables is None:
            variables = {}

        logger.info(f"Starting chain processing with {len(prompts)} prompts")
        logger.debug(f"Initial variables: {variables}")

        result = {}
        total_prompts = len(prompts)

        for index, prompt in enumerate(prompts, 1):
            prompt_name = prompt['name']
            logger.info(
                f"Processing prompt {index}/{total_prompts}: {prompt_name}"
            )
            logger.debug(f"Current variables: {variables}")

            # Create PromptTemplate
            template = PromptTemplate(
                input_variables=list(variables.keys()),
                template=prompt['template']
            )

            # Format prompt with variables
            formatted_prompt = template.format(**variables)
            logger.debug(f"Formatted prompt: {formatted_prompt}")

            # Get completion from provider
            try:
                logger.info(f"Calling LLM for prompt: {prompt_name}")
                output = self.provider.complete(formatted_prompt, **kwargs)
                logger.debug(f"LLM response for {prompt_name}: {output}")

                result[prompt_name] = output

                # Update variables for next prompt
                variables[prompt_name] = output
                logger.info(
                    f"Successfully processed prompt {index}/{total_prompts}: "
                    f"{prompt_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to execute prompt {index}/{total_prompts} "
                    f"({prompt_name}): {str(e)}"
                )
                logger.error(f"Current variables: {variables}")
                logger.error(f"Formatted prompt: {formatted_prompt}")
                raise Exception(f"Prompt execution failed: {str(e)}")

        logger.info("Chain processing completed successfully")
        logger.debug(f"Final result: {result}")
        return result
