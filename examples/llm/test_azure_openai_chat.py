"""Example of using LLM service with Azure OpenAI.

This example demonstrates how to use the LLM service with Azure OpenAI
provider. Make sure to set the following environment variables before running:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_API_BASE
    - AZURE_OPENAI_DEPLOYMENT
"""

import logging
import os
from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.service import LLMService


def main():
    """Example of using LLM service with Azure OpenAI.

    This example shows how to:
    1. Initialize the LLM service with Azure OpenAI config
    2. Set up chat messages
    3. Get response from the model
    """
    try:
        # Initialize Azure OpenAI configuration
        config = AzureOpenAIConfig(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize LLM service
        service = LLMService(config)

        # Define chat messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant powered by Azure."
            },
            {
                "role": "user",
                "content": "What is Python programming language?"
            }
        ]

        # Get response from the model
        response = service.chat(messages)
        print(f"\nLLM Response:\n{response}")

    except Exception as e:
        print(f"Error: {str(e)}")
        logging.exception(e)


if __name__ == "__main__":
    main()
