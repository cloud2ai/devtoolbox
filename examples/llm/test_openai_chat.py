"""Example of using LLM service with OpenAI.

This example demonstrates how to use the LLM service with OpenAI provider.
Make sure to set your OPENAI_API_KEY and OPENAI_API_BASE environment
variables before running.
"""

import os
from devtoolbox.llm.openai_provider import OpenAIConfig
from devtoolbox.llm.service import LLMService


def main():
    """Example of using LLM service.

    This example shows how to:
    1. Initialize the LLM service with OpenAI config
    2. Set up chat messages
    3. Get response from the model
    """
    try:
        # Initialize OpenAI configuration
        config = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize LLM service
        service = LLMService(config)

        # Define chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python programming language?"}
        ]

        # Get response from the model
        response = service.chat(messages)
        print(f"\nLLM Response:\n{response}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
