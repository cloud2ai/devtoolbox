"""Example of using LLM service with Deepseek.

This example demonstrates how to use the LLM service with Deepseek provider.
Make sure to set your DEEPSEEK_API_KEY environment variable before running.
"""

import os
from devtoolbox.llm.deepseek_provider import DeepSeekConfig
from devtoolbox.llm.service import LLMService


def main():
    """Example of using LLM service with Deepseek.

    This example shows how to:
    1. Initialize the LLM service with Deepseek config
    2. Set up chat messages
    3. Get response from the model
    """
    try:
        # Initialize Deepseek configuration
        config = DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize LLM service
        service = LLMService(config)

        # Define chat messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant powered by Deepseek."
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


if __name__ == "__main__":
    main()
