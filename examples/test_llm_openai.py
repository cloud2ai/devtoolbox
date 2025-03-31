import os
import logging
from devtoolbox.llm.openai import OpenAIClient, AzureOpenAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_openai_client():
    """Example usage of OpenAIClient."""
    try:
        # Initialize OpenAI client with API key
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAIClient(
            api_key=api_key,
            model="gpt-3.5-turbo",  # You can specify different model
            temperature=0.7
        )

        # Example system and human prompts
        system_prompt = (
            "You are a helpful assistant that provides concise answers."
        )
        human_prompt = "What is Python programming language?"

        # Get response from the model
        response = client.ask(system_prompt, human_prompt)
        logging.info(f"OpenAI Response: {response}")

    except Exception as e:
        logging.error(f"Error in OpenAI test: {str(e)}")


def test_azure_openai_client():
    """Example usage of AzureOpenAIClient."""
    try:
        # Initialize Azure OpenAI client
        # (using environment variables for credentials)
        client = AzureOpenAIClient(
            temperature=0.7,
            max_tokens=150
        )

        # Example system and human prompts
        system_prompt = (
            "You are a technical expert who provides detailed explanations."
        )
        human_prompt = "Explain the concept of REST APIs."

        # Get response from the model
        response = client.ask(system_prompt, human_prompt)
        logging.info(f"Azure OpenAI Response: {response}")

    except Exception as e:
        logging.error(f"Error in Azure OpenAI test: {str(e)}")


def main():
    """Main function to run the examples."""
    logging.info("Starting OpenAI client tests...")
    
    # Test standard OpenAI client
    logging.info("\nTesting OpenAI client:")
    test_openai_client()
    
    # Test Azure OpenAI client
    logging.info("\nTesting Azure OpenAI client:")
    test_azure_openai_client()


if __name__ == "__main__":
    main() 
