import os
import logging
from devtoolbox.llm.deepseek import DeepseekClient

# Configure logging to show informative messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def simple_chat_example():
    """Basic example of using DeepseekClient for a simple chat."""
    try:
        # Initialize the client with API key from environment variable
        client = DeepseekClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7  # Make responses more creative
        )

        # Define prompts
        system_prompt = "You are a helpful AI assistant."
        human_prompt = "What are the key features of Python language?"

        # Get response
        response = client.ask(system_prompt, human_prompt)
        print(f"\nResponse: {response}\n")

    except Exception as e:
        logging.error(f"Error in simple chat: {str(e)}")


def code_assistant_example():
    """Example of using DeepseekClient as a code assistant."""
    try:
        client = DeepseekClient(
            temperature=0.1,  # Lower temperature for more precise responses
            max_tokens=500    # Limit response length
        )

        system_prompt = """
        You are an expert Python programmer. Provide clear, concise, and 
        well-documented code examples.
        """
        
        human_prompt = """
        Write a Python function that implements binary search. Include 
        comments explaining the code.
        """

        response = client.ask(system_prompt, human_prompt)
        print(f"\nCode Response:\n{response}\n")

    except Exception as e:
        logging.error(f"Error in code assistant: {str(e)}")


def reasoner_example():
    """Example of using DeepseekClient with Reasoner model for complex reasoning."""
    try:
        client = DeepseekClient(
            model="deepseek-reasoner",  # Using Reasoner model
            temperature=0.3,  # Balanced between creativity and precision
            max_tokens=1000   # Allow longer responses for detailed reasoning
        )

        system_prompt = """
        You are an expert at breaking down complex problems and solving them 
        step by step with logical reasoning.
        """
        
        human_prompt = """
        A farmer needs to transport a wolf, a goat, and a cabbage across a 
        river. The boat can only carry the farmer and one item at a time. 
        If left unattended, the wolf would eat the goat, and the goat would 
        eat the cabbage. How can the farmer transport everything safely to 
        the other side? Please explain your reasoning step by step.
        """

        response = client.ask(system_prompt, human_prompt)
        print(f"\nReasoner Response:\n{response}\n")

    except Exception as e:
        logging.error(f"Error in Reasoner example: {str(e)}")


def main():
    """Run all examples."""
    logging.info("Starting Deepseek Client Examples")

    logging.info("\n1. Running simple chat example...")
    simple_chat_example()

    logging.info("\n2. Running code assistant example...")
    code_assistant_example()

    logging.info("\n3. Running Reasoner example...")
    reasoner_example()

    logging.info("\nAll examples completed.")


if __name__ == "__main__":
    main()
