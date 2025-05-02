import os
from devtoolbox.webhook import Webhook


def main():
    # Get webhook URL from environment variable
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        print("Please set WEBHOOK_URL environment variable")
        return

    # Initialize Webhook
    webhook = Webhook(webhook_url)

    # Send text message
    webhook.send_text_message("This is a test message")

    # Send markdown message
    markdown_content = """
    # Title
    - Item 1
    - Item 2
    """
    webhook.send_markdown_message(markdown_content)


if __name__ == "__main__":
    main()