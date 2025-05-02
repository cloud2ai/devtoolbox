import logging
import json
import requests

logger = logging.getLogger(__name__)


class Webhook:
    """
    A class for sending messages to a webhook endpoint.
    Supports various message types including text, markdown, image and file.
    """

    def __init__(self, webhook_url):
        """
        Initialize the Webhook class with the provided webhook URL.
        :param webhook_url: The URL of the webhook.
        """
        self.webhook_url = webhook_url

    def send_text_message(
        self, content, mentioned_list=None, mentioned_mobile_list=None
    ):
        """
        Send a text message via the webhook.
        :param content: The text content of the message.
        :param mentioned_list: List of userids to mention in the message.
        :param mentioned_mobile_list: List of mobile numbers to mention in
                                      the message.
        """
        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list,
                "mentioned_mobile_list": mentioned_mobile_list,
            },
        }
        self._send_request(payload)

    def send_markdown_message(self, content):
        """
        Send a Markdown formatted message via the webhook.
        :param content: The Markdown content of the message.
        """
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        self._send_request(payload)

    def send_image_message(self, base64, md5):
        """
        Send an image message via the webhook.
        :param base64: Base64-encoded content of the image.
        :param md5: MD5 hash of the image content before Base64 encoding.
        """
        payload = {"msgtype": "image", "image": {"base64": base64, "md5": md5}}
        self._send_request(payload)

    def send_file_message(self, media_id):
        """
        Send a file message via the webhook.
        :param media_id: Media ID of the file obtained through the file
                         upload interface.
        """
        payload = {"msgtype": "file", "file": {"media_id": media_id}}
        self._send_request(payload)

    def _send_request(self, payload):
        """
        Internal method to send an HTTP POST request to the webhook.
        Handles request sending, error logging and exception handling.
        :param payload: The payload to be sent in the request.
        :raises: requests.exceptions.RequestException if request fails
        """
        # Set request headers
        headers = {"Content-Type": "application/json"}

        # Log the request details
        logger.debug(
            "Sending data %s to webhook %s...",
            payload,
            self.webhook_url
        )

        try:
            # Send POST request to webhook
            response = requests.post(
                self.webhook_url, headers=headers, data=json.dumps(payload)
            )
            # Check for HTTP errors
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Log error details and re-raise the exception
            logger.error(
                "Failed to send message. Error: %s, "
                "Status: %s",
                str(e),
                getattr(e.response, "status_code", "N/A")
            )
            raise