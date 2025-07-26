# -*- coding: utf-8 -*-
"""
Webhook utility for sending messages to various platforms.

Note:
    The send_text_message, send_markdown_message, send_image_message,
    and send_file_message methods are designed for WeCom (WeChat Work)
    webhook API. The payload formats and parameters follow the official
    WeCom documentation.

    The send_feishu_card_message method is designed for Feishu (Lark)
    webhook API, supporting interactive cards with markdown content.

    For other webhook platforms (e.g., Slack, DingTalk, custom HTTP
    endpoints), you can use the send_request method directly to send
    custom payloads. In the future, this module may be extended to
    support more platforms and message types.
"""
import logging
import json
import requests

logger = logging.getLogger(__name__)


class Webhook:
    """
    A class for sending messages to a webhook endpoint.

    Most send_xxx methods (text, markdown, image, file) are designed
    for WeCom (WeChat Work/企业微信) webhook API, and may not be
    compatible with other platforms' webhook formats.

    For other webhook platforms, you can call _send_request directly
    with your own payload structure.

    This class can be extended in the future to support more platforms
    and message types as needed.
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
        logger.debug(
            f"Sending WeCom text message with {len(content)} characters"
        )

        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list,
                "mentioned_mobile_list": mentioned_mobile_list,
            },
        }
        self.send_request(payload)

        logger.info(
            f"Successfully sent WeCom text message to webhook "
            f"{self.webhook_url}"
        )

    def send_markdown_message(self, content):
        """
        Send a Markdown formatted message via the webhook.
        :param content: The Markdown content of the message.
        """
        logger.debug(
            f"Sending WeCom markdown message with {len(content)} characters"
        )

        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        self.send_request(payload)

        logger.info(
            f"Successfully sent WeCom markdown message to webhook "
            f"{self.webhook_url}"
        )

    def send_image_message(self, base64, md5):
        """
        Send an image message via the webhook.
        :param base64: Base64-encoded content of the image.
        :param md5: MD5 hash of the image content before Base64 encoding.
        """
        logger.debug(
            f"Sending WeCom image message with md5: {md5}"
        )

        payload = {"msgtype": "image", "image": {"base64": base64, "md5": md5}}
        self.send_request(payload)

        logger.info(
            f"Successfully sent WeCom image message to webhook "
            f"{self.webhook_url}"
        )

    def send_file_message(self, media_id):
        """
        Send a file message via the webhook.
        :param media_id: Media ID of the file obtained through the file
                         upload interface.
        """
        logger.debug(
            f"Sending WeCom file message with media_id: {media_id}"
        )

        payload = {"msgtype": "file", "file": {"media_id": media_id}}
        self.send_request(payload)

        logger.info(
            f"Successfully sent WeCom file message to webhook "
            f"{self.webhook_url}"
        )

    def send_feishu_card_message(
        self,
        title: str,
        markdown_content: str,
        template_color: str = "blue",
        wide_screen_mode: bool = True
    ):
        """
        Send a Feishu (Lark) interactive card message via the webhook.

        This method creates an interactive card with markdown content,
        which is a common format for Feishu webhook messages.

        :param title: The title of the card header
        :param markdown_content: Markdown formatted content for the card body
        :param template_color: Color template for the header
                               (blue, green, red, grey)
        :param wide_screen_mode: Whether to enable wide screen mode
        """
        logger.debug(
            f"Sending Feishu card message: title='{title}', "
            f"color='{template_color}', wide_screen={wide_screen_mode}"
        )

        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": wide_screen_mode
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": template_color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": markdown_content
                        }
                    }
                ]
            }
        }

        logger.debug(
            f"Feishu card payload prepared with {len(markdown_content)} "
            f"characters of markdown content"
        )

        self.send_request(payload)

        logger.info(
            f"Successfully sent Feishu card message: '{title}' "
            f"to webhook {self.webhook_url}"
        )

    def send_request(self, payload):
        """
        Internal method to send an HTTP POST request to the webhook.
        Handles request sending, error logging and exception handling.
        :param payload: The payload to be sent in the request.
        :raises: requests.exceptions.RequestException if request fails
        """
        # Set request headers
        headers = {"Content-Type": "application/json"}

        # Log the request details
        logger.debug(f"Sending data {payload} to "
                     f"webhook {self.webhook_url}...")

        try:
            # Send POST request to webhook
            response = requests.post(
                self.webhook_url,
                headers=headers,
                data=json.dumps(payload)
            )
            # Check for HTTP errors
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Log error details and re-raise the exception
            logger.error(
                f"Failed to send message. Error: {str(e)}, "
                f"Status: {getattr(e.response, 'status_code', 'N/A')}"
            )
            raise