import json
import logging
from typing import Any, Dict, Optional, Union

import requests

# Configure logger
logger = logging.getLogger(__name__)


class DifyError(Exception):
    """
    Custom exception for Dify API errors.

    This exception is raised when the API returns an error response.
    It includes the HTTP status code and error message from the API.

    Attributes:
        status_code (int): HTTP status code of the error response
        message (str): Error message from the API response
    """
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class DifyClient:
    """
    Base client for interacting with the Dify API.

    This class provides the core functionality for making HTTP requests to the
    Dify API. It handles authentication, request formatting, and response
    parsing.

    Attributes:
        api_key (str): API key used for authentication
        base_url (str): Base URL for the Dify API
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1"
    ):
        """
        Initialize DifyClient.

        Args:
            api_key: API key for authentication. This is required for all API
                requests.
            base_url: Base URL for the API. Defaults to the official Dify API
                endpoint.
        """
        self.api_key = api_key
        self.base_url = base_url
        logger.info(f"Initialized DifyClient with base_url: {base_url}")

    def _send_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], requests.Response]:
        """
        Send a request to the Dify API.

        This method handles the core HTTP request functionality, including:
        - Adding authentication headers
        - Formatting request data
        - Handling response parsing
        - Error handling

        Args:
            method: HTTP method to use (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint to call (e.g., "/messages")
            json_data: JSON data to send in the request body (for POST/PUT
                requests)
            params: URL parameters to include in the request
            stream: Whether to stream the response. If True, returns the raw
                Response object.

        Returns:
            If stream=True, returns the raw Response object for streaming.
            Otherwise returns the parsed JSON response as a dictionary.

        Raises:
            DifyError: If the API returns an error response (non-2xx status
                code)
            requests.RequestException: If there's a network error or request
                fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{endpoint}"

        # Log request details
        logger.info(f"Making {method} request to {url}")
        if params:
            logger.debug(
                f"Request parameters: {json.dumps(params, indent=2)}"
            )
        if json_data:
            logger.debug(
                f"Request body: {json.dumps(json_data, indent=2)}"
            )
        logger.debug(
            f"Request headers: {json.dumps(headers, indent=2)}"
        )

        try:
            response = requests.request(
                method,
                url,
                json=json_data,
                params=params,
                headers=headers,
                stream=stream
            )

            # Log response details
            logger.info(f"Response status: {response.status_code}")
            logger.debug(
                f"Response headers: "
                f"{json.dumps(dict(response.headers), indent=2)}"
            )

            if not stream:
                try:
                    response_json = response.json()
                    logger.debug(
                        f"Response body: {json.dumps(response_json, indent=2)}"
                    )
                except ValueError:
                    logger.debug(
                        f"Response body (not JSON): {response.text}"
                    )

            # Handle error responses
            if not response.ok:
                error_msg = "Unknown error"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                    logger.error(f"API error: {error_msg}")
                    logger.error(
                        f"Error details: {json.dumps(error_data, indent=2)}"
                    )
                except ValueError:
                    error_msg = response.text or error_msg
                    logger.error(f"API error (non-JSON): {error_msg}")
                raise DifyError(response.status_code, error_msg)

            # Return response based on stream parameter
            if stream:
                return response
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def _send_request_with_files(
        self,
        method: str,
        endpoint: str,
        data: Dict,
        files: Dict
    ) -> Dict[str, Any]:
        """
        Send a request with file uploads to the Dify API.

        This method is specifically designed for handling file uploads, which
        require multipart/form-data encoding. It handles:
        - File upload formatting
        - Multipart request construction
        - Response parsing
        - Error handling

        Args:
            method: HTTP method to use (typically POST for file uploads)
            endpoint: API endpoint for file upload
            data: Form data to send along with the files
            files: Dictionary of files to upload, where keys are field names
                and values are file objects or (filename, fileobj) tuples

        Returns:
            Parsed JSON response from the API as a dictionary

        Raises:
            DifyError: If the API returns an error response (non-2xx status
                code)
            requests.RequestException: If there's a network error or request
                fails
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        url = f"{self.base_url}{endpoint}"

        # Log request details
        logger.info(f"Making {method} request to {url} with files")
        logger.debug(
            f"Request data: {json.dumps(data, indent=2)}"
        )
        logger.debug(f"Files to upload: {list(files.keys())}")
        logger.debug(
            f"Request headers: {json.dumps(headers, indent=2)}"
        )

        try:
            response = requests.request(
                method,
                url,
                data=data,
                headers=headers,
                files=files
            )

            # Log response details
            logger.info(f"Response status: {response.status_code}")
            logger.debug(
                f"Response headers: "
                f"{json.dumps(dict(response.headers), indent=2)}"
            )

            try:
                response_json = response.json()
                logger.debug(
                    f"Response body: {json.dumps(response_json, indent=2)}"
                )
            except ValueError:
                logger.debug(
                    f"Response body (not JSON): {response.text}"
                )

            # Handle error responses
            if not response.ok:
                error_msg = "Unknown error"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                    logger.error(f"API error: {error_msg}")
                    logger.error(
                        f"Error details: {json.dumps(error_data, indent=2)}"
                    )
                except ValueError:
                    error_msg = response.text or error_msg
                    logger.error(f"API error (non-JSON): {error_msg}")
                raise DifyError(response.status_code, error_msg)

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def message_feedback(self, message_id, rating, user):
        data = {"rating": rating, "user": user}
        return self._send_request(
            "POST",
            f"/messages/{message_id}/feedbacks",
            json_data=data
        )

    def get_application_parameters(self, user):
        params = {"user": user}
        return self._send_request("GET", "/parameters", params=params)

    def file_upload(self, user, files):
        data = {"user": user}
        return self._send_request_with_files(
            "POST",
            "/files/upload",
            data=data,
            files=files
        )

    def text_to_audio(self, text: str, user: str, streaming: bool = False):
        data = {"text": text, "user": user, "streaming": streaming}
        return self._send_request("POST", "/text-to-audio", json_data=data)

    def get_meta(self, user):
        params = {"user": user}
        return self._send_request("GET", "/meta", params=params)


class CompletionClient(DifyClient):
    def create_completion_message(
        self,
        inputs,
        response_mode,
        user,
        files=None
    ):
        """
        Create a completion message.

        Args:
            inputs: Input data for the completion
            response_mode: Mode of response (e.g., "blocking", "streaming")
            user: User identifier
            files: Optional files to include with the request

        Returns:
            Response from the API
        """
        data = {
            "inputs": inputs,
            "response_mode": response_mode,
            "user": user,
            "files": files,
        }
        return self._send_request(
            "POST",
            "/completion-messages",
            json_data=data,
            stream=True if response_mode == "streaming" else False,
        )


class ChatClient(DifyClient):
    def create_chat_message(
        self,
        inputs,
        query,
        user,
        response_mode="blocking",
        conversation_id=None,
        files=None,
    ):
        """
        Create a chat message.

        Args:
            inputs: Input data for the chat
            query: Query text
            user: User identifier
            response_mode: Mode of response (e.g., "blocking", "streaming")
            conversation_id: Optional conversation identifier
            files: Optional files to include with the request

        Returns:
            Response from the API
        """
        data = {
            "inputs": inputs,
            "query": query,
            "user": user,
            "response_mode": response_mode,
            "files": files,
        }
        if conversation_id:
            data["conversation_id"] = conversation_id

        return self._send_request(
            "POST",
            "/chat-messages",
            json_data=data,
            stream=True if response_mode == "streaming" else False,
        )

    def get_suggested(self, message_id, user: str):
        """
        Get suggested responses for a message.

        Args:
            message_id: ID of the message
            user: User identifier

        Returns:
            Response from the API
        """
        params = {"user": user}
        return self._send_request(
            "GET",
            f"/messages/{message_id}/suggested",
            params=params
        )

    def stop_message(self, task_id, user):
        """
        Stop a message processing task.

        Args:
            task_id: ID of the task to stop
            user: User identifier

        Returns:
            Response from the API
        """
        data = {"user": user}
        return self._send_request(
            "POST",
            f"/chat-messages/{task_id}/stop",
            json_data=data
        )

    def get_conversations(
        self,
        user,
        last_id=None,
        limit=None,
        pinned=None
    ):
        """
        Get list of conversations.

        Args:
            user: User identifier
            last_id: ID of the last conversation to start from
            limit: Maximum number of conversations to return
            pinned: Whether to return only pinned conversations

        Returns:
            Response from the API
        """
        params = {
            "user": user,
            "last_id": last_id,
            "limit": limit,
            "pinned": pinned
        }
        return self._send_request("GET", "/conversations", params=params)

    def get_conversation_messages(
        self,
        user,
        conversation_id=None,
        first_id=None,
        limit=None
    ):
        """
        Get messages from a conversation.

        Args:
            user: User identifier
            conversation_id: ID of the conversation
            first_id: ID of the first message to start from
            limit: Maximum number of messages to return

        Returns:
            Response from the API
        """
        params = {"user": user}

        if conversation_id:
            params["conversation_id"] = conversation_id
        if first_id:
            params["first_id"] = first_id
        if limit:
            params["limit"] = limit

        return self._send_request("GET", "/messages", params=params)

    def rename_conversation(
        self,
        conversation_id,
        name,
        auto_generate: bool,
        user: str
    ):
        """
        Rename a conversation.

        Args:
            conversation_id: ID of the conversation
            name: New name for the conversation
            auto_generate: Whether to auto-generate the name
            user: User identifier

        Returns:
            Response from the API
        """
        data = {
            "name": name,
            "auto_generate": auto_generate,
            "user": user
        }
        return self._send_request(
            "POST",
            f"/conversations/{conversation_id}/name",
            json_data=data
        )

    def delete_conversation(self, conversation_id, user):
        """
        Delete a conversation.

        Args:
            conversation_id: ID of the conversation
            user: User identifier

        Returns:
            Response from the API
        """
        data = {"user": user}
        return self._send_request(
            "DELETE",
            f"/conversations/{conversation_id}",
            json_data=data
        )

    def audio_to_text(self, audio_file, user):
        """
        Convert audio to text.

        Args:
            audio_file: Audio file to convert
            user: User identifier

        Returns:
            Response from the API
        """
        data = {"user": user}
        files = {"audio_file": audio_file}
        return self._send_request_with_files(
            "POST",
            "/audio-to-text",
            data,
            files
        )


class WorkflowClient(DifyClient):
    def run(
        self,
        inputs: dict,
        response_mode: str = "streaming",
        user: str = "abc-123"
    ):
        """
        Run a workflow.

        Args:
            inputs: Input data for the workflow
            response_mode: Mode of response (e.g., "streaming")
            user: User identifier

        Returns:
            Response from the API
        """
        data = {
            "inputs": inputs,
            "response_mode": response_mode,
            "user": user
        }
        return self._send_request("POST", "/workflows/run", json_data=data)

    def stop(self, task_id, user):
        """
        Stop a workflow task.

        Args:
            task_id: ID of the task to stop
            user: User identifier

        Returns:
            Response from the API
        """
        data = {"user": user}
        return self._send_request(
            "POST",
            f"/workflows/tasks/{task_id}/stop",
            json_data=data
        )

    def get_result(self, workflow_run_id):
        """
        Get the result of a workflow run.

        Args:
            workflow_run_id: ID of the workflow run

        Returns:
            Response from the API
        """
        return self._send_request(
            "GET",
            f"/workflows/run/{workflow_run_id}"
        )


class KnowledgeBaseClient(DifyClient):
    def __init__(
        self,
        api_key,
        base_url: str = "https://api.dify.ai/v1",
        dataset_id: str | None = None,
    ):
        """
        Construct a KnowledgeBaseClient object.

        Args:
            api_key (str): API key of Dify.
            base_url (str, optional): Base URL of Dify API. Defaults to
                'https://api.dify.ai/v1'.
            dataset_id (str, optional): ID of the dataset. Defaults to None.
                You don't need this if you just want to create a new dataset
                or list datasets. Otherwise you need to set this.
        """
        super().__init__(api_key=api_key, base_url=base_url)
        self.dataset_id = dataset_id

    def _get_dataset_id(self):
        """
        Get the dataset ID.

        Returns:
            str: The dataset ID

        Raises:
            ValueError: If dataset_id is not set
        """
        if self.dataset_id is None:
            raise ValueError("dataset_id is not set")
        return self.dataset_id

    def create_dataset(self, name: str, **kwargs):
        """
        Create a new dataset.

        Args:
            name: Name of the dataset
            **kwargs: Additional parameters to pass to the API, such as:
                - indexing_technique: Technique used for indexing
                - permission: Permission setting
                - provider: Provider type
                - description: Description of the dataset
                - external_knowledge_api_id: ID of external knowledge API
                - external_knowledge_id: ID of external knowledge

        Returns:
            Response from the API
        """
        data = {"name": name}
        data.update(kwargs)
        return self._send_request("POST", "/datasets", json_data=data)

    def list_datasets(self, page: int = 1, page_size: int = 20, **kwargs):
        """
        List datasets.

        Args:
            page: Page number for pagination
            page_size: Number of items per page
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        return self._send_request(
            "GET",
            f"/datasets?page={page}&limit={page_size}",
            **kwargs
        )

    def create_document_by_text(
        self,
        name,
        text,
        extra_params: dict | None = None,
        **kwargs
    ):
        """
        Create a document by text.

        Args:
            name: Name of the document
            text: Text content of the document
            extra_params: Extra parameters to pass to the API, such as:
                - indexing_technique: Technique used for indexing
                - process_rule: Rules for processing the document
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        data = {
            "indexing_technique": "high_quality",
            "process_rule": {"mode": "automatic"},
            "name": name,
            "text": text,
        }
        if extra_params is not None and isinstance(extra_params, dict):
            data.update(extra_params)
        url = (
            f"/datasets/{self._get_dataset_id()}/"
            f"document/create_by_text"
        )
        return self._send_request("POST", url, json_data=data, **kwargs)

    def update_document_by_text(
        self,
        document_id,
        name,
        text,
        extra_params: dict | None = None,
        **kwargs
    ):
        """
        Update a document by text.

        Args:
            document_id: ID of the document
            name: Name of the document
            text: Text content of the document
            extra_params: Extra parameters to pass to the API, such as:
                - indexing_technique: Technique used for indexing
                - process_rule: Rules for processing the document
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        data = {"name": name, "text": text}
        if extra_params is not None and isinstance(extra_params, dict):
            data.update(extra_params)
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/update_by_text"
        )
        return self._send_request("POST", url, json_data=data, **kwargs)

    def create_document_by_file(
        self,
        file_path,
        original_document_id=None,
        extra_params: dict | None = None
    ):
        """
        Create a document by file.

        Args:
            file_path: Path to the file
            original_document_id: ID of the original document to replace
            extra_params: Extra parameters to pass to the API, such as:
                - indexing_technique: Technique used for indexing
                - process_rule: Rules for processing the document

        Returns:
            Response from the API
        """
        files = {"file": open(file_path, "rb")}
        data = {
            "process_rule": {"mode": "automatic"},
            "indexing_technique": "high_quality",
        }
        if extra_params is not None and isinstance(extra_params, dict):
            data.update(extra_params)
        if original_document_id is not None:
            data["original_document_id"] = original_document_id
        url = (
            f"/datasets/{self._get_dataset_id()}/"
            f"document/create_by_file"
        )
        return self._send_request_with_files(
            "POST",
            url,
            {"data": json.dumps(data)},
            files
        )

    def update_document_by_file(
        self,
        document_id,
        file_path,
        extra_params: dict | None = None
    ):
        """
        Update a document by file.

        Args:
            document_id: ID of the document
            file_path: Path to the file
            extra_params: Extra parameters to pass to the API, such as:
                - indexing_technique: Technique used for indexing
                - process_rule: Rules for processing the document

        Returns:
            Response from the API
        """
        files = {"file": open(file_path, "rb")}
        data = {}
        if extra_params is not None and isinstance(extra_params, dict):
            data.update(extra_params)
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/update_by_file"
        )
        return self._send_request_with_files(
            "POST",
            url,
            {"data": json.dumps(data)},
            files
        )

    def batch_indexing_status(self, batch_id: str, **kwargs):
        """
        Get the status of the batch indexing.

        Args:
            batch_id: ID of the batch uploading
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{batch_id}/indexing-status"
        )
        return self._send_request("GET", url, **kwargs)

    def delete_dataset(self):
        """
        Delete this dataset.

        Returns:
            Response from the API
        """
        url = f"/datasets/{self._get_dataset_id()}"
        return self._send_request("DELETE", url)

    def delete_document(self, document_id):
        """
        Delete a document.

        Args:
            document_id: ID of the document

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}"
        )
        return self._send_request("DELETE", url)

    def list_documents(
        self,
        page: int | None = None,
        page_size: int | None = None,
        keyword: str | None = None,
        get_all: bool = False,
        **kwargs,
    ):
        """
        Get a list of documents in this dataset.

        Args:
            page: Page number for pagination
            page_size: Number of items per page
            keyword: Search keyword
            get_all: Whether to get all documents by automatically handling
                pagination
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API. If get_all is True, returns all documents
            combined.
        """
        if get_all:
            all_documents = []
            current_page = 1
            page_size = page_size or 20  # Default page size if not specified

            while True:
                params = {
                    "page": current_page,
                    "limit": page_size
                }
                if keyword is not None:
                    params["keyword"] = keyword

                url = (
                    f"/datasets/{self._get_dataset_id()}/"
                    f"documents"
                )
                response = self._send_request(
                    "GET",
                    url,
                    params=params,
                    **kwargs
                )

                # Add documents from current page
                all_documents.extend(response.get("data", []))

                # Check if there are more pages
                if not response.get("has_more", False):
                    break

                current_page += 1

            # Return combined response
            return {
                "data": all_documents,
                "has_more": False,
                "limit": page_size,
                "total": len(all_documents),
                "page": 1
            }
        else:
            # Original pagination logic
            params = {}
            if page is not None:
                params["page"] = page
            if page_size is not None:
                params["limit"] = page_size
            if keyword is not None:
                params["keyword"] = keyword
            url = (
                f"/datasets/{self._get_dataset_id()}/"
                f"documents"
            )
            return self._send_request(
                "GET",
                url,
                params=params,
                **kwargs
            )

    def add_segments(self, document_id, segments, **kwargs):
        """
        Add segments to a document.

        Args:
            document_id: ID of the document
            segments: List of segments to add, example:
                [{"content": "1", "answer": "1", "keyword": ["a"]}]
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        data = {"segments": segments}
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/segments"
        )
        return self._send_request("POST", url, json_data=data, **kwargs)

    def query_segments(
        self,
        document_id,
        keyword: str | None = None,
        status: str | None = None,
        **kwargs,
    ):
        """
        Query segments in this document.

        Args:
            document_id: ID of the document
            keyword: Query keyword, optional
            status: Status of the segment, optional, e.g. completed
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/segments"
        )
        params = {}
        if keyword is not None:
            params["keyword"] = keyword
        if status is not None:
            params["status"] = status
        if "params" in kwargs:
            params.update(kwargs["params"])
        return self._send_request("GET", url, params=params, **kwargs)

    def delete_document_segment(self, document_id, segment_id):
        """
        Delete a segment from a document.

        Args:
            document_id: ID of the document
            segment_id: ID of the segment

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/segments/{segment_id}"
        )
        return self._send_request("DELETE", url)

    def update_document_segment(
        self,
        document_id,
        segment_id,
        segment_data,
        **kwargs
    ):
        """
        Update a segment in a document.

        Args:
            document_id: ID of the document
            segment_id: ID of the segment
            segment_data: Data of the segment, example:
                {"content": "1", "answer": "1", "keyword": ["a"],
                "enabled": True}
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response from the API
        """
        data = {"segment": segment_data}
        url = (
            f"/datasets/{self._get_dataset_id()}/documents/"
            f"{document_id}/segments/{segment_id}"
        )
        return self._send_request("POST", url, json_data=data, **kwargs)

    def add_metadata_field(self, type: str, name: str, **kwargs):
        """
        Add a metadata field to the knowledge base.

        Args:
            type: Type of the metadata field (e.g., "string", "number", "time")
            name: Name of the metadata field
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API containing the created metadata field
        """
        data = {"type": type, "name": name}
        url = f"/datasets/{self._get_dataset_id()}/metadata"
        return self._send_request("POST", url, json_data=data, **kwargs)

    def update_metadata_field(self, metadata_id: str, name: str, **kwargs):
        """
        Update a metadata field in the knowledge base.

        Args:
            metadata_id: ID of the metadata field to update
            name: New name for the metadata field
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API containing the updated metadata field
        """
        data = {"name": name}
        url = (
            f"/datasets/{self._get_dataset_id()}/metadata/"
            f"{metadata_id}"
        )
        return self._send_request("PATCH", url, json_data=data, **kwargs)

    def delete_metadata_field(self, metadata_id: str, **kwargs):
        """
        Delete a metadata field from the knowledge base.

        Args:
            metadata_id: ID of the metadata field to delete
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/metadata/"
            f"{metadata_id}"
        )
        return self._send_request("DELETE", url, **kwargs)

    def toggle_built_in_fields(self, action: str, **kwargs):
        """
        Enable or disable built-in fields in the knowledge base.

        Args:
            action: Action to perform ("enable" or "disable")
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API
        """
        url = (
            f"/datasets/{self._get_dataset_id()}/metadata/"
            f"built-in/{action}"
        )
        return self._send_request("DELETE", url, **kwargs)

    def update_document_metadata(
        self,
        operation_data: list[dict],
        **kwargs
    ):
        """
        Modify metadata for one or more documents.

        Args:
            operation_data: List of operations to perform, each containing:
                - document_id: ID of the document
                - metadata_list: List of metadata to update, each containing:
                    - id: ID of the metadata field
                    - value: Value to set
                    - name: Name of the metadata field
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API
        """
        data = {"operation_data": operation_data}
        url = f"/datasets/{self._get_dataset_id()}/documents/metadata"
        return self._send_request("POST", url, json_data=data, **kwargs)

    def list_metadata_fields(self, **kwargs):
        """
        Get the list of metadata fields in the dataset.

        Returns:
            Response from the API containing:
                - doc_metadata: List of metadata fields
                - built_in_field_enabled: Whether built-in fields are enabled
        """
        url = f"/datasets/{self._get_dataset_id()}/metadata"
        return self._send_request("GET", url, **kwargs)