"""
AzureClient: Encapsulates all Azure REST API and Blob Storage operations
for speech batch transcription and related tasks.

This class is responsible for low-level Azure access, including blob
upload, SAS URL generation, batch transcription submission, polling,
file/result retrieval, and deletion.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Tuple, TYPE_CHECKING

import requests
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)
from azure.core.exceptions import ServiceResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from devtoolbox.speech.clients.azure_errors import (
    AzureRecognitionError,
    AzureNetworkError,
    AzureUploadError
)

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from devtoolbox.speech.azure_provider import AzureConfig

def _log_retry_attempt(retry_state):
    """
    Custom retry logging function to show retry attempts and details.

    Args:
        retry_state: The retry state object from tenacity
    """
    attempt_number = retry_state.attempt_number
    max_attempts = retry_state.retry_object.stop.max_attempt_number
    exception = retry_state.outcome.exception()
    wait_time = retry_state.next_action.sleep

    logger.warning(
        f"[BLOB_UPLOAD_RETRY] Attempt {attempt_number}/{max_attempts} failed. "
        f"Exception: {type(exception).__name__}: {exception}. "
        f"Waiting {wait_time:.2f}s before next attempt."
    )

class AzureClient:
    """
    Azure REST API and Blob Storage client for speech batch transcription.
    """
    DEFAULT_CONTENT_TYPE = "audio/wav"

    # Retry configuration constants (only for upload operations)
    UPLOAD_RETRY_ATTEMPTS = 3
    UPLOAD_RETRY_MIN_WAIT = 4
    UPLOAD_RETRY_MAX_WAIT = 10

    def __init__(self, config: "AzureConfig"):
        """
        Initialize AzureClient with configuration.

        Args:
            config: AzureConfig instance with credentials and settings.
        """
        self.config = config
        self.endpoint = (
            f"https://{config.service_region}.api.cognitive.microsoft.com"
        )
        self.headers = {
            "Ocp-Apim-Subscription-Key": config.subscription_key
        }
        self.blob_service_client = BlobServiceClient(
            f"https://{config.storage_account}.blob.core.windows.net",
            credential=config.storage_key
        )

        # Load retry configuration from config or use defaults
        self.upload_retry_attempts = getattr(
            config, 'upload_retry_attempts', self.UPLOAD_RETRY_ATTEMPTS
        )
        self.upload_retry_min_wait = getattr(
            config, 'upload_retry_min_wait', self.UPLOAD_RETRY_MIN_WAIT
        )
        self.upload_retry_max_wait = getattr(
            config, 'upload_retry_max_wait', self.UPLOAD_RETRY_MAX_WAIT
        )

    @retry(
        retry=retry_if_exception_type(AzureUploadError),
        stop=stop_after_attempt(UPLOAD_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=UPLOAD_RETRY_MIN_WAIT,
            max=UPLOAD_RETRY_MAX_WAIT
        ),
        before_sleep=_log_retry_attempt,
    )
    def upload_blob(self, file_path: str) -> Tuple[str, str]:
        """
        Upload a file to Azure Blob Storage and return (blob_name, sas_url).

        Args:
            file_path (str): Local file path to upload.

        Returns:
            tuple[str, str]: (blob_name, sas_url)
        """
        logger.info(
            f"[BLOB_UPLOAD] Starting upload with retry config: "
            f"max_attempts={self.upload_retry_attempts}, "
            f"wait_range={self.upload_retry_min_wait}-{self.upload_retry_max_wait}s"
        )

        blob_name = self._generate_random_blob_name(file_path)
        container_client = self.blob_service_client.get_container_client(
            self.config.container_name
        )

        # Create container if it doesn't exist
        try:
            container_client.create_container()
        except Exception as e:
            # Log and ignore if already exists
            logger.info(
                f"Create container exception: {e}"
            )

        # Upload blob with retry logic
        try:
            logger.info(
                f"[BLOB_UPLOAD] Starting upload attempt for file={file_path}, "
                f"blob_name={blob_name}"
            )
            with open(file_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True,
                    content_settings=ContentSettings(
                        content_type=self.DEFAULT_CONTENT_TYPE
                    ),
                    # Add timeout configuration
                    timeout=300  # 5 minutes timeout
                )
            logger.info(
                f"[BLOB_UPLOAD] Upload successful for blob_name={blob_name}"
            )
        except (ServiceResponseError, TimeoutError, ConnectionError) as e:
            logger.error(
                f"[BLOB_UPLOAD] Upload failed (network error): {e} "
                f"file={file_path}, blob_name={blob_name}"
            )
            # Raise specific error for retry mechanism
            raise AzureUploadError(
                f"Network error during upload: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"[BLOB_UPLOAD] Upload failed (unexpected error): {e} "
                f"file={file_path}, blob_name={blob_name}"
            )
            # Don't retry for unexpected errors
            raise AzureUploadError(
                f"Unexpected error during upload: {e}"
            ) from e

        # Generate SAS token and URL
        sas_token = generate_blob_sas(
            account_name=self.config.storage_account,
            container_name=self.config.container_name,
            blob_name=blob_name,
            account_key=self.config.storage_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=2)
        )
        sas_url = (
            f"https://{self.config.storage_account}.blob.core.windows.net/"
            f"{self.config.container_name}/{blob_name}?{sas_token}"
        )
        logger.info(
            f"[BLOB_UPLOAD] name={blob_name} sas_url={sas_url}"
        )
        return blob_name, sas_url

    def delete_blob(self, blob_name: str) -> None:
        """
        Delete a blob from Azure Blob Storage by name.

        Args:
            blob_name (str): The name of the blob to delete.
        """
        container_client = self.blob_service_client.get_container_client(
            self.config.container_name
        )
        try:
            container_client.delete_blob(blob_name)
        except Exception as e:
            logger.warning(
                f"Delete blob failed: {e} blob_name={blob_name}"
            )

    def _check_blob_url_accessible(self, blob_url: str) -> None:
        """
        Check if the given blob_url is accessible (HTTP 200) before submitting
        batch transcription. Raise an exception if not accessible.

        Args:
            blob_url (str): The full SAS URL to the blob.

        Raises:
            AzureRecognitionError: If the blob is not accessible.
        """
        try:
            resp = requests.get(blob_url, timeout=10)
            if resp.status_code != 200:
                raise AzureRecognitionError(
                    f"Blob not accessible: url={blob_url} "
                    f"status={resp.status_code}"
                )
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            logger.error(
                f"Blob accessibility check failed (network error): {e} "
                f"url={blob_url}"
            )
            raise AzureNetworkError(
                f"Network error checking blob accessibility: {e}"
            ) from e
        except Exception as e:
            raise AzureRecognitionError(
                f"Blob not accessible: url={blob_url} error={e}"
            )

    def submit_batch_transcription(
        self, sas_url: str, properties: dict
    ) -> str:
        """
        Submit a batch transcription job and return the transcription ID.

        Args:
            sas_url (str): The SAS URL of the audio blob.
            properties (dict): Additional properties for the batch job.

        Returns:
            str: The transcription job ID.
        """
        self._check_blob_url_accessible(sas_url)
        url = f"{self.endpoint}/speechtotext/v3.0/transcriptions"
        body = {
            "contentUrls": [sas_url],
            "locale": self.config.locale,
            "displayName": "BatchTranscription",
            "properties": properties or {},
        }
        try:
            resp = requests.post(
                url, headers=self.headers, json=body, timeout=30
            )
            resp.raise_for_status()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            logger.error(
                f"Batch transcription submit failed (network error): {e} "
                f"url={url}"
            )
            raise AzureNetworkError(
                f"Network error during batch submission: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Batch transcription submit failed (unexpected error): {e} "
                f"url={url}"
            )
            raise
        location = resp.headers.get("Location")
        if not location:
            raise Exception("No Location header in batch submit response")
        transcription_id = location.rstrip("/").split("/")[-1]
        return transcription_id

    def get_transcription_status(self, transcription_id: str) -> dict:
        """
        Get the status and metadata of a batch transcription job.

        Args:
            transcription_id (str): The transcription job ID.

        Returns:
            dict: The job status and metadata.
        """
        url = (
            f"{self.endpoint}/speechtotext/v3.0/transcriptions/"
            f"{transcription_id}"
        )
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            logger.error(
                f"Get transcription status failed (network error): "
                f"{e} id={transcription_id}"
            )
            raise AzureNetworkError(
                f"Network error getting status: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Get transcription status failed (unexpected error): "
                f"{e} id={transcription_id}"
            )
            raise

    def get_transcription_files(self, transcription_id: str) -> dict:
        """
        Get the files metadata for a batch transcription job.

        Args:
            transcription_id (str): The transcription job ID.

        Returns:
            dict: The files metadata.
        """
        url = (
            f"{self.endpoint}/speechtotext/v3.0/transcriptions/"
            f"{transcription_id}/files"
        )
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            logger.error(
                f"Get transcription files failed (network error): "
                f"{e} id={transcription_id}"
            )
            raise AzureNetworkError(
                f"Network error getting files: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Get transcription files failed (unexpected error): "
                f"{e} id={transcription_id}"
            )
            raise

    def download_transcription_result(self, result_url: str) -> str:
        """
        Download the transcription result text from the given URL.

        Args:
            result_url (str): The URL to the result file.

        Returns:
            str: The transcription result text.
        """
        try:
            resp = requests.get(result_url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            logger.error(
                f"Download transcription result failed (network error): "
                f"{e} url={result_url}"
            )
            raise AzureNetworkError(
                f"Network error downloading result: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Download transcription result failed (unexpected error): "
                f"{e} url={result_url}"
            )
            raise

    def _generate_random_blob_name(self, original_path: str) -> str:
        """
        Generate a random blob name based on the original file path.

        Args:
            original_path (str): The original file path.

        Returns:
            str: The generated random blob name.
        """
        ext = ".wav"
        return f"{uuid.uuid4().hex}{ext}"
