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

from devtoolbox.speech.clients.azure_errors import AzureRecognitionError

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from devtoolbox.speech.azure_provider import AzureConfig

class AzureClient:
    """
    Azure REST API and Blob Storage client for speech batch transcription.
    """
    DEFAULT_CONTENT_TYPE = "audio/wav"

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

    def upload_blob(self, file_path: str) -> Tuple[str, str]:
        """
        Upload a file to Azure Blob Storage and return (blob_name, sas_url).

        Args:
            file_path (str): Local file path to upload.

        Returns:
            tuple[str, str]: (blob_name, sas_url)
        """
        blob_name = self._generate_random_blob_name(file_path)
        container_client = self.blob_service_client.get_container_client(
            self.config.container_name
        )
        try:
            container_client.create_container()
        except Exception as e:
            # Log and ignore if already exists
            logger.info(
                f"Create container exception: {e}"
            )
        try:
            with open(file_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True,
                    content_settings=ContentSettings(
                        content_type=self.DEFAULT_CONTENT_TYPE
                    )
                )
        except Exception as e:
            logger.error(
                f"Upload blob failed: {e} file={file_path}"
            )
            raise
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
        except Exception as e:
            logger.error(
                f"Batch transcription submit failed: {e} url={url}"
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
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

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
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def download_transcription_result(self, result_url: str) -> str:
        """
        Download the transcription result text from the given URL.

        Args:
            result_url (str): The URL to the result file.

        Returns:
            str: The transcription result text.
        """
        resp = requests.get(result_url, timeout=30)
        resp.raise_for_status()
        return resp.text

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