import glob
import logging
from io import BytesIO
import os
from pathlib import Path
import shutil
import fnmatch

from minio import Minio, error

# Content type constants
TEXT_CONTENT_TYPE = "text"
TEXT_CONTENT_TYPES = [TEXT_CONTENT_TYPE]

# Object Storage Service content type mappings
OSS_TEXT_CONTENT_TYPE = "text/plain"
OSS_CONTENT_TYPES = {
    "text": OSS_TEXT_CONTENT_TYPE,
    "image": "image/png"
}

logger = logging.getLogger(__name__)


class BaseStorage:
    """Base storage class that defines the interface for all storage
    implementations.

    This class provides a common interface for different storage backends
    like local filesystem and object storage services. It defines the basic
    operations that any storage implementation must support:

    - Reading files
    - Writing files
    - Checking file existence
    - Getting full file paths
    - Copying files
    - Listing files

    Storage implementations should inherit from this class and implement
    all methods.
    """

    def __init__(self, base_path, *args, **kwargs):
        """Initialize storage.

        Args:
            base_path (str): For FileStorage, the base path to store files.
            For ObjectStorage, the name of the bucket.
        """
        logger.debug(f"Initializing BaseStorage with base_path: "
                     f"{base_path}")
        self.base_path = base_path

    def read(self, path, *args, **kwargs):
        """Read content from the specified path."""
        raise NotImplementedError("read() method needs to be implemented")

    def write(self, path, content,
              content_type=TEXT_CONTENT_TYPE, *args, **kwargs):
        """Write content to the specified path."""
        raise NotImplementedError("write() method needs to be implemented")

    def exists(self, path, *args, **kwargs):
        """Check if the specified path exists."""
        raise NotImplementedError("exists() method needs to be implemented")

    def full_path(self, path):
        """Return full path for given path."""
        raise NotImplementedError("full_path() method needs to be implemented")

    def cp_from_path(self, src_full_path, dest_path,
                     content_type=TEXT_CONTENT_TYPE, *args, **kwargs):
        """Copy/Upload file from given path."""
        raise NotImplementedError(
            "cp_from_path() method needs to be implemented"
        )

    def ls(self, path=None, pattern="*"):
        """List all files and return their full paths."""
        raise NotImplementedError("ls() method needs to be implemented")


class ObjectStorage(BaseStorage):
    """Storage implementation for object storage services like S3, MinIO,
    etc.

    This class provides an implementation of the storage interface for
    cloud object storage services. It supports:

    - Reading/writing files to cloud storage
    - Generating pre-signed URLs for temporary access
    - Permanent URLs for public access
    - Listing objects with prefix and pattern matching
    - Content type handling for proper file storage
    - Virtual host style endpoints

    The implementation uses MinIO client which is compatible with S3 and
    other object storage services that implement the S3 protocol.
    """

    def __init__(self, base_path, *args, **kwargs):
        """Initialize object storage.

        Args:
            base_path (str): The bucket name.
            endpoint (str): The endpoint URL.
            access_key (str): The access key.
            secret_key (str): The secret key.
            region (str, optional): The region.
            use_virtual_style (bool, optional): Whether to use virtual style
                endpoint.
        """
        logger.info(f"Initializing ObjectStorage with bucket: {base_path}")

        # Try to get config from kwargs first, then environment variables
        self.endpoint = (
            kwargs.get("endpoint") or
            os.environ.get("OSS_ENDPOINT")
        )
        access_key = (
            kwargs.get("access_key") or
            os.environ.get("OSS_ACCESS_KEY")
        )
        secret_key = (
            kwargs.get("secret_key") or
            os.environ.get("OSS_SECRET_KEY")
        )
        region = (
            kwargs.get("region") or
            os.environ.get("OSS_REGION")
        )
        use_virtual_style = kwargs.get("use_virtual_style", False)

        # Validate required parameters
        if not self.endpoint:
            logger.error("ObjectStorage initialization failed: "
                         "endpoint is required")
            raise ValueError("endpoint is required")
        if not access_key:
            logger.error("ObjectStorage initialization failed: "
                         "access_key is required")
            raise ValueError("access_key is required")
        if not secret_key:
            logger.error("ObjectStorage initialization failed: "
                         "secret_key is required")
            raise ValueError("secret_key is required")

        self.bucket = base_path
        logger.debug(f"Using endpoint: {self.endpoint}, bucket: "
                     f"{self.bucket}")

        # Initialize Minio client
        try:
            self.client = Minio(
                self.endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region
            )
            logger.debug("Minio client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Minio client: {str(e)}")
            raise

        if use_virtual_style:
            logger.info("Enable virtual style endpoint for object storage")
            self.client.enable_virtual_style_endpoint()

        super().__init__(base_path, *args, **kwargs)

    def read(self, path, *args, **kwargs):
        """Read content from object storage.

        Args:
            path (str): The path to the object in storage.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
                content_type (str): The content type of the file. If not
                                  provided, will try to determine from file
                                  extension.

        Returns:
            str or bytes: The content of the file. For text files, returns a
                        string. For binary files, returns bytes.

        Raises:
            Exception: If there is an error reading the file.
        """
        logger.info(
            f"Reading from object storage: bucket={self.bucket}, path={path}"
        )
        try:
            response = self.client.get_object(self.bucket, path)
            data = response.data

            # Get content type from kwargs or try to determine from file
            # extension
            content_type = kwargs.get('content_type')
            if not content_type:
                content_type = self._get_content_type_from_path(path)

            # Only decode if it's a text content type
            if content_type in TEXT_CONTENT_TYPES:
                data = data.decode()
                logger.debug(
                    f"Successfully read and decoded {len(data)} bytes "
                    f"from {path}"
                )
            else:
                logger.debug(
                    f"Successfully read {len(data)} bytes from {path}"
                )

            return data
        except Exception as e:
            logger.error(f"Error reading from object storage: {str(e)}")
            raise
        finally:
            if 'response' in locals():
                logger.debug(f"Closing connection for {path}")
                response.close()
                response.release_conn()

    def _get_content_type_from_path(self, path):
        """Determine content type from file extension.

        Args:
            path (str): The file path.

        Returns:
            str: The content type based on file extension.
        """
        # Get file extension
        ext = os.path.splitext(path)[1].lower()

        # Map common extensions to content types
        content_type_map = {
            '.txt': TEXT_CONTENT_TYPE,
            '.md': TEXT_CONTENT_TYPE,
            '.json': TEXT_CONTENT_TYPE,
            '.csv': TEXT_CONTENT_TYPE,
            '.docx': (
                'application/vnd.openxmlformats-officedocument.'
                'wordprocessingml.document'
            ),
            '.xlsx': (
                'application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.sheet'
            ),
            '.pptx': (
                'application/vnd.openxmlformats-officedocument.'
                'presentationml.presentation'
            ),
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }

        return content_type_map.get(ext, 'application/octet-stream')

    def write(self, path, content,
              content_type=TEXT_CONTENT_TYPE, *args, **kwargs):
        """Write content to object storage."""
        # Set content type for saving
        write_content_type = OSS_CONTENT_TYPES.get(content_type,
                                                   OSS_TEXT_CONTENT_TYPE)
        logger.info(f"Writing to object storage: bucket={self.bucket}, "
                    f"path={path}, content_type={content_type} -> "
                    f"{write_content_type}")

        # Prepare content based on content type
        if content_type in TEXT_CONTENT_TYPES:
            content_encode = content.encode()
            write_content = BytesIO(content_encode)
            content_length = len(content_encode)
            logger.debug(f"Encoded text content, length: {content_length} "
                         "bytes")
        else:
            write_content = BytesIO(content)
            content_length = len(content)
            logger.debug(f"Binary content, length: {content_length} bytes")

        # Add more logger output for better debugging
        logger.info(
            f"Preparing to write object to storage: bucket={self.bucket}, "
            f"path={path}, content_length={content_length}, "
            f"content_type={write_content_type}"
        )
        logger.debug(
            f"Write content type: {type(write_content)}, "
            f"Content length: {content_length}"
        )
        try:
            logger.info(
                f"Calling put_object on Minio client for {path} in bucket "
                f"{self.bucket}"
            )
            result = self.client.put_object(
                self.bucket, path, write_content,
                content_length, content_type=write_content_type
            )
            logger.info(
                f"Successfully wrote object to storage: bucket={self.bucket}, "
                f"path={path}, etag={getattr(result, 'etag', None)}"
            )
            logger.debug(f"Put object result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error writing to object storage: {str(e)}")
            logger.exception(e)
            raise

    def exists(self, path, *args, **kwargs):
        """Check if object exists in storage."""
        logger.info(f"Checking if object exists: bucket={self.bucket}, "
                    f"path={path}")
        try:
            self.client.stat_object(self.bucket, path)
            logger.debug(f"Object exists: {path}")
            return True
        except error.S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"Object does not exist: {path}")
                return False
            else:
                logger.error(f"Error checking if object exists: {str(e)}")
                raise e

    def full_path(self, path, *args, **kwargs):
        """Generate a link for the object."""
        is_permanent = kwargs.get("permanent", True)
        logger.info(f"Generating full path for: {path}, permanent="
                    f"{is_permanent}")

        if is_permanent:
            # Sample url:
            # https://<bucket>.cos.ap-beijing.myqcloud.com/images/object
            base_url = f"{self.bucket}.{self.endpoint}"
            url = os.path.join("https://", base_url, path)
            logger.debug(f"Generated permanent URL: {url}")
        else:
            url = self.client.get_presigned_url("GET", self.bucket, path)
            logger.debug(f"Generated presigned URL: {url}")

        return url

    def cp_from_path(self, src_full_path, dest_path,
                     content_type=TEXT_CONTENT_TYPE, *args, **kwargs):
        """Copy file from local path to object storage."""
        logger.info(f"Copying {src_full_path} to {dest_path}...")
        write_content_type = OSS_CONTENT_TYPES.get(
            content_type, OSS_TEXT_CONTENT_TYPE)
        logger.debug(f"Using content type: {write_content_type}")

        try:
            result = self.client.fput_object(
                self.bucket, dest_path, src_full_path,
                content_type=write_content_type)
            logger.debug(f"Successfully copied to {dest_path}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error copying file to object storage: {str(e)}")
            raise

    def ls(self, path=None, pattern="*"):
        """List objects in storage."""
        objects = []
        prefix = path or ""
        logger.info(f"Listing objects in bucket={self.bucket}, "
                    f"prefix={prefix}, "
                    f"pattern={pattern}")

        try:
            for obj in self.client.list_objects(
                    self.bucket, prefix=prefix, recursive=True):
                if pattern == "*" or glob.fnmatch.fnmatch(obj.object_name,
                                                          pattern):
                    objects.append(obj.object_name)
                    logger.debug(f"Found object: {obj.object_name}")

            logger.info(f"Found {len(objects)} objects matching pattern")
            return objects
        except Exception as e:
            logger.error(f"Error listing objects: {str(e)}")
            raise


class FileStorage(BaseStorage):
    """File system storage implementation.

    This class provides a file system-based implementation of the BaseStorage
    interface. It supports:
    - Reading/writing files with proper content type handling
    - Directory creation and path management
    - File existence checks
    - File copying
    - File listing with pattern matching
    - Full path resolution

    The implementation uses standard Python file and path operations to
    handle local files and directories.
    """

    def read(self, path, *args, **kwargs):
        """Read content from file system."""
        target_path = os.path.join(self.base_path, path)
        # check if full path passed, use full path by default
        if self.base_path in path:
            target_path = path
        logger.info(f"Reading from storage: {target_path}...")

        try:
            # Always use binary mode for reading
            with open(target_path, "rb") as f:
                content = f.read()
            logger.debug(f"Read {len(content)} bytes from {target_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading from file: {target_path}, "
                         f"error: {str(e)}")
            raise

    def write(self, path, content,
              content_type=TEXT_CONTENT_TYPE, *args, **kwargs):
        """Write content to file system."""
        target_path = os.path.join(self.base_path, path)
        self._ensure_path_exists(os.path.dirname(target_path))
        logger.info(f"Writing content type {content_type} to storage: "
                    f"{target_path}...")

        # Always use binary mode for writing
        try:
            with open(target_path, "wb") as f:
                if isinstance(content, str):
                    content = content.encode()
                bytes_written = f.write(content)
            logger.debug(f"Successfully wrote {bytes_written} bytes to "
                         f"{target_path}")
            return bytes_written
        except Exception as e:
            logger.error(f"Error writing to file: {target_path}, "
                         f"error: {str(e)}")
            raise

    def exists(self, path, *args, **kwargs):
        """Check if file exists in file system."""
        target_path = os.path.join(self.base_path, path)
        exists = os.path.exists(target_path)
        logger.info(f"Checking if file exists: {target_path} -> {exists}")
        return exists

    def _ensure_path_exists(self, path):
        """Ensure directory path exists."""
        logger.debug(f"Ensuring path exists: {path}")
        Path(path).mkdir(parents=True, exist_ok=True)

    def full_path(self, path):
        """Return full path for given path."""
        full_path = os.path.join(self.base_path, path)
        logger.debug(f"Full path for {path} is {full_path}")
        return full_path

    def cp_from_path(self, src_full_path, dest_path, **kwargs):
        """Copy file from source path to destination path."""
        target_path = os.path.join(self.base_path, dest_path)
        self._ensure_path_exists(os.path.dirname(target_path))
        logger.info(f"Copy {src_full_path} to {target_path}...")

        try:
            shutil.copy(src_full_path, target_path)
            logger.debug(f"Successfully copied file to {target_path}")
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            raise

    def ls(self, path="", pattern=None):
        """List files in directory."""
        target_path = os.path.join(self.base_path, path)
        logger.info(f"Listing files in: {target_path}")

        try:
            files = []
            for root, _, filenames in os.walk(target_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.base_path)
                    if pattern is None or fnmatch.fnmatch(rel_path, pattern):
                        files.append(rel_path)
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing files in: {target_path}, "
                         f"error: {str(e)}")
            raise
