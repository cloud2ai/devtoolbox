"""
Storage related commands
"""
import typer
from pathlib import Path
import logging
from typing import Optional
from devtoolbox.storage import ObjectStorage, FileStorage
from devtoolbox.cli.utils import setup_logging


# Configure logging
logger = logging.getLogger("devtoolbox.storage")
app = typer.Typer(help="Storage related commands")


@app.callback()
def callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
):
    """
    Storage command line tool
    """
    global logger
    logger = setup_logging(debug, "devtoolbox.storage")


def _get_storage(storage_type: str, bucket: Optional[str] = None,
                 endpoint: Optional[str] = None, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, region: Optional[str] = None,
                 use_virtual_style: bool = False, base_path: str = "."):
    """
    Helper function to create storage instance based on type and parameters
    """
    if storage_type.lower() == "file":
        return FileStorage(base_path)
    elif storage_type.lower() == "object":
        if not all([bucket, endpoint, access_key, secret_key]):
            raise ValueError(
                "bucket, endpoint, access_key, and secret_key are "
                "required for object storage"
            )
        return ObjectStorage(
            bucket,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            use_virtual_style=use_virtual_style
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


@app.command("upload")
def upload(
    source: Path = typer.Argument(
        ...,
        help="Source file path",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    dest: str = typer.Argument(
        ...,
        help="Destination path in storage",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Upload a file to storage
    """
    logger.debug(
        "Uploading file %s to %s using %s storage",
        source, dest, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style, str(source.parent)
        )
        storage.cp_from_path(str(source), dest)
        typer.echo(f"Successfully uploaded {source} to {dest}")
    except Exception as e:
        logger.error(
            "Failed to upload file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to upload file: {str(e)}")
        raise typer.Exit(1)


@app.command("download")
def download(
    source: str = typer.Argument(
        ...,
        help="Source path in storage",
    ),
    dest: Path = typer.Argument(
        ...,
        help="Destination file path",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Download a file from storage
    """
    logger.debug(
        "Downloading file %s to %s using %s storage",
        source, dest, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style, str(dest.parent)
        )

        if storage_type.lower() == "object":
            # Object storage has download method
            storage.download(source, str(dest))
        else:
            # File storage - use read and write
            content = storage.read(source)
            with open(dest, 'wb') as f:
                if isinstance(content, str):
                    f.write(content.encode())
                else:
                    f.write(content)

        typer.echo(f"Successfully downloaded {source} to {dest}")
    except Exception as e:
        logger.error(
            "Failed to download file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to download file: {str(e)}")
        raise typer.Exit(1)


@app.command("read")
def read_file(
    path: str = typer.Argument(
        ...,
        help="Path to read from storage",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path (if not specified, prints to stdout)",
    ),
    content_type: str = typer.Option(
        "text",
        "-c", "--content-type",
        help="Content type of the file",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Read content from storage
    """
    logger.debug(
        "Reading file %s using %s storage",
        path, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style
        )
        content = storage.read(path, content_type=content_type)

        if output_file:
            # Write to file
            with open(output_file, 'wb') as f:
                if isinstance(content, str):
                    f.write(content.encode())
                else:
                    f.write(content)
            typer.echo(f"Successfully read {path} to {output_file}")
        else:
            # Print to stdout
            if isinstance(content, bytes):
                typer.echo(content.decode())
            else:
                typer.echo(content)
    except Exception as e:
        logger.error(
            "Failed to read file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to read file: {str(e)}")
        raise typer.Exit(1)


@app.command("write")
def write_file(
    path: str = typer.Argument(
        ...,
        help="Path to write to storage",
    ),
    content: str = typer.Argument(
        ...,
        help="Content to write",
    ),
    content_type: str = typer.Option(
        "text",
        "-c", "--content-type",
        help="Content type of the file",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Write content to storage
    """
    logger.debug(
        "Writing content to %s using %s storage",
        path, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style
        )
        result = storage.write(path, content, content_type=content_type)
        typer.echo(f"Successfully wrote {len(content)} characters to {path}")
    except Exception as e:
        logger.error(
            "Failed to write file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to write file: {str(e)}")
        raise typer.Exit(1)


@app.command("url")
def get_url(
    path: str = typer.Argument(
        ...,
        help="Path to generate URL for",
    ),
    permanent: bool = typer.Option(
        True,
        "-p", "--permanent",
        help="Generate permanent URL (for object storage)",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Generate URL for file in storage
    """
    logger.debug(
        "Generating URL for %s using %s storage",
        path, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style
        )

        if storage_type.lower() == "object":
            url = storage.full_path(path, permanent=permanent)
        else:
            # For file storage, just return the full path
            url = storage.full_path(path)

        typer.echo(url)
    except Exception as e:
        logger.error(
            "Failed to generate URL: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to generate URL: {str(e)}")
        raise typer.Exit(1)


@app.command("presigned-url")
def get_presigned_url(
    path: str = typer.Argument(
        ...,
        help="Path to generate presigned URL for",
    ),
    operation: str = typer.Option(
        "get",
        "-o", "--operation",
        help="Operation type (get or put)",
        case_sensitive=False,
    ),
    expires_minutes: int = typer.Option(
        10,
        "-e", "--expires",
        help="Expiration time in minutes",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Generate presigned URL for file in storage
    """
    logger.debug(
        "Generating presigned URL for %s using %s storage",
        path, storage_type
    )

    try:
        if storage_type.lower() != "object":
            typer.echo("Presigned URLs are only supported for object storage")
            raise typer.Exit(1)

        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style
        )

        if operation.lower() == "get":
            url = storage.get_presigned_download_url(path, expires_minutes)
        elif operation.lower() == "put":
            url = storage.get_presigned_upload_url(path, expires_minutes)
        else:
            typer.echo(f"Unsupported operation: {operation}")
            raise typer.Exit(1)

        typer.echo(url)
    except Exception as e:
        logger.error(
            "Failed to generate presigned URL: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to generate presigned URL: {str(e)}")
        raise typer.Exit(1)


@app.command("info")
def get_info(
    path: str = typer.Argument(
        ...,
        help="Path to get info for",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Get information about file in storage
    """
    logger.debug(
        "Getting info for %s using %s storage",
        path, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style
        )

        exists = storage.exists(path)
        if not exists:
            typer.echo(f"File {path} does not exist")
            raise typer.Exit(1)

        # Get file info
        if isinstance(storage, ObjectStorage):
            # For object storage, get object stats
            stat = storage.client.stat_object(storage.bucket, path)
            typer.echo(f"File: {path}")
            typer.echo(f"Size: {stat.size} bytes")
            typer.echo(f"Last Modified: {stat.last_modified}")
            typer.echo(f"ETag: {stat.etag}")
            typer.echo(f"Content Type: {stat.content_type}")
        else:
            # For file storage, get file stats
            full_path = storage.full_path(path)
            import os
            stat = os.stat(full_path)
            typer.echo(f"File: {path}")
            typer.echo(f"Size: {stat.st_size} bytes")
            typer.echo(f"Last Modified: {stat.st_mtime}")
            typer.echo(f"Full Path: {full_path}")

    except Exception as e:
        logger.error(
            "Failed to get file info: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to get file info: {str(e)}")
        raise typer.Exit(1)


@app.command("list")
def list_files(
    path: str = typer.Argument(
        "",
        help="Path to list files from",
    ),
    pattern: str = typer.Option(
        "*",
        "-p", "--pattern",
        help="File pattern to match",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    List files in storage
    """
    logger.debug(
        "Listing files in %s with pattern %s using %s storage",
        path, pattern, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style, path
        )
        files = storage.ls(path, pattern)
        if files:
            for file in files:
                typer.echo(file)
        else:
            typer.echo("No files found")
    except Exception as e:
        logger.error(
            "Failed to list files: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to list files: {str(e)}")
        raise typer.Exit(1)


@app.command("delete")
def delete(
    path: str = typer.Argument(
        ...,
        help="Path to delete",
    ),
    recursive: bool = typer.Option(
        False,
        "-r", "--recursive",
        help="Recursively delete directory",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Delete a file or directory from storage
    """
    logger.debug(
        "Deleting %s (recursive=%s) using %s storage",
        path, recursive, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style, path
        )
        storage.rm(path, recursive)
        typer.echo(f"Successfully deleted {path}")
    except Exception as e:
        logger.error(
            "Failed to delete: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to delete: {str(e)}")
        raise typer.Exit(1)


@app.command("exists")
def exists(
    path: str = typer.Argument(
        ...,
        help="Path to check",
    ),
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Check if a file exists in storage
    """
    logger.debug(
        "Checking if %s exists using %s storage",
        path, storage_type
    )

    try:
        storage = _get_storage(
            storage_type, bucket, endpoint, access_key, secret_key,
            region, use_virtual_style, path
        )
        if storage.exists(path):
            typer.echo(f"{path} exists")
        else:
            typer.echo(f"{path} does not exist")
    except Exception as e:
        logger.error(
            "Failed to check existence: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to check existence: {str(e)}")
        raise typer.Exit(1)


@app.command("temp-dir")
def get_temp_dir(
    storage_type: str = typer.Option(
        "file",
        "-t", "--type",
        help="Storage type (file or object)",
        case_sensitive=False,
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "-b", "--bucket",
        help="Bucket name for object storage",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "-e", "--endpoint",
        help="Endpoint URL for object storage",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "-k", "--access-key",
        help="Access key for object storage",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "-s", "--secret-key",
        help="Secret key for object storage",
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r", "--region",
        help="Region for object storage",
    ),
    use_virtual_style: bool = typer.Option(
        False,
        "-v", "--virtual-style",
        help="Use virtual style endpoint for object storage",
    ),
):
    """
    Get temporary directory information for uploads and downloads
    """
    logger.debug(
        "Getting temp directory info using %s storage",
        storage_type
    )

    try:
        import tempfile
        import os

        # Get system temp directory
        temp_dir = tempfile.gettempdir()
        typer.echo(f"System temporary directory: {temp_dir}")

        if storage_type.lower() == "object":
            # For object storage, show presigned URL info
            if not all([bucket, endpoint, access_key, secret_key]):
                typer.echo("Object storage credentials required for presigned URLs")
                typer.echo("Use --bucket, --endpoint, --access-key, --secret-key options")
                raise typer.Exit(1)

            storage = _get_storage(
                storage_type, bucket, endpoint, access_key, secret_key,
                region, use_virtual_style
            )

            # Generate example presigned URLs
            example_path = "temp/example.txt"
            try:
                upload_url = storage.get_presigned_upload_url(example_path, 10)
                download_url = storage.get_presigned_download_url(example_path, 10)

                typer.echo(f"\nObject Storage Temporary URLs (example):")
                typer.echo(f"Upload URL: {upload_url}")
                typer.echo(f"Download URL: {download_url}")
                typer.echo(f"\nTo generate presigned URLs for specific files:")
                typer.echo(f"  devtoolbox storage presigned-url <path> --operation put")
                typer.echo(f"  devtoolbox storage presigned-url <path> --operation get")
            except Exception as e:
                typer.echo(f"Could not generate example presigned URLs: {str(e)}")
        else:
            # For file storage, show local temp directory
            typer.echo(f"\nFile Storage Temporary Directory:")
            typer.echo(f"Local temp directory: {temp_dir}")
            typer.echo(f"Storage base path: {os.getcwd()}")
            typer.echo(f"\nTo upload files:")
            typer.echo(f"  devtoolbox storage upload <local_file> <storage_path>")
            typer.echo(f"To download files:")
            typer.echo(f"  devtoolbox storage download <storage_path> <local_file>")

    except Exception as e:
        logger.error(
            "Failed to get temp directory info: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to get temp directory info: {str(e)}")
        raise typer.Exit(1)