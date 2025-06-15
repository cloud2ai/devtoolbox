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
        if storage_type.lower() == "file":
            storage = FileStorage(str(source.parent))
        elif storage_type.lower() == "object":
            if not all([bucket, endpoint, access_key, secret_key]):
                raise ValueError(
                    "bucket, endpoint, access_key, and secret_key are "
                    "required for object storage"
                )
            storage = ObjectStorage(
                bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                use_virtual_style=use_virtual_style
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

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
        if storage_type.lower() == "file":
            storage = FileStorage(str(dest.parent))
        elif storage_type.lower() == "object":
            if not all([bucket, endpoint, access_key, secret_key]):
                raise ValueError(
                    "bucket, endpoint, access_key, and secret_key are "
                    "required for object storage"
                )
            storage = ObjectStorage(
                bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                use_virtual_style=use_virtual_style
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        storage.download(source, str(dest))
        typer.echo(f"Successfully downloaded {source} to {dest}")
    except Exception as e:
        logger.error(
            "Failed to download file: %s",
            str(e),
            exc_info=True
        )
        typer.echo(f"Failed to download file: {str(e)}")
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
        if storage_type.lower() == "file":
            storage = FileStorage(path)
        elif storage_type.lower() == "object":
            if not all([bucket, endpoint, access_key, secret_key]):
                raise ValueError(
                    "bucket, endpoint, access_key, and secret_key are "
                    "required for object storage"
                )
            storage = ObjectStorage(
                bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                use_virtual_style=use_virtual_style
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

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
        if storage_type.lower() == "file":
            storage = FileStorage(path)
        elif storage_type.lower() == "object":
            if not all([bucket, endpoint, access_key, secret_key]):
                raise ValueError(
                    "bucket, endpoint, access_key, and secret_key are "
                    "required for object storage"
                )
            storage = ObjectStorage(
                bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                use_virtual_style=use_virtual_style
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

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
        if storage_type.lower() == "file":
            storage = FileStorage(path)
        elif storage_type.lower() == "object":
            if not all([bucket, endpoint, access_key, secret_key]):
                raise ValueError(
                    "bucket, endpoint, access_key, and secret_key are "
                    "required for object storage"
                )
            storage = ObjectStorage(
                bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                region=region,
                use_virtual_style=use_virtual_style
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

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