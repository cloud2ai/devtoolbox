import os
import logging
from devtoolbox.storage import FileStorage, ObjectStorage, TEXT_CONTENT_TYPE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_file_storage():
    """Example usage of FileStorage."""
    try:
        # Initialize FileStorage with a base path
        base_path = "./test_storage"
        fs = FileStorage(base_path)

        # Test writing text content
        content = "Hello, this is a test file!"
        fs.write("test.txt", content, content_type=TEXT_CONTENT_TYPE)
        logging.info("Successfully wrote text file")

        # Test reading content
        read_content = fs.read("test.txt")
        logging.info(f"Read content: {read_content}")

        # Test file existence
        exists = fs.exists("test.txt")
        logging.info(f"File exists: {exists}")

        # Test listing files
        files = fs.ls(pattern="*.txt")
        logging.info(f"Found files: {files}")

        # Test copying files
        src_path = os.path.join(base_path, "test.txt")
        fs.cp_from_path(src_path, "test_copy.txt")
        logging.info("Successfully copied file")

    except Exception as e:
        logging.error(f"Error in file storage test: {str(e)}")


def test_object_storage():
    """Example usage of ObjectStorage."""
    try:
        # Initialize ObjectStorage with configuration
        bucket_name = os.environ.get("OSS_BUCKET", "test-bucket-1318151887")
        os_storage = ObjectStorage(
            base_path=bucket_name,
            endpoint=os.environ.get("OSS_ENDPOINT", "storage.example.com"),
            access_key=os.environ.get("OSS_ACCESS_KEY", "your-access-key"),
            secret_key=os.environ.get("OSS_SECRET_KEY", "your-secret-key"),
            region=os.environ.get("OSS_REGION", "us-east-1"),
            use_virtual_style=True
        )

        # Test writing text content
        content = "Hello, this is a test object!"
        os_storage.write(
            "test.txt",
            content,
            content_type=TEXT_CONTENT_TYPE
        )
        logging.info("Successfully wrote object")

        # Test reading content
        read_content = os_storage.read("test.txt")
        logging.info(f"Read content: {read_content}")

        # Test object existence
        exists = os_storage.exists("test.txt")
        logging.info(f"Object exists: {exists}")

        # Test listing objects
        objects = os_storage.ls(pattern="*.txt")
        logging.info(f"Found objects: {objects}")

        # Test getting object URL
        url = os_storage.full_path("test.txt", permanent=True)
        logging.info(f"Object URL: {url}")

    except Exception as e:
        logging.error(f"Error in object storage test: {str(e)}")


def main():
    """Run all storage tests."""
    logging.info("Starting storage tests")

    logging.info("\n1. Testing FileStorage...")
    test_file_storage()

    logging.info("\n2. Testing ObjectStorage...")
    test_object_storage()

    logging.info("\nAll tests completed.")


if __name__ == "__main__":
    main()