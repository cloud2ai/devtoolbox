import unittest
import os
import tempfile
from devtoolbox.markdown.formatter import MarkdownFormatter
from tests.utils.test_logging import setup_test_logging


# Initialize logging
logger = setup_test_logging()


class TestMarkdownFormatter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        logger.info("Cleaning up test fixtures")
        # Remove the temporary directory and all its contents
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_title_formatting(self):
        """Test formatting of titles with blank lines."""
        logger.info("Testing title formatting")
        
        # Test content with titles
        content = """# Title 1
Some text
## Title 2
More text
### Title 3
Even more text"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if blank lines are added around titles
        self.assertIn("# Title 1\n\n", formatted_content)
        self.assertIn("\n## Title 2\n\n", formatted_content)
        self.assertIn("\n### Title 3\n\n", formatted_content)
        logger.debug("Verified title formatting")

    def test_paragraph_formatting(self):
        """Test formatting of paragraphs with blank lines."""
        logger.info("Testing paragraph formatting")
        
        # Test content with paragraphs
        content = """First paragraph
Second paragraph
Third paragraph"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if blank lines are added between paragraphs
        self.assertIn("First paragraph\n\n", formatted_content)
        self.assertIn("\nSecond paragraph\n\n", formatted_content)
        self.assertIn("\nThird paragraph", formatted_content)
        logger.debug("Verified paragraph formatting")

    def test_code_block_formatting(self):
        """Test formatting of code blocks with blank lines."""
        logger.info("Testing code block formatting")
        
        # Test content with code blocks
        content = """Some text
```python
def test():
    pass
```
More text
```bash
echo "test"
```
Even more text"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if blank lines are added around code blocks
        self.assertIn("\n```python\n", formatted_content)
        self.assertIn("\n```\n\n", formatted_content)
        self.assertIn("\n```bash\n", formatted_content)
        logger.debug("Verified code block formatting")

    def test_list_formatting(self):
        """Test formatting of lists with blank lines."""
        logger.info("Testing list formatting")
        
        # Test content with lists
        content = """Some text
- Item 1
- Item 2
- Item 3
More text
1. First
2. Second
3. Third
Even more text"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if lists are preserved and formatted correctly
        self.assertIn("\n- Item 1\n", formatted_content)
        self.assertIn("\n- Item 2\n", formatted_content)
        self.assertIn("\n- Item 3\n", formatted_content)
        self.assertIn("\n1. First\n", formatted_content)
        self.assertIn("\n2. Second\n", formatted_content)
        self.assertIn("\n3. Third\n", formatted_content)
        logger.debug("Verified list formatting")

    def test_separator_formatting(self):
        """Test formatting of separators with blank lines."""
        logger.info("Testing separator formatting")
        
        # Test content with separators
        content = """Some text
---
More text
===
Even more text"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if blank lines are added around separators
        self.assertIn("\n---\n\n", formatted_content)
        self.assertIn("\n===\n\n", formatted_content)
        logger.debug("Verified separator formatting")

    def test_mixed_content_formatting(self):
        """Test formatting of mixed content types."""
        logger.info("Testing mixed content formatting")
        
        # Test content with mixed elements
        content = """# Title
First paragraph

- List item 1
- List item 2

```python
def test():
    pass
```

Second paragraph
---
Third paragraph"""
        
        # Create formatter and format content
        formatter = MarkdownFormatter(content=content)
        formatted_content = formatter.format()
        
        # Check if all elements are properly formatted
        self.assertIn("# Title\n\n", formatted_content)
        self.assertIn("\nFirst paragraph\n\n", formatted_content)
        self.assertIn("\n- List item 1\n", formatted_content)
        self.assertIn("\n- List item 2\n\n", formatted_content)
        self.assertIn("\n```python\n", formatted_content)
        self.assertIn("\n```\n\n", formatted_content)
        self.assertIn("\nSecond paragraph\n\n", formatted_content)
        self.assertIn("\n---\n\n", formatted_content)
        self.assertIn("\nThird paragraph", formatted_content)
        logger.debug("Verified mixed content formatting")

    def test_file_handling(self):
        """Test formatting when reading from and writing to a file."""
        logger.info("Testing file handling")
        
        # Create a test markdown file
        test_file = os.path.join(self.test_dir, "test.md")
        content = """# Test File
Some content
- List item
```python
pass
```"""
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Create formatter with file path
        formatter = MarkdownFormatter(path=test_file)
        formatted_content = formatter.format()
        
        # Check if file was updated
        with open(test_file, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        self.assertEqual(formatted_content, file_content)
        logger.debug("Verified file handling")


if __name__ == '__main__':
    unittest.main() 