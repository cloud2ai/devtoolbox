#!/usr/bin/env python3
"""
Example script demonstrating the usage of MarkdownFormatter.

This script shows how to use the MarkdownFormatter class to format
Markdown files by adding appropriate blank lines around different elements.
"""

import os
import sys
import logging
from pathlib import Path

from devtoolbox.markdown.formatter import MarkdownFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_unformatted_markdown():
    """Create an unformatted Markdown file for demonstration."""

    # Sample unformatted Markdown content
    content = """# Markdown Formatter Example
This paragraph has no blank line above it.
## Subheading
This paragraph has no blank lines around it.
* List item 1
* List item 2
Code block with no blank lines:
```python
def example():
    return "This is an example"
```
Another paragraph with no blank line above.
---
Text after horizontal rule with no blank line.
## Another Subheading
Final paragraph."""

    # Create the file
    file_path = Path(__file__).parent / "unformatted.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created unformatted Markdown file: {file_path}")
    return file_path

def format_markdown_file(file_path):
    """Format a Markdown file using MarkdownFormatter."""
    print(f"\nFormatting file: {file_path}")

    # Initialize formatter with file path
    formatter = MarkdownFormatter(path=str(file_path))

    # Format the file
    formatter.format()

    print(f"File formatted successfully: {file_path}")
    return file_path

def format_markdown_content():
    """Demonstrate formatting Markdown content as a string."""
    print("\nFormatting Markdown content from string:")

    # Sample unformatted content
    content = """# String Content Example
This is a paragraph.
## Subheading
* List item
```
Code block
```
<code_block_to_apply_changes_from>
```
"""

    print("\nBEFORE FORMATTING:")
    print("-" * 40)
    print(content)
    print("-" * 40)

    # Format the content
    formatter = MarkdownFormatter(content=content)
    formatted = formatter.format()

    print("\nAFTER FORMATTING:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)

def main():
    """Main function to demonstrate MarkdownFormatter."""
    print("MARKDOWN FORMATTER DEMONSTRATION")
    print("=" * 40)

    # Example 1: Format a file
    file_path = create_unformatted_markdown()
    formatted_file = format_markdown_file(file_path)

    # Display the formatted file content
    print("\nFormatted file content:")
    print("-" * 40)
    with open(formatted_file, "r", encoding="utf-8") as f:
        print(f.read())
    print("-" * 40)

    # Example 2: Format content from string
    format_markdown_content()

    print("\nDemonstration completed!")

if __name__ == "__main__":
    main()