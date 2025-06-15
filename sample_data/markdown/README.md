# Sample Markdown Files

This directory contains sample markdown files for testing and development purposes.

## File Descriptions

### 1. basic.md
A basic markdown file demonstrating common markdown elements:
- Text formatting
- Lists (ordered and unordered)
- Code blocks
- Links and images
- Tables
- Blockquotes
- Task lists

### 2. with_images.md
A markdown file focused on image handling:
- Multiple image examples
- Different image sizes
- Image galleries
- Inline images
- Images with captions
- Image grids
- Images with links

### 3. technical.md
A technical documentation example:
- System architecture diagrams
- API documentation
- Code examples (Python and JavaScript)
- Configuration examples
- Troubleshooting guides
- Best practices

## Usage Examples

### Download Images
```bash
# Download images from basic.md
devtoolbox markdown download-images basic.md

# Download images with custom output directory
devtoolbox markdown download-images with_images.md --output-dir "images"

# Download images with debug logging
devtoolbox markdown --debug download-images technical.md
```

### Convert to Other Formats
```bash
# Convert to Word document
devtoolbox markdown convert basic.md --format docx

# Convert to PDF
devtoolbox markdown convert technical.md --format pdf

# Convert to HTML
devtoolbox markdown convert with_images.md --format html

# Convert without downloading images
devtoolbox markdown convert basic.md --no-download-images
```

## Notes

- All sample files use [Lorem Picsum](https://picsum.photos) for random images
- Images are generated with different sizes and aspect ratios
- Code examples are for demonstration purposes only
- Some features (like Mermaid diagrams) may require additional processing