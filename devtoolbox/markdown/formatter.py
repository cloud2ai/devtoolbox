import re
import logging

from devtoolbox.markdown.base import MarkdownBase


class MarkdownFormatter(MarkdownBase):
    """
    Formats a Markdown file by ensuring appropriate blank lines around titles,
    paragraphs, code blocks, and separators.

    Inherits from MarkdownBase to utilize common markdown handling functionality.
    """

    def __init__(self, path=None, content=None):
        """
        Initializes the MarkdownFormatter with either a path to a Markdown file
        or markdown content.

        Args:
            path (str, optional): Path to markdown file to load.
            content (str, optional): Raw markdown content string to use.
        """
        super().__init__(path, content)
        self.in_code_block = False
        logging.debug("MarkdownFormatter initialized")

    def _add_blank_lines(self, lines, index, reason):
        """
        Adds blank lines before and after the specified line if needed and logs
        the action.

        Args:
            lines (list): The list of lines from the Markdown file.
            index (int): The index of the current line in the list.
            reason (str): The reason for adding a blank line.
        """
        logging.debug(
            f"Checking blank lines needed for line {index + 1}: {reason}"
        )

        # Add a blank line before the current line if needed
        if index > 0 and not re.match(r'^\s*$', lines[index - 1].strip()):
            lines.insert(index, '\n')
            logging.debug(f"Added blank line before line {index + 1}: {reason}")

        # Add a blank line after the current line if needed
        if index == len(lines) - 1:
            # Special case for the last line
            lines.append('\n')
            logging.debug(
                f"Added blank line after the last line {index + 1}: {reason}"
            )
        elif (index < len(lines) - 1 and
              not re.match(r'^\s*$', lines[index + 1].strip())):
            lines.insert(index + 1, '\n')
            logging.debug(f"Added blank line after line {index + 1}: {reason}")

    def _process_line(self, lines, line, index):
        """
        Processes each line and determines if blank lines need to be added.

        Args:
            lines (list): The list of lines from the Markdown file.
            line (str): The current line being processed.
            index (int): The index of the current line in the list.
        """
        stripped_line = line.strip()
        logging.debug(f"Processing line {index + 1}: {stripped_line}")

        # Handle code blocks
        if re.match(r'^\s*```', stripped_line):
            logging.debug(f"Code block detected at line {index + 1}")
            self._handle_code_block(lines, line, index)

        # Handle list items and tables
        elif (re.match(r'^\s*[-*|]\s', stripped_line) or
              re.match(r'^\s*\d+\.\s', stripped_line)):
            logging.debug(f"List item or table detected at line {index + 1}")
            lines.append(line)

        # Handle titles and horizontal rules
        elif (re.match(r'^#+\s', stripped_line) or
              re.match(r'^\s*[-=]+$', stripped_line)):
            logging.debug(f"Title or separator detected at line {index + 1}")
            self._add_blank_lines(lines, len(lines), "Title or separator")
            lines.append(line)
            self._add_blank_lines(lines, len(lines) - 1, "Title or separator")

        # Handle separators (horizontal rules)
        elif re.match(r'^\s*---+', stripped_line):
            logging.debug(f"Separator detected at line {index + 1}")
            self._add_blank_lines(lines, len(lines), "Separator")
            lines.append(line)
            self._add_blank_lines(lines, len(lines) - 1, "Separator")

        # Handle blank lines
        elif re.match(r'^\s*$', stripped_line):
            logging.debug(f"Blank line detected at line {index + 1}")
            lines.append(line)

        # Handle regular paragraph text
        else:
            if not self.in_code_block:
                logging.debug(f"Paragraph detected at line {index + 1}")
                self._add_blank_lines(lines, len(lines), "Paragraph")
            lines.append(line)

    def _handle_code_block(self, lines, line, index):
        """
        Handles the formatting of code blocks, ensuring blank lines before and
        after.

        Args:
            lines (list): The list of lines from the Markdown file.
            line (str): The current line being processed.
            index (int): The index of the current line in the list.
        """
        # Handle code block start
        if not self.in_code_block:
            logging.debug(f"Starting code block at line {index + 1}")
            if len(lines) > 0 and not re.match(r'^\s*$', lines[-1].strip()):
                lines.append('\n')
                logging.debug(
                    f"Added blank line before code block at line {index + 1}"
                )

        # Add the code block delimiter line
        lines.append(line)

        # Handle code block end
        if self.in_code_block:
            logging.debug(f"Ending code block at line {index + 1}")
            if (index + 1 < len(lines) and
                not re.match(r'^\s*$', lines[index + 1].strip())):
                lines.append('\n')
                logging.debug(
                    f"Added blank line after code block at line {index + 1}"
                )

        # Toggle code block state
        self.in_code_block = not self.in_code_block
        logging.debug(f"Code block state is now: {self.in_code_block}")

    def format(self):
        """
        Formats the Markdown content by processing it and updating the content
        attribute. If a file path was provided, writes the formatted content
        back to the file.

        Returns:
            str: The formatted markdown content.
        """
        # Split content into lines
        lines = self.content.splitlines(True)
        logging.debug(f"Processing {len(lines)} lines of markdown content")

        formatted_lines = []
        for i, line in enumerate(lines):
            self._process_line(formatted_lines, line, i)

        # Remove redundant blank lines
        formatted_content = re.sub(r'\n{3,}', '\n\n', ''.join(formatted_lines))
        logging.debug("Removed redundant blank lines from formatted content")

        # Update the content attribute
        self.content = formatted_content
        logging.debug("Updated content attribute with formatted content")

        # Write back to file if path was provided
        if self.path:
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(formatted_content)
                logging.info(
                    f"Formatted markdown content written to: {self.path}"
                )

        return formatted_content
