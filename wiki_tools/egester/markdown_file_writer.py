# egestion/markdown_file_writer.py

from pathlib import Path
from typing import List, Optional, Union, Dict, Any

class MarkdownFileWriter:
    """Class for writing and editing markdown files."""

    # I will be adding additional methods in here for editing the markdown file
    # instead of making a new one or saving
    # TODO: Add templates?
    # TODO: logic for editing specific sections

    def save_file(
        self,
        content: str,
        file_path: Union[str, Path],
        append: bool = False
    ) -> None:
        """Save content to a markdown file."""
        
        file_path = Path(file_path)

        mode = 'a' if append else 'w'

        if not file_path.exists() and append:
            print(f"File {file_path} does not exist. Creating a new file.")
            mode = 'w'

        if file_path.exists() and not append:
            print(f"Overwriting existing file {file_path}.")

        with file_path.open(mode, encoding='utf-8') as f:
            f.write(content)
            f.write('\n')