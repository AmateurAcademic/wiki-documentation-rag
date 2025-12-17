# ingestion/markdown_processor.py
import os
import glob
import hashlib
from typing import List, Dict, Any, Optional

class MarkdownProcessor:
    """Handles file loading, text splitting, and ID generation."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
    
    def load_file_safely(self, file_path: str) -> Optional[str]:
        """Load file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {file_path} with encoding {encoding}: {str(e)}")
                return None
        return None
    
    def list_all_markdown_files(self) -> List[str]:
        """List all markdown files in the directory"""
        files = []
        pattern = os.path.join(self.base_dir, "**", "*.md")
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.exists(filepath) and os.path.isfile(filepath):
                files.append(filepath)
        return files
    
    def recursive_character_text_splitter(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Mimics LangChain's RecursiveCharacterTextSplitter"""
        separators = ["\n\n", "\n", " ", ""]
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            # Try each separator
            split_pos = end
            for separator in separators:
                if separator and separator in text[start:end]:
                    # Find last occurrence of separator
                    pos = text.rfind(separator, start, end)
                    if pos != -1:
                        split_pos = pos + len(separator)
                        break
            
            chunk = text[start:split_pos]
            chunks.append({
                'content': chunk,
                'metadata': {'start_index': str(start)}
            })
            
            # Move start with overlap
            start = max(start + chunk_size - overlap, split_pos)
            if start >= len(text):
                break
        
        return chunks
    
    def generate_content_based_id(
        self, 
        content: str, 
        source_path: str, 
        chunk_index: int
    ) -> str:
        """Generate stable ID based on content to prevent duplicates"""
        normalized_source = os.path.normpath(source_path)
        identifier = f"{normalized_source}_{chunk_index}_{content}"
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:32]
