# ingester/ingestion_service.py
import os
import time
from typing import List, Dict, Any
from utils.git_handler import GitHandler
from .markdown_processor import MarkdownProcessor
from .embedding_service import EmbeddingService
from .chroma_store import ChromaStore

class MarkdownIngestionService:
    """Orchestrates the document ingestion workflow."""
    
    def __init__(
        self,
        git_repo: GitHandler,
        md_processor: MarkdownProcessor,
        embed_service: EmbeddingService,
        chroma_store: ChromaStore
    ):
        self.git_repo = git_repo
        self.md_processor = md_processor
        self.embed_service = embed_service
        self.chroma_store = chroma_store
    
    def process_git_delta(self) -> None:
        """Process documents based on Git changes."""
        print("=== GIT-BASED DOCUMENT PROCESSING STARTED ===")
        
        # Verify Git setup
        if not self.git_repo.verify_git_installed():
            print("Git is not installed. Falling back to full processing.")
            self.process_full_rebuild()
            return
        
        if not self.git_repo.configure_safe_directory():
            print("Failed to configure Git safe directory. Falling back to full processing.")
            self.process_full_rebuild()
            return
        
        if not self.git_repo.is_git_repo():
            print("This is not a Git repository. Falling back to full processing.")
            self.process_full_rebuild()
            return
        
        try:
            self.chroma_store.connect()
            
            current_commit = self.git_repo.get_current_commit()
            last_commit = self.git_repo.load_last_processed_commit()
            
            if last_commit is None:
                print("No last processed commit found, processing all documents.")
                changed_files = self.md_processor.list_all_markdown_files()
                deleted_files = []
                is_first_run = True
            elif last_commit == current_commit:
                print("No new commits to process.")
                return
            else:
                print(f"Processing changes from {last_commit} to {current_commit}")
                changed_files, deleted_files = self.git_repo.get_changed_files(last_commit, current_commit)
                is_first_run = False
            
            # Process deletions first
            if deleted_files:
                print(f"Deleting chunks for {len(deleted_files)} deleted files...")
                self.chroma_store.delete_chunks_for_files(deleted_files)
            
            # Process changes/additions
            if changed_files:
                print(f"Processing {len(changed_files)} changed/added files...")
                # Critical fix: delete old chunks before reprocessing
                self.chroma_store.delete_chunks_for_files(changed_files)
                self._process_files(changed_files)
            
            # Update state if we processed anything
            if current_commit and (changed_files or deleted_files or is_first_run):
                self.git_repo.save_last_processed_commit(current_commit)
                print(f"Updated last processed commit to {current_commit}")
            
            print("Git-based document processing complete!")
        except Exception as e:
            print(f"Error in Git-based document processing: {str(e)}")
    
    def process_full_rebuild(self) -> None:
        """Fallback to full rebuild with content-based IDs"""
        print("=== FALLBACK DOCUMENT PROCESSING STARTED ===")
        try:
            self.chroma_store.connect()
            
            # Clear existing chunks to prevent duplicates
            try:
                print("Clearing existing chunks for clean transition...")
                existing_docs = self.chroma_store.collection.get(include=['metadatas'])
                if existing_docs['ids']:
                    sources = list(set([meta.get('source') for meta in existing_docs['metadatas'] if meta.get('source')]))
                    self.chroma_store.delete_chunks_for_files(sources)
                    print(f"Cleared {len(sources)} file sources from database")
            except Exception as e:
                print(f"Warning: Could not clear existing chunks: {e}")
            
            # Process all files
            file_paths = self.md_processor.list_all_markdown_files()
            if not file_paths:
                print("No markdown files found to process")
                return
            
            self._process_files(file_paths)
            print("Document processing complete!")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
    
    def process_single_file(self, file_path: str) -> None:
        """Process a single markdown file immediately."""
        print(f"=== IMMEDIATE PROCESSING FOR FILE: {file_path} ===")
        try:
            self.chroma_store.connect()
            
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist, skipping processing.")
                return
            
            if os.path.getsize(file_path) == 0:
                print(f"File {file_path} is empty, skipping processing.")
                return
            
            content = self.md_processor.load_file_safely(file_path)
            if content is None:
                print(f"Could not load content from {file_path}, skipping processing.")
                return
            
            self.chroma_store.delete_chunks_for_files([file_path])
            
            chunks = self.md_processor.recursive_character_text_splitter(content)
            if not chunks:
                print(f"No chunks generated for {file_path}, skipping upsert.")
                return
            
            all_contents = [chunk['content'] for chunk in chunks]
            all_ids = [
                self.md_processor.generate_content_based_id(chunk['content'], file_path, i)
                for i, chunk in enumerate(chunks)
            ]
            all_metadatas = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': file_path,
                    'chunk_index': str(i),
                    'original_length': str(len(content)),
                    'processed_at': str(int(time.time()))
                }
                all_metadatas.append(metadata)
            
            embeddings = self.embed_service.generate_embeddings(all_contents)
            self.chroma_store.upsert_chunks(
                embeddings=embeddings,
                documents=all_contents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Immediate processing complete for file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    def delete_file(self, file_path: str) -> None:
        """Delete all chunks for a file from ChromaDB."""
        print(f"=== DELETING INDEXED CHUNKS FOR FILE: {file_path} ===")
        try:
            self.chroma_store.connect()
            self.chroma_store.delete_chunks_for_files([file_path])
            print(f"Deleted chunks for {file_path}")
        except Exception as e:
            print(f"Error deleting chunks for {file_path}: {str(e)}")
    
    def _process_files(self, file_paths: List[str]) -> None:
        """Helper to process multiple files."""
        all_chunks = []
        all_contents = []
        all_metadatas = []
        all_ids = []
        
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            content = self.md_processor.load_file_safely(file_path)
            if content is None:
                continue
            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                continue
            
            chunks = self.md_processor.recursive_character_text_splitter(content)
            print(f"Split into {len(chunks)} chunks")
            
            for idx, chunk in enumerate(chunks):
                chunk_id = self.md_processor.generate_content_based_id(
                    chunk['content'],
                    file_path,
                    idx
                )
                metadata = {
                    'source': file_path,
                    'chunk_index': str(idx),
                    'original_length': str(len(content)),
                    'processed_at': str(int(time.time()))
                }
                
                all_chunks.append(chunk)
                all_contents.append(chunk['content'])
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)
        
        if not all_chunks:
            print(f"No valid chunks generated for any file")
            return
        
        print(f"Generating embeddings for {len(all_contents)} chunks...")
        all_embeddings = self.embed_service.generate_embeddings(all_contents)
        
        print(f"Upserting {len(all_chunks)} chunks to ChromaDB...")
        self.chroma_store.upsert_chunks(
            embeddings=all_embeddings,
            documents=all_contents,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Processed {len(file_paths)} files")
