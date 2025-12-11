import os
import time
import glob
import json
import hashlib
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
import chromadb

class MarkdownHandler(FileSystemEventHandler):
    """Handles file system events for markdown files, processes them, and stores them in ChromaDB."""
    def __init__(self):
        self.last_processed = 0
        self.data_dir = "/app/data"
        self.markdown_dir = "/app/data/markdown"
        self.state_dir = "/app/state"
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file = os.path.join(self.state_dir, ".git_processing_state.json")
        self.branch_name = None

    def _verify_git_installed(self):
        """Verify Git is installed and working"""
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


    def _configure_git_safe_directory(self) -> bool:
        """Configure Git to trust the markdown directory (CVE-2022-24765)"""
        try:
            self._run_git_command("config", "--global", "--add", "safe.directory", self.markdown_dir)
            print(f"Configured Git safe directory: {self.markdown_dir}")
            return True
        except Exception as e:
            print(f"Failed to configure Git safe directory: {e}")
            return False

    def _is_git_repo(self):
        """Verify we're in a Git repository"""
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.markdown_dir,  # Use markdown_dir, not data_dir
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _save_last_processed_commit(self, commit_hash):
        """Atomic write to prevent state file corruption"""
        temp_file = self.state_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({'last_processed_commit': commit_hash}, f)
            os.replace(temp_file, self.state_file)
        except Exception:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            raise

    def _load_last_processed_commit(self):
        """Load the last commit processed"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    return state.get('last_processed_commit')
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading state file: {str(e)}")
        return None

    def _run_git_command(self, *args, max_retries=3):
        """Run Git command with lock file with retries"""
        delay = 1
        for attempt in range(max_retries):
            try:
                return subprocess.run(
                    ["git", *map(str, args)],
                    cwd=self.markdown_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except subprocess.CalledProcessError as e:
                if "index.lock" in e.stderr.lower() and attempt < max_retries - 1:
                    print(
                        f"Git index.lock detected, retrying in {delay} seconds "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                # Preserve original traceback for debugging
                raise
            except subprocess.TimeoutExpired as exc:
                cmd_str = " ".join(map(str, args))
                raise RuntimeError(f"Git command timed out: {cmd_str}") from exc

        raise RuntimeError("Unexpected error in Git command execution")


    def _detect_git_branch(self):
        """Auto-detect Git branch (main or master)"""
        if self.branch_name:
            return self.branch_name
        
        for branch in ["main", "master"]:
            try:
                self._run_git_command("rev-parse", "--verify", branch)
                self.branch_name = branch
                return branch
            except subprocess.CalledProcessError:
                continue
        raise ValueError("Could not detect Git branch (tried 'main' and 'master')")

    
    def _get_current_commit(self):
        """Get the current HEAD commit hash"""
        branch = self._detect_git_branch()
        result = self._run_git_command("rev-parse", branch)
        return result.stdout.strip()

    def _get_changed_files(self, old_commit: str, new_commit: str):
        """
        Get changed files using Git diff with proper status handling.

        Returns:
            (changed_files, deleted_files) where each list contains absolute paths
            under self.markdown_dir to .md files.
        """
        try:
            # Use --name-status to get file status (M=modified, D=deleted, R=renamed, etc.)
            result = self._run_git_command(
                "diff",
                "--name-status",
                "--find-renames",
                f"{old_commit}..{new_commit}",
            )
        except Exception as exc:  # fallback is part of the design here
            print(f"Error getting changed files: {exc}")
            # Fallback to scanning all files
            return self._list_all_markdown_files(), []

        changed_files = []
        deleted_files = []

        stdout = result.stdout.strip()
        if not stdout:
            print("Git diff found 0 changed files, 0 deleted files")
            return changed_files, deleted_files

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status = parts[0]
            file_path = parts[1]

            # Deleted
            if status.startswith("D"):
                if file_path.endswith(".md"):
                    deleted_files.append(os.path.join(self.markdown_dir, file_path))
                continue

            # Renamed (RXXX where XXX is similarity %)
            if status.startswith("R"):
                old_path = parts[1]
                new_path = parts[2] if len(parts) > 2 else file_path

                if old_path.endswith(".md"):
                    deleted_files.append(os.path.join(self.markdown_dir, old_path))
                if new_path.endswith(".md"):
                    changed_files.append(os.path.join(self.markdown_dir, new_path))
                continue

            # Modified (M), Added (A), Copied (C), etc.
            if file_path.endswith(".md"):
                changed_files.append(os.path.join(self.markdown_dir, file_path))

        print(
            f"Git diff found {len(changed_files)} changed files, "
            f"{len(deleted_files)} deleted files"
        )
        return changed_files, deleted_files

    def _list_all_markdown_files(self):
        """List all markdown files for first-run fallback"""
        files = []
        for filepath in glob.glob(f"{self.markdown_dir}/**/*.md", recursive=True):
            if os.path.exists(filepath) and os.path.isfile(filepath):
                files.append(filepath)
        return files

    def _generate_content_based_id(self, content, source_path, chunk_index):
        """Generate stable ID based on content to prevent duplicates"""
        normalized_source = os.path.normpath(source_path)
        identifier = f"{normalized_source}_{chunk_index}_{content}"
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:32]

    def _delete_chunks_for_files(self, file_paths, collection):
        """Delete all chunks in ChromaDB for the given file paths"""
        if not file_paths:
            return
        
        try:
            collection.delete(where={"source": {"$in": file_paths}})
            print(f"Deleted chunks for {len(file_paths)} files from ChromaDB")
        except Exception as e:
            print(f"Error deleting chunks from ChromaDB: {str(e)}")

    def _load_file_safely(self, file_path):
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

    def _process_and_upsert_files(self, file_paths, openai_client, chroma_collection):
        """Process files and upsert to ChromaDB"""
        all_chunks = []
        all_contents = []
        all_metadatas = []
        all_ids = []

        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            content = self._load_file_safely(file_path)
            if content is None:
                continue

            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                continue

            chunks = self.recursive_character_text_splitter(content)
            print(f"Split into {len(chunks)} chunks")


            for idx, chunk in enumerate(chunks):
                chunk_id = self._generate_content_based_id(
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
        all_embeddings = self.generate_embeddings(openai_client, all_contents)

        print(f"Upserting {len(all_chunks)} chunks to ChromaDB...")
        chroma_collection.upsert(
            embeddings=all_embeddings,
            documents=all_contents,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(f"Processed {len(file_paths)} files")


    def get_chroma_client(self, host, port, max_wait_time=120, initial_delay=2, max_delay=10):
        """
        Wait for ChromaDB to be fully ready before returning a client.
        """
        start_time = time.time()
        delay = initial_delay

        print(f"Waiting for ChromaDB at {host}:{port} to become available (max wait: {max_wait_time}s)...", flush=True)

        while time.time() - start_time < max_wait_time:
            try:
                print("Creating Chroma HttpClient...", flush=True)
                client = chromadb.HttpClient(host=host, port=port)

                print("Calling Chroma heartbeat()...", flush=True)
                # Prefer heartbeat over get_user_identity for readiness
                try:
                    hb = client.heartbeat()
                    print(f"Chroma heartbeat OK: {hb}", flush=True)
                except Exception as e:
                    # Log but still treat as failure and retry
                    print(f"Heartbeat call failed: {e}", flush=True)
                    raise

                elapsed = time.time() - start_time
                print(f"Successfully connected to ChromaDB at {host}:{port}! (took {elapsed:.1f}s)", flush=True)
                return client

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ChromaDB not ready yet ({elapsed:.1f}s elapsed) - error: {repr(e)}", flush=True)
                print(f"Retrying in {delay} seconds...", flush=True)

                time.sleep(delay)
                delay = min(delay * 1.5, max_delay)

        # If we timed out
        raise ConnectionError(f"ChromaDB at {host}:{port} not available after {max_wait_time} seconds")



    def process_documents_fallback(self):
        """Fallback to original processing method with content-based IDs"""
        print("=== FALLBACK DOCUMENT PROCESSING STARTED ===")
        try:
            # Debug: Check directory structure
            data_dir = self.data_dir
            markdown_dir = self.markdown_dir
            print(f"Data directory exists: {os.path.exists(data_dir)}")
            print(f"Markdown directory exists: {os.path.exists(markdown_dir)}")
            if os.path.exists(markdown_dir):
                print(f"Markdown dir contents: {os.listdir(markdown_dir)}")
            else:
                print("Creating markdown directory...")
                os.makedirs(markdown_dir, exist_ok=True)
            
            # Validate API key
            nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
            if not nebius_api_key:
                raise ValueError("NEBIUS_API_KEY environment variable is not set")
            print("API key found, initializing clients...")
            
            # Get ChromaDB configuration
            chroma_host = os.getenv("CHROMA_HOST", "chroma")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            
            # Wait for ChromaDB with our dedicated function
            print("Waiting for ChromaDB to be ready...")
            chroma_client = self.get_chroma_client(chroma_host, chroma_port)
            
            # Initialize OpenAI client
            client = OpenAI(
                api_key=nebius_api_key,
                base_url="https://api.studio.nebius.com/v1/"
            )
            
            # Clear existing chunks to prevent duplicates (transition strategy)
            try:
                collection = chroma_client.get_collection("documents")
                print("Clearing existing chunks for clean transition...")
                # Get all current document sources
                existing_docs = collection.get(include=['metadatas'])
                if existing_docs['ids']:
                    sources = list(set([meta.get('source') for meta in existing_docs['metadatas'] if meta.get('source')]))
                    self._delete_chunks_for_files(sources, collection)
                    print(f"Cleared {len(sources)} file sources from database")
            except Exception as e:
                print(f"Warning: Could not clear existing chunks: {e}")
                # Create new collection if needed
                collection = chroma_client.create_collection(
                    "documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
            
            documents = self.load_markdown_files(markdown_dir)
            if not documents:
                return
            
            # Split documents with content-based IDs
            all_chunks = []
            all_chunk_ids = []
            
            for document in documents:
                chunks = self.recursive_character_text_splitter(
                    document['content'], 
                    chunk_size=1000, 
                    overlap=200
                )
                for i, chunk in enumerate(chunks):
                    # Generate content-based IDs
                    chunk_id = self._generate_content_based_id(
                        chunk['content'], 
                        document['metadata']['source'], 
                        i
                    )
                    
                    chunk['metadata'].update({
                        'source': document['metadata']['source'],
                        'original_length': str(len(document['content'])),
                        'chunk_index': str(i)
                    })
                    
                    all_chunks.append(chunk)
                    all_chunk_ids.append(chunk_id)
            
            print(f"Split into {len(all_chunks)} chunks")
            
            if not all_chunks:
                return
            
            contents = [chunk['content'] for chunk in all_chunks]
            
            # Generate embeddings
            all_embeddings = self.generate_embeddings(client, contents)
            
            # Use content-based IDs in upsert
            print("Upserting documents to ChromaDB with content-based IDs...")
            collection.upsert(
                embeddings=all_embeddings,
                documents=contents,
                metadatas=[chunk['metadata'] for chunk in all_chunks],
                ids=all_chunk_ids  # Content-based IDs instead of sequential
            )
            
            print("Document processing complete!")
            try:
                print(f"Collection now contains {collection.count()} documents")
            except Exception as e:
                print(f"Could not get collection count: {e}")
                
        except Exception as e:
            print(f"Error processing documents: {str(e)}")


    def process_git_based_documents(self):
        """Process documents based on Git changes."""
        print("=== GIT-BASED DOCUMENT PROCESSING STARTED ===")
        # Verify Git setup
        if not self._verify_git_installed():
            print("Git is not installed. Falling back to full processing.")
            self.process_documents_fallback()
            return

        if not self._configure_git_safe_directory():
            print("Failed to configure Git safe directory. Falling back to full processing.")
            self.process_documents_fallback()
            return

        if not self._is_git_repo():
            print("This is not a Git repository. Falling back to full processing.")
            self.process_documents_fallback()
            return

        try:
            current_commit = self._get_current_commit()
            last_commit = self._load_last_processed_commit()

            if last_commit is None:
                print("No last processed commit found, processing all documents.")
                changed_files = self._list_all_markdown_files()
                deleted_files = []
                is_first_run = True
            elif last_commit == current_commit:
                print("No new commits to process.")
                return
            else:
                print(f"Processing changes from {last_commit} to {current_commit}")
                changed_files, deleted_files = self._get_changed_files(last_commit, current_commit)
                is_first_run = False

            nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
            if not nebius_api_key:
                raise ValueError("NEBIUS_API_KEY environment variable is not set")
            
            chroma_host = os.getenv("CHROMA_HOST", "chroma")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            chroma_client = self.get_chroma_client(chroma_host, chroma_port)

            client = OpenAI(
                api_key=nebius_api_key,
                base_url="https://api.studio.nebius.com/v1/"
            )

            try:
                collection = chroma_client.get_collection("documents")
                print("Using existing collection")
            except Exception as e:
                print(f"Creating new collection: {e}")
                collection = chroma_client.create_collection(
                    "documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
                print("Created new collection")

            if deleted_files:
                print(f"Deleting chunks for {len(deleted_files)} deleted files...")
                self._delete_chunks_for_files(deleted_files, collection)

            if changed_files:
                print(f"Processing {len(changed_files)} changed/added files...")
                self._delete_chunks_for_files(changed_files, collection)
                self._process_and_upsert_files(changed_files, client, collection)

            if current_commit and (changed_files or deleted_files or is_first_run):
                self._save_last_processed_commit(current_commit)
                print(f"Updated last processed commit to {current_commit}")

            
            print("Git-based document processing complete!")

        except Exception as e:
            print(f"Error in Git-based document processing: {str(e)}")

    def process_single_file_immediately(self, file_path):
        """Process a single markdown file immediately without Git. This handles real-time file changes from watchdog."""
        print(f"=== IMMEDIATE PROCESSING FOR FILE: {file_path} ===")
        try:
            nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
            if not nebius_api_key:
                raise ValueError("NEBIUS_API_KEY environment variable is not set")
            
            chroma_host = os.getenv("CHROMA_HOST", "chroma")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            chroma_client = self.get_chroma_client(chroma_host, chroma_port)

            client = OpenAI(
                api_key=nebius_api_key,
                base_url="https://api.studio.nebius.com/v1/"
            )

            try:
                collection = chroma_client.get_collection("documents")
                print("Using existing collection")
            except Exception as e:
                print(f"Creating new collection: {e}")
                collection = chroma_client.create_collection(
                    "documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
                print("Created new collection")

            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist, skipping processing.")
                return 

            if os.path.getsize(file_path) == 0:
                print(f"File {file_path} is empty, skipping processing.")
                return

            content = self._load_file_safely(file_path)
            if content is None:
                print(f"Could not load content from {file_path}, skipping processing.")
                return

            self._delete_chunks_for_files([file_path], collection)
            chunks = self.recursive_character_text_splitter(content)
            if not chunks:
                print(f"No chunks generated for {file_path}, skipping upsert.")
                return

            all_contents = [chunk['content'] for chunk in chunks]
            all_ids = [self._generate_content_based_id(chunk['content'], file_path, i)
                          for i, chunk in enumerate(chunks)]
            all_metadatas = []

            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': file_path,
                    'chunk_index': str(i),
                    'original_length': str(len(content)),
                    'processed_at': str(int(time.time()))
                }
                all_metadatas.append(metadata)

            embeddings = self.generate_embeddings(client, all_contents)

            collection.upsert(
                embeddings=embeddings,
                documents=all_contents,
                metadatas=all_metadatas,
                ids=all_ids
            )

            print(f"Immediate processing complete for file: {file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")


    def load_markdown_files(self, markdown_dir):
        """Load markdown files from the specified directory"""
        print(f"Searching for markdown files in: {markdown_dir}")
        documents = []
        for filepath in glob.glob(f"{markdown_dir}/**/*.md", recursive=True):
            print(f"Loading file: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'metadata': {'source': filepath}
                    })
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                continue
        
        print(f"Loaded {len(documents)} documents")
        
        if not documents:
            print("No documents to process - checking if this is expected")
            return
        
        return documents
        
    def on_modified(self, event):
        """Handle file modification events with immediate processing"""
        if event.is_directory:
            return

        if not event.src_path.endswith('.md'):
            return

        current_time = time.time()
        if current_time - self.last_processed > 5:
            print(f"Detected modification: {event.src_path}")
            self.process_single_file_immediately(event.src_path)
            self.last_processed = current_time
                
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return

        print(f"Detected new file: {event.src_path}")
        self.process_single_file_immediately(event.src_path)


    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.is_directory or not event.src_path.endswith('.md'):
            return

        print(f"Detected deleted file: {event.src_path}")
        self.process_single_file_immediately(event.src_path)
            
    def recursive_character_text_splitter(self, text, chunk_size=1000, overlap=200):
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
    
    def generate_embeddings(self, client, contents, embedding_dimensions=4096, model="Qwen/Qwen3-Embedding-8B"):
        """Generate embeddings using the OpenAI API, defaulting to Qwen3 Embedding-8B with 4096 dimensions"""
        # Generate embeddings
        
        print("Generating embeddings...")
        
        # Batch process to avoid rate limits
        all_embeddings = []
        batch_size = 10
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            try:
                print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(contents)-1)//batch_size + 1}")
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Verify embedding dimensions
                if batch_embeddings and len(batch_embeddings[0]) != embedding_dimensions:
                    print(f"Warning: Embedding dimension mismatch. Expected {embedding_dimensions}, got {len(batch_embeddings[0])}")
                    embedding_dimensions = len(batch_embeddings[0])
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                # Use correct dimension even on error
                all_embeddings.extend([[0.0] * embedding_dimensions for _ in batch])
        
        print(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings


def main():
    """Main function to start the markdown ingester and watcher."""
    print("Waiting for ChromaDB to be ready...", flush=True)
    time.sleep(3)  # Small delay for service readiness

    
    print("Checking for pending changes since last run..", flush=True)
    handler = MarkdownHandler()
    handler.process_git_based_documents()


    print("Starting document watcher for ongoing changes...", flush=True)
    observer = Observer()
    observer.schedule(handler, handler.markdown_dir, recursive=True)
    observer.start()
    print ("File watcher started - monitoring for changes...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopping document watcher...", flush=True)
    observer.join()

    return 0

if __name__ == "__main__":
    main()