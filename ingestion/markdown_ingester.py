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
        self.state_file = "/app/data/.git_processing_state.json"
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


    def get_chroma_client(self, host, port, max_wait_time=120, initial_delay=2, max_delay=10):
        """
        Wait for ChromaDB to be fully ready before returning a client.
            
        Args:
            host: ChromaDB host
            port: ChromaDB port
            max_wait_time: Maximum time to wait in seconds
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            
        Returns:
            A working ChromaDB client
            """
        start_time = time.time()
        delay = initial_delay
        
        print(f"Waiting for ChromaDB at {host}:{port} to become available (max wait: {max_wait_time}s)...", flush=True)
        
        while time.time() - start_time < max_wait_time:
            try:
                # Create client
                client = chromadb.HttpClient(host=host, port=port)
                
                # CRITICAL: Actually verify connectivity with a lightweight API call
                # This triggers the real connection attempt
                client.get_user_identity()
                elapsed = time.time() - start_time
                print(f"Successfully connected to ChromaDB at {host}:{port}! (took {elapsed:.1f}s)")
                return client
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ChromaDB not ready yet ({elapsed:.1f}s elapsed) - error: {str(e)}")
                print(f"Retrying in {delay} seconds...")
                
                time.sleep(delay)
                
                # Exponential backoff with ceiling
                delay = min(delay * 1.5, max_delay)
        
        # If we timed out
        raise ConnectionError(f"ChromaDB at {host}:{port} not available after {max_wait_time} seconds")

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
        if event.is_directory:
            return
        if event.src_path.endswith('.md'):
            current_time = time.time()
            if current_time - self.last_processed > 5:
                print(f"Detected modification: {event.src_path}")
                self.process_documents()
                self.last_processed = current_time
                
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.md'):
            print(f"Detected new file: {event.src_path}")
            self.process_documents()
            
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
    
    def process_documents(self):
        """Process markdown documents and store them in ChromaDB."""
        print("=== DOCUMENT PROCESSING STARTED ===")
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
                 
            documents = self.load_markdown_files(markdown_dir)
            if not documents:
                return
            
            # Split documents
            all_chunks = []
            for document in documents:
                chunks = self.recursive_character_text_splitter(
                    document['content'], 
                    chunk_size=1000, 
                    overlap=200
                )
                for chunk in chunks:
                    # Convert all metadata values to strings
                    chunk['metadata'].update({
                        'source': document['metadata']['source'],
                        'original_length': str(len(document['content']))
                    })
                all_chunks.extend(chunks)
            
            print(f"Split into {len(all_chunks)} chunks")

            contents = [chunk['content'] for chunk in all_chunks]
            
            # Generate embeddings
            all_embeddings = self.generate_embeddings(client, contents)
            
            # Store in ChromaDB
            try:
                collection = chroma_client.get_collection("documents")
                print("Using existing collection")
            except Exception as e:
                print(f"Creating new collection: {e}")
                # Create new collection with correct dimension
                collection = chroma_client.create_collection(
                    "documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None  # Let server handle embeddings
                )
                print("Created new collection")
            
            print("Upserting documents to ChromaDB...")
            collection.upsert(
                embeddings=all_embeddings,
                documents=contents,
                metadatas=[chunk['metadata'] for chunk in all_chunks],
                ids=[f"doc_{i}" for i in range(len(all_chunks))]
            )
            
            print("Document processing complete!")
            print(f"Collection now contains {collection.count()} documents")
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("Waiting for ChromaDB to be ready...", flush=True)
    time.sleep(3)  # Small delay for service readiness
    print("Starting document watcher...", flush=True)
    print(f"Current working directory: {os.getcwd()}", flush=True)
    handler = MarkdownHandler()
    markdown_dir = MarkdownHandler().markdown_dir
    print(f"Data directory contents: {os.listdir(markdown_dir) if os.path.exists(markdown_dir) else 'NOT FOUND'}", flush=True)
    
    print("Running initial document processing...", flush=True)
    handler.process_documents()  # Initial processing
    
    print("Setting up file watcher...", flush=True)
    observer = Observer()
    observer.schedule(handler, markdown_dir, recursive=True)
    observer.start()
    print("File watcher started - monitoring for changes...", flush=True)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopping document watcher...", flush=True)
    observer.join()

if __name__ == "__main__":
    main()