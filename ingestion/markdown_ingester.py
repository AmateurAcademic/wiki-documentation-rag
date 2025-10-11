import os
import time
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
import chromadb

class MarkdownHandler(FileSystemEventHandler):
    """Handles file system events for markdown files, processes them, and stores them in ChromaDB."""
    def __init__(self):
        self.last_processed = 0
        
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
    
    def process_documents(self):
        print("=== DOCUMENT PROCESSING STARTED ===")
        try:
            # Debug: Check directory structure
            data_dir = "/app/data"
            markdown_dir = "/app/data/markdown"
            
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
            
            # Initialize OpenAI client
            client = OpenAI(
                api_key=nebius_api_key,
                base_url="https://api.studio.nebius.com/v1/"
            )
            
            # Initialize ChromaDB HTTP client
            chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            print(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
            
            # Load markdown files
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
            
            # Split documents
            all_chunks = []
            for doc in documents:
                chunks = self.recursive_character_text_splitter(
                    doc['content'], 
                    chunk_size=1000, 
                    overlap=200
                )
                for chunk in chunks:
                    # Convert all metadata values to strings
                    chunk['metadata'].update({
                        'source': doc['metadata']['source'],
                        'original_length': str(len(doc['content']))
                    })
                all_chunks.extend(chunks)
            
            print(f"Split into {len(all_chunks)} chunks")
            
            # Generate embeddings
            contents = [chunk['content'] for chunk in all_chunks]
            print("Generating embeddings...")
            
            # Batch process to avoid rate limits
            all_embeddings = []
            batch_size = 10
            embedding_dim = 4096  # Qwen3 Embedding-8B dimension
            
            for i in range(0, len(contents), batch_size):
                batch = contents[i:i+batch_size]
                try:
                    print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(contents)-1)//batch_size + 1}")
                    response = client.embeddings.create(
                        input=batch,
                        model="Qwen/Qwen3-Embedding-8B"
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Verify embedding dimensions
                    if batch_embeddings and len(batch_embeddings[0]) != embedding_dim:
                        print(f"Warning: Embedding dimension mismatch. Expected {embedding_dim}, got {len(batch_embeddings[0])}")
                        embedding_dim = len(batch_embeddings[0])
                except Exception as e:
                    print(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                    # Use correct dimension even on error
                    all_embeddings.extend([[0.0] * embedding_dim for _ in batch])
            
            print(f"Generated {len(all_embeddings)} embeddings")
            
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
    print(f"Data directory contents: {os.listdir('/app') if os.path.exists('/app') else 'NOT FOUND'}", flush=True)
    
    handler = MarkdownHandler()
    print("Running initial document processing...", flush=True)
    handler.process_documents()  # Initial processing
    
    print("Setting up file watcher...", flush=True)
    observer = Observer()
    observer.schedule(handler, "/app/data/markdown", recursive=True)
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