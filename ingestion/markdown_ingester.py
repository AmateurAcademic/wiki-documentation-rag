# ingestion/markdown_ingester.py
import os
import time
from watchdog.observers import Observer
from .git_repository import GitRepository
from .markdown_processor import MarkdownProcessor
from .embedding_service import EmbeddingService
from .chroma_store import ChromaStore
from .ingestion_service import MarkdownIngestionService
from .watcher import MarkdownWatchHandler

def main():
    """Main function to start the markdown ingester and watcher."""
    print("Starting document processor...", flush=True)
    
    # Configuration
    data_dir = "/app/data"
    markdown_dir = os.path.join(data_dir, "markdown")
    state_dir = "/app/state"
    os.makedirs(state_dir, exist_ok=True)
    
    # Environment variables
    nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
    if not nebius_api_key:
        raise ValueError("NEBIUS_API_KEY environment variable is not set")
    
    chroma_host = os.getenv("CHROMA_HOST", "chroma")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    
    # Core components
    git_repo = GitRepository(
        repo_dir=markdown_dir,
        state_file=os.path.join(state_dir, ".git_processing_state.json")
    )
    
    md_processor = MarkdownProcessor(
        base_dir=markdown_dir
    )
    
    embed_service = EmbeddingService(
        api_key=nebius_api_key
    )
    
    chroma_store = ChromaStore(
        host=chroma_host,
        port=chroma_port
    )
    
    ingestion_service = MarkdownIngestionService(
        git_repo=git_repo,
        md_processor=md_processor,
        embed_service=embed_service,
        chroma_store=chroma_store
    )
    
    print("Waiting for ChromaDB to be ready...", flush=True)
    time.sleep(3)  # Small delay for service readiness
    
    print("Checking for pending changes since last run...", flush=True)
    ingestion_service.process_git_delta()
    
    print("Starting document watcher for ongoing changes...", flush=True)
    observer = Observer()
    handler = MarkdownWatchHandler(ingestion_service=ingestion_service)
    observer.schedule(handler, markdown_dir, recursive=True)
    observer.start()
    print("File watcher started - monitoring for changes...")
    
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
