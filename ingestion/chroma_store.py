# ingestion/chroma_store.py
import chromadb
import time
from typing import List, Dict, Any

class ChromaStore:
    """Manages interactions with ChromaDB."""
    
    def __init__(
        self,
        host: str = "chroma",
        port: int = 8000,
        collection_name: str = "documents",
        max_wait_time: int = 120
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.max_wait_time = max_wait_time
        self.client = None
        self.collection = None
    
    def connect(self) -> None:
        """Wait for ChromaDB to be fully ready before connecting."""
        start_time = time.time()
        delay = 2
        
        print(f"Waiting for ChromaDB at {self.host}:{self.port} to become available (max wait: {self.max_wait_time}s)...", flush=True)
        
        while time.time() - start_time < self.max_wait_time:
            try:
                # Create client
                self.client = chromadb.HttpClient(host=self.host, port=self.port)
                
                # Verify connectivity with heartbeat
                try:
                    hb = self.client.heartbeat()
                    print(f"Chroma heartbeat OK: {hb}", flush=True)
                except Exception as e:
                    print(f"Heartbeat call failed: {e}", flush=True)
                    raise
                
                elapsed = time.time() - start_time
                print(f"Successfully connected to ChromaDB at {self.host}:{self.port}! (took {elapsed:.1f}s)", flush=True)
                self._get_or_create_collection()
                return
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ChromaDB not ready yet ({elapsed:.1f}s elapsed) - error: {repr(e)}", flush=True)
                print(f"Retrying in {delay} seconds...", flush=True)
                
                time.sleep(delay)
                delay = min(delay * 1.5, 10)
        
        raise ConnectionError(f"ChromaDB at {self.host}:{self.port} not available after {self.max_wait_time} seconds")
    
    def _get_or_create_collection(self) -> None:
        """Get or create the document collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print("Using existing collection")
        except Exception as e:
            print(f"Creating new collection: {e}")
            self.collection = self.client.create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            print("Created new collection")
    
    def delete_chunks_for_files(self, file_paths: List[str]) -> None:
        """Delete all chunks in ChromaDB for the given file paths"""
        if not file_paths:
            return
        
        try:
            self.collection.delete(where={"source": {"$in": file_paths}})
            print(f"Deleted chunks for {len(file_paths)} files from ChromaDB")
        except Exception as e:
            print(f"Error deleting chunks from ChromaDB: {str(e)}")
    
    def upsert_chunks(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, str]],
        ids: List[str]
    ) -> None:
        """Upsert chunks to ChromaDB"""
        print(f"Upserting {len(documents)} chunks to ChromaDB...")
        self.collection.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Could not get collection count: {e}")
            return 0
