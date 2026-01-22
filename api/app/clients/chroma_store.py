import chromadb
import time
from typing import Optional, List, Dict, Any


class ChromaStore:
    """Connection manager for ChromaDB with collection acquisition and backoff.
    
    Responsibilities:
        - Manage ChromaDB client connection
        - Handle collection acquisition with exponential backoff
        - Provide semantic query and BM25 retrieval methods
        
    Assumptions:
        - ChromaDB service is accessible at specified host/port
        - Collection exists and is properly configured
        
    Failure modes:
        - Raises ConnectionError if unable to connect to ChromaDB
        - Raises ValueError if collection is not available after backoff
    """
    
    def __init__(self, host: str, port: int, collection_name: str, bm25_limit: int, backoff: dict):
        """Initialize ChromaDB store.
        
        Args:
            host: ChromaDB host
            port: ChromaDB port
            collection_name: Name of the collection to use
            bm25_limit: Limit for BM25 document retrieval
            backoff: Backoff configuration with keys: initial_delay, multiplier, max_delay, max_attempts
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.bm25_limit = bm25_limit
        self.backoff = backoff
        self.client: Optional[chromadb.Client] = None
        self.collection = None

    def connect(self) -> None:
        """Establish connection to ChromaDB.
        
        Raises:
            ConnectionError: If unable to connect to ChromaDB
        """
        try:
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")

    def get_collection_with_backoff(self) -> None:
        """Acquire collection with exponential backoff.
        
        Raises:
            ValueError: If collection is not available after backoff
        """
        if self.client is None:
            raise ValueError("Client not connected. Call connect() first.")
            
        delay = self.backoff["initial_delay"]
        attempts = 0
        
        while attempts < self.backoff["max_attempts"]:
            try:
                self.collection = self.client.get_collection(
                    self.collection_name,
                    embedding_function=None  # Critical: Match ingestion setting
                )
                return
            except Exception as e:
                attempts += 1
                if attempts >= self.backoff["max_attempts"]:
                    raise ValueError(f"Collection not ready after {attempts} attempts: {e}")
                
                print(f"Collection not ready, retrying in {delay:.2f}s...")
                time.sleep(delay)
                delay = min(delay * self.backoff["multiplier"], self.backoff["max_delay"])

    def semantic_query(self, query_embeddings: List[List[float]], n_results: int) -> Dict[str, Any]:
        """Perform semantic search using pre-computed embeddings.
        
        Args:
            query_embeddings: Pre-computed embedding vectors
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results with documents, distances, and metadata
            
        Raises:
            ValueError: If collection is not available
        """
        if self.collection is None:
            raise ValueError("Collection not available")
            
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )

    def get_for_bm25(self) -> Dict[str, Any]:
        """Retrieve documents for BM25 processing.
        
        Returns:
            Dictionary containing documents and metadata
            
        Raises:
            ValueError: If collection is not available
        """
        if self.collection is None:
            raise ValueError("Collection not available")
            
        return self.collection.get(
            include=["documents", "metadatas"],
            limit=self.bm25_limit
        )