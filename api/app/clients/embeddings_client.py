from openai import OpenAI
from typing import List


class EmbeddingClient:
    """Wrapper for OpenAI/Nebius client that enforces embedding dimension.
    
    Responsibilities:
        - Generate embeddings using the specified model
        - Validate embedding dimensions match expected size
        
    Assumptions:
        - API key and base URL are valid
        - Model specified exists and supports embeddings
        
    Failure modes:
        - Raises ValueError if embedding dimension mismatch occurs
        - Propagates API errors from underlying client
    """
    
    def __init__(self, api_key: str, base_url: str, model: str, embedding_dim: int):
        """Initialize the embedding client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model: Model name to use for embeddings
            embedding_dim: Expected dimension of embeddings
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.embedding_dim = embedding_dim

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.
        
        Args:
            query: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If embedding dimension doesn't match expected size
        """
        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        embedding = response.data[0].embedding
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.embedding_dim}, got {len(embedding)}"
            )
            
        return embedding