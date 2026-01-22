import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Settings:
    """Centralized application configuration with environment variable parsing.
    
    Responsibilities:
        - Parse and validate environment variables
        - Hold application constants
        - Provide a single source of truth for configuration
        
    Assumptions:
        - Environment variables are set before application startup
        - All required environment variables are present
        
    Failure modes:
        - Raises ValueError if required environment variables are missing
    """
    
    # Nebius API configuration
    nebius_api_key: str
    nebius_base_url: str = "https://api.studio.nebius.com/v1/"
    
    # ChromaDB configuration
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    
    # Model configurations
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 4096
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    
    # Search parameters
    bm25_limit: int = 1000
    wiki_base_url: str = ""
    
    # Collection backoff parameters
    collection_backoff: dict = None
    
    def __post_init__(self):
        """Initialize default values for complex types."""
        if self.collection_backoff is None:
            self.collection_backoff = {
                "initial_delay": 0.5,
                "multiplier": 1.5,
                "max_delay": 5.0,
                "max_attempts": 10
            }
        
        # Clean up wiki base URL
        self.wiki_base_url = self.wiki_base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings instance from environment variables.
        
        Returns:
            Settings: Fully configured settings instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
        if not nebius_api_key:
            raise ValueError("NEBIUS_API_KEY environment variable is not set")
            
        return cls(
            nebius_api_key=nebius_api_key,
            nebius_base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
            chroma_host=os.getenv("CHROMA_HOST", "chroma"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "4096")),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            bm25_limit=int(os.getenv("BM25_LIMIT", "1000")),
            wiki_base_url=os.getenv("WIKI_BASE_URL", ""),
        )