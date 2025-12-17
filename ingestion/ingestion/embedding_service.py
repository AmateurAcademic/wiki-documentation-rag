# ingestion/embedding_service.py
from openai import OpenAI
from typing import List

class EmbeddingService:
    """Handles embedding generation using OpenAI API."""
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.studio.nebius.com/v1/",
        model: str = "Qwen/Qwen3-Embedding-8B",
        embedding_dimensions: int = 4096,
        batch_size: int = 10
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model = model
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
    
    def generate_embeddings(self, contents: List[str]) -> List[List[float]]:
        """Generate embeddings for the given contents"""
        print("Generating embeddings...")
        all_embeddings = []
        
        for i in range(0, len(contents), self.batch_size):
            batch = contents[i:i+self.batch_size]
            try:
                print(f"Generating embeddings for batch {i//self.batch_size + 1}/{(len(contents)-1)//self.batch_size + 1}")
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Verify embedding dimensions
                if batch_embeddings and len(batch_embeddings[0]) != self.embedding_dimensions:
                    print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dimensions}, got {len(batch_embeddings[0])}")
                    self.embedding_dimensions = len(batch_embeddings[0])
            except Exception as e:
                print(f"Error generating embeddings for batch {i//self.batch_size}: {str(e)}")
                # Use correct dimension even on error
                all_embeddings.extend([[0.0] * self.embedding_dimensions for _ in batch])
        
        print(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
