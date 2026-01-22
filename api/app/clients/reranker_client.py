from sentence_transformers import CrossEncoder
from typing import List, Tuple


class RerankerClient:
    """Heavy model loader and predictor for re-ranking.
    
    Responsibilities:
        - Load and manage the re-ranker model
        - Predict relevance scores for query-document pairs
        
    Assumptions:
        - Model specified exists and is compatible
        - Input pairs are properly formatted
        
    Failure modes:
        - Propagates model loading errors
        - Propagates prediction errors from underlying model
    """
    
    def __init__(self, model_name: str):
        """Initialize the re-ranker client.
        
        Args:
            model_name: Name of the model to load
        """
        self.model = CrossEncoder(model_name)

    def predict_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Predict relevance scores for query-document pairs.
        
        Args:
            query: Query text
            texts: List of document texts to score
            
        Returns:
            List of relevance scores corresponding to each text
        """
        pairs = [(query, text) for text in texts]
        return self.model.predict(pairs).tolist()