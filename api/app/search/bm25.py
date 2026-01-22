import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi


def get_bm25_results(query: str, documents: List[str], k: int = 10) -> List[Tuple[int, float]]:
    """Get BM25 results for hybrid search.
    
    Args:
        query: Search query text
        documents: List of documents to search through
        k: Number of top results to return
        
    Returns:
        List of tuples containing (index, score) for top k documents
    """
    if not documents:
        return []

    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    # Get top k results
    top_indices = np.argpartition(scores, -k)[-k:] if len(scores) > k else range(len(scores))
    indexes_scores = [(int(i), float(scores[i])) for i in top_indices if i < len(documents)]
    return sorted(indexes_scores, key=lambda x: x[1], reverse=True)