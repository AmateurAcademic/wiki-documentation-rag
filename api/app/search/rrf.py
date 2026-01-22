from typing import List, Tuple, Any, Dict, Union


def _doc_key(doc: Union[Dict[str, Any], str]) -> str:
    """
    Build a stable key for a doc object for use in RRF.
    
    Expects doc to be either:
      - a dict with 'metadata' containing 'source' and 'chunk_index'
      - or a plain string fallback
      
    Args:
        doc: Document object to create key for
        
    Returns:
        String key uniquely identifying the document
    """
    if isinstance(doc, dict):
        meta = doc.get("metadata", {}) or {}
        src = meta.get("source", "")
        idx = meta.get("chunk_index", "")
        if src or idx:
            return f"{src}::{idx}"
        # fallback: use content hash
        content = str(doc.get("content", ""))
        return f"content::{hash(content)}"
    else:
        # plain string fallback
        return f"str::{hash(str(doc))}"


def reciprocal_rank_fusion(results_list: List[List[Tuple[Any, float]]], 
                          weights: List[float] = None, 
                          k: int = 60) -> List[Tuple[Any, float]]:
    """Combine multiple result lists using Reciprocal Rank Fusion.
    
    Args:
        results_list: List of result lists, each containing (document, score) tuples
        weights: Weights for each result list (default: equal weights)
        k: Ranking constant for RRF calculation
        
    Returns:
        List of (document, fused_score) tuples sorted by score descending
    """
    if not results_list or not any(results_list):
        return []

    if weights is None:
        weights = [1.0] * len(results_list)

    # Collect all unique documents
    all_docs = {}
    for i, results in enumerate(results_list):
        for rank, (doc, score) in enumerate(results):
            key = _doc_key(doc)
            if key not in all_docs:
                all_docs[key] = {
                    'doc': doc,
                    'score': 0.0
                }
            all_docs[key]['score'] += weights[i] * (1.0 / (rank + k))

    # Sort by score
    sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]['score'], reverse=True)
    return [(item[1]['doc'], item[1]['score']) for item in sorted_docs]