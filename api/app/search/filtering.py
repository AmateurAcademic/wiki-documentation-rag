from typing import List, Tuple, Any, Dict, Union


def _filter_nonempty_results(results: List[Tuple[Union[Dict[str, Any], str], float]]) -> List[Tuple[Union[Dict[str, Any], str], float]]:
    """
    Remove empty / whitespace-only documents from ranked results.
    Keeps the original (doc, score) pairs but drops useless chunks.
    
    Args:
        results: List of (document, score) tuples
        
    Returns:
        Filtered list with empty documents removed
    """
    filtered = []
    for doc, score in results:
        # Handle dict vs string vs anything-else defensively
        if isinstance(doc, dict):
            content = str(doc.get("content", "")).strip()
        else:
            content = str(doc or "").strip()

        if content:
            filtered.append((doc, float(score)))
    return filtered