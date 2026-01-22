from typing import List, Tuple, Dict, Any


async def run_semantic_search(query_embedding: List[float], k: int, chroma_store) -> List[Tuple[Dict[str, Any], float]]:
    """Run semantic search with ChromaDB using pre-computed embeddings.
    
    Args:
        query_embedding: Pre-computed query embedding vector
        k: Number of results to return
        chroma_store: ChromaStore instance for database access
        
    Returns:
        List of (document, similarity_score) tuples
    """
    try:
        results = chroma_store.semantic_query(
            query_embeddings=[query_embedding],  # Pre-computed embedding
            n_results=k * 2
        )

        if not results['documents'][0]:
            return []

        docs = results['documents'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]

        similarities = [1.0 - d for d in dists] 

        semantic_results = [
            (
                {
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {}
                },
                float(similarities[i])
            )
            for i, doc in enumerate(docs)
        ]

        # Convert to standard format
        return semantic_results
        
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return []