import asyncio
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from app.clients.chroma_store import ChromaStore
from app.clients.embeddings_client import EmbeddingClient
from app.clients.reranker_client import RerankerClient
from app.config import Settings
from app.search.bm25 import get_bm25_results
from app.search.rrf import reciprocal_rank_fusion
from app.search.filtering import _filter_nonempty_results
from app.search.semantic import run_semantic_search


class HybridSearchService:
    """Orchestrator for the complete hybrid search pipeline.
    
    Responsibilities:
        - Coordinate semantic and BM25 search execution
        - Perform result fusion and re-ranking
        - Apply filtering and formatting
        
    Assumptions:
        - All dependencies are properly initialized
        - Collection is available and ready for queries
        
    Failure modes:
        - Propagates errors from underlying components
        - Returns empty list if critical components fail
    """
    
    def __init__(self, settings: Settings, chroma: ChromaStore, embeddings: EmbeddingClient, reranker: RerankerClient):
        """Initialize the hybrid search service.
        
        Args:
            settings: Application configuration
            chroma: ChromaDB store instance
            embeddings: Embedding client instance
            reranker: Re-ranker client instance
        """
        self.settings = settings
        self.chroma = chroma
        self.embeddings = embeddings
        self.reranker = reranker

    async def _run_bm25_search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Run BM25 keyword search.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Get documents for BM25 (exclude embeddings to avoid dimension issues)
            results = self.chroma.get_for_bm25()

            docs = results['documents'] or []
            metas = results['metadatas'] or []
            
            if not docs:
                return []

            bm25_results = get_bm25_results(query, docs, k=k * 2)

            docs_scores = [
                (
                    {
                        "content": docs[i],
                        "metadata": metas[i] if i < len(metas) else {}
                    },
                    score,
                )
                for i, score in bm25_results
                if 0 <= i < len(docs)
            ]
            return docs_scores

        except Exception as e:
            print(f"BM25 search failed: {e}")
            return []

    async def _parallel_search(self, query_embedding: List[float], query_text: str, k: int = 10):
        """Run semantic and BM25 search in parallel.
        
        Args:
            query_embedding: Pre-computed query embedding
            query_text: Original query text
            k: Number of results for each search method
            
        Returns:
            Tuple of (semantic_results, bm25_results)
        """
        # Execute both searches concurrently
        semantic_results, bm25_results = await asyncio.gather(
            run_semantic_search(query_embedding, k, self.chroma),
            self._run_bm25_search(query_text, k),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(semantic_results, Exception):
            print(f"Semantic search failed: {semantic_results}")
            semantic_results = []

        if isinstance(bm25_results, Exception):
            print(f"BM25 search failed: {bm25_results}")
            bm25_results = []

        return semantic_results, bm25_results

    async def search(self, query: str, k: int = 10, rerank_k: Optional[int] = None) -> List[Tuple[Union[Dict[str, Any], str], float]]:
        """
        Full hybrid search implementation.
        
        Args:
            query: Search query text
            k: Maximum number of results to return
            rerank_k: Optional cap on how many of the fused results we re-rank
            
        Returns:
            List of (document, score) tuples, limited to k results
            
        Raises:
            ValueError: If query is empty or k is not positive
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be a positive integer")

        # Determine internal rerank_k (how many to re-rank, not how many to return)
        if rerank_k is None:
            rerank_k = min(k, 5)
        else:
            # Clamp between 1 and k
            rerank_k = max(1, min(rerank_k, k))

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Run semantic and BM25 search in parallel
        semantic_results, bm25_results = await self._parallel_search(query_embedding, query, k)

        # Reciprocal Rank Fusion
        combined_results = reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            weights=[0.5, 0.5]
        )

        # If we have no results at all, just bail out
        if not combined_results:
            return []

        # Re-ranking (limit candidates for performance)
        max_candidates = min(len(combined_results), rerank_k * 2, 50)
        candidate_docs = [doc for doc, _ in combined_results[:max_candidates]]

        if not candidate_docs:
            # Return top-k fused results without re-ranking, after filtering empties
            top_results = combined_results[:min(k, len(combined_results))]
            top_results = _filter_nonempty_results(top_results)
            return top_results[:min(k, len(top_results))]

        # Re-ranking with error handling
        try:
            rerank_pairs = []
            for doc in candidate_docs:
                if isinstance(doc, dict):
                    content = str(doc.get("content", ""))
                else:
                    content = str(doc)
                rerank_pairs.append((query, content))

            rerank_scores = self.reranker.predict_pairs(query, [pair[1] for pair in rerank_pairs])

            # Final scoring
            final_results = []
            for i, (doc, rrf_score) in enumerate(combined_results[:max_candidates]):
                if i < len(rerank_scores):
                    # Normalize re-rank score
                    norm_rerank = 1 / (1 + np.exp(-rerank_scores[i]))
                    # Weighted combination
                    final_score = 0.7 * rrf_score + 0.3 * norm_rerank
                    final_results.append((doc, final_score))
                else:
                    final_results.append((doc, rrf_score))

            # Sort by final score
            final_results.sort(key=lambda x: x[1], reverse=True)

            # Filter out empty/whitespace-only chunks
            final_results = _filter_nonempty_results(final_results)

            # IMPORTANT: clamp output to k here
            return final_results[:min(k, len(final_results))]

        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fall back to fused results without re-ranking, still respecting k
            top_results = combined_results[:min(k, len(combined_results))]
            top_results = _filter_nonempty_results(top_results)
            return top_results[:min(k, len(top_results))]