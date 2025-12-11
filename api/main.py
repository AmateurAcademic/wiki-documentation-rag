import os
import time
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

app = FastAPI(
    title="Minimal Document Retriever API",
    version="1.0.0",
    description="Hybrid search with full ranking and re-ranking pipeline. Results formatted for Open WebUI."
)

# Pydantic models
class RetrievalQueryInput(BaseModel):
    queries: List[str]
    k: int = 10

class RetrievedDoc(BaseModel):
    query: str
    results: List[str]

class RetrievalResponse(BaseModel):
    responses: List[RetrievedDoc]

class ToolRequest(BaseModel):
    body: Dict[str, Any]

# Application state
class AppState:
    client: OpenAI = None
    chroma_client: chromadb.Client = None
    collection = None
    reranker: CrossEncoder = None
    embedding_dim: int = 4096  # Qwen3 Embedding-8B dimension

app_state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        # Validate API key
        nebius_api_key = os.getenv("NEBIUS_API_KEY", "").strip('"').strip("'")
        if not nebius_api_key:
            raise ValueError("NEBIUS_API_KEY environment variable is not set")

        # Get ChromaDB configuration
        chroma_host = os.getenv("CHROMA_HOST", "chroma")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        # Initialize OpenAI client
        app_state.client = OpenAI(
            api_key=nebius_api_key,
            base_url="https://api.studio.nebius.com/v1/"
        )

        # Initialize ChromaDB HTTP client
        app_state.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        print("Waiting for ingester to complete...")
        time.sleep(10)

        # Try to get collection with explicit embedding function setting
        try:
            app_state.collection = app_state.chroma_client.get_collection(
                "documents",
                embedding_function=None  # Critical: Match ingestion setting
            )
        except Exception as e:
            print(f"Collection not ready: {e}")
            app_state.collection = None

        # Initialize re-ranker
        app_state.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

        print("Components initialized successfully")
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        raise


def get_bm25_results(query, documents, k=10):
    """Get BM25 results for hybrid search"""
    if not documents:
        return []

    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    # Get top k results
    top_indices = np.argpartition(scores, -k)[-k:] if len(scores) > k else range(len(scores))
    results = [(documents[i], scores[i]) for i in top_indices if i < len(documents)]
    return sorted(results, key=lambda x: x[1], reverse=True)


def reciprocal_rank_fusion(results_list, weights=None, k=60):
    """Combine multiple result lists using Reciprocal Rank Fusion"""
    if not results_list or not any(results_list):
        return []

    if weights is None:
        weights = [1.0] * len(results_list)

    # Collect all unique documents
    all_docs = {}
    for i, results in enumerate(results_list):
        for rank, (doc, score) in enumerate(results):
            doc_id = hash(doc)
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'doc': doc,
                    'score': 0.0
                }
            all_docs[doc_id]['score'] += weights[i] * (1.0 / (rank + k))

    # Sort by score
    sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]['score'], reverse=True)
    return [(item[1]['doc'], item[1]['score']) for item in sorted_docs]


def format_result_for_openwebui(doc, score: float) -> str:
    """Format document result with natural citation for Open WebUI external tool"""
    source = "document"
    content = str(doc)

    # Handle different document formats
    if isinstance(doc, dict):
        content = doc.get('content', str(doc))
        if 'metadata' in doc and 'source' in doc['metadata']:
            source = os.path.basename(doc['metadata']['source'])
    elif hasattr(doc, 'metadata') and hasattr(doc.metadata, 'get'):
        source_path = doc.metadata.get('source', '')
        if source_path:
            source = os.path.basename(source_path)
    elif isinstance(doc, str) and 'Source: ' in doc:
        # Extract source from formatted string
        source_line = doc.split('\n')[0]
        if source_line.startswith('Source: '):
            source = source_line.replace('Source: ', '').split()[0]

    return f"Source: {source} (relevance: {score:.2f})\n\n{content}"


async def run_semantic_search(query_embedding: List[float], k: int, collection):
    """Run semantic search with ChromaDB using pre-computed embeddings"""
    if collection is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],  # Pre-computed embedding
            n_results=k * 2,
            include=["documents", "distances"]
        )

        if not results['documents'][0]:
            return []

        # Convert to standard format
        return list(zip(
            results['documents'][0],
            [1.0 - d for d in results['distances'][0]]  # Convert distance to similarity
        ))
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return []


async def run_bm25_search(query: str, k: int, collection):
    """Run BM25 keyword search"""
    if collection is None:
        return []

    try:
        # Get documents for BM25 (exclude embeddings to avoid dimension issues)
        results = collection.get(
            include=["documents", "metadatas"],
            limit=1000
        )

        if not results['documents']:
            return []

        # Apply BM25 to document contents
        return get_bm25_results(query, results['documents'], k=k * 2)
    except Exception as e:
        print(f"BM25 search failed: {e}")
        return []


async def parallel_search(query_embedding: List[float], query_text: str, k: int = 10):
    """Run semantic and BM25 search in parallel"""
    # Execute both searches concurrently
    semantic_results, bm25_results = await asyncio.gather(
        run_semantic_search(query_embedding, k, app_state.collection),
        run_bm25_search(query_text, k, app_state.collection),
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


async def hybrid_search(query: str, k: int = 10, rerank_k: Optional[int] = None):
    """
    Full hybrid search implementation.

    k = maximum number of results to return.
    rerank_k = optional cap on how many of the fused results we re-rank
               (does NOT change the max number of results returned).
    """
    try:
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

        # Check if collection exists
        if app_state.collection is None:
            try:
                app_state.collection = app_state.chroma_client.get_collection(
                    "documents",
                    embedding_function=None
                )
            except Exception:
                return []  # Collection not ready yet

        # Generate query embedding
        response = app_state.client.embeddings.create(
            input=[query],
            model="Qwen/Qwen3-Embedding-8B"
        )
        query_embedding = response.data[0].embedding

        # Validate embedding dimension
        if len(query_embedding) != app_state.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {app_state.embedding_dim}, got {len(query_embedding)}"
            )

        # Run semantic and BM25 search in parallel
        semantic_results, bm25_results = await parallel_search(query_embedding, query, k)

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
            # Return top-k fused results without re-ranking
            top_results = combined_results[:min(k, len(combined_results))]
            return [(doc, float(score)) for doc, score in top_results]

        # Re-ranking with error handling
        try:
            rerank_pairs = [(query, doc) for doc in candidate_docs]
            rerank_scores = app_state.reranker.predict(rerank_pairs)

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

            # IMPORTANT: clamp output to k here
            return final_results[:min(k, len(final_results))]

        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fall back to fused results without re-ranking, still respecting k
            top_results = combined_results[:min(k, len(combined_results))]
            return [(doc, float(score)) for doc, score in top_results]

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_docs(input: RetrievalQueryInput):
    """Retrieve documents with full ranking and re-ranking pipeline"""
    try:
        responses = []
        for query in input.queries:
            # hybrid_search now guarantees at most input.k results
            ranked_results = await hybrid_search(query, input.k)

            # Format results for regular API usage
            results = [
                f"[Score: {score:.2f}] {doc[:200]}..."  # Truncated preview
                for doc, score in ranked_results
            ]
            responses.append(RetrievedDoc(query=query, results=results))
        return RetrievalResponse(responses=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: ToolRequest):
    """Search with full ranking pipeline for Open WebUI external tool"""
    try:
        query = request.body.get("query", "")
        k = request.body.get("k", 10)
        rerank_k = request.body.get("rerank_k", 5)

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        # hybrid_search will clamp rerank_k to [1, k] and always return <= k docs
        ranked_results = await hybrid_search(query, k, rerank_k)

        # Format results with natural citation for Open WebUI external tool
        results = [
            format_result_for_openwebui(doc, score)
            for doc, score in ranked_results
        ]

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/specification")
async def get_specification():
    return {
        "name": "Minimal Document Retriever",
        "description": "Semantic search with hybrid ranking and re-ranking. Results include natural citation formatting for Open WebUI.",
        "endpoints": [{
            "name": "search_documents",
            "method": "POST",
            "path": "/search",
            "description": "Search documents with full hybrid ranking pipeline",
            "parameters": [
                {"name": "query", "type": "string", "required": True},
                {"name": "k", "type": "integer", "required": False, "default": 10},
                {"name": "rerank_k", "type": "integer", "required": False, "default": 5}
            ]
        }]
    }


@app.get("/openapi.json")
async def get_openapi_spec():
    return app.openapi()


@app.get("/")
async def root():
    return {"message": "Minimal Document Retriever API for Open WebUI"}


@app.get("/health")
async def health_check():
    collection_status = "not_ready"
    if app_state.collection is not None:
        try:
            # Check if collection has documents
            count = app_state.collection.count()
            collection_status = f"ready ({count} documents)"
        except Exception:
            collection_status = "ready (count unavailable)"

    return {
        "status": "healthy",
        "components": {
            "openai": app_state.client is not None,
            "chromadb": collection_status,
            "reranker": app_state.reranker is not None,
            "embedding_dimension": app_state.embedding_dim
        }
    }


# Test endpoint for debugging
@app.get("/test_embedding")
async def test_embedding():
    if not app_state.client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    try:
        response = app_state.client.embeddings.create(
            input=["test query"],
            model="Qwen/Qwen3-Embedding-8B"
        )
        embedding = response.data[0].embedding
        return {
            "dimension": len(embedding),
            "sample_values": embedding[:5]  # First 5 values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")
