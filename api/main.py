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

WIKI_BASE_URL = os.getenv("WIKI_BASE_URL", "").rstrip("/")
def _build_wiki_url_from_source(source_path: str) -> Optional[str]:
    """
    Convert a local markdown file path into a wiki URL.

    Example:
      source_path = "/app/data/markdown/hardware/magic-mirror.md"
      WIKI_BASE_URL = "https://wiki.home"

      -> "https://wiki.home/hardware/magic-mirror"
    """
    if not WIKI_BASE_URL or not source_path:
        return None

    # Normalize separators just in case
    sp = str(source_path).replace("\\", "/")

    marker = "/markdown/"
    idx = sp.find(marker)
    if idx != -1:
        # keep everything after ".../markdown/"
        rel = sp[idx + len(marker):]           # "hardware/magic-mirror.md"
    else:
        # fallback: just use the basename
        rel = os.path.basename(sp)            # "magic-mirror.md"

    # Strip extension
    rel_no_ext, _ext = os.path.splitext(rel)  # "hardware/magic-mirror"

    # Ensure it starts with a slash for clean joining
    if not rel_no_ext.startswith("/"):
        rel_no_ext = "/" + rel_no_ext         # "/hardware/magic-mirror"

    return f"{WIKI_BASE_URL}{rel_no_ext}"


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
    indexes_scores = [(int(i), float(scores[i])) for i in top_indices if i < len(documents)]
    return sorted(indexes_scores, key=lambda x: x[1], reverse=True)

def _doc_key(doc):
    """
    Build a stable key for a doc object for use in RRF.
    Expects doc to be either:
      - a dict with 'metadata' containing 'source' and 'chunk_index'
      - or a plain string fallback
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


def format_result_for_openwebui(doc, score: float) -> str:
    """Format document result with natural citation for Open WebUI external tool."""
    # Default values
    source = "document"
    content = ""
    url = None

    if isinstance(doc, dict):
        content = str(doc.get("content", ""))
        meta = doc.get("metadata", {}) or {}

        source_path = meta.get("source", "")
        if source_path:
            base = os.path.basename(source_path)   # "magic-mirror.md"
            name, _ext = os.path.splitext(base)    # "magic-mirror"
            source = name or "document"
            url = _build_wiki_url_from_source(source_path)
        else:
            # No source in metadata – just leave source="document"
            pass
    else:
        # Extremely defensive fallback; shouldn't really happen with current code
        content = str(doc)

    header = f"Source: {source} (relevance: {score:.2f})"
    if url:
        header += f"\nURL: {url}"

    return f"{header}\n\n{content}"


async def run_semantic_search(query_embedding: List[float], k: int, collection):
    """Run semantic search with ChromaDB using pre-computed embeddings"""
    if collection is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],  # Pre-computed embedding
            n_results=k * 2,
            include=["documents", "distances", "metadatas"]
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

def _filter_nonempty_results(results):
    """
    Remove empty / whitespace-only documents from ranked results.
    Keeps the original (doc, score) pairs but drops useless chunks.
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

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_docs(input: RetrievalQueryInput):
    """Retrieve documents with full ranking and re-ranking pipeline"""
    try:
        responses = []
        for query in input.queries:
            # hybrid_search now guarantees at most input.k results
            ranked_results = await hybrid_search(query, input.k)

            # Build string results (with preview) safely for dict docs
            results = []
            for doc, score in ranked_results:
                if isinstance(doc, dict):
                    text = str(doc.get("content", ""))
                else:
                    text = str(doc)
                preview = text[:200]
                results.append(f"[Score: {score:.2f}] {preview}...")

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
