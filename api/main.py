import os
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import time

app = FastAPI(
    title="Optimized Document Retriever API",
    version="1.0.0",
    description="Hybrid search with parallel processing and caching",
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
    embeddings: OpenAIEmbeddings = None
    vectorstore: Chroma = None
    reranker: CrossEncoder = None
    _cached_documents = None
    _last_cache_update = 0

app_state = AppState()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        # Validate API key
        nebius_api_key = os.getenv("NEBIUS_API_KEY")
        if not nebius_api_key:
            raise ValueError("NEBIUS_API_KEY environment variable is not set")
        
        app_state.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            openai_api_key=nebius_api_key,
            openai_api_base="https://api.studio.nebius.com/v1/",
            tiktoken_enabled=False
        )
        
        app_state.vectorstore = Chroma(
            persist_directory="/app/chroma_data",
            embedding_function=app_state.embeddings
        )
        
        app_state.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        
        print("Components initialized successfully")
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        raise

def get_cached_documents():
    """Cache all documents with metadata"""
    current_time = time.time()
    if app_state._cached_documents is None or (current_time - app_state._last_cache_update) > 30:
        try:
            app_state._cached_documents = app_state.vectorstore.get()
            app_state._last_cache_update = current_time
        except Exception as e:
            print(f"Error refreshing document cache: {e}")
            if app_state._cached_documents is None:
                raise
    
    return app_state._cached_documents

def reciprocal_rank_fusion(results_list: List[List], weights: List[float] = None, k: int = 60) -> List:
    """Combine multiple result lists using Reciprocal Rank Fusion"""
    if weights is None:
        weights = [1.0] * len(results_list)
    
    # Collect all unique documents
    all_docs = {}
    for results in results_list:
        for doc in results:
            doc_id = hash(doc.page_content) if hasattr(doc, 'page_content') else hash(str(doc))
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
    
    # Calculate RRF scores
    scores = {doc_id: 0.0 for doc_id in all_docs}
    
    for i, results in enumerate(results_list):
        weight = weights[i] if i < len(weights) else 1.0
        for rank, doc in enumerate(results):
            doc_id = hash(doc.page_content) if hasattr(doc, 'page_content') else hash(str(doc))
            if doc_id in scores:
                scores[doc_id] += weight * (1.0 / (rank + k))
    
    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(all_docs[doc_id], score) for doc_id, score in sorted_docs]

def format_result_for_openwebui(doc, score: float) -> str:
    """Format document result with natural citation for Open WebUI external tool"""
    source = doc.metadata.get('source', 'document')
    filename = os.path.basename(source) if source else 'document'
    return f"Source: {filename} (relevance: {score:.2f})\n\n{doc.page_content}"

async def optimized_hybrid_search(query: str, k: int = 10, rerank_k: int = 5):
    """Parallel hybrid search with caching and re-ranking"""
    try:
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if k <= 0 or rerank_k <= 0:
            raise ValueError("k and rerank_k must be positive integers")
        
        # Get cached documents
        all_docs_data = get_cached_documents()
        
        if not all_docs_data or not all_docs_data.get('documents'):
            return []  # No documents to search
        
        # Prepare documents for BM25
        docs_for_bm25 = []
        documents_content = all_docs_data['documents']
        metadatas = all_docs_data.get('metadatas', [])
        
        for i, content in enumerate(documents_content):
            metadata = metadatas[i] if i < len(metadatas) else {}
            doc_obj = type('Document', (), {
                'page_content': content,
                'metadata': metadata
            })()
            docs_for_bm25.append(doc_obj)
        
        if not docs_for_bm25:
            return []
        
        # Parallel search execution
        async def run_semantic_search():
            try:
                return app_state.vectorstore.similarity_search_with_score(
                    query, 
                    k=min(k*2, len(docs_for_bm25))
                )
            except Exception as e:
                print(f"Semantic search error: {e}")
                return []
        
        async def run_keyword_search():
            try:
                bm25 = BM25Retriever.from_documents(docs_for_bm25)
                bm25.k = min(k*2, len(docs_for_bm25))
                return bm25.get_relevant_documents(query)
            except Exception as e:
                print(f"Keyword search error: {e}")
                return []
        
        # Execute both searches concurrently
        semantic_results, bm25_results = await asyncio.gather(
            run_semantic_search(),
            run_keyword_search(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(semantic_results, Exception):
            print(f"Semantic search failed: {semantic_results}")
            semantic_results = []
        
        if isinstance(bm25_results, Exception):
            print(f"Keyword search failed: {bm25_results}")
            bm25_results = []
        
        # Extract documents from semantic results
        semantic_docs = [doc for doc, _ in semantic_results] if semantic_results else []
        
        # Hybrid fusion
        if not semantic_docs and not bm25_results:
            return []
        elif not semantic_docs:
            hybrid_results = [(doc, 1.0/idx) for idx, doc in enumerate(bm25_results, 1)]
        elif not bm25_results:
            hybrid_results = [(doc, score) for doc, score in zip(semantic_docs, [1.0]*len(semantic_docs))]
        else:
            hybrid_results = reciprocal_rank_fusion(
                [semantic_docs, bm25_results],
                weights=[0.5, 0.5],
                k=60
            )
        
        # Prepare for re-ranking
        max_candidates = min(len(hybrid_results), k*2, 100)
        candidate_docs = [doc for doc, _ in hybrid_results[:max_candidates]]
        candidate_contents = [getattr(doc, 'page_content', str(doc)) for doc in candidate_docs]
        
        if not candidate_contents:
            top_results = hybrid_results[:min(rerank_k, len(hybrid_results))]
            return [(doc, float(score)) for doc, score in top_results]
        
        # Re-ranking with error handling
        try:
            rerank_pairs = [(query, content) for content in candidate_contents]
            rerank_scores = app_state.reranker.predict(rerank_pairs)
            
            # Combine scores
            final_results = []
            for i, (doc, hybrid_score) in enumerate(hybrid_results[:max_candidates]):
                if i < len(rerank_scores):
                    # Normalize re-rank score using sigmoid
                    norm_rerank = 1 / (1 + np.exp(-rerank_scores[i]))
                    # Weighted combination
                    final_score = 0.7 * hybrid_score + 0.3 * norm_rerank
                    final_results.append((doc, final_score))
                else:
                    final_results.append((doc, hybrid_score))
            
            # Sort by final score and return top results
            final_results.sort(key=lambda x: x[1], reverse=True)
            return final_results[:min(rerank_k, len(final_results))]
            
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fall back to hybrid results without re-ranking
            top_results = hybrid_results[:min(rerank_k, len(hybrid_results))]
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
            ranked_results = await optimized_hybrid_search(query, input.k, min(5, input.k))
            
            # Format results for regular API usage
            results = [
                f"[Score: {score:.2f}] {doc.page_content[:200]}..."  # Truncated preview
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
        
        ranked_results = await optimized_hybrid_search(query, k, rerank_k)
        
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
        "name": "Ranked Document Retriever",
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
    return {"message": "Ranked Document Retriever API for Open WebUI"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "components": {
            "embeddings": app_state.embeddings is not None,
            "vectorstore": app_state.vectorstore is not None,
            "reranker": app_state.reranker is not None
        }
    }