from fastapi import FastAPI, HTTPException
from app.models import ToolRequest
from app.dependencies import build_app_state
from app.search.formatting import format_result_for_openwebui


app = FastAPI(
    title="Minimal Document Retriever API",
    version="1.0.0",
    description="Hybrid search with full ranking and re-ranking pipeline. Results formatted for Open WebUI.",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Wire dependencies at startup; collection readiness handled via backoff."""
    app.state.state = build_app_state()


@app.post("/search")
async def search_documents(request: ToolRequest) -> dict:
    """Search with full ranking pipeline for Open WebUI external tool.
    
    Args:
        request: ToolRequest containing query parameters
        
    Returns:
        Dictionary with formatted search results
        
    Raises:
        HTTPException: For various error conditions (400 for bad request, 500 for server errors)
    """
    try:
        query = request.body.get("query", "")
        k = int(request.body.get("k", 10))
        rerank_k = int(request.body.get("rerank_k", 5))

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        st = app.state.state
        ranked = await st.search_service.search(query=query, k=k, rerank_k=rerank_k)

        results = [
            format_result_for_openwebui(doc, score, st.settings.wiki_base_url)
            for doc, score in ranked
        ]

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/specification")
async def get_specification():
    """Return API specification for Open WebUI integration."""
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
    """Return OpenAPI specification."""
    return app.openapi()


@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {"message": "Minimal Document Retriever API for Open WebUI"}


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    st = app.state.state
    collection_status = "not_ready"
    if st.chroma.collection is not None:
        try:
            # Check if collection has documents
            count = st.chroma.collection.count()
            collection_status = f"ready ({count} documents)"
        except Exception:
            collection_status = "ready (count unavailable)"

    return {
        "status": "healthy",
        "components": {
            "openai": st.embeddings is not None,
            "chromadb": collection_status,
            "reranker": st.reranker is not None,
            "embedding_dimension": st.settings.embedding_dim
        }
    }