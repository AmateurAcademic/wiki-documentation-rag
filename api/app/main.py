from fastapi import FastAPI, HTTPException
from app.models import ToolRequest, NoteStageResult, NoteCommitResult
from app.dependencies import build_app_state
from app.search.formatting import format_result_for_openwebui


app = FastAPI(
    title="RAG Retriever and Note Writer API",
    version="1.0.1",
    description="RAG Document retriever API and a note Writing API for markdown-based wikis as an OpenWebUI Tool",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Wire dependencies at startup; collection readiness handled via backoff."""
    app.state.state = build_app_state()


@app.post("/search")
async def search_documents(request: ToolRequest) -> dict:
    """
    OpenWebUI external tool to search documents with RAG.
    
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


@app.post("/notes/stage", response_model=NoteStageResult)
async def stage_note(request: ToolRequest) -> NoteStageResult:
    """
    Stage a note for later committing.
    
    Args:
        request: ToolRequest containing note parameters
    
    Returns:
        NoteStageResult with staging details

    Raises:
        HTTPException: For various error conditions (400 for bad request, 500 for server errors)
    """

    body = request.body
    file_name = body.get("file_name", "")
    content = body.get("content", "")
    append = bool(body.get("append", False))

    if not file_name:
        raise HTTPException(status_code=400, detail="file_name is required")

    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    app_state = app.state.state
    stage_id, path, expires_at = app_state.notes_service.stage_note(
        file_name=file_name,
        content=content,
        append=append
    )
    return NoteStageResult(
        stage_id=stage_id,
        path=path,
        expires_at=expires_at
    )

@app.post("/notes/commit", response_model=NoteCommitResult)
async def commit_note(request: ToolRequest) -> NoteCommitResult:
    """
    Commit a staged note to the wiki repository.

    Args:
        request: ToolRequest containing stage_id of the note to commit
    
    Returns:
        NoteCommitResult with commit details
    
    Raises:
        HTTPException: For various error conditions (400 for bad request, 500 for server errors)
    """
    body = request.body
    stage_id = body.get("stage_id", "")

    if not stage_id:
        raise HTTPException(status_code=400, detail="stage_id is required")

    app_state = app.state.state
    try:
        path = await app_state.notes_service.commit_note(stage_id=stage_id)
        return NoteCommitResult(
            ok=True,
            path=path
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Commit failed: {str(e)}")

@app.get("/specification")
async def get_specification():
    """Return API specification for Open WebUI integration."""
    return {
        "name": "RAG Retriever and Note Writer API",
        "description": "RAG Document retriever API and a note Writing API for markdown-based wikis as an OpenWebUI Tool",
        "endpoints": [{
            "name": "search_documents",
            "method": "POST",
            "path": "/search",
            "description": "OpenWebUI external tool to search documents with RAG",
            "parameters": [
                {"name": "query", "type": "string", "required": True},
                {"name": "k", "type": "integer", "required": False, "default": 10},
                {"name": "rerank_k", "type": "integer", "required": False, "default": 5}
            ]
        },
        {
            "name": "stage_note",
            "method": "POST",
            "path": "/notes/stage",
            "description": "Stage a note for later commit (use only after the user approves the content)",
            "parameters": [
                {"name": "file_name", "type": "string", "required": True},
                {"name": "content", "type": "string", "required": True},
                {"name": "append", "type": "boolean", "required": False, "default": False},
            ],
        },
        {
            "name": "commit_note",
            "method": "POST",
            "path": "/notes/commit",
            "description": "Commit a previously staged note (use only after explicit user confirmation)",
            "parameters": [
                {"name": "stage_id", "type": "string", "required": True},
            ],
        }
        ]
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
    app_state = app.state.state
    collection_status = "not_ready"
    if app_state.chroma.collection is not None:
        try:
            # Check if collection has documents
            count = app_state.chroma.collection.count()
            collection_status = f"ready ({count} documents)"
        except Exception:
            collection_status = "ready (count unavailable)"

    return {
        "status": "healthy",
        "components": {
            "openai": app_state.embeddings is not None,
            "chromadb": collection_status,
            "reranker": app_state.reranker is not None,
            "embedding_dimension": app_state.settings.embedding_dim
        }
    }