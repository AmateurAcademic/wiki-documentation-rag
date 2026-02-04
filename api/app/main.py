from fastapi import FastAPI, HTTPException

from app.dependencies import build_app_state
from app.models import ToolRequest, NoteCommitResult, NoteStageResult
from app.search.formatting import format_result_for_openwebui

app = FastAPI(
    title="RAG Retriever and Note Writer API",
    version="1.0.1",
    description="Open WebUI tool API that can search an ingested markdown wiki and stage/commit approved notes into the wiki.",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Wire dependencies at startup; collection readiness handled via backoff."""
    app.state.state = build_app_state()


@app.post("/search")
async def search_documents(request: ToolRequest) -> dict:
    """
    Open WebUI external tool endpoint to search ingested wiki documents.

    Expected ToolRequest body:
      {"query": "...", "k": 10, "rerank_k": 5}
    """
    try:
        query = request.body.get("query", "")
        k = int(request.body.get("k", 10))
        rerank_k = int(request.body.get("rerank_k", 5))

        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        app_state = app.state.state
        ranked = await app_state.search_service.search(query=query, k=k, rerank_k=rerank_k)

        results = [
            format_result_for_openwebui(doc, score, app_state.settings.wiki_base_url)
            for doc, score in ranked
        ]
        return {"results": results}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.post("/notes/stage", response_model=NoteStageResult)
async def stage_note(request: ToolRequest) -> NoteStageResult:
    """
    Stage approved markdown content for later commit.

    Expected ToolRequest body:
      {"file_name": "Some Note.md", "content": "...markdown...", "append": false}

    Rules:
      - Use only after the user approves the content in chat.
      - This does not write to git; it only stages server-side.
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
    try:
        stage_id, path, expires_at = app_state.notes_service.stage_note(
            file_name=file_name,
            content=content,
            append=append,
        )
        return NoteStageResult(stage_id=stage_id, path=path, expires_at=expires_at)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stage failed: {exc}") from exc


@app.post("/notes/commit", response_model=NoteCommitResult)
async def commit_note(request: ToolRequest) -> NoteCommitResult:
    """
    Commit a previously staged note into the wiki via the egester service.

    Expected ToolRequest body:
      {"stage_id": "stg_..."}

    Rules:
      - Use only after explicit user confirmation.
      - This writes the staged content into ai_notes/ and creates a git commit (via egester).
    """
    body = request.body
    stage_id = body.get("stage_id", "")

    if not stage_id:
        raise HTTPException(status_code=400, detail="stage_id is required")

    app_state = app.state.state
    try:
        path = await app_state.notes_service.commit_note(stage_id=stage_id)
        return NoteCommitResult(ok=True, path=path)
    except ValueError as exc:
        # Your NotesService raises ValueError for "not found or expired"
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Commit failed: {exc}") from exc


@app.get("/specification")
async def get_specification() -> dict:
    """Return a compact tool specification for Open WebUI integration."""
    return {
        "name": "RAG Retriever and Note Writer API",
        "description": "Search ingested wiki documents and stage/commit user-approved notes into the wiki (notes are written under ai_notes/).",
        "endpoints": [
            {
                "name": "search_documents",
                "method": "POST",
                "path": "/search",
                "description": "Search ingested wiki documents for RAG use.",
                "parameters": [
                    {"name": "query", "type": "string", "required": True},
                    {"name": "k", "type": "integer", "required": False, "default": 10},
                    {"name": "rerank_k", "type": "integer", "required": False, "default": 5},
                ],
            },
            {
                "name": "stage_note",
                "method": "POST",
                "path": "/notes/stage",
                "description": "Stage a note (use only after the user approves the exact markdown content).",
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
                "description": "Commit a previously staged note (use only after explicit user confirmation).",
                "parameters": [
                    {"name": "stage_id", "type": "string", "required": True},
                ],
            },
        ],
    }


@app.get("/openapi.json")
async def get_openapi_spec() -> dict:
    """Return OpenAPI specification."""
    return app.openapi()


@app.get("/")
async def root() -> dict:
    """Root endpoint for basic health check."""
    return {"message": "RAG Retriever and Note Writer API for Open WebUI"}


@app.get("/health")
async def health_check() -> dict:
    """Detailed health check endpoint."""
    app_state = app.state.state

    collection_status = "not_ready"
    if app_state.chroma.collection is not None:
        try:
            count = app_state.chroma.collection.count()
            collection_status = f"ready ({count} documents)"
        except Exception:
            collection_status = "ready (count unavailable)"

    return {
        "status": "healthy",
        "components": {
            "chromadb": collection_status,
            "notes_service": app_state.notes_service is not None,
            "wiki_base_url": app_state.settings.wiki_base_url,
        },
    }
