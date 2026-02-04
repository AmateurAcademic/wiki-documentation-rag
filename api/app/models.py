from pydantic import BaseModel
from typing import List, Dict, Any


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

class NoteStageResult(BaseModel):
    stage_id: str
    path: str
    expires_at: str

class NoteCommitResult(BaseModel):
    ok: bool
    path: str