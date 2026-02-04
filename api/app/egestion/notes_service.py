import os
from datetime import datetime
from typing import Tuple

from app.clients.egester_client import EgesterClient
from app.egestion.stage_store import StageStore

class NotesService:
    """Service for staging and committing notes to the wiki repository."""

    def __init__(self, stage_store: StageStore, egester_client: EgesterClient):
        self.stage_store = stage_store
        self.egester_client = egester_client

    def normalize_file_name(self, file_name: str) -> str:
        """Normalize a user provided file name into a safe markdown file name."""

        cleaned = (file_name or "").strip()
        if not cleaned:
            raise ValueError("file_name cannot be empty")

        if "/" in cleaned or "\\" in cleaned:
            raise ValueError("file_name cannot contain path separators")
        
        if cleaned in {".", ".."}:
            raise ValueError("file_name cannot be . or ..")

        if not cleaned.lower().endswith(".md"):
            cleaned += ".md"

        return cleaned

    def build_note_path(self, normalized_file_name: str) -> str:
        """Build the wiki-relative path for a note given its normalized file name."""
        return os.path.join("ai_notes", normalized_file_name).replace("\\", "/")


    def stage_note(self, file_name: str, content: str, append: bool) -> Tuple[str, str, str]:
        """Stage a note for later committing."""
        normalized_file_name = self.normalize_file_name(file_name)
        path = self.build_note_path(normalized_file_name)
        stage_id, expires_at = self.stage_store.stage_note(
            file_name=normalized_file_name,
            path=path,
            content=content,
            append=append
        )
        return stage_id, path, expires_at.isoformat()

    async def commit_note(self, stage_id: str) -> str:
        """Commit a staged note to the wiki repository."""
        staged_note = self.stage_store.consume_staged_note(stage_id)
        if not staged_note:
            raise ValueError("Staged note not found or expired")

        await self.egester_client.write_note(
            file_name=staged_note.file_name,
            content=staged_note.content,
            append=staged_note.append
        )

        return staged_note.path