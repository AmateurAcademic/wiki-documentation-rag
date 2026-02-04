from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from typing import Dict, Optional, Tuple

@dataclass(frozen=True)
class StagedNote:
    file_name: str
    path: str
    content: str
    append: bool
    expires_at: datetime


class StageStore:
    """Store for staging notes before committing to the wiki repository."""
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, StagedNote] = {}
    
    def stage_note(self, file_name: str, path: str, content: str, append: bool) -> Tuple[str, datetime]:
        """Stage a note for later committing."""
        stage_id = self._new_stage_id()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
        staged_note = StagedNote(
            file_name=file_name,
            path=path,
            content=content,
            append=append,
            expires_at=expires_at
        )
        self._store[stage_id] = staged_note
        return stage_id, expires_at

    def consume_staged_note(self, stage_id: str) -> Optional[StagedNote]:
        """Consume and remove a staged note by its stage ID."""
        self._evict_expired()
        return self._store.pop(stage_id, None)

    def _new_stage_id(self) -> str:
        """Generate a new unique stage ID."""
        return "stg_" + secrets.token_urlsafe(24)
    
    def _evict_expired(self) -> None:
        """Evict expired staged notes from the store."""
        now = datetime.now(timezone.utc)
        expired_keys = [key for key, note in self._store.items() if note.expires_at <= now]
        for key in expired_keys:
            self._store.pop(key, None)
