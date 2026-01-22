from dataclasses import dataclass
from app.search.service import HybridSearchService
from app.clients.chroma_store import ChromaStore
from app.config import Settings


@dataclass
class AppState:
    """Holds fully-wired application dependencies."""
    settings: Settings
    chroma: ChromaStore
    search_service: HybridSearchService