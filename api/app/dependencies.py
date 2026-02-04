from app.config import Settings
from app.state import AppState
from app.clients.embeddings_client import EmbeddingClient
from app.clients.chroma_store import ChromaStore
from app.clients.reranker_client import RerankerClient
from app.search.service import HybridSearchService

from app.clients.egester_client import EgesterClient
from app.egestion.stage_store import StageStore
from app.egestion.notes_service import NotesService


def build_app_state() -> AppState:
    """Composition root: only place that reads env-derived Settings and constructs AppState."""
    
    settings = Settings.from_env()

    embeddings = EmbeddingClient(
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    chroma = ChromaStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name="documents",
        bm25_limit=settings.bm25_limit,
        backoff=settings.collection_backoff,
    )

    # Connect to ChromaDB and acquire collection with backoff
    chroma.connect()
    chroma.get_collection_with_backoff()

    reranker = RerankerClient(settings.reranker_model)

    search_service = HybridSearchService(
        settings=settings,
        chroma=chroma,
        embeddings=embeddings,
        reranker=reranker,
    )

    stage_store = StageStore(ttl_seconds=settings.note_stage_ttl_seconds)

    egester_client = EgesterClient(
        base_url=settings.egester_base_url
        )

    notes_service = NotesService(
        stage_store=stage_store,
        egester_client=egester_client
    )

    return AppState(
        settings=settings,
        chroma=chroma,
        search_service=search_service,
        notes_service=notes_service
        )