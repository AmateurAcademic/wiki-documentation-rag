import os
from typing import Union, Dict, Any


def _build_wiki_url_from_source(source_path: str, wiki_base_url: str) -> str:
    """
    Convert a local markdown file path into a wiki URL.
    
    Args:
        source_path: Path to the source markdown file
        wiki_base_url: Base URL for the wiki
        
    Returns:
        Complete wiki URL or empty string if conversion fails
        
    Example:
        source_path = "/app/data/markdown/hardware/magic-mirror.md"
        wiki_base_url = "https://wiki.home"
        -> "https://wiki.home/hardware/magic-mirror"
    """
    if not wiki_base_url or not source_path:
        return ""

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

    return f"{wiki_base_url}{rel_no_ext}"


def format_result_for_openwebui(doc: Union[Dict[str, Any], str], score: float, wiki_base_url: str = "") -> str:
    """Format document result with natural citation for Open WebUI external tool.
    
    Args:
        doc: Document to format (dict with content/metadata or string)
        score: Relevance score
        wiki_base_url: Base URL for wiki links
        
    Returns:
        Formatted string result for Open WebUI
    """
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
            url = _build_wiki_url_from_source(source_path, wiki_base_url)
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