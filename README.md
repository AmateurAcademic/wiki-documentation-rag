# Documentation RAG Pipeline

A complete semantic search system for markdown documentation using Nebius Qwen3 embeddings, ChromaDB, and Open WebUI integration.

The project is designed primarily for a self-hosted personal wiki, but can be adapted to other markdown-based documentation setups.

- Ingestion is Git-aware (for startup sync) and still uses a file watcher for live changes.
- Retrieval uses hybrid search (semantic + BM25) with Reciprocal Rank Fusion and re-ranking.
- The `/search` endpoint is optimized for use as an Open WebUI external tool.

I built this myself for my technical documentation wiki at home and integrated it as an external tool in Open WebUI. I still wouldn't consider this production. 

I basically looked at how Open WebUI does RAG and copied that. I first built it with LangChain. I found the images to be too bloated. In addition, I didn't really understand how it worked under the hood so well. Therefore, I re-implemented by rolling it myself. This project isn't vibe coded, but I did do some small refactors with LLMs. My personal goal was to really understand how it all works and to ensure it works how I want it to. The codebase sorta exploded from the prototype I originally made. I will need to properly re-write the codebase to make it easier to understand. Once I do that, I will probably call it production ready and call it a day.

Thanks to Open WebUI and LangChain for allowing me to not totally reinvent the wheel. I am grateful I could study in detail how they do things to make my own heavily inspired from that.

## Overview

This project provides an automated pipeline that:

- Watches for changes in markdown files
- Uses Git history at startup to process only changed / deleted files
- Processes documents and generates embeddings using Nebius Qwen3 via an OpenAI-compatible API
- Stores vectors in ChromaDB for efficient retrieval
- Implements hybrid search with **semantic + keyword (BM25)** retrieval
- Re-ranks results with a cross-encoder for better relevance
- Exposes a search API tailored for Open WebUI as an external tool

## Architecture

```text
Markdown Files (Git repo)
        │
        ▼
 Ingestion Service
  - Git-based startup sync
  - Watchdog file watcher (runtime changes)
        │
        ▼
     ChromaDB
  (HTTP vector DB)
        │
        ▼
    API Service
  - Hybrid search (semantic + BM25)
  - Reciprocal Rank Fusion
  - Cross-encoder re-ranking
        │
        ▼
   Open WebUI Tool
  - External tool /search
  - Custom model with system prompt
````

## Key Features

* **Git-Aware Ingestion (Startup)**
  Uses the markdown directory as a Git repo:

  * On startup, compares the last processed commit to the current HEAD
  * Only re-embeds changed/added files
  * Removes deleted/renamed files from ChromaDB
  * Falls back to full scan if Git is unavailable

* **File Watcher (Runtime)**
  Uses `watchdog` to:

  * React to file creations, modifications, deletions
  * Reprocess just the affected file(s) on the fly

* **Qwen3 Embeddings via Nebius**

  * Uses `Qwen/Qwen3-Embedding-8B` via Nebius (OpenAI-compatible endpoint)
  * Fixed embedding dimension: **4096**

* **Hybrid Search**

  * Semantic search over ChromaDB vectors
  * BM25 keyword search over raw documents
  * Combined via Reciprocal Rank Fusion (RRF)

* **Re-Ranking**

  * Uses `BAAI/bge-reranker-v2-m3` (CrossEncoder) to re-score top candidates
  * Final score = combination of RRF score + normalized re-rank score

* **Wiki-Friendly Output**

  * Each result from `/search` is formatted like:
    `Source: <page_name> (relevance: 0.87)`
    `URL: https://wiki...` *(if WIKI_BASE_URL is set)*
  * This is optimized for LLM consumption in Open WebUI.

## Prerequisites

* Docker and Docker Compose
* Nebius AI API key (or compatible OpenAI-style API that supports the Qwen3 embedding model)
* A markdown-based wiki stored under a Git repository (for Git-based change detection)


## Project Structure

```text
.
├── docker-compose.yml
├── chroma_data/           # Persistent ChromaDB storage
├── data/
│   └── markdown/          # Gollum / wiki repo (Git)
├── api/
│   ├── main.py            # FastAPI search server
│   ├── Dockerfile
│   └── requirements.txt
└── ingestion/
    ├── markdown_ingester.py   # Git-aware ingest + watcher
    ├── Dockerfile
    └── requirements.txt
```

> Note: names may vary slightly if you’ve renamed files, but conceptually this is the layout.


## Environment Variables

### Core

* `NEBIUS_API_KEY`
  Nebius AI API key (or compatible OpenAI-style API key).

* `CHROMA_HOST`
  Hostname for ChromaDB HTTP service (default: `chroma`).

* `CHROMA_PORT`
  Port for ChromaDB HTTP service (default: `8000`).

### Wiki URL Generation

* `WIKI_BASE_URL` *(optional but recommended)*
  Base URL for your wiki, e.g.:

  ```env
  WIKI_BASE_URL=https://wiki.home.dopest.cloud
  ```

  When set, the API derives a human-clickable wiki URL for each chunk, based on the `source` metadata path stored in Chroma, e.g.:

  * `source = "/app/data/markdown/hardware/magic-mirror.md"`
  * `WIKI_BASE_URL = "https://wiki.home.dopest.cloud"`
    → `URL: https://wiki.home.dopest.cloud/hardware/magic-mirror`

  This URL is included in `/search` results to make it easy for the LLM (and you) to link back to the full page.


## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/AmateurAcademic/wiki-documentation-rag.git
   cd wiki-documentation-rag
   ```

2. **Prepare markdown data**

   * The ingestion container expects a Git repository mounted at `./data/markdown/`.
   * Typically this is your Gollum wiki or equivalent.

3. **Create `.env`**

   ```bash
   cp .env.example .env
   ```

   Fill in:

   ```env
   NEBIUS_API_KEY=your-nebius-api-key-here
   WIKI_BASE_URL=https://wiki.home.dopest.cloud
   ```

4. **Start the stack**

   ```bash
   docker compose up -d
   ```

   This will start:

   * ChromaDB
   * Ingestion service
   * API service


## Services

### 1. ChromaDB

* Vector database service (HTTP)
* Persistent storage: `./chroma_data`
* Configured for external embedding generation (no embedding function attached in Chroma itself)
* Used for:

  * Semantic vector search
  * Metadata storage (`source`, `chunk_index`, etc.)


### 2. Ingestion Service

* Monitors `./data/markdown/` for markdown files.

* Uses **Git-based processing on startup**:

  * Reads a state file with the last processed commit hash.
  * Compares to current `HEAD`.
  * For new/changed files: re-chunks and re-embeds.
  * For deleted/renamed files: removes associated chunks from Chroma.
  * Saves the new commit hash atomically to avoid corruption.

* Uses **watchdog file watcher at runtime**:

  * `on_created`: embeds new file
  * `on_modified`: re-embeds that file
  * `on_deleted`: removes chunks for that file
  * This logic currently works independently of Git commits for live changes.

* Chunking:

  * Typ. `chunk_size ~ 1000`, `overlap ~ 200` (config in code)
  * IDs are **content-based** (SHA-256 over content + source + chunk_index)
    → avoids duplicates when reprocessing

* Stored metadata per chunk:

  * `source`: full path to the markdown file inside the container
  * `chunk_index`: index of chunk in that file
  * `original_length`: size of full document
  * `processed_at`: timestamp (for debugging/future use)


### 3. API Service

* **FastAPI** server
* Provides search & debug endpoints
* Uses:

  * Nebius OpenAI-compatible client for embeddings
  * Chroma HTTP client
  * `BAAI/bge-reranker-v2-m3` CrossEncoder for re-ranking
  * BM25 via `rank_bm25` for keyword search


## API Endpoints

### Public / Tool-Facing

These are intended for Open WebUI and external use:

* `POST /search`
  Primary search endpoint for Open WebUI external tools.

  * Runs hybrid search (semantic + BM25)
  * Combines with Reciprocal Rank Fusion
  * Re-ranks top candidates with cross-encoder
  * Returns an array of formatted strings, e.g.:

    ```text
    Source: magic-mirror (relevance: 0.87)
    URL: https://wiki.home.dopest.cloud/hardware/magic-mirror

    <chunk content here...>
    ```

* `GET /openapi.json`
  OpenAPI specification used by Open WebUI to discover the tool.

* `GET /`
  Simple root message.

* `GET /specification`
  Optional spec helper for tool registration (not strictly required if using `/openapi.json` directly).

### Internal / Debug (hidden from OpenAPI)

These are useful during development but are **hidden from schema** to avoid LLMs calling them as tools:

* `POST /retrieve` *(include_in_schema=False)*

  * Multi-query retrieval
  * Returns quick text previews with scores
  * Useful for debugging hybrid_search behavior

* `GET /health` *(include_in_schema=False)*

  * Health check for:

    * OpenAI client
    * ChromaDB connectivity
    * Re-ranker initialization
    * Embedding dimension

* `GET /test_embedding` *(include_in_schema=False)*

  * Simple check that the embedding endpoint works and returns the expected dimension.


## Search Pipeline (API)

For `POST /search`:

1. **Embedding**

   * Generate query embedding with `Qwen/Qwen3-Embedding-8B`.
   * Validate the embedding dimension (4096).

2. **Parallel Retrieval**

   * Semantic search over Chroma with the query embedding.
   * BM25 keyword search over stored documents.
   * Both return `(doc, score)` pairs, where `doc` is a dict:

     ```json
     {
       "content": "...",
       "metadata": {
         "source": "/app/data/markdown/...",
         "chunk_index": "0",
         "original_length": "12345",
         "processed_at": "..."
       }
     }
     ```

3. **Fusion (RRF)**

   * Combine semantic + BM25 rankings via Reciprocal Rank Fusion.
   * Use a stable doc key based on `(source, chunk_index)` so the same chunk from both pipelines merges.

4. **Re-Ranking**

   * Take top fused candidates (limited by `rerank_k` and hard-capped to avoid large batches).
   * Run `BAAI/bge-reranker-v2-m3` on `(query, content)` pairs.
   * Combine RRF score + normalized re-rank score into a final score.

5. **Filtering & Truncation**

   * Drop empty/whitespace-only chunks.
   * Sort by final score.
   * Return at most `k` results.

6. **Formatting**

   * For each `(doc, score)`:

     * Derive `source` from the markdown filename (without extension).
     * Construct a wiki `URL` using `WIKI_BASE_URL` and the relative path if available.
     * Render as a text block for Open WebUI.


## Open WebUI Integration

### 1. Add Tool Server

In **Open WebUI**:

1. Go to **Admin Panel → External Tools → Manage Tool Servers**.
2. Add a new tool server:

   * **URL**: `http://<api-host>:<port>`
     e.g. `http://api:8000` (inside Docker network) or `http://localhost:8000`.
   * **OpenAPI Spec Path**: `/openapi.json`
3. Save.
   Open WebUI will discover the `/search` endpoint automatically.

> You typically won’t expose `/retrieve`, `/health`, or `/test_embedding` as tools because they’re hidden from the OpenAPI schema.


### 2. Create a Custom Model with System Prompt

To get the best behavior, create a **custom model definition** (In your "Workspace" or “evaluation” style model) that:

* Wraps your base chat model (e.g. Qwen, GPT, etc.).
* Uses a **fixed system prompt** that explains how and when to call the wiki search tool.
* Has the external tool server (this API) enabled.

A good system prompt template for this project:

```text
You are an assistant that answers user questions and can call tools defined in the attached OpenAPI specification.

You have access to an HTTP tool whose purpose is to search my personal wiki. It is the POST /search endpoint in the API (described in the OpenAPI spec as “Search documents with full ranking pipeline”). It returns text chunks that usually start with a line like:
  Source: <name> (relevance: <score>)
and often include a URL line, for example:
  URL: https://...

## When to use the wiki search tool

- Use the wiki search tool whenever the user is asking about:
  - my systems, servers, homelab, personal setup, or internal tools
  - things that look like they could be documented in my wiki (hardware, services, configs, how-to steps)
- If you are unsure whether the answer is in the wiki, prefer to CALL THE TOOL first.
- Only rely on your general knowledge if:
  - the tool returns no relevant results, or
  - the user’s question is clearly general / unrelated to my personal environment.

## How to use the wiki results

When the tool returns results:

1. Read the returned text and answer the user’s question using that content.
2. If a result includes a URL line (for example, `URL: https://...`), reuse that URL as a normal Markdown link in your answer, such as:
   - `More details here: [BlackMirror](https://...)`
3. Keep the protocol (`http` vs `https`) exactly as given in the context whenever you reuse a URL.
4. You may combine or summarize multiple results, but stay faithful to what the wiki text actually says.
5. Prefer higher-relevance results first.

If the tool returns nothing useful, explain briefly that the wiki didn’t have relevant information and then, if appropriate, use your general knowledge to help.

## Style and behavior

- Match the user’s language and level of detail.
- Be direct and practical; include commands or steps when appropriate.
- Use real hostnames/IPs/paths from the wiki when present; do not invent or “sanitize” them.
- Do not mention tools, OpenAPI, or internal implementation details in your answer.
- Do not invent links or URLs that are not present in the wiki/tool output.
```

Attach this system prompt in the model configuration (e.g., in the model’s “params” / “system prompt” field).


### 3. Make the Model Easy to Use

* Add the custom model to a **Workspace** or pin it in the **sidebar** so it appears prominently in the model list.
* Ensure:

  * The base model has **Tool Usage** enabled.
  * The external tool server (this API) is **enabled** for that model.

Then, when you select this model in chat, it will:

* Automatically see the `/search` tool (from OpenAPI).
* Decide when to call it based on your prompt.
* Generate answers that include clickable wiki links.


## Notes / Future Work

* **Refactor into Classes / Modules**
  The current API and ingester work, but the next step is splitting into:

  * `GitHandler`, `MarkdownIO`, `EmbeddingService`, `ChromaStore`, `MarkdownIngestionService`, `MarkdownWatchHandler`, etc.
    for better structure and testability.

* **Replace Runtime Watcher with Git-Based Polling**
  At the moment:

  * Startup sync is Git-based.
  * Runtime updates are watcher-based.
    In a future version, runtime could also be Git-aware (polling or hooks) to reduce reliance on filesystem events.

* **Type Hints & Linting**
  The code has been written with type-hinting in mind and can be progressively annotated with `mypy` / `ruff` rules later on.


If you’re reading this in the repo and want to adapt it for your own wiki, the main knobs are:

* Where your markdown lives (`./data/markdown`).
* What your wiki base URL is (`WIKI_BASE_URL`).
* Which base model you wrap in Open WebUI with the system prompt above.

```
