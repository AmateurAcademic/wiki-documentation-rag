# Documentation RAG Pipeline
A complete semantic search system for markdown documentation using Nebius Qwen3 embeddings, ChromaDB, and Open WebUI integration. I use Nebius API for the Qwen3 embedder, but you could alwyas use something else that is OAI API-based. I originally used langchain to try it out again (I haven't checked it out in a long while), but I found the dependencies too large for what it does, and I just implemented everything myself. 

## Overview
This project provides an automated pipeline that:
- Watches for changes in markdown files
- Processes documents and generates embeddings using Nebius Qwen3 via OAI-compatible API
- Stores vectors in ChromaDB for efficient retrieval
- Implements hybrid search with semantic + keyword retrieval
- Exposes a semantic search API for integration with Open WebUI

## Architecture
```
Markdown Files → Ingestion Service → ChromaDB → API Service → Open WebUI
                     │                      │
                     │                      └── Hybrid Search (Semantic + BM25)
                     └── File Watcher        └── Re-ranking (Cross-Encoder)
```

## Key Features
- **Automated Ingestion**: Real-time processing of markdown file changes
- **Qwen3 Embeddings**: State-of-the-art embeddings via Nebius API
- **Hybrid Search**: Combines semantic search with keyword-based BM25
- **Re-ranking**: Uses BAAI/bge-reranker-v2-m3 for improved relevance
- **Open WebUI Integration**: Seamless RAG capabilities in chat interface

## Prerequisites
- Docker and Docker Compose
- Nebius AI API key with Qwen3 embedding access

## Setup
1. Clone the repository
2. Place your markdown files in `./data/markdown/`
3. Create a `.env` file with your Nebius API key:
   ```env
   NEBIUS_API_KEY=your-nebius-api-key-here
   ```
4. Start the services:
   ```bash
   docker compose up -d
   ```

## Services
1. **ChromaDB**: Vector database service (port 8001)
   - Persistent storage in `./chroma_data`
   - Pre-configured for Qwen3 embeddings (4096 dimensions)

2. **Ingestion Service**: Processes markdown files
   - Monitors `./data/markdown` for changes
   - Splits documents using recursive text splitting
   - Generates embeddings via Nebius API
   - Stores embeddings in ChromaDB

3. **API Service**: Search and retrieval endpoints (port 8000)
   - FastAPI server with OpenAPI documentation
   - Hybrid search with semantic + keyword retrieval
   - Reciprocal Rank Fusion for result combination
   - Cross-encoder re-ranking

## Open WebUI Integration
1. Navigate to Open WebUI Admin Panel → External Tools → Manage Tool Servers
2. Add New Tool Server with:
   - **URL**: `http://localhost:8000` (or `http://api:8000` if in same Docker network)
   - **OpenAPI Spec**: `/openapi.json`
   - **Name**: "Documentation Retriever"
   - **Description**: "Semantic search over documentation"
3. The tool will automatically appear in chat interface for RAG queries

## API Endpoints
- `POST /search` - Primary search endpoint (Open WebUI integration)
- `POST /retrieve` - Retrieve documents for multiple queries
- `GET /specification` - Open WebUI tool specification
- `GET /openapi.json` - OpenAPI specification
- `GET /health` - Service health check

## Search Pipeline
1. **Query Processing**:
   - Generate Qwen3 embeddings for query
   - Validate embedding dimensions (4096)
2. **Hybrid Retrieval**:
   - Semantic search via ChromaDB (cosine similarity)
   - Keyword search via BM25
3. **Result Fusion**:
   - Combine results using Reciprocal Rank Fusion
4. **Re-ranking**:
   - Score candidate documents with Cross-Encoder
   - Combine RRF and re-ranker scores
5. **Formatting**:
   - Prepare results with source attribution
   - Structure for Open WebUI display

## Configuration
Environment variables:
- `NEBIUS_API_KEY`: Nebius AI API key
- `CHROMA_HOST`: ChromaDB hostname (default: `chroma`)
- `CHROMA_PORT`: ChromaDB port (default: `8000`)

## Project Structure
```
.
├── docker-compose.yml
├── .env.example
├── chroma_data/ (persistent ChromaDB storage)
├── data/
│   └── markdown/ (user-provided markdown files)
├── api/
│   ├── main.py (FastAPI server)
│   ├── Dockerfile
│   └── requirements.txt
└── ingestion/
    ├── ingestion.py (file watcher and processor)
    ├── Dockerfile
    └── requirements.txt
```

## How It Works
1. **Ingestion**:
   - Service watches `./data/markdown/` for file changes
   - Processes new/modified markdown files
   - Splits documents into chunks (1000 chars, 200 overlap)
   - Generates Qwen3 embeddings via Nebius API
   - Stores embeddings in ChromaDB with metadata

2. **Search**:
   - Receives query from Open WebUI
   - Generates Qwen3 embedding for query
   - Performs hybrid search (semantic + keyword)
   - Fuses and re-ranks results
   - Returns formatted responses with sources