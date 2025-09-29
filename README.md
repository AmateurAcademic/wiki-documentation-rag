# Markdown RAG Pipeline
A complete semantic search system for markdown documents using Nebius Qwen3 embeddings, ChromaDB, and Open WebUI integration.

## Overview
This project provides an automated pipeline that:
- Watches for changes in markdown files
- Processes documents and generates embeddings using Nebius Qwen3 via OAI-compatible API
- Stores vectors in ChromaDB for efficient retrieval
- Exposes a semantic search API for integration with Open WebUI

## Architecture
```Markdown Files → Ingestion Service → ChromaDB → API Service → Open WebUI```

The system consists of three main components:
1. **Ingestion Service**: Monitors markdown files and updates the vector database
2. **ChromaDB**: Persistent vector storage for embeddings
3. **API Service**: FastAPI server providing semantic search endpoints

## Prerequisites
- Docker and Docker Compose
- Nebius AI API key with Qwen3 embedding access

## Setup
1. Clone the repository
2. Place your markdown files in `./data/markdown/`
3. Create a `.env` file with your Nebius API key:
   ```
   NEBIUS_API_KEY=your-nebius-api-key-here
   ```
4. Start the services:
   ```bash
   docker compose up -d
   ```

## Usage
- The ingestion service automatically processes markdown files when they change
- Access the API at `http://localhost:8000`
- Integrate with Open WebUI using the external tool configuration

## Project Structure
```
project/
├── docker-compose.yml
├── ingestion/
│   ├── markdown_ingester.py
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── chroma-data/ (generated)
└── data/
    └── markdown/ (place your .md files here)
```

## API Endpoints
- `POST /retrieve` - Retrieve documents for multiple queries
- `POST /search` - Search documents (Open WebUI integration)
- `GET /specification` - Open WebUI tool specification
- `GET /openapi.json` - OpenAPI specification

## Configuration
The system uses environment variables:
- `NEBIUS_API_KEY`: Your Nebius AI API key

## How It Works
1. **Ingestion**: The ingestion service watches `./data/markdown/` for file changes
2. **Processing**: When files change, they're automatically processed and embedded
3. **Storage**: Embeddings are stored in the persistent ChromaDB volume
4. **Retrieval**: The API service provides semantic search over the stored embeddings
5. **Integration**: Open WebUI connects to the API service for RAG capabilities

## Development
To rebuild and restart services:
```bash
docker compose up -d --build
```