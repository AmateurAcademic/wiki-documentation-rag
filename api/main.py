import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(
    title="Markdown Document Retriever API",
    version="1.0.0",
    description="Semantic search over markdown documentation for Open WebUI integration",
)

class RetrievalQueryInput(BaseModel):
    queries: List[str] = Field(..., description="List of queries to retrieve from the vectorstore")
    k: int = Field(3, description="Number of results per query")

class RetrievedDoc(BaseModel):
    query: str
    results: List[str]

class RetrievalResponse(BaseModel):
    responses: List[RetrievedDoc]

class ToolRequest(BaseModel):
    body: Dict[str, Any]

def get_retriever():
    """Initialize retriver on startup"""
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-8B",
        openai_api_key=os.getenv("NEBIUS_API_KEY"),
        openai_api_base="https://api.studio.nebius.com/v1/"
    )
    vectorstore = Chroma(
        persist_directory="../chroma-data", 
        embedding_function=embeddings
    )
    return vectorstore.as_retriever()

retriever = get_retriever()

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_docs(input: RetrievalQueryInput):
    """Given a list of user queries, returns top-k retrieved documents per query."""
    try:
        responses = []
        for query in input.queries:
            docs = retriever.get_relevant_documents(query)[:input.k]
            results = [doc.page_content for doc in docs]
            responses.append(RetrievedDoc(query=query, results=results))
        return RetrievalResponse(responses=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Open WebUI External Tool endpoints
@app.post("/search")
async def search_documents(request: ToolRequest):
    query = request.body.get("query", "")
    k = request.body.get("k", 3)
    
    docs = retriever.get_relevant_documents(query)[:k]
    results = [doc.page_content for doc in docs]
    
    return {"results": results}

@app.get("/specification")
async def get_specification():
    return {
        "name": "Document Retriever",
        "description": "Semantic search over markdown documentation",
        "endpoints": [{
            "name": "search_documents",
            "method": "POST", 
            "path": "/search",
            "description": "Search documents semantically",
            "parameters": [
                {"name": "query", "type": "string", "required": True},
                {"name": "k", "type": "integer", "required": False, "default": 3}
            ]
        }]
    }

@app.get("/openapi.json")
async def get_openapi_spec():
    return app.openapi()

@app.get("/")
async def root():
    return {"message": "Document Retriever API for Open WebUI"}