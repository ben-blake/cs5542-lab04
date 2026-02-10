import sys
import os

# Add the parent directory to sys.path so we can import 'rag'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from rag.pipeline import load_and_index_data_core, build_context, simple_extractive_answer

# Global storage for the index
DATA_PACK = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data on startup
    print("Loading RAG index...")
    global DATA_PACK
    DATA_PACK = load_and_index_data_core()
    print("RAG index loaded.")
    yield
    # Clean up on shutdown if needed
    DATA_PACK.clear()

app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    method: str = "hybrid"
    chunking: str = "page"
    top_k_text: int = 5
    top_k_images: int = 3
    top_k_evidence: int = 8

class QueryResponse(BaseModel):
    answer: str
    context: str
    evidence: List[Dict[str, Any]]
    image_paths: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not DATA_PACK:
        raise HTTPException(status_code=503, detail="Index not ready")
    
    # 1. Retrieve
    try:
        result = build_context(
            query=req.query,
            data_pack=DATA_PACK,
            method=req.method,
            chunking=req.chunking,
            top_k_text=req.top_k_text,
            top_k_images=req.top_k_images,
            top_k_evidence=req.top_k_evidence
        )
        
        # 2. Generate
        answer = simple_extractive_answer(req.query, result["context"], result["evidence"])
        
        return {
            "answer": answer,
            "context": result["context"],
            "evidence": result["evidence"],
            "image_paths": result["image_paths"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "indexed": bool(DATA_PACK)}
