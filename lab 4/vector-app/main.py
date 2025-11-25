from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
import os
from typing import List, Optional

app = FastAPI(title="Simple Vector API", version="1.0.0")

# In-memory storage for vectors (для демонстрации)
vector_storage = {}

class VectorCreateRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[dict] = None

class VectorResponse(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    vector: List[float]
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    similarity: float
    metadata: Optional[dict] = None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

@app.get("/")
async def root():
    return {"message": "Simple Vector API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/vectors/", response_model=VectorResponse)
async def create_vector(request: VectorCreateRequest):
    """Store a vector"""
    if request.id in vector_storage:
        raise HTTPException(status_code=400, detail=f"Vector with id {request.id} already exists")
    
    vector_storage[request.id] = {
        "vector": request.vector,
        "metadata": request.metadata or {}
    }
    
    return VectorResponse(
        id=request.id,
        vector=request.vector,
        metadata=request.metadata
    )

@app.get("/vectors/{vector_id}", response_model=VectorResponse)
async def get_vector(vector_id: str):
    """Retrieve a vector by ID"""
    if vector_id not in vector_storage:
        raise HTTPException(status_code=404, detail="Vector not found")
    
    data = vector_storage[vector_id]
    return VectorResponse(
        id=vector_id,
        vector=data["vector"],
        metadata=data["metadata"]
    )

@app.post("/search/", response_model=List[SearchResult])
async def search_similar(request: SearchRequest):
    """Search for similar vectors"""
    if not vector_storage:
        return []
    
    results = []
    for vector_id, data in vector_storage.items():
        similarity = cosine_similarity(request.vector, data["vector"])
        results.append(SearchResult(
            id=vector_id,
            similarity=similarity,
            metadata=data["metadata"]
        ))
    
    # Sort by similarity (descending) and return top_k
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:request.top_k]

@app.delete("/vectors/{vector_id}")
async def delete_vector(vector_id: str):
    """Delete a vector"""
    if vector_id not in vector_storage:
        raise HTTPException(status_code=404, detail="Vector not found")
    
    del vector_storage[vector_id]
    return {"message": f"Vector {vector_id} deleted"}

@app.get("/vectors/")
async def list_vectors():
    """List all vector IDs"""
    return {"vector_ids": list(vector_storage.keys())}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)