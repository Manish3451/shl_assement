# backend/routers/recommendation.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from backend.services.retriever import retrieve_documents
from backend.services.chat import call_chat, format_recommendations
from backend.config import DEFAULT_ALPHA, DEFAULT_K

router = APIRouter()

class RecommendRequest(BaseModel):
    query: str = Field(..., description="User's assessment search query", min_length=1)
    alpha: Optional[float] = Field(
        DEFAULT_ALPHA, 
        description="Weight for dense retriever (0-1)", 
        ge=0.0, 
        le=1.0
    )
    k: Optional[int] = Field(
        DEFAULT_K, 
        description="Number of results to retrieve and use", 
        ge=1, 
        le=50
    )
    mode: Optional[str] = Field(
        "hybrid", 
        description="Retrieval mode: dense, sparse, or hybrid"
    )
    temperature: Optional[float] = Field(
        0.0, 
        description="LLM temperature for response generation", 
        ge=0.0, 
        le=2.0
    )
    max_tokens: Optional[int] = Field(
        1000, 
        description="Maximum tokens in LLM response", 
        ge=100, 
        le=4000
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I need to assess problem-solving skills for software engineers",
                "alpha": 0.7,
                "k": 10,
                "mode": "hybrid",
                "temperature": 0.0
            }
        }

class RecommendResponse(BaseModel):
    query: str
    candidates_used: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    explanation: str
    llm_response: Dict[str, Any]
    formatted_output: str
    retrieval_info: Dict[str, Any]
    success: bool

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Get assessment recommendations based on user query.
    
    Process:
    1. Retrieve relevant assessments using hybrid search (dense + sparse)
    2. Pass top results to LLM for analysis and recommendations
    3. Return structured recommendations with justifications
    """
    # Validate query
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Validate mode
    if req.mode not in ["dense", "sparse", "hybrid"]:
        raise HTTPException(
            status_code=400, 
            detail="Mode must be one of: dense, sparse, hybrid"
        )
    
    try:
        # 1. Retrieve candidate documents
        candidates = retrieve_documents(
            query_text=q,
            alpha=req.alpha,
            k=req.k,
            mode=req.mode
        )
        
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant assessments found. Try rephrasing your query."
            )
        
        # 2. Call LLM with retrieved candidates
        chat_result = call_chat(
            query=q,
            candidates=candidates,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        
        if not chat_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"LLM processing failed: {chat_result.get('error', 'Unknown error')}"
            )
        
        # 3. Extract parsed response
        parsed = chat_result.get("parsed_response", {})
        recommendations = parsed.get("recommendations", [])
        explanation = parsed.get("explanation", "")
        
        # 4. Format candidates for response
        candidates_summary = [
            {
                "name": c.get("name", "Unknown"),
                "url": c.get("metadata", {}).get("url", ""),
                "has_content": bool(c.get("text", "").strip())
            }
            for c in candidates
        ]
        
        # 5. Build response
        return RecommendResponse(
            query=q,
            candidates_used=candidates_summary,
            recommendations=recommendations,
            explanation=explanation,
            llm_response={
                "raw_answer": chat_result.get("answer", ""),
                "usage": chat_result.get("usage"),
                "elapsed_seconds": chat_result.get("elapsed"),
                "model": chat_result.get("model")
            },
            formatted_output=format_recommendations(parsed),
            retrieval_info={
                "mode": req.mode,
                "alpha": req.alpha,
                "k": req.k,
                "num_candidates_retrieved": len(candidates),
                "num_recommendations": len(recommendations)
            },
            success=True
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other errors
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation pipeline failed: {str(e)}"
        )


class HealthResponse(BaseModel):
    status: str
    message: str

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the recommendation service is healthy."""
    return HealthResponse(
        status="healthy",
        message="Recommendation service is running"
    )


# Optional: Simple search endpoint without LLM
@router.post("/search")
def search_only(req: RecommendRequest):
    """
    Search for assessments without LLM recommendations.
    Returns raw retrieval results only.
    """
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        candidates = retrieve_documents(
            query_text=q,
            alpha=req.alpha,
            k=req.k,
            mode=req.mode
        )
        
        return {
            "query": q,
            "results": candidates,
            "count": len(candidates),
            "retrieval_info": {
                "mode": req.mode,
                "alpha": req.alpha,
                "k": req.k
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )