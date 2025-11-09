# backend/routers/recommendation.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.services.retriever import retrieve_documents
from src.services.chat import call_chat, format_recommendations, get_test_type_probs_via_llm
from src.config import DEFAULT_ALPHA, DEFAULT_K

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


@router.post("/recommend")
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
        test_type_probs = get_test_type_probs_via_llm(q)

        # 1. Retrieve candidate documents
        candidates = retrieve_documents(
            query_text=q,
            alpha=req.alpha,
            k=req.k,
            mode=req.mode,
            test_type_probs=test_type_probs
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
            max_tokens=req.max_tokens,
            test_type_probs=test_type_probs
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
        
        # 4. Build response in required format
        formatted_recommendations = []
        for rec in recommendations:
            # Find matching candidate to get metadata
            matching_candidate = next(
                (c for c in candidates if c.get("name") == rec.get("name")),
                None
            )
            
            if matching_candidate:
                metadata = matching_candidate.get("metadata", {})
                
                formatted_rec = {
                    "url": metadata.get("url", ""),
                    "name": rec.get("name", ""),
                    "adaptive_support": metadata.get("adaptive_support", "No"),
                    "description": rec.get("justification", ""),
                    "duration": metadata.get("duration", "Not specified"),
                    "remote_support": metadata.get("remote_support", "No"),
                    "test_type": metadata.get("test_type", [])
                }
                formatted_recommendations.append(formatted_rec)

        return {
            "recommended_assessments": formatted_recommendations
        }
        
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
    Returns raw retrieval results with scores.
    """
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get test type probabilities for better scoring
        test_type_probs = get_test_type_probs_via_llm(q)
        
        # Retrieve candidates with scoring
        candidates = retrieve_documents(
            query_text=q,
            alpha=req.alpha,
            k=req.k,
            mode=req.mode,
            test_type_probs=test_type_probs
        )
        
        # Format results to include all scoring details
        formatted_results = []
        for idx, candidate in enumerate(candidates, 1):
            formatted_results.append({
                "rank": idx,
                "name": candidate.get("name", "Unknown"),
                "url": candidate.get("metadata", {}).get("url", ""),
                "similarity_score": round(candidate.get("score", 0.0), 4),
                "type_alignment_score": round(candidate.get("s_type", 0.0), 4),
                "final_score": round(candidate.get("final_score", 0.0), 4),
                "test_type": candidate.get("metadata", {}).get("test_type", []),
                "description_preview": candidate.get("text", "")[:200]
            })
        
        return {
            "query": q,
            "test_type_probs": test_type_probs,
            "results": formatted_results,
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