"""
Pydantic schemas for API request/response models.

Provides validation and serialization for API data.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., min_length=2, max_length=1000, description="User query string")
    include_metrics: bool = Field(default=True, description="Include performance metrics in response")
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query string."""
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v


class MetricsResponse(BaseModel):
    """Performance metrics response model."""
    
    query: str
    timestamp: str
    response_time_ms: float
    embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
    llm_time_ms: Optional[float] = None
    similarity_score: Optional[float] = None
    matches_found: int
    status: str
    response_length: int = 0


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    query: str
    response: str
    status: str = Field(..., description="Query status: success, not_found, or error")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    matches_found: int = Field(..., ge=0)
    meets_threshold: bool
    metrics: Optional[Dict[str, Any]] = None


class IndexResponse(BaseModel):
    """Response model for index endpoint."""
    
    status: str
    message: str
    chunks_indexed: int = Field(..., ge=0)


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    
    initialized: bool
    data_indexed: bool
    ollama_available: bool
    embedding_model: str
    llm_model: str
    chunks_count: Optional[int] = Field(None, ge=0)


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = "healthy"


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str
    detail: Optional[str] = None
    status_code: int

