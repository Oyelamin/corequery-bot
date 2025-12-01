"""
FastAPI application for Core Query Bot.

Defines all API routes and request handling.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.core.agent import QueryAgent
from src.config import settings
from src.utils.exceptions import (
    CoreQueryBotError,
    DataLoadError,
    EmptyDataError,
    FileValidationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMTimeoutError,
    NotIndexedError,
    QueryValidationError,
)
from src.utils.logger import logger
from src.api.schemas import (
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)

# Singleton agent instance
_agent: QueryAgent = None


def get_agent() -> QueryAgent:
    """Get or create the agent singleton."""
    global _agent
    if _agent is None:
        _agent = QueryAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    logger.info("Starting Core Query Bot API...")
    
    # Initialize agent
    agent = get_agent()
    agent.initialize()
    
    logger.info("API ready to accept requests")
    yield
    
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description=settings.api.description,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Exception Handlers ---

def handle_core_error(error: CoreQueryBotError, status_code: int) -> HTTPException:
    """Convert CoreQueryBotError to HTTPException."""
    detail = error.message
    if error.details:
        detail = f"{error.message}: {error.details}"
    return HTTPException(status_code=status_code, detail=detail)


# --- Routes ---

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api.title,
        "version": settings.api.version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
    """Get agent status and health information."""
    try:
        return get_agent().get_status()
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@app.post("/index", response_model=IndexResponse, tags=["Data"])
async def index_file(file: UploadFile = File(...)):
    """
    Index a CSV or Excel file.
    
    Upload a file to be indexed for querying. Supported formats: .csv, .xlsx, .xls
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .csv, .xlsx, .xls"
        )
    
    # Read and validate file size
    content = await file.read()
    max_size = settings.data.max_file_size_bytes
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {settings.data.max_file_size_mb}MB limit"
        )
    
    # Save to temp file and process
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        result = get_agent().index_data(tmp_path)
        return IndexResponse(**result)
        
    except FileValidationError as e:
        raise handle_core_error(e, 400)
    except (DataLoadError, EmptyDataError) as e:
        raise handle_core_error(e, 422)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing failed")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/index/path", response_model=IndexResponse, tags=["Data"])
async def index_file_path(file_path: str = Form(...)):
    """
    Index a file by filesystem path.
    
    Provide a path to a CSV or Excel file on the server.
    """
    try:
        result = get_agent().index_data(file_path)
        return IndexResponse(**result)
    except FileValidationError as e:
        raise handle_core_error(e, 400)
    except (DataLoadError, EmptyDataError) as e:
        raise handle_core_error(e, 422)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing failed")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the indexed data.
    
    Ask a question about the indexed data. Returns a response generated
    using semantic search and LLM.
    """
    try:
        result = get_agent().query(
            user_query=request.query,
            include_metrics=request.include_metrics
        )
        return QueryResponse(**result)
        
    except QueryValidationError as e:
        raise handle_core_error(e, 400)
    except NotIndexedError as e:
        raise handle_core_error(e, 400)
    except LLMConnectionError as e:
        raise handle_core_error(e, 503)
    except (LLMTimeoutError, LLMGenerationError) as e:
        raise handle_core_error(e, 502)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/query/simple", tags=["Query"])
async def query_simple(q: str, include_metrics: bool = False):
    """
    Simple query endpoint using GET request.
    
    Query using URL parameters instead of JSON body.
    """
    try:
        result = get_agent().query(
            user_query=q,
            include_metrics=include_metrics
        )
        return result
        
    except QueryValidationError as e:
        raise handle_core_error(e, 400)
    except NotIndexedError as e:
        raise handle_core_error(e, 400)
    except LLMConnectionError as e:
        raise handle_core_error(e, 503)
    except (LLMTimeoutError, LLMGenerationError) as e:
        raise handle_core_error(e, 502)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")
