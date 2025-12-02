"""
FastAPI application for Core Query Bot.

Defines all API routes and request handling.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
    WebSocketMessage,
    WebSocketResponse,
)
from src.core.session_manager import session_manager

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
# For WebSocket support, we allow all origins if none specified
# WebSocket connections need more permissive CORS settings
allowed_origins = settings.api.allowed_origins
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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
    using semantic search and LLM. Optionally provide a session_id for
    conversational context.
    """
    try:
        result = get_agent().query(
            user_query=request.query,
            include_metrics=request.include_metrics,
            session_id=request.session_id
        )
        if request.session_id:
            result["session_id"] = request.session_id
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversational queries.
    
    Requires a session_id in the initial connection query parameter or in messages.
    Maintains conversation history per session.
    
    Note: WebSocket connections bypass CORS restrictions, so this endpoint
    accepts connections from any origin for easier development and usage.
    """
    # Accept WebSocket connection from any origin
    # WebSocket connections don't use CORS, but we accept all origins here
    await websocket.accept()

    try:
        # Get session_id from query parameter or wait for it in first message
        query_params = dict(websocket.query_params)
        session_id = query_params.get("session_id")
        
        if not session_id:
            # Wait for first message to get session_id
            initial_data = await websocket.receive_text()
            try:
                initial_msg = json.loads(initial_data)
                session_id = initial_msg.get("session_id")
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON. Please provide session_id in query parameter or first message."
                })
                await websocket.close()
                return
        
        if not session_id:
            await websocket.send_json({
                "type": "error",
                "error": "session_id is required. Provide it in query parameter (?session_id=xxx) or in first message."
            })
            await websocket.close()
            return
        
        # Create or get session
        session_manager.get_or_create_session(session_id)
        logger.info(f"WebSocket connected for session: {session_id}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected successfully. You can now send queries."
        })
        
        # Main message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "query")
                    
                    if msg_type == "ping":
                        # Respond to ping
                        await websocket.send_json({
                            "type": "pong",
                            "session_id": session_id
                        })
                        continue
                    
                    elif msg_type == "clear_history":
                        # Clear conversation history
                        session_manager.clear_session(session_id)
                        await websocket.send_json({
                            "type": "response",
                            "response": "Conversation history cleared.",
                            "status": "success",
                            "session_id": session_id
                        })
                        continue
                    
                    elif msg_type == "query":
                        # Process query
                        query_text = message.get("query")
                        if not query_text:
                            await websocket.send_json({
                                "type": "error",
                                "error": "Query text is required for 'query' type messages.",
                                "session_id": session_id
                            })
                            continue
                        
                        include_metrics = message.get("include_metrics", False)
                        
                        # Process query with agent
                        try:
                            result = get_agent().query(
                                user_query=query_text,
                                include_metrics=include_metrics,
                                session_id=session_id
                            )
                            
                            # Send response
                            response = {
                                "type": "response",
                                "query": query_text,
                                "response": result.get("response", ""),
                                "status": result.get("status", "success"),
                                "session_id": session_id,
                                "similarity_score": result.get("similarity_score"),
                                "matches_found": result.get("matches_found", 0),
                                "meets_threshold": result.get("meets_threshold", False)
                            }
                            
                            if include_metrics and result.get("metrics"):
                                response["metrics"] = result["metrics"]
                            
                            await websocket.send_json(response)
                            
                        except Exception as e:
                            logger.error(f"Query processing error: {e}", exc_info=True)
                            await websocket.send_json({
                                "type": "error",
                                "error": f"Failed to process query: {str(e)}",
                                "session_id": session_id
                            })
                    
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Unknown message type: {msg_type}",
                            "session_id": session_id
                        })
                
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid JSON format",
                        "session_id": session_id
                    })
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Internal error: {str(e)}",
                        "session_id": session_id
                    })
                except:
                    pass
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass
