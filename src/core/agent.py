"""
Main agent orchestration class.

Coordinates all components to process queries and generate responses.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings
from src.data.loader import DataLoader
from src.data.embeddings import EmbeddingManager
from src.utils.exceptions import NotIndexedError
from src.llm.client import LLMClient
from src.utils.logger import logger
from src.utils.performance import PerformanceTracker
from src.core.query_processor import QueryProcessor
from src.core.session_manager import session_manager


class QueryAgent:
    """
    Main agent that orchestrates all components.
    
    Provides a unified interface for indexing data and processing queries.
    """
    
    def __init__(self) -> None:
        """Initialize the query agent with all components."""
        self._data_loader = DataLoader()
        self._embedding_manager = EmbeddingManager()
        self._llm_client = LLMClient()
        self._performance_tracker = PerformanceTracker()
        self._query_processor: Optional[QueryProcessor] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize all components.
        
        This is called automatically when needed, but can be called
        explicitly to pre-load models.
        """
        if self._initialized:
            return
        
        logger.info("Initializing QueryAgent...")
        
        # Initialize embedding manager (loads model and creates collection)
        _ = self._embedding_manager.embedding_model
        _ = self._embedding_manager.collection
        
        # Initialize query processor
        self._query_processor = QueryProcessor(
            self._embedding_manager.embedding_model,
            self._embedding_manager.collection
        )
        
        # Auto-index if configured
        auto_index_file = settings.data.auto_index_file
        if auto_index_file:
            auto_index_path = Path(auto_index_file)
            if not auto_index_path.is_absolute():
                # Relative to project root
                auto_index_path = settings.paths.base_dir / auto_index_file
            
            if auto_index_path.exists():
                # Check if we should re-index (only if no data exists)
                if not self._embedding_manager.has_data():
                    logger.info(f"Auto-indexing configured file: {auto_index_path}")
                    try:
                        result = self.index_data(str(auto_index_path))
                        logger.info(f"Auto-indexing completed: {result.get('chunks_indexed', 0)} chunks indexed")
                    except Exception as e:
                        logger.error(f"Auto-indexing failed: {e}", exc_info=True)
                        logger.error("Please check the error above and ensure all dependencies are installed (e.g., openpyxl for Excel files)")
                else:
                    logger.info(f"Data already indexed ({self._embedding_manager.get_collection_count()} chunks). Skipping auto-index.")
                    logger.info(f"To re-index '{auto_index_path.name}', clear existing data first or use the /index endpoint.")
            else:
                logger.warning(f"Auto-index file not found: {auto_index_path}")
                logger.warning("Please check your .env file and ensure AUTO_INDEX_FILE path is correct.")
        else:
            if not self._embedding_manager.has_data():
                logger.warning("No data indexed and AUTO_INDEX_FILE not configured. Please index a CSV/Excel file first.")
        
        if not self._llm_client.check_health():
            logger.warning("Ollama is not running. LLM features unavailable.")
        
        self._initialized = True
        logger.info("QueryAgent initialized successfully")
    
    def _ensure_initialized(self) -> None:
        """Ensure agent is initialized."""
        if not self._initialized:
            self.initialize()
    
    def index_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load and index data from a CSV/Excel file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with indexing results
        """
        self._ensure_initialized()
        logger.info(f"Starting data indexing: {file_path}")
        
        # Load and process data
        chunks = self._data_loader.load_and_process(file_path)
        
        # Index chunks (pass source file for tracking)
        count = self._embedding_manager.index_chunks(chunks, source_file=file_path)
        
        # Reinitialize query processor with updated collection
        self._query_processor = QueryProcessor(
            self._embedding_manager.embedding_model,
            self._embedding_manager.collection
        )
        
        result = {
            "status": "success",
            "message": f"Successfully indexed {count} rows",
            "chunks_indexed": count
        }
        
        logger.info(f"Indexing completed: {count} chunks")
        return result
    
    def query(
        self, 
        user_query: str, 
        include_metrics: bool = True,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: The user's question
            include_metrics: Whether to include performance metrics
            session_id: Optional session ID for conversation history
            
        Returns:
            Dictionary with response and metadata
        """
        self._ensure_initialized()
        start_time = self._performance_tracker.start_timer()
        
        # Check if data is indexed
        if not self._embedding_manager.has_data():
            raise NotIndexedError(
                "No data indexed",
                details="Please index a CSV/Excel file using the /index endpoint"
            )
        
        # Process query
        query_result = self._query_processor.process(user_query)
        
        # Get conversation history if session ID provided
        conversation_history = None
        if session_id:
            conversation_history = session_manager.get_history(session_id, max_messages=10)
        
        # Determine response
        # For short queries with matches, even if below threshold, 
        # still try to generate a response if we have any matches
        if not query_result.meets_threshold:
            # If we have matches but they're below threshold, still use them for short queries
            query_length = len(user_query.split())
            if query_length <= 2 and query_result.matches:
                logger.info(f"Short query '{user_query}' below threshold but has matches, using them anyway")
                # Use the matches even though below threshold
                llm_result = self._llm_client.generate(
                    user_query,
                    query_result.context,
                    conversation_history=conversation_history
                )
                response_text = llm_result["response"]
                llm_time = llm_result["generation_time"]
                status = "success"
            else:
                response_text = self._llm_client.get_not_found_response()
                status = "not_found"
                llm_time = 0.0
        else:
            llm_result = self._llm_client.generate(
                user_query, 
                query_result.context,
                conversation_history=conversation_history
            )
            response_text = llm_result["response"]
            llm_time = llm_result["generation_time"]
            status = "success"
        
        # Store messages in session history if session ID provided
        if session_id:
            session_manager.add_message(session_id, "user", user_query)
            session_manager.add_message(session_id, "assistant", response_text)
        
        # Create metrics
        metrics = self._performance_tracker.create_metrics(
            query=user_query,
            start_time=start_time,
            embedding_time=query_result.embedding_time,
            search_time=query_result.search_time,
            llm_time=llm_time if query_result.meets_threshold else None,
            similarity_score=query_result.max_similarity,
            matches_found=len(query_result.matches),
            status=status,
            response=response_text
        )
        
        # Save metrics
        # self._performance_tracker.save_metrics(metrics)
        
        # Build response
        result = {
            "query": user_query,
            "response": response_text,
            "status": status,
            "similarity_score": query_result.max_similarity,
            "matches_found": len(query_result.matches),
            "meets_threshold": query_result.meets_threshold
        }
        
        if include_metrics:
            result["metrics"] = metrics.to_dict()
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status and health information.
        
        Returns:
            Dictionary with status information
        """
        self._ensure_initialized()
        
        has_data = self._embedding_manager.has_data()
        ollama_available = self._llm_client.check_health()
        
        status = {
            "initialized": self._initialized,
            "data_indexed": has_data,
            "ollama_available": ollama_available,
            "embedding_model": settings.embedding.model_name,
            "llm_model": self._llm_client.model
        }
        
        if has_data:
            status["chunks_count"] = self._embedding_manager.get_collection_count()
            
            # Get indexed file information
            file_info = self._embedding_manager.get_indexed_file_info()
            if file_info:
                status["indexed_file"] = file_info.get("source_file")
                status["indexed_file_path"] = file_info.get("source_path")
                status["indexed_at"] = file_info.get("indexed_at")
        
        return status
