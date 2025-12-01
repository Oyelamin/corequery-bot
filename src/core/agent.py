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
        if not self._embedding_manager.has_data():
            auto_index_file = settings.data.auto_index_file
            if auto_index_file:
                auto_index_path = Path(auto_index_file)
                if not auto_index_path.is_absolute():
                    # Relative to project root
                    auto_index_path = settings.paths.base_dir / auto_index_file
                
                if auto_index_path.exists():
                    logger.info(f"Auto-indexing configured file: {auto_index_path}")
                    try:
                        result = self.index_data(str(auto_index_path))
                        logger.info(f"Auto-indexing completed: {result.get('chunks_indexed', 0)} chunks indexed")
                    except Exception as e:
                        logger.error(f"Auto-indexing failed: {e}", exc_info=True)
                else:
                    logger.warning(f"Auto-index file not found: {auto_index_path}")
            else:
                logger.warning("No data indexed. Please index a CSV/Excel file first.")
        
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
        
        # Index chunks
        count = self._embedding_manager.index_chunks(chunks)
        
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
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: The user's question
            include_metrics: Whether to include performance metrics
            
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
        
        # Determine response
        if not query_result.meets_threshold:
            response_text = self._llm_client.get_not_found_response()
            status = "not_found"
            llm_time = 0.0
        else:
            llm_result = self._llm_client.generate(user_query, query_result.context)
            response_text = llm_result["response"]
            llm_time = llm_result["generation_time"]
            status = "success"
        
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
        
        return status
