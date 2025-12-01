"""
Performance monitoring and metrics tracking.

Provides utilities for tracking query performance and logging metrics.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings
from src.utils.logger import logger


@dataclass
class QueryMetrics:
    """Container for query performance metrics."""
    
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: float = 0.0
    embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
    llm_time_ms: Optional[float] = None
    similarity_score: Optional[float] = None
    matches_found: int = 0
    status: str = "success"
    response_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceTracker:
    """
    Tracks and logs performance metrics for queries.
    
    Provides methods for creating, formatting, and persisting metrics.
    """
    
    def __init__(self, enabled: Optional[bool] = None) -> None:
        """
        Initialize performance tracker.
        
        Args:
            enabled: Whether tracking is enabled (default from config)
        """
        self._enabled = enabled if enabled is not None else settings.logging.enable_performance_tracking
    
    @property
    def enabled(self) -> bool:
        """Check if performance tracking is enabled."""
        return self._enabled
    
    @staticmethod
    def start_timer() -> float:
        """Start a performance timer using perf_counter for precision."""
        return time.perf_counter()
    
    def create_metrics(
        self,
        query: str,
        start_time: float,
        embedding_time: Optional[float] = None,
        search_time: Optional[float] = None,
        llm_time: Optional[float] = None,
        similarity_score: Optional[float] = None,
        matches_found: int = 0,
        status: str = "success",
        response: Optional[str] = None
    ) -> QueryMetrics:
        """
        Create a metrics object for a query.
        
        Args:
            query: User query
            start_time: Timer start value from start_timer()
            embedding_time: Time for embedding generation (seconds)
            search_time: Time for vector search (seconds)
            llm_time: Time for LLM generation (seconds)
            similarity_score: Maximum similarity score
            matches_found: Number of matches found
            status: Query status (success, not_found, error)
            response: LLM response text
            
        Returns:
            QueryMetrics instance
        """
        total_time = time.perf_counter() - start_time
        
        return QueryMetrics(
            query=query,
            response_time_ms=round(total_time * 1000, 2),
            embedding_time_ms=round(embedding_time * 1000, 2) if embedding_time else None,
            search_time_ms=round(search_time * 1000, 2) if search_time else None,
            llm_time_ms=round(llm_time * 1000, 2) if llm_time else None,
            similarity_score=round(similarity_score, 3) if similarity_score else None,
            matches_found=matches_found,
            status=status,
            response_length=len(response) if response else 0
        )
    
    def save_metrics(self, metrics: QueryMetrics) -> Optional[Path]:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: QueryMetrics instance
            
        Returns:
            Path to saved file, or None if saving failed
        """
        if not self._enabled:
            return None
        
        try:
            # Ensure directory exists
            metrics_dir = settings.paths.metrics_dir
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = metrics_dir / f"query_{timestamp}.json"
            
            # Save
            with open(filepath, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            logger.debug(f"Metrics saved to {filepath}")
            return filepath
            
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save metrics: {e}")
            return None
    
    def format_console_output(self, metrics: QueryMetrics) -> str:
        """
        Format metrics for console display.
        
        Args:
            metrics: QueryMetrics instance
            
        Returns:
            Formatted string
        """
        lines = [
            "Performance Metrics:",
            f"  Total time: {metrics.response_time_ms / 1000:.2f}s"
        ]
        
        if metrics.embedding_time_ms:
            lines.append(f"  - Embedding: {metrics.embedding_time_ms / 1000:.2f}s")
        if metrics.search_time_ms:
            lines.append(f"  - Search: {metrics.search_time_ms / 1000:.2f}s")
        if metrics.llm_time_ms:
            lines.append(f"  - LLM: {metrics.llm_time_ms / 1000:.2f}s")
        
        if metrics.similarity_score is not None:
            confidence = self._get_confidence_label(metrics.similarity_score)
            lines.append(f"  Similarity: {metrics.similarity_score:.2f} ({confidence})")
        
        lines.append(f"  Status: {metrics.status}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _get_confidence_label(score: float) -> str:
        """Get human-readable confidence label."""
        if score >= 0.8:
            return "high"
        elif score >= 0.7:
            return "medium"
        return "low"
