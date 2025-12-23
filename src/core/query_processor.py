"""
Query processing and semantic search.

Handles query validation, embedding generation, and similarity search.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils.exceptions import QueryValidationError
from src.utils.logger import logger


@dataclass
class QueryResult:
    """Container for query processing results."""
    
    query: str
    matches: List[Dict[str, Any]]
    max_similarity: float
    meets_threshold: bool
    context: str
    processing_time: float
    embedding_time: float
    search_time: float


class QueryProcessor:
    """
    Handles query processing and semantic search.
    
    Provides methods for validating queries, generating embeddings,
    and searching the vector database.
    """
    
    MIN_QUERY_LENGTH = 2
    MAX_QUERY_LENGTH = 1000
    
    def __init__(
        self, 
        embedding_model: SentenceTransformer, 
        collection: Any
    ) -> None:
        """
        Initialize QueryProcessor.
        
        Args:
            embedding_model: Initialized sentence transformer model
            collection: ChromaDB collection instance
        """
        self._model = embedding_model
        self._collection = collection
        self._threshold = settings.vector_search.similarity_threshold
        self._top_k = settings.vector_search.top_k_matches
    
    def validate_query(self, query: str) -> str:
        """
        Validate and sanitize user query.
        
        Args:
            query: Raw user query
            
        Returns:
            Sanitized query string
            
        Raises:
            QueryValidationError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise QueryValidationError("Query must be a non-empty string")
        
        query = query.strip()
        
        if len(query) < self.MIN_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query must be at least {self.MIN_QUERY_LENGTH} characters"
            )
        
        if len(query) > self.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters"
            )
        
        return query
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: Optional[int] = None,
        query_text: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search for similar content in vector database.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            query_text: Original query text (for exact matching on short queries)
            
        Returns:
            Tuple of (matches list, max similarity score)
        """
        top_k = top_k or self._top_k
        
        # Check collection count
        count = self._collection.count()
        if count == 0:
            logger.warning("Collection is empty")
            return [], 0.0
        
        # For very short queries (1-2 words), get more results to improve matching
        # This helps with greetings and simple queries
        if query_text:
            query_words = query_text.strip().lower().split()
            if len(query_words) <= 2:
                # Get more results for short queries to find better matches
                top_k = min(top_k * 2, count)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count)
        )
        
        return self._format_results(results, query_text=query_text)
    
    def _format_results(
        self, 
        results: Dict[str, Any],
        query_text: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Format ChromaDB results into match dictionaries.
        
        Args:
            results: ChromaDB query results
            query_text: Original query text for exact matching boost
        """
        matches = []
        max_score = 0.0
        
        if not results.get("ids") or not results["ids"][0]:
            return matches, max_score
        
        query_lower = query_text.lower().strip() if query_text else ""
        
        for i in range(len(results["ids"][0])):
            # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: similarity = 1 - (distance / 2)
            # This gives us a 0-1 similarity score where 1 = identical
            distance = results["distances"][0][i] if results.get("distances") else 0
            # Cosine distance ranges from 0 to 2, convert to similarity (0 to 1)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            # Boost similarity for exact matches in Question field (for short queries)
            if query_text and len(query_text.split()) <= 2:
                document_text = results["documents"][0][i].lower()
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                
                # Check if query appears in Question field (from original_data)
                if metadata.get("original_data"):
                    try:
                        import json
                        original_data = json.loads(metadata.get("original_data", "{}"))
                        question = original_data.get("Question", "").lower()
                        if query_lower in question or question in query_lower:
                            # Boost similarity for exact/partial matches in Question
                            similarity = min(1.0, similarity + 0.2)
                            logger.debug(f"Boosted similarity for query '{query_text}' in Question field")
                    except (json.JSONDecodeError, AttributeError):
                        pass
                
                # Also check if query appears directly in document text
                if query_lower in document_text:
                    similarity = min(1.0, similarity + 0.15)
            
            max_score = max(max_score, similarity)
            
            matches.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "similarity": round(similarity, 4),
                "distance": round(distance, 4)
            })
        
        return matches, max_score
    
    def format_context(self, matches: List[Dict[str, Any]]) -> str:
        """Format matches into context string for LLM."""
        if not matches:
            return ""
        
        parts = []
        for i, match in enumerate(matches, 1):
            parts.append(
                f"Match {i} (relevance: {match['similarity']:.0%}):\n{match['text']}"
            )
        
        return "\n\n".join(parts)
    
    def process(self, query: str) -> QueryResult:
        """
        Complete query processing pipeline.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult with matches, scores, and timing
        """
        start_time = time.perf_counter()
        
        # Validate
        query = self.validate_query(query)
        
        # Generate embedding
        embed_start = time.perf_counter()
        query_embedding = self.generate_embedding(query)
        embedding_time = time.perf_counter() - embed_start
        
        # Search (pass query text for exact matching on short queries)
        search_start = time.perf_counter()
        matches, max_score = self.search(query_embedding, query_text=query)
        search_time = time.perf_counter() - search_start
        
        # Check threshold with special handling for short queries
        # Short queries (like "Hello", "Hi") may have lower similarity scores
        # but should still be processed if they have any matches
        query_length = len(query.split())
        if query_length <= 2 and matches:
            # For very short queries (1-2 words), use a lower threshold
            # This helps with greetings and simple queries
            adjusted_threshold = max(0.3, self._threshold - 0.3)
            meets_threshold = max_score >= adjusted_threshold
            if meets_threshold:
                logger.debug(f"Short query '{query}' matched with adjusted threshold {adjusted_threshold:.2f}")
        else:
            meets_threshold = max_score >= self._threshold
        
        # Format context
        context = self.format_context(matches) if matches else ""
        
        total_time = time.perf_counter() - start_time
        
        logger.info(
            f"Query processed: similarity={max_score:.3f}, "
            f"matches={len(matches)}, threshold_met={meets_threshold}"
        )
        
        return QueryResult(
            query=query,
            matches=matches,
            max_similarity=max_score,
            meets_threshold=meets_threshold,
            context=context,
            processing_time=total_time,
            embedding_time=embedding_time,
            search_time=search_time
        )
