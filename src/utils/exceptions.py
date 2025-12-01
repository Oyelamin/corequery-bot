"""
Custom exceptions for the Core Query Bot.

Provides specific exception types for better error handling and debugging.
"""

from typing import Optional


class CoreQueryBotError(Exception):
    """Base exception for all Core Query Bot errors."""
    
    def __init__(self, message: str, details: Optional[str] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)


class DataLoadError(CoreQueryBotError):
    """Raised when data loading fails."""
    pass


class FileValidationError(CoreQueryBotError):
    """Raised when file validation fails."""
    pass


class EmptyDataError(CoreQueryBotError):
    """Raised when data source is empty."""
    pass


class EmbeddingError(CoreQueryBotError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(CoreQueryBotError):
    """Raised when vector store operations fail."""
    pass


class QueryProcessingError(CoreQueryBotError):
    """Raised when query processing fails."""
    pass


class QueryValidationError(CoreQueryBotError):
    """Raised when query validation fails."""
    pass


class LLMConnectionError(CoreQueryBotError):
    """Raised when LLM connection fails."""
    pass


class LLMGenerationError(CoreQueryBotError):
    """Raised when LLM response generation fails."""
    pass


class LLMTimeoutError(CoreQueryBotError):
    """Raised when LLM request times out."""
    pass


class NotIndexedError(CoreQueryBotError):
    """Raised when querying without indexed data."""
    pass

