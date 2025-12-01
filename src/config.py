"""
Configuration management for the AI agent.

This module handles all configuration settings loaded from environment variables.
Directories are created lazily when needed, not at import time.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env_list(key: str, default: Optional[str] = None) -> Optional[List[str]]:
    """Parse comma-separated environment variable into list."""
    value = os.getenv(key, default)
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return None


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_env_int(key: str, default: int) -> int:
    """Parse integer environment variable with validation."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Parse float environment variable with validation."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class PathConfig:
    """Path configuration - directories created lazily."""
    
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def chroma_db_path(self) -> Path:
        return self.base_dir / "chroma_db"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    @property
    def metrics_dir(self) -> Path:
        return self.base_dir / "metrics"
    
    @property
    def log_file(self) -> Path:
        return self.logs_dir / "app.log"
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [self.data_dir, self.chroma_db_path, self.logs_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""
    
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 32))


@dataclass(frozen=True)
class VectorSearchConfig:
    """Vector search configuration."""
    
    similarity_threshold: float = field(default_factory=lambda: _get_env_float("SIMILARITY_THRESHOLD", 0.7))
    top_k_matches: int = field(default_factory=lambda: _get_env_int("TOP_K_MATCHES", 5))
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            object.__setattr__(self, 'similarity_threshold', 0.7)
        if self.top_k_matches < 1:
            object.__setattr__(self, 'top_k_matches', 5)


@dataclass(frozen=True)
class LLMConfig:
    """LLM (Ollama) configuration."""
    
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    timeout: int = field(default_factory=lambda: _get_env_int("LLM_TIMEOUT", 30))


@dataclass(frozen=True)
class DataConfig:
    """Data processing configuration."""
    
    search_columns: Optional[List[str]] = field(default_factory=lambda: _get_env_list("SEARCH_COLUMNS"))
    max_file_size_mb: int = field(default_factory=lambda: _get_env_int("MAX_FILE_SIZE_MB", 100))
    auto_index_file: Optional[str] = field(default_factory=lambda: os.getenv("AUTO_INDEX_FILE", None))
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    enable_performance_tracking: bool = field(default_factory=lambda: _get_env_bool("ENABLE_PERFORMANCE_TRACKING", True))


@dataclass(frozen=True)
class APIConfig:
    """FastAPI configuration."""
    
    title: str = "Core Query Bot API"
    version: str = "1.0.0"
    description: str = "AI Agent for querying structured CSV/Excel data using RAG"
    allowed_origins: List[str] = field(default_factory=lambda: _get_env_list("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000") or [])


@dataclass
class Settings:
    """Application settings container."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global settings instance
settings = Settings()
