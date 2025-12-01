"""
Embedding generation and vector database storage.

Handles embedding model initialization, vector generation, and ChromaDB operations.
"""

import json
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import settings
from src.utils.exceptions import EmbeddingError, VectorStoreError
from src.utils.logger import logger


class EmbeddingManager:
    """
    Manages embedding generation and vector database storage.
    
    Provides lazy initialization of models and database connections.
    """
    
    COLLECTION_NAME = "data_chunks"
    
    def __init__(self) -> None:
        """Initialize EmbeddingManager with lazy loading."""
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._initialize_embedder()
        return self._embedding_model
    
    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            self._initialize_chroma()
        return self._collection
    
    def _initialize_embedder(self) -> None:
        """Initialize the embedding model."""
        model_name = settings.embedding.model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self._embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {model_name}", details=str(e))
    
    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        db_path = settings.paths.chroma_db_path
        logger.info(f"Initializing ChromaDB at: {db_path}")
        
        try:
            # Ensure directory exists
            db_path.mkdir(parents=True, exist_ok=True)
            
            self._chroma_client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "CSV/Excel data chunks"}
            )
            
            logger.info(f"ChromaDB collection ready: {self.COLLECTION_NAME}")
            
        except Exception as e:
            raise VectorStoreError("Failed to initialize ChromaDB", details=str(e))
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = settings.embedding.batch_size
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            embeddings = []
            iterator = range(0, len(texts), batch_size)
            
            if show_progress:
                iterator = tqdm(iterator, desc="Generating embeddings")
            
            for i in iterator:
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings.tolist())
            
            logger.info("Embedding generation completed")
            return embeddings
            
        except Exception as e:
            raise EmbeddingError("Failed to generate embeddings", details=str(e))
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index chunks into ChromaDB (replaces existing data).
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        try:
            # Clear existing collection
            self._clear_collection()
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Prepare data for ChromaDB
            ids = [f"chunk_{chunk['row_index']}" for chunk in chunks]
            
            # ChromaDB metadata must be flat (no nested dicts)
            # Convert metadata to ChromaDB-compatible format
            metadatas = []
            for chunk in chunks:
                metadata = chunk["metadata"].copy()
                # Store original_data as JSON string (ChromaDB doesn't support nested dicts)
                if "original_data" in metadata:
                    metadata["original_data"] = json.dumps(metadata["original_data"])
                # Ensure all values are ChromaDB-compatible types
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        clean_metadata[key] = value
                    elif isinstance(value, list):
                        # Convert lists to comma-separated string
                        clean_metadata[key] = ",".join(str(v) for v in value)
                    else:
                        clean_metadata[key] = str(value)
                metadatas.append(clean_metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            return len(chunks)
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"ChromaDB indexing error: {e}", exc_info=True)
            raise VectorStoreError("Failed to index chunks", details=str(e))
    
    def _clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            if self._chroma_client:
                self._chroma_client.delete_collection(self.COLLECTION_NAME)
                self._collection = self._chroma_client.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "CSV/Excel data chunks"}
                )
                logger.info("Cleared existing collection")
        except Exception:
            # Collection might not exist, that's fine
            pass
    
    def get_collection_count(self) -> int:
        """Get the number of items in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def has_data(self) -> bool:
        """Check if collection has indexed data."""
        return self.get_collection_count() > 0
