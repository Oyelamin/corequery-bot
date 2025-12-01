"""
Data loading and preprocessing for CSV/Excel files.

Handles file loading, validation, preprocessing, and chunk creation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import settings
from src.utils.exceptions import DataLoadError, EmptyDataError, FileValidationError
from src.utils.logger import logger


class DataLoader:
    """
    Handles loading and preprocessing of CSV/Excel files.
    
    Provides methods for loading files, preprocessing data, and creating
    searchable text chunks for embedding.
    """
    
    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
    ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252"]
    
    def __init__(self, search_columns: Optional[List[str]] = None) -> None:
        """
        Initialize DataLoader.
        
        Args:
            search_columns: Specific columns to use for search. If None, uses all columns.
        """
        self._search_columns = search_columns or settings.data.search_columns
    
    def validate_file_path(self, file_path: str) -> Path:
        """
        Validate file path and check file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Validated Path object
            
        Raises:
            FileValidationError: If file is invalid or doesn't exist
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileValidationError(f"File not found: {path}")
        
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise FileValidationError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > settings.data.max_file_size_bytes:
            raise FileValidationError(
                f"File size ({file_size / (1024*1024):.1f}MB) exceeds "
                f"limit of {settings.data.max_file_size_mb}MB"
            )
        
        return path
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV or Excel file into DataFrame.
        
        Args:
            file_path: Path to the CSV or Excel file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            DataLoadError: If file loading fails
        """
        path = self.validate_file_path(file_path)
        logger.info(f"Loading file: {path.name}")
        
        try:
            if path.suffix.lower() == ".csv":
                df = self._load_csv(path)
            else:
                df = self._load_excel(path)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except FileValidationError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to load file: {path.name}", details=str(e))
    
    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV with encoding fallback and proper quoting."""
        for encoding in self.ENCODING_FALLBACKS:
            try:
                # Try standard CSV parsing first
                df = pd.read_csv(
                    path, 
                    encoding=encoding,
                    quotechar='"',
                    escapechar='\\'
                )
                # Reset index to ensure clean integer indices
                df = df.reset_index(drop=True)
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                # If parsing fails, try with more lenient settings
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        on_bad_lines='skip',
                        engine='python'
                    )
                    df = df.reset_index(drop=True)
                    return df
                except Exception as e:
                    if encoding == self.ENCODING_FALLBACKS[-1]:
                        # Last encoding attempt failed
                        raise DataLoadError(
                            "CSV parsing error. Ensure fields with commas are properly quoted with double quotes.",
                            details=str(e)
                        )
                    continue
        
        raise DataLoadError("Could not decode CSV file with any supported encoding")
    
    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(path)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataframe.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Preprocessing data...")
        
        # Work on copy
        df = df.copy()
        
        # Fill NaN and convert to string
        df = df.fillna("")
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        
        logger.info("Preprocessing completed")
        return df
    
    def create_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create searchable text chunks from dataframe rows.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of chunk dictionaries with text and metadata
            
        Raises:
            EmptyDataError: If no valid chunks can be created
        """
        logger.info("Creating text chunks...")
        
        columns = self._get_search_columns(df)
        chunks = []
        
        for idx, row in df.iterrows():
            # Ensure idx is an integer (handle any index type)
            row_idx = int(idx) if isinstance(idx, (int, float)) else len(chunks)
            
            text_parts = [
                f"{col}: {row[col]}"
                for col in columns
                if col in df.columns and str(row[col]).strip()
            ]
            
            if not text_parts:
                continue
            
            chunk = {
                "text": " | ".join(text_parts),
                "row_index": row_idx,
                "metadata": {
                    "row_index": row_idx,
                    "columns": columns,
                    "original_data": {k: str(v) for k, v in row.to_dict().items()}
                }
            }
            chunks.append(chunk)
        
        if not chunks:
            raise EmptyDataError("No valid data found. File appears to be empty.")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _get_search_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to use for search."""
        if self._search_columns:
            return [col for col in self._search_columns if col in df.columns]
        return df.columns.tolist()
    
    def load_and_process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Complete pipeline: load, preprocess, and create chunks.
        
        Args:
            file_path: Path to CSV or Excel file
            
        Returns:
            List of text chunks ready for embedding
        """
        df = self.load_file(file_path)
        df = self.preprocess(df)
        return self.create_chunks(df)
