"""
LLM client for Ollama integration.

Provides a client for communicating with the Ollama API.
"""

import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from src.config import settings
from src.utils.exceptions import LLMConnectionError, LLMGenerationError, LLMTimeoutError
from src.utils.logger import logger


class LLMClient:
    """
    Client for interacting with Ollama LLM.
    
    Provides methods for health checks, model verification, and response generation.
    """
    
    NOT_FOUND_MESSAGE = (
        "I'm unable to process this request as the information "
        "is not available in my knowledge base."
    )
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> None:
        """
        Initialize LLM client.
        
        Args:
            model: Ollama model name (default from config)
            base_url: Ollama API base URL (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        self._model = model or settings.llm.model
        self._base_url = base_url or settings.llm.base_url
        self._timeout = timeout or settings.llm.timeout
    
    @property
    def model(self) -> str:
        """Get the configured model name."""
        return self._model
    
    def check_health(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is available
        """
        try:
            response = requests.get(
                urljoin(self._base_url, "/api/tags"),
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """Get list of available models."""
        try:
            response = requests.get(
                urljoin(self._base_url, "/api/tags"),
                timeout=10
            )
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except requests.RequestException:
            pass
        return []
    
    def is_model_available(self) -> bool:
        """Check if configured model is available."""
        models = self.list_models()
        # Check for exact match or model without tag
        return self._model in models or any(
            m.startswith(f"{self._model}:") for m in models
        )
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""Based on the following information from the knowledge base:

        {context}
        
        User's question: {query}
        
        Instructions:
        - Answer using ONLY the information provided above
        - If the information doesn't contain the answer, say: "{self.NOT_FOUND_MESSAGE}"
        - Be concise and accurate
        - Synthesize information from multiple matches if relevant
        """
    
    def generate(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Args:
            query: User query
            context: Retrieved context from vector search
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            LLMConnectionError: If Ollama is not accessible
            LLMTimeoutError: If request times out
            LLMGenerationError: If generation fails
        """
        if not self.check_health():
            raise LLMConnectionError(
                "Ollama is not running",
                details="Please start Ollama with 'ollama serve'"
            )
        
        start_time = time.perf_counter()
        prompt = self._build_prompt(query, context)
        
        try:
            response = requests.post(
                urljoin(self._base_url, "/api/generate"),
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self._timeout
            )
            
            if response.status_code != 200:
                raise LLMGenerationError(
                    f"Ollama API error: {response.status_code}",
                    details=response.text
                )
            
            result = response.json()
            response_text = result.get("response", "").strip()
            generation_time = time.perf_counter() - start_time
            
            logger.info(f"LLM response generated in {generation_time:.2f}s")
            
            return {
                "response": response_text,
                "model": self._model,
                "generation_time": generation_time,
                "tokens_used": result.get("eval_count", 0)
            }
            
        except requests.Timeout:
            raise LLMTimeoutError(
                f"Request timed out after {self._timeout}s",
                details="Consider increasing LLM_TIMEOUT or using a faster model"
            )
        except requests.RequestException as e:
            raise LLMGenerationError("Failed to generate response", details=str(e))
    
    def get_not_found_response(self) -> str:
        """Get the standard 'not found' response message."""
        return self.NOT_FOUND_MESSAGE
