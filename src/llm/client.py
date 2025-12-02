"""
LLM client for Ollama integration.

Provides a client for communicating with the Ollama API.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
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
    
    def _build_prompt(self, query: str, context: str, conversation_history: Optional[list] = None) -> str:
        """
        Build the prompt for the LLM with optional conversation history.
        
        Args:
            query: Current user query
            context: Retrieved context from knowledge base
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content')
        """
        # Build conversation history section if available
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_lines = []
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"You: {content}")
            
            if history_lines:
                history_section = f"""
PREVIOUS CONVERSATION HISTORY (for context):
{chr(10).join(history_lines)}

---
CURRENT QUESTION (respond to this):
"""
        
        # Adjust instructions based on whether there's history
        follow_up_instructions = ""
        if conversation_history and len(conversation_history) > 0:
            follow_up_instructions = """
IMPORTANT FOR FOLLOW-UP CONVERSATIONS:
- This is a follow-up question in an ongoing conversation
- Reference previous messages naturally using pronouns (e.g., "As I mentioned earlier...", "That process I described...")
- If the user asks "What about...?" or "How about...?", they're likely referring to something from the previous conversation
- Build on previous answers - don't repeat everything, just add to what was already discussed
- Use natural transitions like "Sure!", "Absolutely!", "Great follow-up question!", "To add to what I said..."
- If the current question relates to a previous topic, acknowledge the connection naturally
"""
        
        return f"""You are a friendly and helpful customer support agent for the GATEWAY program platform. Your goal is to assist users with their questions in a warm, conversational, and supportive manner.

{history_section}Here is the relevant information from our knowledge base:

{context}

User's current question: {query}

{follow_up_instructions}
Instructions:
- Answer the user's question using ONLY the information provided above from the knowledge base
- Write in a friendly, conversational tone as if you're a helpful support agent
- Use a warm and approachable voice (like talking to a friend)
- Be encouraging and positive
- If this is a follow-up question, reference the previous conversation naturally using pronouns and context
- If the information doesn't contain the answer, respond with: "{self.NOT_FOUND_MESSAGE}"
- Keep responses concise but friendly
- If multiple pieces of information are relevant, synthesize them naturally
- Use emojis sparingly if appropriate (😊, ✅, etc.)
- Start with a friendly greeting or acknowledgment when appropriate

Example responses:
- First question: "Great question! To create an account, you can click 'Sign Up' on the homepage..."
- Follow-up: "Sure! After you create the account, you'll need to complete the computer literacy test..."
- Follow-up with reference: "Yes, that's the one I mentioned! The test takes about 10 minutes and..."

Remember: Be helpful, friendly, and make the user feel supported! Maintain natural conversation flow and use context from previous messages when relevant.
"""
    
    def generate(self, query: str, context: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
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
