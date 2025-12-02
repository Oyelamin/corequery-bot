"""
Session manager for conversation history.

Manages conversation sessions and maintains chat history per session ID.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from src.config import settings
from src.utils.logger import logger


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """Represents a conversation session with history."""
    session_id: str
    messages: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=20))  # (role, content)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append((role, content))
        self.last_activity = time.time()
        logger.debug(f"Added {role} message to session {self.session_id}")
    
    def get_history(self, max_messages: int = 10) -> list:
        """
        Get conversation history formatted for LLM.
        
        Args:
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of message dictionaries with role and content
        """
        recent_messages = list(self.messages)[-max_messages:]
        return [
            {"role": role, "content": content}
            for role, content in recent_messages
        ]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        logger.info(f"Cleared history for session {self.session_id}")


class SessionManager:
    """Manages conversation sessions and their history."""
    
    def __init__(self, max_sessions: int = 1000, session_timeout: int = 3600):
        """
        Initialize SessionManager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout: Session timeout in seconds (default 1 hour)
        """
        self._sessions: Dict[str, ConversationSession] = {}
        self._max_sessions = max_sessions
        self._session_timeout = session_timeout
    
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationSession object
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Update last activity
            session.last_activity = time.time()
            return session
        
        # Create new session
        if len(self._sessions) >= self._max_sessions:
            # Remove oldest inactive session
            self._cleanup_inactive_sessions()
        
        session = ConversationSession(session_id=session_id)
        self._sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession or None if not found
        """
        return self._sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
        """
        session = self.get_or_create_session(session_id)
        session.add_message(role, content)
    
    def get_history(self, session_id: str, max_messages: int = 10) -> list:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_history(max_messages)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was found and cleared, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.clear_history()
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def _cleanup_inactive_sessions(self) -> None:
        """Remove inactive sessions that have timed out."""
        current_time = time.time()
        inactive_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if current_time - session.last_activity > self._session_timeout
        ]
        
        for session_id in inactive_sessions:
            self.delete_session(session_id)
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
    
    def get_active_sessions_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()

