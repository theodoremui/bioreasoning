# ------------------------------------------------------------------------------
# bioagents/judge/interfaces.py
#
# Abstract interfaces for response judging systems.
# Defines contracts for response evaluation to enable different implementations
# while maintaining consistent APIs across the system.
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from bioagents.agents.common import AgentResponse
from .models import AgentJudgment


class ResponseJudgeInterface(ABC):
    """Abstract interface for response judgment systems.
    
    Defines the contract that all response judges must implement.
    Follows Interface Segregation Principle by providing focused,
    single-purpose methods for response evaluation.
    """
    
    @abstractmethod
    async def judge_response(
        self, 
        capability: str, 
        response: AgentResponse, 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentJudgment:
        """Judge a single agent response using structured evaluation criteria.
        
        Args:
            capability: The capability/agent type being evaluated 
                       (e.g., 'graph', 'llama_rag', 'biomcp', 'web').
            response: The agent response to evaluate.
            query: The original user query for relevance assessment.
            context: Optional context information for specialized judging.
            
        Returns:
            Structured judgment with scores, justifications, and summary.
            
        Raises:
            JudgmentError: When evaluation fails critically.
        """
        pass

    @abstractmethod
    def create_fallback_judgment(
        self, 
        capability: str, 
        response: AgentResponse, 
        query: str,
        reason: str = "Evaluation system unavailable"
    ) -> AgentJudgment:
        """Create a fallback judgment when primary evaluation fails.
        
        Args:
            capability: The capability/agent type being evaluated.
            response: The agent response to evaluate.
            query: The original user query.
            reason: Reason for fallback (for transparency).
            
        Returns:
            Basic judgment using heuristic scoring methods.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the underlying model name used for judgment."""
        pass

    @property  
    @abstractmethod
    def timeout(self) -> int:
        """Get the timeout value for judgment operations."""
        pass


class JudgmentError(Exception):
    """Custom exception for response judgment failures.
    
    Used when the judgment system encounters critical errors
    that prevent evaluation from proceeding.
    """
    
    def __init__(self, message: str, capability: str = "", original_error: Optional[Exception] = None):
        """Initialize judgment error with context.
        
        Args:
            message: Human-readable error description.
            capability: The capability being judged when error occurred.
            original_error: The underlying exception that caused this error.
        """
        self.capability = capability
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.capability:
            base_msg = f"[{self.capability}] {base_msg}"
        if self.original_error:
            base_msg = f"{base_msg} (caused by: {self.original_error})"
        return base_msg
