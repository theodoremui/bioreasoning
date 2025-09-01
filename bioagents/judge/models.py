# ------------------------------------------------------------------------------
# bioagents/judge/models.py
#
# Pydantic models for structured response judgment and scoring.
# These models define the schema for evaluating agent responses across
# multiple criteria with justifications and overall scores.
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

from typing import Dict
from pydantic import BaseModel, Field


class JudgmentScores(BaseModel):
    """Structured scores for evaluating agent response quality.
    
    All scores are normalized between 0.0 (lowest) and 1.0 (highest).
    These criteria provide comprehensive evaluation across multiple dimensions
    of response quality for multi-agent systems.
    """
    
    accuracy: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How factually accurate is the response? Evaluates correctness of information."
    )
    completeness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How complete is the response? Evaluates coverage of the query requirements."
    )
    groundedness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How well-grounded is the response in evidence? Evaluates citation quality and support."
    )
    professional_tone: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How professional and appropriate is the tone? Evaluates communication quality."
    )
    clarity_coherence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How clear and coherent is the response? Evaluates readability and flow."
    )
    relevance: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How relevant is the response to the query? Evaluates topical alignment."
    )
    usefulness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How useful is the response to the user? Evaluates practical value."
    )

    def get_average_score(self) -> float:
        """Calculate the average score across all criteria with equal weighting."""
        scores = [
            self.accuracy, self.completeness, self.groundedness,
            self.professional_tone, self.clarity_coherence, 
            self.relevance, self.usefulness
        ]
        return sum(scores) / len(scores)

    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score using custom weights for each criterion.
        
        Args:
            weights: Dictionary mapping criterion names to weight values.
                    Weights should sum to 1.0 for proper normalization.
                    
        Returns:
            Weighted average score between 0.0 and 1.0.
            
        Example:
            weights = {
                'accuracy': 0.25,
                'groundedness': 0.20,
                'relevance': 0.20,
                'usefulness': 0.15,
                'completeness': 0.10,
                'clarity_coherence': 0.05,
                'professional_tone': 0.05
            }
        """
        weighted_sum = (
            self.accuracy * weights.get('accuracy', 0.0) +
            self.completeness * weights.get('completeness', 0.0) +
            self.groundedness * weights.get('groundedness', 0.0) +
            self.professional_tone * weights.get('professional_tone', 0.0) +
            self.clarity_coherence * weights.get('clarity_coherence', 0.0) +
            self.relevance * weights.get('relevance', 0.0) +
            self.usefulness * weights.get('usefulness', 0.0)
        )
        return max(0.0, min(1.0, weighted_sum))


class JudgmentJustifications(BaseModel):
    """Textual justifications explaining the reasoning behind each judgment score.
    
    Provides transparency and interpretability for automated scoring decisions.
    Each justification should be concise but informative about the scoring rationale.
    """
    
    accuracy: str = Field(
        ..., 
        description="Explanation for the accuracy score - why this level of factual correctness"
    )
    completeness: str = Field(
        ..., 
        description="Explanation for the completeness score - what was covered/missing"
    )
    groundedness: str = Field(
        ..., 
        description="Explanation for the groundedness score - quality of evidence/citations"
    )
    professional_tone: str = Field(
        ..., 
        description="Explanation for the professional tone score - appropriateness of communication"
    )
    clarity_coherence: str = Field(
        ..., 
        description="Explanation for the clarity score - readability and logical flow"
    )
    relevance: str = Field(
        ..., 
        description="Explanation for the relevance score - alignment with query intent"
    )
    usefulness: str = Field(
        ..., 
        description="Explanation for the usefulness score - practical value to user"
    )
    
    def __str__(self) -> str:
        judge_lines = [
            f"Accuracy: {self.accuracy}",
            f"Completeness: {self.completeness}",
            f"Groundedness: {self.groundedness}",
            f"Professional Tone: {self.professional_tone}",
            f"Clarity & Coherence: {self.clarity_coherence}",
            f"Relevance: {self.relevance}",
            f"Usefulness: {self.usefulness}"
        ]
        return ", ".join(judge_lines)

class AgentJudgment(BaseModel):
    """Complete structured judgment of a single agent's response quality.
    
    Combines numerical scores with textual justifications and an overall assessment.
    Used for evaluating individual agent contributions in multi-agent systems.
    """

    agent_name: str = Field(
        "",
        description="Name of the agent being evaluated"
    )
    response_str: str = Field(
        "",
        description="The response text being evaluated"
    )
    prose_summary: str = Field(
        ..., 
        description="2-3 sentence summary highlighting key strengths and improvement areas"
    )
    scores: JudgmentScores = Field(
        ..., 
        description="Numerical scores across all evaluation criteria"
    )
    overall_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall weighted score representing response quality"
    )
    justifications: JudgmentJustifications = Field(
        ..., 
        description="Detailed explanations for each individual score"
    )

    class Config:
        extra = "forbid"  # Strict validation - no extra fields allowed
        
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if this judgment represents a high-quality response.
        
        Args:
            threshold: Minimum score threshold for high quality (default 0.8).
            
        Returns:
            True if overall_score >= threshold.
        """
        return self.overall_score >= threshold
        
    def get_improvement_areas(self, threshold: float = 0.6) -> Dict[str, float]:
        """Identify criteria that scored below the improvement threshold.
        
        Args:
            threshold: Minimum acceptable score (default 0.6).
            
        Returns:
            Dictionary of criteria names and their scores that need improvement.
        """
        improvement_areas = {}
        scores_dict = {
            'accuracy': self.scores.accuracy,
            'completeness': self.scores.completeness,
            'groundedness': self.scores.groundedness,
            'professional_tone': self.scores.professional_tone,
            'clarity_coherence': self.scores.clarity_coherence,
            'relevance': self.scores.relevance,
            'usefulness': self.scores.usefulness
        }
        
        for criterion, score in scores_dict.items():
            if score < threshold:
                improvement_areas[criterion] = score
                
        return improvement_areas
    
    def __str__(self) -> str:
        """Convert the judgment to a string representation."""
        judge_lines = [
            f"**Score**: {self.overall_score:.2f}",
            f"**Assessment**: {self.prose_summary}",
            f"- {self.agent_name}:",
            f"  - Response: {len(self.response_str)} chars",
            f"  - Score: {self.overall_score:.2f}",
            f"  - Justification: {self.prose_summary} ({str(self.justifications)})"
        ]

        return "\n".join(judge_lines)


class HALOJudgmentSummary(BaseModel):
    """Summary of all subagent judgments for multi-agent orchestration systems.
    
    Aggregates individual agent judgments to provide system-wide evaluation
    and synthesis guidance for hierarchical orchestration (e.g., HALO systems).
    """
    
    individual_judgments: Dict[str, AgentJudgment] = Field(
        ..., 
        description="Judgment for each capability/agent (e.g., 'graph', 'llama_rag', 'biomcp')"
    )
    overall_halo_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall system score averaged across all agents"
    )
    synthesis_notes: str = Field(
        ..., 
        description="Qualitative assessment and guidance for response synthesis"
    )
    
    class Config:
        extra = "forbid"

    def get_top_performers(self, count: int = 2) -> Dict[str, AgentJudgment]:
        """Get the top-performing agents by overall score.
        
        Args:
            count: Number of top performers to return.
            
        Returns:
            Dictionary of top agent names and their judgments.
        """
        sorted_agents = sorted(
            self.individual_judgments.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        return dict(sorted_agents[:count])

    def get_low_performers(self, threshold: float = 0.5) -> Dict[str, AgentJudgment]:
        """Get agents that performed below the specified threshold.
        
        Args:
            threshold: Minimum acceptable score (default 0.5).
            
        Returns:
            Dictionary of underperforming agent names and their judgments.
        """
        return {
            name: judgment 
            for name, judgment in self.individual_judgments.items()
            if judgment.overall_score < threshold
        }

    def get_score_distribution(self) -> Dict[str, float]:
        """Get statistical distribution of agent scores.
        
        Returns:
            Dictionary with min, max, mean, and median scores.
        """
        scores = [j.overall_score for j in self.individual_judgments.values()]
        if not scores:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
            
        scores.sort()
        n = len(scores)
        median = scores[n//2] if n % 2 == 1 else (scores[n//2-1] + scores[n//2]) / 2
        
        return {
            'min': min(scores),
            'max': max(scores),
            'mean': sum(scores) / len(scores),
            'median': median
        }
