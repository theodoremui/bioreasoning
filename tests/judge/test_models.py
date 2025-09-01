# ------------------------------------------------------------------------------
# tests/judge/test_models.py
#
# Comprehensive test suite for judgment models (Pydantic).
# Tests all model functionality, validation, and utility methods.
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

import pytest
from pydantic import ValidationError
from typing import Dict

from bioagents.judge.models import (
    JudgmentScores,
    JudgmentJustifications, 
    AgentJudgment,
    HALOJudgmentSummary
)


class TestJudgmentScores:
    """Test the JudgmentScores Pydantic model."""
    
    def test_valid_scores(self):
        """Test creation with valid scores."""
        scores = JudgmentScores(
            accuracy=0.8,
            completeness=0.7,
            groundedness=0.9,
            professional_tone=0.8,
            clarity_coherence=0.8,
            relevance=0.9,
            usefulness=0.8
        )
        
        assert scores.accuracy == 0.8
        assert scores.groundedness == 0.9
        assert scores.usefulness == 0.8
    
    def test_score_validation_boundaries(self):
        """Test that scores are validated within [0.0, 1.0] range."""
        # Test lower boundary
        scores = JudgmentScores(
            accuracy=0.0, completeness=0.0, groundedness=0.0,
            professional_tone=0.0, clarity_coherence=0.0,
            relevance=0.0, usefulness=0.0
        )
        assert all(getattr(scores, field) == 0.0 for field in scores.__fields__)
        
        # Test upper boundary
        scores = JudgmentScores(
            accuracy=1.0, completeness=1.0, groundedness=1.0,
            professional_tone=1.0, clarity_coherence=1.0,
            relevance=1.0, usefulness=1.0
        )
        assert all(getattr(scores, field) == 1.0 for field in scores.__fields__)
    
    def test_invalid_scores_below_zero(self):
        """Test that scores below 0.0 are rejected."""
        with pytest.raises(ValidationError):
            JudgmentScores(
                accuracy=-0.1, completeness=0.5, groundedness=0.5,
                professional_tone=0.5, clarity_coherence=0.5,
                relevance=0.5, usefulness=0.5
            )
    
    def test_invalid_scores_above_one(self):
        """Test that scores above 1.0 are rejected."""
        with pytest.raises(ValidationError):
            JudgmentScores(
                accuracy=1.1, completeness=0.5, groundedness=0.5,
                professional_tone=0.5, clarity_coherence=0.5,
                relevance=0.5, usefulness=0.5
            )
    
    def test_get_average_score(self):
        """Test the get_average_score method."""
        scores = JudgmentScores(
            accuracy=1.0, completeness=0.8, groundedness=0.6,
            professional_tone=0.4, clarity_coherence=0.2,
            relevance=0.0, usefulness=0.7
        )
        
        expected_avg = (1.0 + 0.8 + 0.6 + 0.4 + 0.2 + 0.0 + 0.7) / 7
        assert abs(scores.get_average_score() - expected_avg) < 1e-10
    
    def test_get_weighted_score(self):
        """Test the get_weighted_score method."""
        scores = JudgmentScores(
            accuracy=0.8, completeness=0.7, groundedness=0.9,
            professional_tone=0.6, clarity_coherence=0.7,
            relevance=0.8, usefulness=0.7
        )
        
        # Test equal weights (should match average)
        equal_weights = {field: 1/7 for field in scores.__fields__}
        weighted_equal = scores.get_weighted_score(equal_weights)
        average = scores.get_average_score()
        assert abs(weighted_equal - average) < 1e-10
        
        # Test custom weights
        custom_weights = {
            'accuracy': 0.3, 'groundedness': 0.3, 'relevance': 0.2,
            'usefulness': 0.1, 'completeness': 0.05,
            'clarity_coherence': 0.03, 'professional_tone': 0.02
        }
        
        expected_weighted = (
            0.8 * 0.3 + 0.9 * 0.3 + 0.8 * 0.2 + 0.7 * 0.1 + 
            0.7 * 0.05 + 0.7 * 0.03 + 0.6 * 0.02
        )
        weighted_custom = scores.get_weighted_score(custom_weights)
        assert abs(weighted_custom - expected_weighted) < 1e-10
    
    def test_weighted_score_clamping(self):
        """Test that weighted scores are clamped to [0.0, 1.0]."""
        scores = JudgmentScores(
            accuracy=1.0, completeness=1.0, groundedness=1.0,
            professional_tone=1.0, clarity_coherence=1.0,
            relevance=1.0, usefulness=1.0
        )
        
        # Weights that sum to more than 1.0
        over_weights = {field: 0.2 for field in scores.__fields__}  # Sum = 1.4
        weighted = scores.get_weighted_score(over_weights)
        assert weighted == 1.0  # Should be clamped to 1.0


class TestJudgmentJustifications:
    """Test the JudgmentJustifications Pydantic model."""
    
    def test_valid_justifications(self):
        """Test creation with valid justifications."""
        justifications = JudgmentJustifications(
            accuracy="Response is factually correct",
            completeness="Covers all key points",
            groundedness="Strong citation support",
            professional_tone="Appropriate and respectful",
            clarity_coherence="Clear logical flow",
            relevance="Directly addresses the query",
            usefulness="Provides actionable information"
        )
        
        assert "factually correct" in justifications.accuracy
        assert "citation support" in justifications.groundedness
        assert "actionable" in justifications.usefulness
    
    def test_empty_justifications(self):
        """Test creation with empty justifications."""
        justifications = JudgmentJustifications(
            accuracy="", completeness="", groundedness="",
            professional_tone="", clarity_coherence="",
            relevance="", usefulness=""
        )
        
        assert justifications.accuracy == ""
        assert justifications.completeness == ""
    
    def test_missing_fields_validation(self):
        """Test that all justification fields are required."""
        with pytest.raises(ValidationError):
            JudgmentJustifications(
                accuracy="Good",
                completeness="Good",
                # Missing other required fields
            )


class TestAgentJudgment:
    """Test the AgentJudgment Pydantic model."""
    
    @pytest.fixture
    def valid_judgment(self):
        """Fixture providing a valid AgentJudgment."""
        scores = JudgmentScores(
            accuracy=0.8, completeness=0.7, groundedness=0.9,
            professional_tone=0.8, clarity_coherence=0.8,
            relevance=0.9, usefulness=0.8
        )
        justifications = JudgmentJustifications(
            accuracy="High accuracy", completeness="Good coverage",
            groundedness="Strong evidence", professional_tone="Appropriate",
            clarity_coherence="Clear", relevance="Very relevant",
            usefulness="Highly useful"
        )
        
        return AgentJudgment(
            agent_name="test",
            response_str="text",
            prose_summary="Excellent response with good coverage and citations.",
            scores=scores,
            overall_score=0.81,
            justifications=justifications
        )
    
    def test_valid_judgment(self, valid_judgment):
        """Test creation of valid judgment."""
        assert valid_judgment.overall_score == 0.81
        assert "Excellent response" in valid_judgment.prose_summary
        assert valid_judgment.scores.accuracy == 0.8
        assert valid_judgment.justifications.groundedness == "Strong evidence"
    
    def test_is_high_quality(self, valid_judgment):
        """Test the is_high_quality method."""
        # Default threshold (0.8)
        assert valid_judgment.is_high_quality() == True
        
        # Custom threshold
        assert valid_judgment.is_high_quality(threshold=0.7) == True
        assert valid_judgment.is_high_quality(threshold=0.9) == False
        
        # Edge case - exactly at threshold
        valid_judgment.overall_score = 0.8
        assert valid_judgment.is_high_quality(threshold=0.8) == True
    
    def test_get_improvement_areas(self, valid_judgment):
        """Test the get_improvement_areas method."""
        # With default threshold (0.6), only completeness should need improvement
        improvement = valid_judgment.get_improvement_areas()
        assert len(improvement) == 0  # All scores are above 0.6
        
        # With higher threshold (0.8), some areas should need improvement
        improvement = valid_judgment.get_improvement_areas(threshold=0.8)
        assert 'completeness' in improvement  # 0.7 < 0.8
        assert 'accuracy' not in improvement  # 0.8 >= 0.8
        assert 'groundedness' not in improvement  # 0.9 >= 0.8
        
        # Verify returned scores are correct
        assert improvement['completeness'] == 0.7
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        scores = JudgmentScores(
            accuracy=0.8, completeness=0.7, groundedness=0.9,
            professional_tone=0.8, clarity_coherence=0.8,
            relevance=0.9, usefulness=0.8
        )
        justifications = JudgmentJustifications(
            accuracy="Good", completeness="Good", groundedness="Good",
            professional_tone="Good", clarity_coherence="Good",
            relevance="Good", usefulness="Good"
        )
        
        with pytest.raises(ValidationError):
            AgentJudgment(
                prose_summary="Test",
                scores=scores,
                overall_score=0.8,
                justifications=justifications,
                extra_field="This should not be allowed"
            )
    
    def test_overall_score_validation(self):
        """Test that overall_score is validated within [0.0, 1.0]."""
        scores = JudgmentScores(
            accuracy=0.8, completeness=0.7, groundedness=0.9,
            professional_tone=0.8, clarity_coherence=0.8,
            relevance=0.9, usefulness=0.8
        )
        justifications = JudgmentJustifications(
            accuracy="Good", completeness="Good", groundedness="Good",
            professional_tone="Good", clarity_coherence="Good",
            relevance="Good", usefulness="Good"
        )
        
        # Test invalid scores
        with pytest.raises(ValidationError):
            AgentJudgment(
                prose_summary="Test",
                scores=scores,
                overall_score=-0.1,  # Below 0
                justifications=justifications
            )
        
        with pytest.raises(ValidationError):
            AgentJudgment(
                prose_summary="Test",
                scores=scores,
                overall_score=1.1,  # Above 1
                justifications=justifications
            )


class TestHALOJudgmentSummary:
    """Test the HALOJudgmentSummary Pydantic model."""
    
    @pytest.fixture
    def sample_judgments(self):
        """Fixture providing sample agent judgments."""
        scores_high = JudgmentScores(
            accuracy=0.9, completeness=0.8, groundedness=0.9,
            professional_tone=0.9, clarity_coherence=0.9,
            relevance=0.9, usefulness=0.9
        )
        scores_med = JudgmentScores(
            accuracy=0.7, completeness=0.7, groundedness=0.7,
            professional_tone=0.7, clarity_coherence=0.7,
            relevance=0.7, usefulness=0.7
        )
        scores_low = JudgmentScores(
            accuracy=0.4, completeness=0.4, groundedness=0.4,
            professional_tone=0.4, clarity_coherence=0.4,
            relevance=0.4, usefulness=0.4
        )
        
        justifications = JudgmentJustifications(
            accuracy="Test", completeness="Test", groundedness="Test",
            professional_tone="Test", clarity_coherence="Test",
            relevance="Test", usefulness="Test"
        )
        
        return {
            "high_agent": AgentJudgment(
                agent_name="high_agent", response_str="r",
                prose_summary="Excellent", scores=scores_high,
                overall_score=0.89, justifications=justifications
            ),
            "med_agent": AgentJudgment(
                agent_name="med_agent", response_str="r",
                prose_summary="Good", scores=scores_med,
                overall_score=0.7, justifications=justifications
            ),
            "low_agent": AgentJudgment(
                agent_name="low_agent", response_str="r",
                prose_summary="Poor", scores=scores_low,
                overall_score=0.4, justifications=justifications
            )
        }
    
    def test_valid_summary(self, sample_judgments):
        """Test creation of valid HALO judgment summary."""
        summary = HALOJudgmentSummary(
            individual_judgments=sample_judgments,
            overall_halo_score=0.66,
            synthesis_notes="Mixed performance across agents"
        )
        
        assert len(summary.individual_judgments) == 3
        assert summary.overall_halo_score == 0.66
        assert "Mixed performance" in summary.synthesis_notes
    
    def test_get_top_performers(self, sample_judgments):
        """Test the get_top_performers method."""
        summary = HALOJudgmentSummary(
            individual_judgments=sample_judgments,
            overall_halo_score=0.66,
            synthesis_notes="Test"
        )
        
        # Get top 2 performers
        top_2 = summary.get_top_performers(count=2)
        assert len(top_2) == 2
        assert "high_agent" in top_2
        assert "med_agent" in top_2
        assert "low_agent" not in top_2
        
        # Get top 1 performer
        top_1 = summary.get_top_performers(count=1)
        assert len(top_1) == 1
        assert "high_agent" in top_1
        
        # Request more than available
        top_all = summary.get_top_performers(count=10)
        assert len(top_all) == 3  # Should return all available
    
    def test_get_low_performers(self, sample_judgments):
        """Test the get_low_performers method."""
        summary = HALOJudgmentSummary(
            individual_judgments=sample_judgments,
            overall_halo_score=0.66,
            synthesis_notes="Test"
        )
        
        # Default threshold (0.5)
        low_performers = summary.get_low_performers()
        assert len(low_performers) == 1
        assert "low_agent" in low_performers
        
        # Custom threshold (0.8)
        low_performers_high_thresh = summary.get_low_performers(threshold=0.8)
        assert len(low_performers_high_thresh) == 2
        assert "med_agent" in low_performers_high_thresh
        assert "low_agent" in low_performers_high_thresh
        assert "high_agent" not in low_performers_high_thresh
        
        # Very low threshold
        low_performers_low_thresh = summary.get_low_performers(threshold=0.1)
        assert len(low_performers_low_thresh) == 0
    
    def test_get_score_distribution(self, sample_judgments):
        """Test the get_score_distribution method."""
        summary = HALOJudgmentSummary(
            individual_judgments=sample_judgments,
            overall_halo_score=0.66,
            synthesis_notes="Test"
        )
        
        distribution = summary.get_score_distribution()
        
        assert distribution['min'] == 0.4
        assert distribution['max'] == 0.89
        assert abs(distribution['mean'] - 0.663) < 0.01  # (0.89 + 0.7 + 0.4) / 3
        assert distribution['median'] == 0.7  # Middle value
    
    def test_get_score_distribution_empty(self):
        """Test score distribution with no judgments."""
        summary = HALOJudgmentSummary(
            individual_judgments={},
            overall_halo_score=0.0,
            synthesis_notes="No agents"
        )
        
        distribution = summary.get_score_distribution()
        assert distribution['min'] == 0.0
        assert distribution['max'] == 0.0
        assert distribution['mean'] == 0.0
        assert distribution['median'] == 0.0
    
    def test_get_score_distribution_single_judgment(self, sample_judgments):
        """Test score distribution with single judgment."""
        single_judgment = {"only_agent": sample_judgments["high_agent"]}
        
        summary = HALOJudgmentSummary(
            individual_judgments=single_judgment,
            overall_halo_score=0.89,
            synthesis_notes="Single agent"
        )
        
        distribution = summary.get_score_distribution()
        assert distribution['min'] == 0.89
        assert distribution['max'] == 0.89
        assert distribution['mean'] == 0.89
        assert distribution['median'] == 0.89
    
    def test_extra_fields_forbidden(self, sample_judgments):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            HALOJudgmentSummary(
                individual_judgments=sample_judgments,
                overall_halo_score=0.66,
                synthesis_notes="Test",
                extra_field="Not allowed"
            )
    
    def test_overall_score_validation(self, sample_judgments):
        """Test overall HALO score validation."""
        with pytest.raises(ValidationError):
            HALOJudgmentSummary(
                individual_judgments=sample_judgments,
                overall_halo_score=-0.1,  # Below 0
                synthesis_notes="Test"
            )
        
        with pytest.raises(ValidationError):
            HALOJudgmentSummary(
                individual_judgments=sample_judgments,
                overall_halo_score=1.1,  # Above 1
                synthesis_notes="Test"
            )
