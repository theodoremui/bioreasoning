"""
Summary Generation

This module provides LLM-based summary generation for community data,
following the Single Responsibility Principle and Dependency Inversion Principle.

Author: Theodore Mui
Date: 2025-08-24
"""

import re
from typing import Dict, List

from llama_index.core.llms import LLM, ChatMessage
from llama_index.llms.openai import OpenAI

from .constants import COMMUNITY_SUMMARY_SYSTEM_PROMPT
from .interfaces import ISummaryGenerator


class LLMSummaryGenerator(ISummaryGenerator):
    """LLM-based summary generator for community relationships.

    Uses a language model to generate coherent summaries of relationship
    data within graph communities. Follows DIP by depending on the LLM
    abstraction rather than concrete implementations.
    """

    def __init__(self, llm: LLM, system_prompt: str = COMMUNITY_SUMMARY_SYSTEM_PROMPT):
        """Initialize with LLM and system prompt.

        Args:
            llm: Language model for summary generation
            system_prompt: System prompt for guiding summary generation
        """
        self._llm = llm
        self._system_prompt = system_prompt

    def generate_summary(self, text: str) -> str:
        """Generate a summary for the given relationship text.

        Args:
            text: Text containing relationship descriptions

        Returns:
            Generated summary text

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate summary for empty text")

        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=text.strip()),
        ]

        try:
            response = self._llm.chat(messages)
            # Clean response by removing any "assistant:" prefix
            clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
            return clean_response
        except Exception as e:
            # Return a fallback summary if LLM fails
            return f"Summary generation failed: {str(e)}"

    def get_system_prompt(self) -> str:
        """Get the system prompt used for summary generation.

        Returns:
            System prompt text
        """
        return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Set a new system prompt.

        Args:
            prompt: New system prompt text

        Raises:
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("System prompt cannot be empty")
        self._system_prompt = prompt.strip()


class CommunitySummarizer:
    """High-level community summarization coordinator.

    Coordinates the summarization of multiple communities using a summary
    generator. Handles batch processing and error recovery.
    """

    def __init__(self, summary_generator: ISummaryGenerator):
        """Initialize with a summary generator.

        Args:
            summary_generator: Generator for creating summaries
        """
        self._generator = summary_generator

    def summarize_communities(
        self, community_info: Dict[int, List[Dict[str, any]]]
    ) -> Dict[int, str]:
        """Generate summaries for all communities.

        Args:
            community_info: Dictionary mapping community ID to relationship details

        Returns:
            Dictionary mapping community ID to summary text
        """
        summaries = {}

        for community_id, details in community_info.items():
            try:
                # Extract text from details (support both old and new formats)
                if details and isinstance(details[0], dict):
                    text_lines = [
                        d.get("detail", "") for d in details if d.get("detail")
                    ]
                else:
                    text_lines = [str(d) for d in details if str(d).strip()]

                if text_lines:
                    details_text = "\n".join(text_lines) + "."
                    summary = self._generator.generate_summary(details_text)
                    summaries[community_id] = summary
                else:
                    summaries[community_id] = "No relationship details available."

            except Exception as e:
                # Log error and provide fallback summary
                summaries[community_id] = (
                    f"Summary generation failed for community {community_id}: {str(e)}"
                )

        return summaries

    def summarize_single_community(
        self, community_id: int, details: List[Dict[str, any]]
    ) -> str:
        """Generate summary for a single community.

        Args:
            community_id: ID of the community
            details: List of relationship details

        Returns:
            Generated summary text
        """
        try:
            # Extract text from details
            if details and isinstance(details[0], dict):
                text_lines = [d.get("detail", "") for d in details if d.get("detail")]
            else:
                text_lines = [str(d) for d in details if str(d).strip()]

            if text_lines:
                details_text = "\n".join(text_lines) + "."
                return self._generator.generate_summary(details_text)
            else:
                return "No relationship details available."

        except Exception as e:
            return f"Summary generation failed for community {community_id}: {str(e)}"

    def get_generator_info(self) -> Dict[str, any]:
        """Get information about the summary generator.

        Returns:
            Dictionary with generator information
        """
        return {
            "generator_type": type(self._generator).__name__,
            "system_prompt": self._generator.get_system_prompt(),
        }


class SummaryGeneratorFactory:
    """Factory for creating summary generators.

    Implements the Factory Pattern to provide different summary generation
    strategies based on requirements and available resources.
    """

    @staticmethod
    def create_openai_generator(
        model: str = "gpt-4.1-mini",
        system_prompt: str = COMMUNITY_SUMMARY_SYSTEM_PROMPT,
    ) -> ISummaryGenerator:
        """Create an OpenAI-based summary generator.

        Args:
            model: OpenAI model to use
            system_prompt: System prompt for summary generation

        Returns:
            OpenAI summary generator instance
        """
        llm = OpenAI(model=model)
        return LLMSummaryGenerator(llm, system_prompt)

    @staticmethod
    def create_custom_generator(
        llm: LLM, system_prompt: str = COMMUNITY_SUMMARY_SYSTEM_PROMPT
    ) -> ISummaryGenerator:
        """Create a summary generator with custom LLM.

        Args:
            llm: Custom language model
            system_prompt: System prompt for summary generation

        Returns:
            Custom summary generator instance
        """
        return LLMSummaryGenerator(llm, system_prompt)
