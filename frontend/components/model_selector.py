"""
Model Selector Component for the BioReasoning application.

This module provides a reusable UI component for selecting different LLM models,
following the Single Responsibility Principle by handling only model selection logic.

Author: Theodore Mui
Date: 2025-04-26
"""

import streamlit as st
import sys
import os
from typing import Dict, Any

# Add project root to Python path for bioagents import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from bioagents.models.llms import LLM


class ModelSelector:
    """
    A component responsible for handling model selection in the Streamlit interface.
    
    This class encapsulates the model selection logic and provides a clean interface
    for integrating model selection into different pages of the application.
    """
    
    def __init__(self):
        """Initialize the ModelSelector with available model options."""
        self._model_options = {
            "GPT-4.1 Mini": LLM.GPT_4_1_MINI,
            "GPT-4.1 Nano": LLM.GPT_4_1_NANO,
            "GPT-4.1": LLM.GPT_4_1,
            "GPT-4o": LLM.GPT_4O
        }
        self._default_model_index = 1  # Default to GPT-4.1 Nano
    
    def render(self, container=None) -> str:
        """
        Render the model selection UI component.
        
        Args:
            container: Optional Streamlit container to render within
            
        Returns:
            str: The selected model identifier
        """
        if container:
            with container:
                return self._render_content()
        else:
            return self._render_content()
    
    def _render_content(self) -> str:
        """
        Render the actual model selection content.
        
        Returns:
            str: The selected model identifier
        """        
        model_selection = st.selectbox(
            "Select LLM Model",
            list(self._model_options.keys()),
            index=self._default_model_index,
            help="Choose the language model for processing your requests"
        )
        
        selected_model = self._model_options[model_selection]
        self._update_session_model(selected_model)
        
        return selected_model
    
    def _update_session_model(self, model: str) -> None:
        """
        Update the model in the session state if it has changed.
        
        Args:
            model: The selected model identifier
        """
        if "llm_client" in st.session_state:
            if st.session_state.llm_client._model_name != model:
                st.session_state.llm_client._model_name = model
    
    def get_model_options(self) -> Dict[str, str]:
        """
        Get the available model options.
        
        Returns:
            Dict[str, str]: Dictionary mapping display names to model identifiers
        """
        return self._model_options.copy()
    
    def get_current_model(self) -> str:
        """
        Get the currently selected model from session state.
        
        Returns:
            str: Current model identifier, or default if not set
        """
        if "llm_client" in st.session_state:
            return st.session_state.llm_client._model_name
        return self._model_options[list(self._model_options.keys())[self._default_model_index]] 