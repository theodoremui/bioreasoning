"""
Session Manager Component for the BioReasoning application.

This module provides centralized session state management, following the Single
Responsibility Principle by handling only session-related operations.

Author: Theodore Mui
Date: 2025-04-26
"""

import streamlit as st
import sys
import os
from typing import Any, Optional

# Add project root to Python path for bioagents import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from bioagents.models.llms import LLM
from bioagents.agents.bio_concierge import BioConciergeAgent


class SessionManager:
    """
    A component responsible for managing session state across the application.
    
    This class encapsulates session state initialization and management logic,
    ensuring consistent state management across all pages of the application.
    """
    
    @staticmethod
    def initialize_session() -> None:
        """
        Initialize the session state with required components.
        
        This method sets up the core components (LLM client, reasoner, messages)
        if they don't already exist in the session state.
        """
        # Initialize LLM client
        if "llm_client" not in st.session_state:
            st.session_state.llm_client = LLM(model_name=LLM.GPT_4_1_NANO)
        
        # Initialize BioConcierge agent
        if "reasoner" not in st.session_state:
            st.session_state.reasoner = BioConciergeAgent(name="BioConcierge")
        
        # Initialize chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize uploaded files tracking
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
    
    @staticmethod
    def get_llm_client() -> LLM:
        """
        Get the LLM client from session state.
        
        Returns:
            LLM: The initialized LLM client
            
        Raises:
            KeyError: If session is not properly initialized
        """
        if "llm_client" not in st.session_state:
            SessionManager.initialize_session()
        return st.session_state.llm_client
    
    @staticmethod
    def get_reasoner() -> BioConciergeAgent:
        """
        Get the BioConcierge agent from session state.
        
        Returns:
            BioConciergeAgent: The initialized reasoning agent
            
        Raises:
            KeyError: If session is not properly initialized
        """
        if "reasoner" not in st.session_state:
            SessionManager.initialize_session()
        return st.session_state.reasoner
    
    @staticmethod
    def get_messages() -> list:
        """
        Get the chat messages from session state.
        
        Returns:
            list: List of chat messages
        """
        if "messages" not in st.session_state:
            SessionManager.initialize_session()
        return st.session_state.messages
    
    @staticmethod
    def add_message(role: str, content: str, **kwargs) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The message content
            **kwargs: Additional message metadata (citations, route, etc.)
        """
        if "messages" not in st.session_state:
            SessionManager.initialize_session()
        
        message = {
            "role": role,
            "content": content,
            **kwargs
        }
        st.session_state.messages.append(message)
    
    @staticmethod
    def clear_messages() -> None:
        """Clear all chat messages from session state."""
        st.session_state.messages = []
    
    @staticmethod
    def get_uploaded_files() -> list:
        """
        Get the list of uploaded files from session state.
        
        Returns:
            list: List of uploaded file information
        """
        if "uploaded_files" not in st.session_state:
            SessionManager.initialize_session()
        return st.session_state.uploaded_files
    
    @staticmethod
    def add_uploaded_file(file_info: dict) -> None:
        """
        Add uploaded file information to session state.
        
        Args:
            file_info: Dictionary containing file metadata
        """
        if "uploaded_files" not in st.session_state:
            SessionManager.initialize_session()
        st.session_state.uploaded_files.append(file_info)
    
    @staticmethod
    def remove_uploaded_file(file_name: str) -> bool:
        """
        Remove uploaded file information from session state.
        
        Args:
            file_name: Name of the file to remove (can be original name or safe_name)
            
        Returns:
            bool: True if file was removed, False if not found
        """
        if "uploaded_files" not in st.session_state:
            return False
        
        original_length = len(st.session_state.uploaded_files)
        st.session_state.uploaded_files = [
            f for f in st.session_state.uploaded_files 
            if f.get('name') != file_name and f.get('safe_name') != file_name
        ]
        return len(st.session_state.uploaded_files) < original_length
    
    @staticmethod
    def validate_uploaded_file(file_info: dict) -> dict:
        """
        Validate and normalize uploaded file information.
        
        Args:
            file_info: Dictionary containing file information
            
        Returns:
            dict: Validated and normalized file information
        """
        # Ensure required fields exist with defaults
        validated = {
            'name': file_info.get('name', 'Unknown'),
            'safe_name': file_info.get('safe_name', file_info.get('name', 'Unknown')),
            'path': file_info.get('path', ''),
            'size': file_info.get('size', 0),
            'type': file_info.get('type', 'Unknown'),
            'upload_time': file_info.get('upload_time', ''),
            'was_renamed': file_info.get('was_renamed', False)
        }
        
        return validated 