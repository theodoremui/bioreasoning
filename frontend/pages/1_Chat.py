"""
Chat Page for the BioReasoning multi-page Streamlit application.

This module provides the chat interface for interacting with the BioReasoning Agent,
refactored from the original bioreasoning.py to follow better architectural principles.

Author: Theodore Mui
Date: 2025-04-26
"""

import streamlit as st
import asyncio
from typing import Optional

# Import components and utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from components.model_selector import ModelSelector
from components.session_manager import SessionManager
from bioagents.agents.common import AgentResponse


class ChatPage:
    """
    Chat page implementation for the BioReasoning application.
    
    This class encapsulates the chat functionality, providing a clean interface
    for user interaction with the BioReasoning Agent.
    """
    
    def __init__(self):
        """Initialize the chat page with required components."""
        self.model_selector = ModelSelector()
        SessionManager.initialize_session()
    
    def render(self) -> None:
        """
        Render the complete chat page interface.
        
        This method handles the layout, sidebar, chat history, and new message processing.
        """
        self._render_header()
        self._render_sidebar()
        self._render_chat_interface()
    
    def _render_header(self) -> None:
        """Render the page header and title."""
        st.title("💬 Chat with BioReasoning Agent")
        st.markdown("Ask me anything about medicine, genetics, drug design, and clinical trials!")
        st.markdown("---")
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with model selection and chat controls."""
        with st.sidebar:
            st.markdown("## Chat Settings")
            
            # Model selection
            self.model_selector.render()
            
            st.markdown("---")
            
            # Chat controls
            st.markdown("### Chat Controls")
            if st.button("Clear Chat History", type="secondary"):
                SessionManager.clear_messages()
                st.rerun()
            
            # Display chat statistics
            messages = SessionManager.get_messages()
            st.markdown(f"**Messages in conversation:** {len(messages)}")
            
            user_messages = len([m for m in messages if m["role"] == "user"])
            assistant_messages = len([m for m in messages if m["role"] == "assistant"])
            
            st.markdown(f"- User messages: {user_messages}")
            st.markdown(f"- Assistant messages: {assistant_messages}")
    
    def _render_chat_interface(self) -> None:
        """Render the main chat interface with message history and input."""
        # Display chat history
        messages = SessionManager.get_messages()
        self._render_message_history(messages)
        
        # Handle new user input
        self._handle_user_input()
    
    def _render_message_history(self, messages: list) -> None:
        """
        Render the chat message history.
        
        Args:
            messages: List of chat messages to display
        """
        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display citations if available for assistant messages
                if (message["role"] == "assistant" and 
                    "citations" in message and 
                    message["citations"]):
                    self._render_citations(message["citations"])
    
    def _render_citations(self, citations: list) -> None:
        """
        Render citations in an expandable section.
        
        Args:
            citations: List of citation objects to display
        """
        with st.expander("📚 Citations", expanded=False):
            for i, citation in enumerate(citations):
                st.markdown(f"**{i+1}.** [{citation.title}]({citation.url})")
                if hasattr(citation, 'snippet') and citation.snippet:
                    st.markdown(f"*{citation.snippet}*")
                if i < len(citations) - 1:  # Add separator except for last citation
                    st.markdown("---")
    
    def _handle_user_input(self) -> None:
        """Handle new user input and generate assistant response."""
        if prompt := st.chat_input("How can I help you?"):
            # Add user message to history
            SessionManager.add_message("user", prompt)
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            self._generate_assistant_response(prompt)
    
    def _generate_assistant_response(self, prompt: str) -> None:
        """
        Generate and display the assistant's response to user input.
        
        Args:
            prompt: The user's input prompt
        """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get the reasoner and generate response
                    reasoner = SessionManager.get_reasoner()
                    agent_response: AgentResponse = asyncio.run(reasoner.achat(prompt))
                    
                    # Display the response
                    st.write(agent_response.response_str)
                    
                    # Display citations if available
                    if agent_response.citations:
                        self._render_citations(agent_response.citations)
                    
                    # Add assistant message to history
                    SessionManager.add_message(
                        "assistant",
                        agent_response.response_str,
                        citations=agent_response.citations,
                        route=getattr(agent_response, 'route', None)
                    )
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    SessionManager.add_message("assistant", error_message)


def main():
    """Main function to render the chat page."""
    chat_page = ChatPage()
    chat_page.render()


if __name__ == "__main__":
    main() 