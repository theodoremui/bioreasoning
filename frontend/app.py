"""
Main entry point for the BioReasoning multi-page Streamlit application.

This module serves as the primary navigation hub for the BioReasoning Agent application,
providing access to different functional areas through a clean, organized interface.

Author: Theodore Mui
Date: 2025-04-26
"""

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import streamlit as st

import asyncio
import io
import os
import sys
import time
import tempfile as temp
from pathlib import Path
from typing import Tuple

# Add project root to Python path for bioagents import
sys.path.append(os.path.dirname(__file__))

# Configure Streamlit page settings
st.set_page_config(
    page_title="BioReasoning Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add common sidebar content
with st.sidebar:
    st.markdown("### About BioReasoning")
    st.markdown(
        "Ask me anything about medicine, genetics, drug design, and clinical trials!"
    )

# Define pages using st.Page
chat_page = st.Page("pages/1_Chat.py", title="Chat", icon="💬", default=True)

documents_page = st.Page("pages/2_Documents.py", title="Documents", icon="📁")

# Create navigation
pg = st.navigation([chat_page, documents_page])
pg.run()
