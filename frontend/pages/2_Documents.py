"""
Documents Page for the BioReasoning multi-page Streamlit application.

This module provides the document upload and management interface, allowing users
to upload files for processing by the BioReasoning Agent.

Author: Theodore Mui
Date: 2025-04-26
"""

import streamlit as st
import streamlit.components.v1 as components

from streamlit.runtime.uploaded_file_manager import UploadedFile

import asyncio
import datetime
import io
import os
import time
import tempfile as temp
from typing import Tuple

from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib

# Import components and utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from components.session_manager import SessionManager
from bioagents.models.file_info import FileInfo
from bioagents.utils.async_utils import run_async
from bioagents.utils.audio_manager import AudioFileProcessor


#------------------------------------------------
# Instrumentation
#------------------------------------------------

from bioagents.observability.instrumentation import OtelTracesSqlEngine
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

# Configure OpenTelemetry at SDK level BEFORE any imports
ENABLE_OBSERVABILITY = os.getenv("ENABLE_OBSERVABILITY", "true").lower() == "true"

if not ENABLE_OBSERVABILITY:
    # Disable OpenTelemetry at SDK level
    os.environ["OTEL_TRACES_SAMPLER"] = "off"
    os.environ["OTEL_TRACES_EXPORTER"] = "none"
    os.environ["OTEL_METRICS_EXPORTER"] = "none"
    os.environ["OTEL_LOGS_EXPORTER"] = "none"
    print("📊 OpenTelemetry disabled at SDK level")
else:
    print("📊 OpenTelemetry enabled")

# Configure observability based on environment
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

# Initialize observability only if enabled
instrumentor = None
if ENABLE_OBSERVABILITY:
    try:
        # Test if the OpenTelemetry endpoint is reachable before initializing
        import requests
        # For Jaeger all-in-one, test the main Jaeger API endpoint instead
        # since the OTLP collector doesn't expose a health endpoint
        jaeger_health_url = "http://localhost:16686/api/services"
        try:
            response = requests.get(jaeger_health_url, timeout=2)
            if response.status_code == 200:
                # define a custom span exporter
                span_exporter = OTLPSpanExporter(OTLP_ENDPOINT)
                
                # initialize the instrumentation object
                instrumentor = LlamaIndexOpenTelemetry(
                    service_name_or_resource="agent.traces",
                    span_exporter=span_exporter,
                    debug=True,
                )
                print(f"✅ OpenTelemetry initialized with endpoint: {OTLP_ENDPOINT}")
                print(f"✅ Jaeger UI available at: http://localhost:16686")
            else:
                print(f"⚠️  Jaeger endpoint not healthy: {response.status_code}")
                print("📊 Continuing without observability...")
                instrumentor = None
        except requests.exceptions.RequestException as req_e:
            print(f"⚠️  Jaeger endpoint unreachable: {req_e}")
            print("📊 Continuing without observability...")
            instrumentor = None
    except Exception as e:
        print(f"⚠️  OpenTelemetry initialization failed: {e}")
        print("📊 Continuing without observability...")
        instrumentor = None
else:
    print("📊 Observability disabled via ENABLE_OBSERVABILITY=false")

engine_url = f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}"
sql_engine = OtelTracesSqlEngine(
    engine_url=engine_url,
    table_name="agent_traces",
    service_name="agent.traces",
)


#------------------------------------------------
# NotebookLM Workflow
#------------------------------------------------
from bioagents.docs.documents import ManagedDocument, DocumentManager
from bioagents.docs.audio import PODCAST_GEN, PodcastConfig
from bioagents.docs.workflow import NotebookLMWorkflow, FileInputEvent, NotebookOutputEvent

document_manager = DocumentManager(engine_url=engine_url)

WF = NotebookLMWorkflow(timeout=600)


# Read the HTML file
def read_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def run_workflow(
    file: io.BytesIO, document_title: str
) -> Tuple[str, str, str, str, str]:
    # Create temp file with proper Windows handling
    with temp.NamedTemporaryFile(suffix=".pdf", delete=False) as fl:
        content = file.getvalue()
        fl.write(content)
        fl.flush()  # Ensure data is written
        temp_path = fl.name

    try:
        st_time = int(time.time() * 1000000)
        ev = FileInputEvent(file=temp_path)
        result: NotebookOutputEvent = await WF.run(start_event=ev)

        q_and_a = ""
        for q, a in zip(result.questions, result.answers):
            q_and_a += f"**{q}**\n\n{a}\n\n"
        bullet_points = "## Bullet Points\n\n- " + "\n- ".join(result.highlights)

        mind_map = result.mind_map
        if Path(mind_map).is_file():
            mind_map = read_html_file(mind_map)
            try:
                os.remove(result.mind_map)
            except OSError:
                pass  # File might be locked on Windows

        end_time = int(time.time() * 1000000)
        sql_engine.to_sql_database(start_time=st_time, end_time=end_time)
        document_manager.put_documents(
            [
                ManagedDocument(
                    document_name=document_title,
                    content=result.md_content,
                    summary=result.summary,
                    q_and_a=q_and_a,
                    mindmap=mind_map,
                    bullet_points=bullet_points,
                )
            ]
        )
        return result.md_content, result.summary, q_and_a, bullet_points, mind_map

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            await asyncio.sleep(0.1)
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Give up if still locked


def sync_run_workflow(file: io.BytesIO, document_title: str):
    """Synchronous wrapper for workflow execution that handles event loop properly."""
    return run_async(run_workflow(file, document_title))


async def create_podcast(file_content: str, config: PodcastConfig = None):
    if PODCAST_GEN is None:
        raise ValueError(
            "Podcast generation is not available. Please ensure both ELEVENLABS_API_KEY and OPENAI_API_KEY "
            "environment variables are set."
        )
    audio_fl = await PODCAST_GEN.create_conversation(
        file_transcript=file_content, config=config
    )
    return audio_fl


def sync_create_podcast(file_content: str, config: PodcastConfig = None):
    """Synchronous wrapper for podcast creation that handles event loop properly."""
    return run_async(create_podcast(file_content=file_content, config=config))

#------------------------------------------------
# Initialize Session State
#------------------------------------------------

import randomname

# Initialize session state BEFORE creating the text input
if "workflow_results" not in st.session_state:
    st.session_state.workflow_results = None
if "document_title" not in st.session_state:
    st.session_state.document_title = randomname.get_name(
        adj=("music_theory", "geometry", "emotions"), noun=("cats", "food")
    )

# Use session_state as the value and update it when changed
document_title = st.text_input(
    label="Document Title",
    value=st.session_state.document_title,
    key="document_title_input",
)

# Update session state when the input changes
if document_title != st.session_state.document_title:
    st.session_state.document_title = document_title



#------------------------------------------------
# Document Manager
#------------------------------------------------

class DocumentManager:
    """
    Document management functionality for handling file uploads and storage.
    
    This class encapsulates file handling operations following the Single
    Responsibility Principle.
    """
    
    def __init__(self, upload_directory: str = "data/uploads"):
        """
        Initialize the document manager.
        
        Args:
            upload_directory: Directory path for storing uploaded files
        """
        self.upload_directory = Path(upload_directory)
        self._ensure_upload_directory()
    
    def _ensure_upload_directory(self) -> None:
        """Ensure the upload directory exists."""
        self.upload_directory.mkdir(parents=True, exist_ok=True)
    
    def check_file_exists(self, filename: str) -> bool:
        """
        Check if a file with the given name already exists.
        
        Args:
            filename: The filename to check
            
        Returns:
            bool: True if file exists
        """
        file_path = self.upload_directory / filename
        return file_path.exists()
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename by adding timestamp and hash if needed.
        
        Args:
            original_filename: The original filename
            
        Returns:
            str: Unique filename
        """
        if not self.check_file_exists(original_filename):
            return original_filename
        
        # File exists, generate unique name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        # Split filename and extension
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            unique_filename = f"{base_name}_{timestamp}_{file_hash}.{extension}"
        else:
            unique_filename = f"{original_filename}_{timestamp}_{file_hash}"
        
        return unique_filename
    
    def save_uploaded_file(self, uploaded_file, handle_duplicates: str = "rename") -> Dict[str, Any]:
        """
        Save an uploaded file to the upload directory.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            handle_duplicates: How to handle duplicates ("rename", "overwrite", "skip")
            
        Returns:
            Dict containing file information or error information
        """
        original_filename = uploaded_file.name
        
        # Check for duplicates and handle accordingly
        if self.check_file_exists(original_filename):
            if handle_duplicates == "skip":
                return {
                    "error": True,
                    "message": f"File '{original_filename}' already exists. Skipped.",
                    "name": original_filename
                }
            elif handle_duplicates == "rename":
                safe_filename = self.generate_unique_filename(original_filename)
            elif handle_duplicates == "overwrite":
                safe_filename = original_filename
            else:
                safe_filename = self.generate_unique_filename(original_filename)
        else:
            safe_filename = original_filename
        
        file_path = self.upload_directory / safe_filename
        
        try:
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Return file information
            file_info = {
                "name": uploaded_file.name,
                "safe_name": safe_filename,
                "path": str(file_path),
                "size": uploaded_file.size,
                "type": uploaded_file.type,
                "upload_time": datetime.datetime.now().isoformat(),
                "was_renamed": safe_filename != original_filename
            }
            
            return file_info
            
        except Exception as e:
            return {
                "error": True,
                "message": f"Failed to save '{original_filename}': {str(e)}",
                "name": original_filename
            }
    
    def delete_file(self, safe_filename: str) -> bool:
        """
        Delete a file from the upload directory.
        
        Args:
            safe_filename: The safe filename to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            file_path = self.upload_directory / safe_filename
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception:
            pass
        return False
    
    def get_file_info(self, safe_filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored file.
        
        Args:
            safe_filename: The safe filename to query
            
        Returns:
            Dict with file information or None if not found
        """
        file_path = self.upload_directory / safe_filename
        if file_path.exists():
            stat = file_path.stat()
            return {
                "safe_name": safe_filename,
                "path": str(file_path),
                "size": stat.st_size,
                "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        return None


class DocumentsPage:
    """
    Documents page implementation for the BioReasoning application.
    
    This class provides the interface for file upload, management, and processing.
    """
    
    def __init__(self):
        """Initialize the documents page."""
        self.document_manager = DocumentManager()
        self.audio_processor = AudioFileProcessor()
        SessionManager.initialize_session()
        self._cleanup_processed_files_state()
    
    def _cleanup_processed_files_state(self) -> None:
        """Clean up processed files state for files that no longer exist."""
        if "files_being_processed" not in st.session_state:
            return
        
        # Get list of files that still exist
        existing_files = set()
        for file_info in SessionManager.get_uploaded_files():
            existing_files.add(file_info.get('name', ''))
            existing_files.add(file_info.get('safe_name', ''))
        
        # Remove processed files that no longer exist
        files_to_remove = st.session_state.files_being_processed - existing_files
        for file_name in files_to_remove:
            st.session_state.files_being_processed.discard(file_name)
    
    def render(self) -> None:
        """Render the complete documents page interface."""
        self._render_header()
        self._render_upload_section()
        self._render_workflow_results()
        self._render_file_management()
    
    def _render_header(self) -> None:
        """Render the page header and description."""
        st.title("📁 Document Management")
        st.markdown(
            "Upload documents for processing by the BioReasoning Agent. "
            "Supported formats include PDF, TXT, DOCX, and more."
        )
        st.markdown("---")
    
    def _render_upload_section(self) -> None:
        """Render the file upload interface."""
        st.markdown("## Upload Documents")
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'docx', 'doc', 'csv', 'md'],
                help="Select one or more files to upload for processing"
            )
            
            if uploaded_files:
                # Initialize processing state in session if not exists
                if "files_being_processed" not in st.session_state:
                    st.session_state.files_being_processed = set()
                
                # Check if these files are already being processed or have been processed
                file_names = [f.name for f in uploaded_files]
                already_processed = all(name in st.session_state.files_being_processed for name in file_names)
                
                if not already_processed:
                    # Check for duplicate files only for new uploads
                    duplicate_files = [
                        file for file in uploaded_files 
                        if self.document_manager.check_file_exists(file.name)
                    ]
                    
                    # Only show duplicate handling options if there are actual duplicates
                    if duplicate_files:
                        st.warning(f"⚠️ {len(duplicate_files)} file(s) already exist: {', '.join([f.name for f in duplicate_files])}")
                        duplicate_option = st.radio(
                            "How to handle duplicate files:",
                            ["Rename automatically", "Overwrite existing", "Skip duplicates"],
                            index=0,
                            help="Choose how to handle files with names that already exist"
                        )
                        
                        # Map user-friendly options to internal values
                        duplicate_mapping = {
                            "Rename automatically": "rename",
                            "Overwrite existing": "overwrite", 
                            "Skip duplicates": "skip"
                        }
                        duplicate_handling = duplicate_mapping[duplicate_option]
                    else:
                        # No duplicates, use default handling
                        duplicate_handling = "rename"
                    
                    self._handle_file_uploads(uploaded_files, duplicate_handling)
                else:
                    st.info("✅ These files have already been processed. Please upload new files to process them.")
                    if st.button("🔄 Reset Upload State", help="Clear processing history to re-upload these files"):
                        # Clear the processing state for these files
                        for name in file_names:
                            st.session_state.files_being_processed.discard(name)
                        st.rerun()
        
        with col2:
            st.markdown("**Supported Formats**")
            st.markdown("""\
            - **PDF** (.pdf)
            - **Text** (.txt, .md)
            - **Word** (.docx, .doc)
            """)

    def _render_workflow_results(self) -> None:
        """Render workflow results if available."""
        # Display results if available
        if st.session_state.workflow_results:
            st.markdown("---")
            
            # Header with clear button
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("## Document Processing Results")
            with col2:
                if st.button("🗑️ Clear Results", help="Clear the processing results"):
                    st.session_state.workflow_results = None
                    st.rerun()
            
            results = st.session_state.workflow_results

            # Summary
            st.markdown("### Summary")
            st.markdown(results["summary"])

            # Bullet Points
            st.markdown(results["bullet_points"])

            # FAQ (toggled)
            with st.expander("FAQ"):
                st.markdown(results["q_and_a"])

            # Mind Map
            if results["mind_map"]:
                st.markdown("### Mind Map")
                components.html(results["mind_map"], height=800, scrolling=True)

            # Podcast Configuration Panel
            st.markdown("---")
            st.markdown("### Podcast Configuration")

            with st.expander("Customize Your Podcast", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    style = st.selectbox(
                        "Conversation Style",
                        ["conversational", "interview", "debate", "educational"],
                        help="The overall style of the podcast conversation",
                    )

                    tone = st.selectbox(
                        "Tone",
                        ["friendly", "professional", "casual", "energetic"],
                        help="The tone of voice for the conversation",
                    )

                    target_audience = st.selectbox(
                        "Target Audience",
                        ["general", "technical", "business", "expert", "beginner"],
                        help="Who is the intended audience for this podcast?",
                    )

                with col2:
                    speaker1_role = st.text_input(
                        "Speaker 1 Role",
                        value="host",
                        help="The role or persona of the first speaker",
                    )

                    speaker2_role = st.text_input(
                        "Speaker 2 Role",
                        value="guest",
                        help="The role or persona of the second speaker",
                    )

                # Focus Topics
                st.markdown("**Focus Topics** (optional)")
                focus_topics_input = st.text_area(
                    "Enter topics to emphasize (one per line)",
                    help="List specific topics you want the podcast to focus on. Leave empty for general coverage.",
                    placeholder="How can this be applied for Machine Learning Applications?\nUnderstand the historical context\nFuture Implications",
                )

                # Parse focus topics
                focus_topics = None
                if focus_topics_input.strip():
                    focus_topics = [
                        topic.strip()
                        for topic in focus_topics_input.split("\n")
                        if topic.strip()
                    ]

                # Custom Prompt
                custom_prompt = st.text_area(
                    "Custom Instructions (optional)",
                    help="Add any additional instructions for the podcast generation",
                    placeholder="Make sure to explain technical concepts simply and include real-world examples...",
                )

                # Create config object
                podcast_config = PodcastConfig(
                    style=style,
                    tone=tone,
                    focus_topics=focus_topics,
                    target_audience=target_audience,
                    custom_prompt=custom_prompt if custom_prompt.strip() else None,
                    speaker1_role=speaker1_role,
                    speaker2_role=speaker2_role,
                )

            # Second button: Generate Podcast
            if st.button("Generate In-Depth Conversation", type="secondary"):
                with st.spinner("Generating podcast... This may take several minutes."):
                    try:
                        # Generate the podcast audio
                        temp_audio_file = sync_create_podcast(
                            results["md_content"], config=podcast_config
                        )
                        
                        # Get the original document name
                        original_document_name = results.get("original_document_name", "unknown_document")
                        
                        # Save the audio file using the audio processor
                        audio_result = self.audio_processor.process_podcast_audio(
                            temp_audio_file,
                            original_document_name,
                            handle_duplicates="rename"
                        )
                        
                        if audio_result.get("error"):
                            st.error(f"Failed to save audio file: {audio_result['message']}")
                        else:
                            st.success("Podcast generated and saved successfully!")
                            
                            # Display audio player
                            st.markdown("#### Generated Podcast")
                            audio_path = audio_result["path"]
                            
                            if os.path.exists(audio_path):
                                with open(audio_path, "rb") as f:
                                    audio_bytes = f.read()
                                st.audio(audio_bytes, format="audio/mp3")
                                
                                # Show file information
                                st.info(f"**Saved as:** {audio_result['filename']}")
                                st.info(f"**Location:** {audio_path}")
                                st.info(f"**Size:** {audio_result['size_bytes']} bytes")
                            else:
                                st.error("Audio file not found after saving.")

                    except Exception as e:
                        st.error(f"Error generating podcast: {str(e)}")
    
    def _process_uploaded_file(self, uploaded_file: UploadedFile) -> None:
        """
        Process an uploaded file.
        """
        with st.spinner(f"Processing {uploaded_file.name}... (this may take a while)"):
            try:
                md_content, summary, q_and_a, bullet_points, mind_map = (
                    sync_run_workflow(uploaded_file, st.session_state.document_title)
                )
                st.session_state.workflow_results = {
                    "md_content": md_content,
                    "summary": summary,
                    "q_and_a": q_and_a,
                    "bullet_points": bullet_points,
                    "mind_map": mind_map,
                    "original_document_name": uploaded_file.name,
                }
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"❌ file `{uploaded_file.name}` processing error: {e}")
            

    def _handle_file_uploads(self, uploaded_files: List, duplicate_handling: str = "rename") -> None:
        """
        Handle the processing of uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            duplicate_handling: How to handle duplicate files ("rename", "overwrite", "skip")
        """
        if st.button("Process Uploaded Files", type="primary"):
            # Mark files as being processed to prevent duplicate UI from showing again
            if "files_being_processed" not in st.session_state:
                st.session_state.files_being_processed = set()
            
            for uploaded_file in uploaded_files:
                st.session_state.files_being_processed.add(uploaded_file.name)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            error_count = 0
            skipped_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                result = self.document_manager.save_uploaded_file(uploaded_file, duplicate_handling)
            
                if result.get("error", False):
                    if "already exists" in result.get("message", ""):
                        st.warning(f"⚠️ {result['message']}")
                        skipped_count += 1
                    else:
                        st.error(f"❌ {result['message']}")
                        error_count += 1
                else:
                    # Validate and add to session state only if no error
                    validated_result = SessionManager.validate_uploaded_file(result)
                    SessionManager.add_uploaded_file(validated_result)
                    
                    # Show appropriate success message
                    if result.get("was_renamed", False):
                        st.success(f"✅ Uploaded '{uploaded_file.name}' as '{result['safe_name']}'")
                    else:
                        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")

                    self._process_uploaded_file(uploaded_file)
                    success_count += 1
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show summary
            status_text.text("Upload complete!")
            
            summary_parts = []
            if success_count > 0:
                summary_parts.append(f"✅ {success_count} uploaded")
            if skipped_count > 0:
                summary_parts.append(f"⚠️ {skipped_count} skipped")
            if error_count > 0:
                summary_parts.append(f"❌ {error_count} failed")
            
            if summary_parts:
                st.info(f"Summary: {', '.join(summary_parts)}")
            
            st.rerun()
    
    def _render_file_management(self) -> None:
        """Render the file management interface."""
        st.markdown("---")
        st.markdown("## Uploaded Files")
        
        uploaded_files = SessionManager.get_uploaded_files()
        
        if not uploaded_files:
            st.info("No files uploaded yet. Use the upload section above to add documents.")
            return
        
        # Display files in a table-like format
        for index, file_info in enumerate(uploaded_files):
            self._render_file_item(file_info, index)
    
    def _render_file_item(self, file_info: Dict[str, Any], index: int) -> None:
        """
        Render a single file item with management controls.
        
        Args:
            file_info: Dictionary containing file information
            index: Unique index for this file item to ensure unique button keys
        """
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{file_info['name']}**")
                
                # Add info if file was renamed due to duplicate
                if file_info.get('was_renamed', False):
                    st.caption(f"📂 Saved as: {file_info['safe_name']}")
                
                st.caption(f"Type: {file_info.get('type', 'Unknown')} | "
                          f"Size: {self._format_file_size(file_info.get('size', 0))} | "
                          f"Uploaded: {self._format_timestamp(file_info.get('upload_time', ''))}")
            
            with col2:
                if st.button("📄 View", key=f"view_{index}_{hash(file_info.get('path', ''))}"):
                    self._show_file_preview(file_info)
            
            with col3:
                if st.button("💬 Analyze", key=f"analyze_{index}_{hash(file_info.get('path', ''))}"):
                    self._analyze_file(file_info)
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{index}_{hash(file_info.get('path', ''))}"):
                    self._delete_file(file_info)
            
            st.markdown("---")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def _format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.datetime.fromisoformat(timestamp_str)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return "Unknown"
    
    def _show_file_preview(self, file_info: Dict[str, Any]) -> None:
        """Show a preview of the file content."""
        st.info(f"File preview for {file_info['name']} will be implemented in future versions.")
    
    def _analyze_file(self, file_info: Dict[str, Any]) -> None:
        """Analyze the file with the BioReasoning Agent."""
        st.info(f"File analysis for {file_info['name']} will be integrated with the chat system.")
        st.markdown("For now, you can mention this file in the Chat page to discuss its contents.")
    
    def _delete_file(self, file_info: Dict[str, Any]) -> None:
        """Delete the file and remove from session state."""
        safe_name = file_info.get('safe_name', file_info.get('name', 'Unknown'))
        original_name = file_info.get('name', 'Unknown')
        
        if self.document_manager.delete_file(safe_name):
            # Remove from session state using both names to ensure removal
            SessionManager.remove_uploaded_file(safe_name)
            
            # Also clear from processed files state to allow re-upload
            if "files_being_processed" in st.session_state:
                st.session_state.files_being_processed.discard(original_name)
                st.session_state.files_being_processed.discard(safe_name)
            
            st.success(f"✅ Deleted {original_name}")
            st.rerun()
        else:
            st.error(f"❌ Failed to delete {original_name}")


def main():
    """Main function to render the documents page."""
    documents_page = DocumentsPage()
    documents_page.render()


if __name__ == "__main__":
    main() 