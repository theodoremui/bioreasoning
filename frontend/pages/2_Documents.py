"""
Documents Page for the BioReasoning multi-page Streamlit application.

This module provides the document upload and management interface, allowing users
to upload files for processing by the BioReasoning Agent.

Author: Theodore Mui
Date: 2025-04-26
"""

import streamlit as st
import os
import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib

# Import components and utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from components.session_manager import SessionManager


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
        SessionManager.initialize_session()
    
    def render(self) -> None:
        """Render the complete documents page interface."""
        self._render_header()
        self._render_upload_section()
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
                # Show duplicate handling options
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
                
                self._handle_file_uploads(uploaded_files, duplicate_mapping[duplicate_option])
        
        with col2:
            st.markdown("### Supported Formats")
            st.markdown("""
            - **PDF** (.pdf)
            - **Text** (.txt, .md)
            - **Word** (.docx, .doc)
            """)
    
    def _handle_file_uploads(self, uploaded_files: List, duplicate_handling: str = "rename") -> None:
        """
        Handle the processing of uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            duplicate_handling: How to handle duplicate files ("rename", "overwrite", "skip")
        """
        if st.button("Process Uploaded Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            error_count = 0
            skipped_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save the file
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