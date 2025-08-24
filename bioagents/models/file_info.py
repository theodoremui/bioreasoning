# ------------------------------------------------------------------------------
# file_info.py
#
# Pydantic model for file information and metadata. This class encapsulates
# all file-related metadata with strong typing, validation, and utility methods.
#
# Author: Theodore Mui
# Date: 2025-07-20
# ------------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator, computed_field
import mimetypes


class FileInfo(BaseModel):
    """
    A strongly typed model representing uploaded file metadata.

    This class encapsulates all information about an uploaded file including
    its original name, sanitized name, storage path, size, MIME type, upload
    timestamp, and whether it was renamed during processing.

    Attributes:
        name: The original filename as provided by the user
        safe_name: The sanitized filename used for storage (filesystem-safe)
        path: The full storage path where the file is located
        size_bytes: The file size in bytes
        mime_type: The MIME type of the file (e.g., 'application/pdf')
        upload_time: The timestamp when the file was uploaded
        was_renamed: Whether the filename was changed during sanitization

    Example:
        >>> file_info = FileInfo(
        ...     name="My Document.pdf",
        ...     safe_name="my_document.pdf",
        ...     path="/uploads/my_document.pdf",
        ...     size_bytes=1024000,
        ...     mime_type="application/pdf",
        ...     upload_time=datetime.now(),
        ...     was_renamed=True
        ... )
        >>> print(file_info.size_human_readable)
        "1.0 MB"
    """

    name: str = Field(
        description="Original filename as provided by the user",
        min_length=1,
        max_length=255,
        examples=["document.pdf", "My Research Paper.docx"],
    )

    safe_name: str = Field(
        description="Sanitized filename used for storage (filesystem-safe)",
        min_length=1,
        max_length=255,
        examples=["document.pdf", "my_research_paper.docx"],
    )

    path: Path = Field(
        description="Full storage path where the file is located",
        examples=["/uploads/documents/file.pdf", "uploads/file.txt"],
    )

    size_bytes: int = Field(
        description="File size in bytes",
        ge=0,  # Must be >= 0
        le=1024 * 1024 * 1024 * 5,  # Max 5GB
        examples=[1024, 2048000, 52428800],
    )

    mime_type: str = Field(
        description="MIME type of the file",
        min_length=1,
        examples=["application/pdf", "text/plain", "image/jpeg"],
    )

    upload_time: datetime = Field(
        description="Timestamp when the file was uploaded",
        examples=["2024-04-26T14:30:00"],
    )

    was_renamed: bool = Field(
        description="Whether the filename was changed during sanitization",
        examples=[True, False],
    )

    class Config:
        """Pydantic configuration for the FileInfo model."""

        # Allow Path objects to be serialized
        arbitrary_types_allowed = True
        # Validate on assignment
        validate_assignment = True
        # Use enum values in serialization
        use_enum_values = True

    @field_validator("name", "safe_name")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """
        Validate that filenames don't contain null bytes or other problematic characters.

        Args:
            v: The filename to validate

        Returns:
            The validated filename

        Raises:
            ValueError: If filename contains invalid characters
        """
        if "\x00" in v:
            raise ValueError("Filename cannot contain null bytes")
        if v.startswith(".") and len(v) == 1:
            raise ValueError("Filename cannot be just a dot")
        if v.startswith("..") and len(v) == 2:
            raise ValueError("Filename cannot be just two dots")
        return v.strip()

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """
        Validate that the MIME type follows the correct format.

        Args:
            v: The MIME type to validate

        Returns:
            The validated MIME type

        Raises:
            ValueError: If MIME type format is invalid
        """
        if "/" not in v:
            raise ValueError(
                "MIME type must contain a forward slash (e.g., 'text/plain')"
            )

        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("MIME type must have exactly one forward slash")

        type_part, subtype_part = parts
        if not type_part or not subtype_part:
            raise ValueError("Both type and subtype must be non-empty")

        return v.lower().strip()

    @computed_field
    @property
    def size_human_readable(self) -> str:
        """
        Get a human-readable representation of the file size.

        Returns:
            File size formatted as human-readable string (e.g., "1.5 MB")
        """
        if self.size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        size = float(self.size_bytes)
        i = 0

        while size >= 1024 and i < len(size_names) - 1:
            size /= 1024
            i += 1

        if i == 0:
            return f"{int(size)} {size_names[i]}"
        else:
            return f"{size:.1f} {size_names[i]}"

    @computed_field
    @property
    def file_extension(self) -> str:
        """
        Get the file extension from the safe filename.

        Returns:
            File extension including the dot (e.g., ".pdf", ".txt")
            Returns empty string if no extension found
        """
        return Path(self.safe_name).suffix.lower()

    @computed_field
    @property
    def is_text_file(self) -> bool:
        """
        Check if the file is a text-based file type.

        Returns:
            True if the file is a text-based format, False otherwise
        """
        text_types = {
            "text/",
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-yaml",
        }
        return any(self.mime_type.startswith(t) for t in text_types)

    @computed_field
    @property
    def is_image(self) -> bool:
        """
        Check if the file is an image.

        Returns:
            True if the file is an image format, False otherwise
        """
        return self.mime_type.startswith("image/")

    @computed_field
    @property
    def is_document(self) -> bool:
        """
        Check if the file is a document format (PDF, Word, etc.).

        Returns:
            True if the file is a document format, False otherwise
        """
        document_types = {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        return self.mime_type in document_types

    def get_type_category(self) -> str:
        """
        Get a general category for the file type.

        Returns:
            String category: 'image', 'document', 'text', 'archive', 'video', 'audio', or 'other'
        """
        if self.is_image:
            return "image"
        elif self.is_document:
            return "document"
        elif self.is_text_file:
            return "text"
        elif self.mime_type.startswith("video/"):
            return "video"
        elif self.mime_type.startswith("audio/"):
            return "audio"
        elif self.mime_type in (
            "application/zip",
            "application/x-rar-compressed",
            "application/x-tar",
            "application/gzip",
        ):
            return "archive"
        else:
            return "other"

    def to_dict(self) -> dict:
        """
        Convert the FileInfo instance to a dictionary compatible with the original format.

        This method provides backward compatibility with existing code that expects
        a dictionary format, while maintaining all the benefits of the Pydantic model.

        Returns:
            Dictionary representation matching the original file_info format
        """
        return {
            "name": self.name,
            "safe_name": self.safe_name,
            "path": str(self.path),
            "size": self.size_bytes,
            "type": self.mime_type,
            "upload_time": self.upload_time.isoformat(),
            "was_renamed": self.was_renamed,
        }

    @classmethod
    def from_upload(
        cls,
        uploaded_file,
        safe_filename: str,
        file_path: Union[str, Path],
        was_renamed: Optional[bool] = None,
    ) -> "FileInfo":
        """
        Create a FileInfo instance from a Streamlit uploaded file object.

        This factory method simplifies creation from Streamlit's UploadedFile objects,
        automatically determining if the file was renamed and setting appropriate defaults.

        Args:
            uploaded_file: Streamlit UploadedFile object
            safe_filename: The sanitized filename used for storage
            file_path: Path where the file is stored
            was_renamed: Optional override for rename detection

        Returns:
            New FileInfo instance

        Example:
            >>> file_info = FileInfo.from_upload(
            ...     uploaded_file=st_uploaded_file,
            ...     safe_filename="document_2024.pdf",
            ...     file_path="/uploads/document_2024.pdf"
            ... )
        """
        if was_renamed is None:
            was_renamed = safe_filename != uploaded_file.name

        return cls(
            name=uploaded_file.name,
            safe_name=safe_filename,
            path=Path(file_path),
            size_bytes=uploaded_file.size,
            mime_type=uploaded_file.type
            or mimetypes.guess_type(uploaded_file.name)[0]
            or "application/octet-stream",
            upload_time=datetime.now(),
            was_renamed=was_renamed,
        )

    def __str__(self) -> str:
        """String representation showing key file information."""
        return f"FileInfo(name='{self.name}', size={self.size_human_readable}, type='{self.get_type_category()}')"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"FileInfo(name='{self.name}', safe_name='{self.safe_name}', "
            f"size_bytes={self.size_bytes}, mime_type='{self.mime_type}', "
            f"was_renamed={self.was_renamed})"
        )
