"""
Document Processors

This module provides document processing functionality for PDF parsing, cleaning,
and node creation with provenance metadata. Follows SOLID principles with clear
separation of concerns.

Author: Theodore Mui
Date: 2025-08-24
"""

from pathlib import Path
from typing import Any, List, Optional

from llama_cloud_services.parse import LlamaParse
from llama_cloud_services.parse.types import JobResult
from llama_index.core.schema import Document, TextNode

from .constants import DOCUMENT_CLEANUP_PATTERNS
from .interfaces import IDocumentProcessor


class DocumentCleaner:
    """Cleans document content by removing unwanted text patterns.

    Follows SRP by focusing solely on document cleaning operations.
    Configurable patterns allow for different cleaning strategies.
    """

    def __init__(self, cleanup_patterns: Optional[List[str]] = None):
        """Initialize document cleaner with patterns.

        Args:
            cleanup_patterns: List of text patterns to remove (optional)
        """
        self.cleanup_patterns = cleanup_patterns or DOCUMENT_CLEANUP_PATTERNS.copy()

    def clean_content(self, content: Any) -> Any:
        """Clean content by removing unwanted patterns.

        Args:
            content: Content to clean (string or object with text/md attributes)

        Returns:
            Cleaned content
        """
        if isinstance(content, str):
            return self._clean_text(content)

        # Handle objects with text and md attributes (like LlamaParse pages)
        if hasattr(content, "text") and content.text:
            content.text = self._clean_text(content.text)

        if hasattr(content, "md") and content.md:
            content.md = self._clean_text(content.md)

        return content

    def _clean_text(self, text: str) -> str:
        """Clean text by removing all configured patterns.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        cleaned = text
        for pattern in self.cleanup_patterns:
            cleaned = cleaned.replace(pattern, "")
        return cleaned.strip()

    def add_cleanup_pattern(self, pattern: str) -> None:
        """Add a new cleanup pattern.

        Args:
            pattern: Text pattern to remove
        """
        if pattern and pattern not in self.cleanup_patterns:
            self.cleanup_patterns.append(pattern)

    def remove_cleanup_pattern(self, pattern: str) -> None:
        """Remove a cleanup pattern.

        Args:
            pattern: Text pattern to stop removing
        """
        if pattern in self.cleanup_patterns:
            self.cleanup_patterns.remove(pattern)


class ProvenanceNodeBuilder:
    """Builds text nodes with provenance metadata from documents.

    Follows SRP by focusing solely on node creation with provenance tracking.
    Handles paragraph-level splitting and metadata enrichment.
    """

    def __init__(self, paragraph_separator: str = "\n\n"):
        """Initialize node builder with configuration.

        Args:
            paragraph_separator: Separator for splitting paragraphs
        """
        self.paragraph_separator = paragraph_separator

    def build_nodes_from_documents(
        self, documents: List[Document], file_path: str
    ) -> List[TextNode]:
        """Build text nodes with provenance from documents.

        Args:
            documents: List of documents (typically one per page)
            file_path: Source file path for provenance

        Returns:
            List of TextNode objects with provenance metadata
        """
        doc_id = Path(file_path).stem
        doc_title = doc_id  # Could be enhanced to extract actual title
        nodes = []

        for page_idx, doc in enumerate(documents):
            page_text = doc.text or ""
            if not page_text.strip():
                continue

            # Split into paragraphs
            paragraphs = [
                p.strip()
                for p in page_text.split(self.paragraph_separator)
                if p.strip()
            ]

            # Create nodes for each paragraph
            offset = 0
            for para_idx, paragraph in enumerate(paragraphs):
                # Find character positions
                start = page_text.find(paragraph, offset)
                end = start + len(paragraph) if start >= 0 else None
                offset = end or offset

                # Create node with provenance metadata
                node = TextNode(
                    text=paragraph,
                    metadata={
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "file_path": file_path,
                        "page_number": page_idx + 1,
                        "paragraph_index": para_idx,
                        "char_start": start,
                        "char_end": end,
                    },
                )
                nodes.append(node)

        return nodes

    def build_single_node_from_document(
        self, document: Document, file_path: str, page_number: int = 1
    ) -> TextNode:
        """Build a single node from a document.

        Args:
            document: Source document
            file_path: Source file path
            page_number: Page number for provenance

        Returns:
            TextNode with provenance metadata
        """
        doc_id = Path(file_path).stem
        doc_title = doc_id

        return TextNode(
            text=document.text or "",
            metadata={
                "doc_id": doc_id,
                "doc_title": doc_title,
                "file_path": file_path,
                "page_number": page_number,
                "paragraph_index": 0,
                "char_start": 0,
                "char_end": len(document.text) if document.text else 0,
            },
        )


class DocumentProcessor(IDocumentProcessor):
    """Main document processor coordinating parsing, cleaning, and node creation.

    Follows the Facade pattern to provide a simple interface for complex document
    processing operations. Uses dependency injection for flexibility and testability.

    Features:
        - PDF parsing with LlamaParse
        - Configurable document cleaning
        - Provenance-rich node creation
        - Error handling and recovery
        - Progress tracking integration

    Example:
        processor = DocumentProcessor(
            api_key="your-key",
            num_workers=8,
            cleanup_patterns=["unwanted text"]
        )
        nodes = await processor.process_document("document.pdf")
    """

    def __init__(
        self,
        api_key: str,
        num_workers: int = 8,
        language: str = "en",
        verbose: bool = False,
        document_cleaner: Optional[DocumentCleaner] = None,
        node_builder: Optional[ProvenanceNodeBuilder] = None,
    ):
        """Initialize document processor with configuration.

        Args:
            api_key: LlamaParse API key
            num_workers: Number of parallel workers for parsing
            language: Document language for parsing
            verbose: Whether to enable verbose logging
            document_cleaner: Custom document cleaner (optional)
            node_builder: Custom node builder (optional)
        """
        self.api_key = api_key
        self.num_workers = num_workers
        self.language = language
        self.verbose = verbose

        # Dependency injection with defaults
        self._cleaner = document_cleaner or DocumentCleaner()
        self._node_builder = node_builder or ProvenanceNodeBuilder()

        # Initialize parser
        self._parser = None
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize LlamaParse parser with configuration."""
        if not self.api_key:
            raise ValueError("API key is required for document processing")

        self._parser = LlamaParse(
            api_key=self.api_key,
            num_workers=self.num_workers,
            verbose=self.verbose,
            language=self.language,
        )

    async def process_document(self, file_path: str) -> List[TextNode]:
        """Process a document and return text nodes with provenance.

        Args:
            file_path: Path to the document file

        Returns:
            List of TextNode objects with provenance metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If processing fails
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        try:
            # Parse document
            results = await self._parser.aparse(file_path)
            if not results:
                raise ValueError(f"Failed to parse document: {file_path}")

            # Clean document content
            cleaned_results = await self.clean_document_content(results)

            # Convert to documents
            documents = cleaned_results.get_markdown_documents(split_by_page=True)

            # Build nodes with provenance
            nodes = self._node_builder.build_nodes_from_documents(documents, file_path)

            return nodes

        except Exception as e:
            raise ValueError(f"Document processing failed: {str(e)}") from e

    async def clean_document_content(self, content: Any) -> Any:
        """Clean document content by removing unwanted text.

        Args:
            content: Document content to clean

        Returns:
            Cleaned document content
        """
        if hasattr(content, "pages") and content.pages:
            # Clean each page
            for page in content.pages:
                self._cleaner.clean_content(page)
        else:
            # Clean content directly
            content = self._cleaner.clean_content(content)

        return content

    def add_cleanup_pattern(self, pattern: str) -> None:
        """Add a cleanup pattern to the document cleaner.

        Args:
            pattern: Text pattern to remove during cleaning
        """
        self._cleaner.add_cleanup_pattern(pattern)

    def remove_cleanup_pattern(self, pattern: str) -> None:
        """Remove a cleanup pattern from the document cleaner.

        Args:
            pattern: Text pattern to stop removing
        """
        self._cleaner.remove_cleanup_pattern(pattern)

    def get_cleanup_patterns(self) -> List[str]:
        """Get current cleanup patterns.

        Returns:
            List of cleanup patterns
        """
        return self._cleaner.cleanup_patterns.copy()

    def update_parser_config(self, **kwargs) -> None:
        """Update parser configuration and reinitialize.

        Args:
            **kwargs: Parser configuration parameters
        """
        # Update configuration
        if "num_workers" in kwargs:
            self.num_workers = kwargs["num_workers"]
        if "language" in kwargs:
            self.language = kwargs["language"]
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

        # Reinitialize parser
        self._initialize_parser()

    def get_processing_stats(self, nodes: List[TextNode]) -> dict:
        """Get statistics about processed nodes.

        Args:
            nodes: Processed text nodes

        Returns:
            Dictionary with processing statistics
        """
        if not nodes:
            return {"total_nodes": 0}

        total_chars = sum(len(node.text) for node in nodes)
        pages = set(node.metadata.get("page_number", 1) for node in nodes)
        docs = set(node.metadata.get("doc_id", "unknown") for node in nodes)

        return {
            "total_nodes": len(nodes),
            "total_characters": total_chars,
            "unique_pages": len(pages),
            "unique_documents": len(docs),
            "avg_chars_per_node": total_chars / len(nodes),
            "nodes_per_page": len(nodes) / len(pages) if pages else 0,
        }
