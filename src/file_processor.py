"""File processor for uploaded documents.

Handles MRT (Meeting Room Transcript) files and converts them to LlamaIndex Documents.
"""

import tempfile
from pathlib import Path
from llama_index.core import Document

from src.ingestion import parse_mrt_file


class FileProcessor:
    """Process uploaded MRT files into Documents."""
    
    @staticmethod
    def process_file(filename: str, content: bytes) -> Document:
        """Process uploaded MRT file into a Document.
        
        Args:
            filename: Original filename (e.g., "Bmr001.mrt")
            content: Raw file content as bytes
            
        Returns:
            Parsed LlamaIndex Document with text and metadata
            
        Raises:
            ValueError: If file is not an .mrt file or parsing fails
        """
        file_extension = Path(filename).suffix.lower()
        
        # Only accept .mrt files
        if file_extension != '.mrt':
            raise ValueError(
                f"Cannot process '{file_extension}' files. "
                f"Only .mrt (Meeting Room Transcript) files are supported."
            )
        
        # Create temporary file for XML parsing
        # (XML parser needs a file path, not bytes)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mrt', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Parse the MRT XML file using existing parser
            parsed_document = parse_mrt_file(temp_file_path)
            
            if parsed_document is None:
                raise ValueError(
                    f"Failed to parse '{filename}'. "
                    f"The file may be corrupted or not a valid MRT file."
                )
            
            # Add the original filename to metadata for tracking
            parsed_document.metadata['uploaded_filename'] = filename
            
            return parsed_document
            
        finally:
            # Always clean up the temporary file
            temp_file_path.unlink(missing_ok=True)
    
    @staticmethod
    def validate_file_size(content: bytes, max_size_mb: int) -> None:
        """Check if file size is within allowed limit.
        
        Args:
            content: File content as bytes
            max_size_mb: Maximum allowed size in megabytes
            
        Raises:
            ValueError: If file exceeds size limit
        """
        # Convert bytes to megabytes
        file_size_mb = len(content) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"File is too large ({file_size_mb:.2f}MB). "
                f"Maximum allowed size is {max_size_mb}MB."
            )
    
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: list) -> None:
        """Check if file type is allowed.
        
        Args:
            filename: Original filename
            allowed_extensions: List of allowed file extensions (e.g., ['.mrt'])
            
        Raises:
            ValueError: If file extension is not in allowed list
        """
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            allowed_types_display = ', '.join(allowed_extensions)
            raise ValueError(
                f"'{file_extension}' files are not allowed. "
                f"Allowed file types: {allowed_types_display}"
            )


