"""Tests for file processor module."""

import pytest
from pathlib import Path
from llama_index.core import Document

from src.file_processor import FileProcessor


class TestProcessFile:
    """Tests for process_file method."""
    
    def test_process_file_invalid_extension(self):
        """Test processing file with invalid extension."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.process_file("test.txt", b"content")
        
        assert ".txt" in str(exc_info.value)
        assert ".mrt" in str(exc_info.value)
    
    def test_process_file_pdf_extension(self):
        """Test processing PDF file (not supported)."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.process_file("document.pdf", b"content")
        
        assert ".pdf" in str(exc_info.value)
    
    def test_process_file_no_extension(self):
        """Test processing file with no extension."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.process_file("noextension", b"content")
        
        assert "Only .mrt" in str(exc_info.value)
    
    def test_process_file_corrupted_xml(self):
        """Test processing corrupted XML file."""
        corrupted_content = b"<Meeting>Not valid XML"
        
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.process_file("corrupted.mrt", corrupted_content)
        
        # Should fail to parse
        assert "Failed to parse" in str(exc_info.value) or "corrupted" in str(exc_info.value).lower()
    
    def test_process_file_empty_content(self):
        """Test processing empty MRT file."""
        empty_content = b""
        
        with pytest.raises(ValueError):
            FileProcessor.process_file("empty.mrt", empty_content)
    
    def test_process_file_adds_uploaded_filename(self):
        """Test that uploaded filename is added to metadata."""
        # Valid minimal MRT content
        valid_mrt = b"""<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Test001">
    <Transcript>
        <Segment Participant="me001" StartTime="0.0" EndTime="1.0">
            Hello world
        </Segment>
    </Transcript>
</Meeting>"""
        
        doc = FileProcessor.process_file("test.mrt", valid_mrt)
        
        assert isinstance(doc, Document)
        assert doc.metadata.get("uploaded_filename") == "test.mrt"


class TestValidateFileSize:
    """Tests for validate_file_size method."""
    
    def test_validate_file_size_within_limit(self):
        """Test file size validation for file within limit."""
        content = b"x" * 1024  # 1 KB
        
        # Should not raise
        FileProcessor.validate_file_size(content, max_size_mb=10)
    
    def test_validate_file_size_exactly_at_limit(self):
        """Test file size exactly at the limit."""
        # 1 MB exactly
        content = b"x" * (1024 * 1024)
        
        # Should not raise
        FileProcessor.validate_file_size(content, max_size_mb=1)
    
    def test_validate_file_size_just_over_limit(self):
        """Test file size just over the limit."""
        # 1 MB + 1 byte
        content = b"x" * (1024 * 1024 + 1)
        
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.validate_file_size(content, max_size_mb=1)
        
        assert "too large" in str(exc_info.value).lower()
        assert "1" in str(exc_info.value)  # Should mention the limit
    
    def test_validate_file_size_way_over_limit(self):
        """Test file size way over the limit."""
        # 100 MB
        content = b"x" * (100 * 1024 * 1024)
        
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.validate_file_size(content, max_size_mb=10)
        
        assert "too large" in str(exc_info.value).lower()
        assert "100" in str(exc_info.value)  # Should show actual size
    
    def test_validate_file_size_empty_file(self):
        """Test validation of empty file."""
        content = b""
        
        # Empty file should pass (0 MB < any limit)
        FileProcessor.validate_file_size(content, max_size_mb=10)
    
    def test_validate_file_size_very_small_limit(self):
        """Test with very small size limit."""
        content = b"x" * 1024  # 1 KB
        
        with pytest.raises(ValueError):
            FileProcessor.validate_file_size(content, max_size_mb=0)  # 0 MB limit


class TestValidateFileType:
    """Tests for validate_file_type method."""
    
    def test_validate_file_type_valid_mrt(self):
        """Test validation of valid .mrt file."""
        # Should not raise
        FileProcessor.validate_file_type("test.mrt", [".mrt"])
    
    def test_validate_file_type_uppercase_extension(self):
        """Test validation with uppercase extension."""
        # Should not raise (case-insensitive)
        FileProcessor.validate_file_type("test.MRT", [".mrt"])
    
    def test_validate_file_type_mixed_case(self):
        """Test validation with mixed case extension."""
        # Should not raise
        FileProcessor.validate_file_type("test.MrT", [".mrt"])
    
    def test_validate_file_type_invalid_extension(self):
        """Test validation of invalid file type."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.validate_file_type("test.txt", [".mrt"])
        
        assert ".txt" in str(exc_info.value)
        assert ".mrt" in str(exc_info.value)
    
    def test_validate_file_type_multiple_allowed(self):
        """Test validation with multiple allowed extensions."""
        # Should not raise
        FileProcessor.validate_file_type("test.mrt", [".mrt", ".xml", ".txt"])
    
    def test_validate_file_type_no_extension(self):
        """Test validation of file with no extension."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.validate_file_type("noextension", [".mrt"])
        
        assert "not allowed" in str(exc_info.value).lower()
    
    def test_validate_file_type_wrong_extension_from_multiple(self):
        """Test wrong extension when multiple are allowed."""
        with pytest.raises(ValueError) as exc_info:
            FileProcessor.validate_file_type("test.pdf", [".mrt", ".xml"])
        
        assert ".pdf" in str(exc_info.value)
        assert ".mrt" in str(exc_info.value)
        assert ".xml" in str(exc_info.value)
    
    def test_validate_file_type_empty_allowed_list(self):
        """Test validation with empty allowed extensions list."""
        with pytest.raises(ValueError):
            FileProcessor.validate_file_type("test.mrt", [])


class TestFileProcessorIntegration:
    """Integration tests for FileProcessor."""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        filename = "test.mrt"
        content = b"x" * 1024  # 1 KB
        
        # All validations should pass
        FileProcessor.validate_file_type(filename, [".mrt"])
        FileProcessor.validate_file_size(content, max_size_mb=10)
    
    def test_validation_order_matters(self):
        """Test that validation can be done in any order."""
        filename = "test.txt"
        content = b"x" * (100 * 1024 * 1024)  # 100 MB
        
        # Type validation fails
        with pytest.raises(ValueError):
            FileProcessor.validate_file_type(filename, [".mrt"])
        
        # Size validation also fails
        with pytest.raises(ValueError):
            FileProcessor.validate_file_size(content, max_size_mb=10)
    
    def test_case_insensitive_extension_handling(self):
        """Test that extension handling is case-insensitive throughout."""
        test_cases = [
            "file.mrt",
            "file.MRT",
            "file.Mrt",
            "file.mRt",
        ]
        
        for filename in test_cases:
            # Should not raise for any case variation
            FileProcessor.validate_file_type(filename, [".mrt"])
