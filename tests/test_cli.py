"""Tests for CLI module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import requests

from src.cli import (
    check_api_health,
    upload_file,
    send_chat_message,
    list_files,
    clear_files,
)


class TestCheckApiHealth:
    """Tests for API health check function."""
    
    @patch('src.cli.requests.get')
    def test_check_api_health_success(self, mock_get):
        """Test successful API health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is True
        mock_get.assert_called_once()
    
    @patch('src.cli.requests.get')
    def test_check_api_health_failure_status(self, mock_get):
        """Test health check with non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is False
    
    @patch('src.cli.requests.get')
    def test_check_api_health_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        
        result = check_api_health()
        
        assert result is False
    
    @patch('src.cli.requests.get')
    def test_check_api_health_timeout(self, mock_get):
        """Test health check with timeout."""
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        result = check_api_health()
        
        assert result is False


class TestUploadFile:
    """Tests for file upload function."""
    
    @patch('src.cli.requests.post')
    @patch('builtins.open', create=True)
    def test_upload_file_success(self, mock_open, mock_post):
        """Test successful file upload."""
        # Mock file reading
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "files_processed": ["test.mrt"],
            "files_cached": [],
            "total_files": 1,
            "status": "ready"
        }
        mock_post.return_value = mock_response
        
        file_path = Path("test.mrt")
        result = upload_file(file_path)
        
        assert result["status"] == "ready"
        assert "test.mrt" in result["files_processed"]
        mock_post.assert_called_once()
    
    @patch('src.cli.requests.post')
    @patch('builtins.open', create=True)
    def test_upload_file_http_error(self, mock_open, mock_post):
        """Test upload with HTTP error."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("Bad request")
        mock_post.return_value = mock_response
        
        file_path = Path("test.mrt")
        
        with pytest.raises(requests.HTTPError):
            upload_file(file_path)
    
    @patch('src.cli.requests.post')
    @patch('builtins.open', create=True)
    def test_upload_file_connection_error(self, mock_open, mock_post):
        """Test upload with connection error."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        
        file_path = Path("test.mrt")
        
        with pytest.raises(requests.ConnectionError):
            upload_file(file_path)


class TestSendChatMessage:
    """Tests for send chat message function."""
    
    @patch('src.cli.requests.post')
    @patch('builtins.print')
    def test_send_chat_message_streaming(self, mock_print, mock_post):
        """Test streaming chat message."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [
            "Hello ",
            "this ",
            "is ",
            "a ",
            "test"
        ]
        mock_post.return_value = mock_response
        
        send_chat_message("Test message")
        
        # Verify streaming was enabled
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['stream'] is True
        
        # Verify all chunks were printed
        assert mock_print.call_count == 5
    
    @patch('src.cli.requests.post')
    def test_send_chat_message_http_error(self, mock_post):
        """Test chat with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.HTTPError("Service unavailable")
        mock_post.return_value = mock_response
        
        with pytest.raises(requests.HTTPError):
            send_chat_message("Test message")
    
    @patch('src.cli.requests.post')
    def test_send_chat_message_timeout(self, mock_post):
        """Test chat with timeout."""
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        with pytest.raises(requests.Timeout):
            send_chat_message("Test message")


class TestListFiles:
    """Tests for list files function."""
    
    @patch('src.cli.requests.get')
    def test_list_files_success(self, mock_get):
        """Test successful file listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "files": ["file1.mrt", "file2.mrt"],
            "count": 2
        }
        mock_get.return_value = mock_response
        
        result = list_files()
        
        assert result["count"] == 2
        assert len(result["files"]) == 2
        mock_get.assert_called_once()
    
    @patch('src.cli.requests.get')
    def test_list_files_empty(self, mock_get):
        """Test listing when no files uploaded."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "files": [],
            "count": 0
        }
        mock_get.return_value = mock_response
        
        result = list_files()
        
        assert result["count"] == 0
        assert result["files"] == []
    
    @patch('src.cli.requests.get')
    def test_list_files_connection_error(self, mock_get):
        """Test list files with connection error."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        
        with pytest.raises(requests.ConnectionError):
            list_files()


class TestClearFiles:
    """Tests for clear files function."""
    
    @patch('src.cli.requests.delete')
    def test_clear_files_success(self, mock_delete):
        """Test successful file clearing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "cleared",
            "message": "All files have been removed"
        }
        mock_delete.return_value = mock_response
        
        result = clear_files()
        
        assert result["status"] == "cleared"
        mock_delete.assert_called_once()
    
    @patch('src.cli.requests.delete')
    def test_clear_files_http_error(self, mock_delete):
        """Test clear files with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        mock_delete.return_value = mock_response
        
        with pytest.raises(requests.HTTPError):
            clear_files()
    
    @patch('src.cli.requests.delete')
    def test_clear_files_connection_error(self, mock_delete):
        """Test clear files with connection error."""
        mock_delete.side_effect = requests.ConnectionError("Connection refused")
        
        with pytest.raises(requests.ConnectionError):
            clear_files()


class TestCLIIntegration:
    """Integration tests for CLI functions."""
    
    @patch('src.cli.requests.get')
    @patch('src.cli.requests.post')
    @patch('builtins.open', create=True)
    def test_full_workflow(self, mock_open, mock_post, mock_get):
        """Test complete CLI workflow: health check, upload, chat."""
        # Mock health check
        health_response = Mock()
        health_response.status_code = 200
        
        # Mock upload
        upload_response = Mock()
        upload_response.status_code = 200
        upload_response.json.return_value = {
            "files_processed": ["test.mrt"],
            "total_files": 1,
            "status": "ready"
        }
        
        # Mock chat
        chat_response = Mock()
        chat_response.status_code = 200
        chat_response.iter_content.return_value = ["Test ", "response"]
        
        mock_get.return_value = health_response
        mock_post.side_effect = [upload_response, chat_response]
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Execute workflow
        assert check_api_health() is True
        
        file_path = Path("test.mrt")
        upload_result = upload_file(file_path)
        assert upload_result["status"] == "ready"
        
        # Chat would print, so we just verify it doesn't raise
        with patch('builtins.print'):
            send_chat_message("Test question")
