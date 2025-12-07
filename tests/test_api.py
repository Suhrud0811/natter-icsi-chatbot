"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api import app, ChatRequest, ChatResponse, HealthResponse


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)
    
    @patch("src.api._chat_engine", None)
    def test_health_when_not_initialized(self, client):
        """Test health check when engine not initialized."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["index_loaded"] is False
    
    @patch("src.api._chat_engine")
    def test_health_when_initialized(self, mock_engine, client):
        """Test health check when engine is initialized."""
        mock_engine.return_value = Mock()
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoint:
    """Tests for /chat endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)
    
    @patch("src.api._chat_engine", None)
    def test_chat_when_not_initialized(self, client):
        """Test chat returns 503 when engine not initialized."""
        response = client.post(
            "/chat",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 503
    
    @patch("src.api._chat_engine")
    def test_chat_empty_message(self, mock_engine, client):
        """Test chat rejects empty message."""
        response = client.post(
            "/chat",
            json={"message": ""}
        )
        
        assert response.status_code == 400
    
    @patch("src.api._chat_engine")
    def test_chat_whitespace_message(self, mock_engine, client):
        """Test chat rejects whitespace-only message."""
        response = client.post(
            "/chat",
            json={"message": "   "}
        )
        
        assert response.status_code == 400
    
    @patch("src.api._chat_engine")
    def test_chat_success(self, mock_engine, client):
        """Test successful chat request with streaming."""
        # Mock the chat_stream generator
        def mock_stream(message):
            yield "This "
            yield "is "
            yield "a "
            yield "test "
            yield "response"
        
        mock_engine.chat_stream = mock_stream
        
        response = client.post(
            "/chat",
            json={"message": "What is this corpus about?"}
        )
        
        assert response.status_code == 200
        # Streaming returns text/plain, not JSON
        assert response.text == "This is a test response"
    
    @patch("src.api._chat_engine")
    def test_chat_handles_exception(self, mock_engine, client):
        """Test chat handles exceptions gracefully."""
        def mock_stream_error(message):
            raise Exception("Test error")
        
        mock_engine.chat_stream = mock_stream_error
        
        response = client.post(
            "/chat",
            json={"message": "What is this?"}
        )
        
        # Streaming catches errors and yields them as text
        assert response.status_code == 200
        assert "Error" in response.text or "Test error" in response.text


class TestRequestModels:
    """Tests for Pydantic request/response models."""
    
    def test_chat_request_validation(self):
        """Test ChatRequest model validation."""
        request = ChatRequest(message="Hello")
        assert request.message == "Hello"
    
    def test_chat_response_model(self):
        """Test ChatResponse model."""
        response = ChatResponse(response="Test response")
        assert response.response == "Test response"
    
    def test_health_response_model(self):
        """Test HealthResponse model."""
        response = HealthResponse(status="healthy", index_loaded=True)
        assert response.status == "healthy"
        assert response.index_loaded is True


class TestAPIIntegration:
    """Integration tests for API (require mocking)."""
    
    @pytest.fixture
    def mock_chat_engine(self):
        """Create mock chat engine."""
        mock = Mock()
        mock.query.return_value = "The ICSI corpus contains meeting transcripts."
        return mock
    
    @patch("src.api._chat_engine")
    def test_full_chat_flow(self, mock_engine):
        """Test complete chat request/response flow with streaming."""
        def mock_stream(message):
            yield "Test "
            yield "response "
            yield "about "
            yield "meetings"
        
        mock_engine.chat_stream = mock_stream
        
        client = TestClient(app)
        
        # First check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Then send chat
        chat_response = client.post(
            "/chat",
            json={"message": "What meetings are in the corpus?"}
        )
        
        assert chat_response.status_code == 200
        assert chat_response.text == "Test response about meetings"


class TestUploadEndpoint:
    """Tests for /upload endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)
    
    @patch("src.api._file_cache")
    @patch("src.api._chat_engine")
    @patch("src.api.FileProcessor")
    @patch("src.api.create_index")
    @patch("src.api.ChatEngine")
    def test_upload_single_file_success(self, mock_chat_engine_class, mock_create_index, 
                                       mock_processor, mock_engine, mock_cache, client):
        """Test successful single file upload."""
        from io import BytesIO
        
        # Mock file cache
        mock_cache.get_file_hash.return_value = "abc123"
        mock_cache.is_cached.return_value = False
        mock_cache.get_all_documents.return_value = [Mock()]
        mock_cache.size.return_value = 1
        
        # Mock file processor
        mock_doc = Mock()
        mock_processor.process_file.return_value = mock_doc
        
        # Mock index creation
        mock_index = Mock()
        mock_create_index.return_value = mock_index
        
        file_content = BytesIO(b"test content")
        
        response = client.post(
            "/upload",
            files={"files": ("test.mrt", file_content, "application/octet-stream")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "files_processed" in data
        assert data["total_files"] == 1
    
    @patch("src.api._file_cache")
    @patch("src.api.create_index")
    @patch("src.api.ChatEngine")
    def test_upload_cached_file(self, mock_chat_engine_class, mock_create_index, mock_cache, client):
        """Test uploading a file that's already cached."""
        from io import BytesIO
        
        # Mock file cache to return cached
        mock_cache.get_file_hash.return_value = "abc123"
        mock_cache.is_cached.return_value = True
        mock_cache.get_all_documents.return_value = [Mock()]
        mock_cache.size.return_value = 1
        
        # Mock index creation
        mock_index = Mock()
        mock_create_index.return_value = mock_index
        
        file_content = BytesIO(b"test content")
        
        response = client.post(
            "/upload",
            files={"files": ("test.mrt", file_content, "application/octet-stream")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "test.mrt" in data["files_cached"]
    
    @patch("src.api.MAX_FILES_PER_UPLOAD", 2)
    def test_upload_too_many_files(self, client):
        """Test uploading more files than allowed."""
        from io import BytesIO
        
        files = [
            ("files", ("test1.mrt", BytesIO(b"content1"), "application/octet-stream")),
            ("files", ("test2.mrt", BytesIO(b"content2"), "application/octet-stream")),
            ("files", ("test3.mrt", BytesIO(b"content3"), "application/octet-stream")),
        ]
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]
    
    @patch("src.api._file_cache")
    @patch("src.api.FileProcessor")
    def test_upload_invalid_file_type(self, mock_processor, mock_cache, client):
        """Test uploading invalid file type."""
        from io import BytesIO
        
        mock_cache.get_file_hash.return_value = "abc123"
        mock_cache.is_cached.return_value = False
        
        # Mock validation error
        mock_processor.validate_file_type.side_effect = ValueError("Invalid file type")
        
        file_content = BytesIO(b"test content")
        
        response = client.post(
            "/upload",
            files={"files": ("test.txt", file_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    @patch("src.api._file_cache")
    @patch("src.api.FileProcessor")
    def test_upload_oversized_file(self, mock_processor, mock_cache, client):
        """Test uploading file that's too large."""
        from io import BytesIO
        
        mock_cache.get_file_hash.return_value = "abc123"
        mock_cache.is_cached.return_value = False
        
        # Mock size validation error
        mock_processor.validate_file_size.side_effect = ValueError("File too large")
        
        file_content = BytesIO(b"x" * (100 * 1024 * 1024))  # 100 MB
        
        response = client.post(
            "/upload",
            files={"files": ("huge.mrt", file_content, "application/octet-stream")}
        )
        
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]


class TestFilesEndpoint:
    """Tests for /files endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)
    
    @patch("src.api._current_files", ["file1.mrt", "file2.mrt"])
    def test_list_files_with_files(self, client):
        """Test listing files when files exist."""
        response = client.get("/files")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert "file1.mrt" in data["files"]
        assert "file2.mrt" in data["files"]
    
    @patch("src.api._current_files", [])
    def test_list_files_empty(self, client):
        """Test listing files when no files uploaded."""
        response = client.get("/files")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["files"] == []
    
    @patch("src.api._file_cache")
    @patch("src.api._chat_engine")
    @patch("src.api._current_files", ["file1.mrt"])
    def test_delete_files(self, mock_engine, mock_cache, client):
        """Test deleting all files."""
        response = client.delete("/files")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        
        # Verify cache was cleared
        mock_cache.clear.assert_called_once()


class TestStreamingEdgeCases:
    """Tests for streaming edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)
    
    @patch("src.api._chat_engine")
    def test_streaming_with_empty_response(self, mock_engine, client):
        """Test streaming when LLM returns empty response."""
        def mock_stream(message):
            return
            yield  # Generator that yields nothing
        
        mock_engine.chat_stream = mock_stream
        
        response = client.post(
            "/chat",
            json={"message": "Test"}
        )
        
        assert response.status_code == 200
        assert response.text == ""
    
    @patch("src.api._chat_engine")
    def test_streaming_with_unicode(self, mock_engine, client):
        """Test streaming with unicode characters."""
        def mock_stream(message):
            yield "Hello "
            yield "世界 "  # Chinese for "world"
            yield "🌍"  # Earth emoji
        
        mock_engine.chat_stream = mock_stream
        
        response = client.post(
            "/chat",
            json={"message": "Test"}
        )
        
        assert response.status_code == 200
        assert "世界" in response.text
        assert "🌍" in response.text
    
    @patch("src.api._chat_engine")
    def test_streaming_with_newlines(self, mock_engine, client):
        """Test streaming with newline characters."""
        def mock_stream(message):
            yield "Line 1\n"
            yield "Line 2\n"
            yield "Line 3"
        
        mock_engine.chat_stream = mock_stream
        
        response = client.post(
            "/chat",
            json={"message": "Test"}
        )
        
        assert response.status_code == 200
        assert response.text.count("\n") == 2

