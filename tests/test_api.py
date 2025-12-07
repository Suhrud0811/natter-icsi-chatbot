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
        """Test successful chat request."""
        mock_engine.query.return_value = "This is a test response"
        
        response = client.post(
            "/chat",
            json={"message": "What is this corpus about?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a test response"
    
    @patch("src.api._chat_engine")
    def test_chat_handles_exception(self, mock_engine, client):
        """Test chat handles exceptions gracefully."""
        mock_engine.query.side_effect = Exception("Test error")
        
        response = client.post(
            "/chat",
            json={"message": "What is this?"}
        )
        
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()


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
        """Test complete chat request/response flow."""
        mock_engine.query.return_value = "Test response about meetings"
        
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
        assert "response" in chat_response.json()
