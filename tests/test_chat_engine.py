"""Tests for chat engine module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.chat_engine import ChatEngine, create_chat_engine


class TestChatEngine:
    """Tests for ChatEngine class."""
    
    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        mock = Mock()
        mock.as_retriever.return_value = Mock()
        mock.as_query_engine.return_value = Mock()
        return mock
    
    @patch("src.chat_engine.load_prompt")
    @patch("src.chat_engine.OpenAI")
    @patch("src.chat_engine.ContextChatEngine")
    def test_chat_engine_initialization(self, mock_context_engine, mock_openai, mock_load_prompt, mock_index):
        """Test ChatEngine initializes correctly."""
        mock_load_prompt.return_value = "Test system prompt"
        engine = ChatEngine(mock_index)
        
        assert engine.index == mock_index
        mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
    
    @patch("src.chat_engine.load_prompt")
    @patch("src.chat_engine.OpenAI")
    @patch("src.chat_engine.ContextChatEngine")
    def test_chat_returns_string(self, mock_context_engine, mock_openai, mock_load_prompt, mock_index):
        """Test chat method returns a string response."""
        mock_load_prompt.return_value = "Test system prompt"
        mock_engine_instance = Mock()
        mock_engine_instance.chat.return_value = "Test response"
        mock_context_engine.from_defaults.return_value = mock_engine_instance
        
        engine = ChatEngine(mock_index)
        response = engine.chat("Hello")
        
        assert isinstance(response, str)
        assert response == "Test response"
    
    @patch("src.chat_engine.load_prompt")
    @patch("src.chat_engine.OpenAI")
    @patch("src.chat_engine.ContextChatEngine")
    def test_reset_clears_memory(self, mock_context_engine, mock_openai, mock_load_prompt, mock_index):
        """Test reset method clears conversation."""
        mock_load_prompt.return_value = "Test system prompt"
        mock_engine_instance = Mock()
        mock_context_engine.from_defaults.return_value = mock_engine_instance
        
        engine = ChatEngine(mock_index)
        engine.reset()
        
        mock_engine_instance.reset.assert_called_once()
    
    @patch("src.chat_engine.load_prompt")
    @patch("src.chat_engine.OpenAI")
    @patch("src.chat_engine.ContextChatEngine")
    def test_query_uses_query_engine(self, mock_context_engine, mock_openai, mock_load_prompt, mock_index):
        """Test query method uses query engine for stateless queries."""
        mock_load_prompt.return_value = "Test system prompt"
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = "Query response"
        mock_index.as_query_engine.return_value = mock_query_engine
        
        engine = ChatEngine(mock_index)
        response = engine.query("What is this about?")
        
        assert response == "Query response"
        mock_index.as_query_engine.assert_called()


class TestCreateChatEngine:
    """Tests for create_chat_engine factory function."""
    
    @patch("src.chat_engine.load_prompt")
    @patch("src.chat_engine.OpenAI")
    @patch("src.chat_engine.ContextChatEngine")
    def test_creates_chat_engine_instance(self, mock_context_engine, mock_openai, mock_load_prompt):
        """Test factory creates ChatEngine instance."""
        mock_load_prompt.return_value = "Test system prompt"
        mock_index = Mock()
        mock_index.as_retriever.return_value = Mock()
        
        engine = create_chat_engine(mock_index)
        
        assert isinstance(engine, ChatEngine)


class TestSystemPrompt:
    """Tests for system prompt loading."""
    
    @patch("src.chat_engine.load_prompt")
    def test_system_prompt_is_loaded(self, mock_load_prompt):
        """Test that system prompt is loaded during initialization."""
        from src.config import load_prompt as real_load_prompt
        
        # Load the actual prompt file
        prompt_content = real_load_prompt("system_prompts")
        
        # Verify it contains expected content
        assert "ICSI" in prompt_content
        assert "meeting" in prompt_content.lower()
        assert "transcript" in prompt_content.lower()
