"""Chat engine for querying ICSI meeting transcripts."""

from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SIMILARITY_TOP_K,
    CHAT_MEMORY_TOKEN_LIMIT,
    load_prompt,
)


class ChatEngine:
    """Chat engine wrapping LlamaIndex for ICSI corpus queries."""
    
    def __init__(self, index: VectorStoreIndex):
        """Initialize the chat engine with a vector index.
        
        Args:
            index: VectorStoreIndex built from ICSI transcripts
        """
        self.index = index
        self.llm = OpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
        )
        
        # Load system prompt (lazy loading to avoid issues during import)
        system_prompt = load_prompt("system_prompts")
        
        # Create retriever
        self.retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
        
        # Create chat memory for conversation context
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKEN_LIMIT)
        
        # Create chat engine
        self._engine = ContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            memory=self.memory,
            system_prompt=system_prompt,
        )
    
    def chat(self, message: str) -> str:
        """Send a message and get a response.
        
        Args:
            message: User's question or message
            
        Returns:
            Assistant's response based on retrieved context
        """
        response = self._engine.chat(message)
        return str(response)
    
    def reset(self) -> None:
        """Reset conversation memory."""
        self._engine.reset()
    
    def query(self, question: str) -> str:
        """Single query without conversation memory.
        
        Useful for one-off questions via the API.
        
        Args:
            question: Question to answer
            
        Returns:
            Response based on retrieved context
        """
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=SIMILARITY_TOP_K,
        )
        response = query_engine.query(question)
        return str(response)


def create_chat_engine(index: VectorStoreIndex) -> ChatEngine:
    """Factory function to create a chat engine.
    
    Args:
        index: VectorStoreIndex to use for retrieval
        
    Returns:
        Configured ChatEngine instance
    """
    return ChatEngine(index)
