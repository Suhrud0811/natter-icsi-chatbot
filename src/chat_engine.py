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
from src.logger import logger


class ChatEngine:
    """Chat engine wrapping LlamaIndex for ICSI corpus queries."""
    
    def __init__(self, index: VectorStoreIndex):
        """Initialize the chat engine with a vector index.
        
        Args:
            index: VectorStoreIndex built from ICSI transcripts
        """
        logger.info("Initializing ChatEngine...")
        self.index = index
        self.llm = OpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
        )
        logger.debug(f"Using OpenAI model: {OPENAI_MODEL}")
        
        # Load system prompt (lazy loading to avoid issues during import)
        system_prompt = load_prompt("system_prompts")
        
        # Create retriever
        self.retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
        logger.debug(f"Retriever configured with top_k={SIMILARITY_TOP_K}")
        
        # Create chat memory for conversation context
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKEN_LIMIT)
        logger.debug(f"Chat memory initialized with token_limit={CHAT_MEMORY_TOKEN_LIMIT}")
        
        # Create chat engine
        self._engine = ContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            memory=self.memory,
            system_prompt=system_prompt,
        )
        logger.info("ChatEngine initialized successfully")
    
    def chat(self, message: str) -> str:
        """Send a message and get a response.
        
        Args:
            message: User's question or message
            
        Returns:
            Assistant's response based on retrieved context
        """
        logger.debug(f"Processing chat message: {message[:100]}...")
        try:
            response = self._engine.chat(message)
            logger.debug("Chat response generated successfully")
            return str(response)
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise
    
    def chat_stream(self, message: str):
        """Stream chat response token by token.
        
        Args:
            message: User's question or message
            
        Yields:
            Response tokens as they're generated
        """
        logger.debug(f"Processing streaming chat message: {message[:100]}...")
        try:
            streaming_response = self._engine.stream_chat(message)
            for token in streaming_response.response_gen:
                yield token
            logger.debug("Streaming chat response completed")
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Reset conversation memory."""
        logger.info("Resetting chat memory")
        self._engine.reset()
    
    def query(self, question: str) -> str:
        """Single query without conversation memory.
        
        Useful for one-off questions via the API.
        
        Args:
            question: Question to answer
            
        Returns:
            Response based on retrieved context
        """
        logger.debug(f"Processing query: {question[:100]}...")
        try:
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=SIMILARITY_TOP_K,
            )
            response = query_engine.query(question)
            logger.debug("Query response generated successfully")
            return str(response)
        except Exception as e:
            logger.error(f"Error in query: {str(e)}", exc_info=True)
            raise


def create_chat_engine(index: VectorStoreIndex) -> ChatEngine:
    """Factory function to create a chat engine.
    
    Args:
        index: VectorStoreIndex to use for retrieval
        
    Returns:
        Configured ChatEngine instance
    """
    return ChatEngine(index)
