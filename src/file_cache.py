"""File cache manager for uploaded documents.

Prevents reprocessing of files by caching them based on content hash.
Uses SHA-256 to detect if the same file content has been uploaded before.
"""

import hashlib
from typing import Dict, List, Optional
from llama_index.core import Document

from src.logger import logger


class FileCache:
    """Smart cache for processed document files.
    
    Tracks uploaded files by their content hash (not filename) so that
    uploading the same file twice doesn't require reprocessing.
    """
    
    def __init__(self):
        """Initialize an empty cache."""
        logger.debug("Initializing FileCache")
        # Map: content_hash -> Document
        self._documents_by_hash: Dict[str, Document] = {}
        
        # Map: filename -> content_hash (for tracking)
        self._hash_by_filename: Dict[str, str] = {}
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Calculate unique hash for file content.
        
        Uses SHA-256 to create a fingerprint of the file.
        Same content = same hash, even if filename differs.
        
        Args:
            file_content: Raw file bytes
            
        Returns:
            64-character hexadecimal hash string
        """
        return hashlib.sha256(file_content).hexdigest()
    
    def is_cached(self, content_hash: str) -> bool:
        """Check if this file content has been processed before.
        
        Args:
            content_hash: SHA-256 hash of file content
            
        Returns:
            True if we've already processed this exact content
        """
        return content_hash in self._documents_by_hash
    
    def add(self, content_hash: str, filename: str, document: Document) -> None:
        """Store a processed document in the cache.
        
        Args:
            content_hash: SHA-256 hash of the file content
            filename: Original filename (for display purposes)
            document: Parsed LlamaIndex Document
        """
        logger.debug(f"Adding to cache: {filename} (hash: {content_hash[:8]}...)")
        self._documents_by_hash[content_hash] = document
        self._hash_by_filename[filename] = content_hash
        logger.info(f"Cached document: {filename} (total: {len(self._documents_by_hash)})")
    
    def get(self, content_hash: str) -> Optional[Document]:
        """Retrieve a cached document by its content hash.
        
        Args:
            content_hash: SHA-256 hash of file content
            
        Returns:
            Cached Document if found, None otherwise
        """
        doc = self._documents_by_hash.get(content_hash)
        if doc:
            logger.debug(f"Cache hit for hash: {content_hash[:8]}...")
        else:
            logger.debug(f"Cache miss for hash: {content_hash[:8]}...")
        return doc
    
    def get_all_documents(self) -> List[Document]:
        """Get all cached documents for indexing.
        
        Returns:
            List of all processed Documents
        """
        return list(self._documents_by_hash.values())
    
    def get_all_filenames(self) -> List[str]:
        """Get names of all uploaded files.
        
        Returns:
            List of filenames that have been uploaded
        """
        return list(self._hash_by_filename.keys())
    
    def clear(self) -> None:
        """Remove all cached documents and start fresh."""
        count = len(self._documents_by_hash)
        logger.info(f"Clearing cache ({count} document(s))")
        self._documents_by_hash.clear()
        self._hash_by_filename.clear()
        logger.debug("Cache cleared successfully")
    
    def size(self) -> int:
        """Get the number of unique documents in cache.
        
        Returns:
            Count of cached documents
        """
        return len(self._documents_by_hash)

