"""FastAPI server for ICSI chatbot."""

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import validate_config, MAX_FILE_SIZE_MB, MAX_FILES_PER_UPLOAD, ALLOWED_FILE_TYPES
from src.ingestion import load_or_create_index, create_index
from src.chat_engine import ChatEngine
from src.file_cache import FileCache
from src.file_processor import FileProcessor
from src.logger import logger


# Global state for the chat engine and file cache
_chat_engine: Optional[ChatEngine] = None
_file_cache: FileCache = FileCache()
_current_files: List[str] = []


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    index_loaded: bool


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    files_processed: List[str]
    files_cached: List[str]
    total_files: int
    status: str


class FilesListResponse(BaseModel):
    """Response model for files list endpoint."""
    files: List[str]
    count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global _chat_engine, _file_cache, _current_files
    
    logger.info("Starting Chatbot API...")
    validate_config()
    
    # Ensure required directories exist
    from src.config import ensure_directories
    ensure_directories()
    logger.info("Required directories created/verified")
    
    # Initialize file cache
    _file_cache = FileCache()
    _current_files = []
    _chat_engine = None
    
    logger.info("API ready - upload files to start chatting")
    yield
    
    # Cleanup
    logger.info("Shutting down API...")


app = FastAPI(
    title="Document Q&A Chatbot",
    description="Upload documents and chat with them using RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns the service status and whether the index is loaded.
    """
    return HealthResponse(
        status="healthy",
        index_loaded=_chat_engine is not None,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files to chat with.
    
    Processes new files and caches them. Skips already-processed files.
    Updates the index with all uploaded documents.
    """
    global _chat_engine, _file_cache, _current_files
    
    logger.info(f"Upload request received: {len(files)} file(s)")
    
    # Validate number of files
    if len(files) > MAX_FILES_PER_UPLOAD:
        logger.warning(f"Too many files uploaded: {len(files)} > {MAX_FILES_PER_UPLOAD}")
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum: {MAX_FILES_PER_UPLOAD}"
        )
    
    processed = []
    cached = []
    errors = []
    
    for file in files:
        try:
            # Validate file type
            FileProcessor.validate_file_type(file.filename, ALLOWED_FILE_TYPES)
            
            # Read file content
            content = await file.read()
            
            # Validate file size
            FileProcessor.validate_file_size(content, MAX_FILE_SIZE_MB)
            
            # Check cache
            file_hash = _file_cache.get_file_hash(content)
            
            if _file_cache.is_cached(file_hash):
                logger.debug(f"File already cached: {file.filename}")
                cached.append(file.filename)
                continue
            
            # Process new file
            logger.info(f"Processing new file: {file.filename}")
            document = FileProcessor.process_file(file.filename, content)
            _file_cache.add(file_hash, file.filename, document)
            processed.append(file.filename)
            
            if file.filename not in _current_files:
                _current_files.append(file.filename)
                
        except ValueError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except Exception as e:
            errors.append(f"{file.filename}: Unexpected error - {str(e)}")
    
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"Errors processing files: {'; '.join(errors)}"
        )
    
    # Rebuild index with all cached documents
    all_docs = _file_cache.get_all_documents()
    if all_docs:
        try:
            logger.info(f"Creating vector index with {len(all_docs)} document(s)")
            index = create_index(all_docs, persist=False)
            _chat_engine = ChatEngine(index)
            logger.info("Chat engine initialized successfully")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error creating index: {str(e)}"
            )
    
    return UploadResponse(
        files_processed=processed,
        files_cached=cached,
        total_files=_file_cache.size(),
        status="ready" if all_docs else "no_files"
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with streaming support.
    
    Send a message and receive a streaming response based on uploaded files.
    Tokens are streamed as they're generated for better UX.
    """
    logger.info(f"Chat request: {request.message[:50]}...")
    
    if _chat_engine is None:
        logger.warning("Chat request rejected: no files uploaded")
        raise HTTPException(
            status_code=503,
            detail="No files uploaded. Please upload files first using /upload endpoint.",
        )
    
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )
    
    try:
        def generate():
            """Generator function for streaming response."""
            try:
                for token in _chat_engine.chat_stream(request.message):
                    yield token
                logger.info("Streaming chat response completed")
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                yield f"\n\nError: {str(e)}"
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/files", response_model=FilesListResponse)
async def list_files():
    """List all uploaded files."""
    return FilesListResponse(
        files=_current_files,
        count=len(_current_files)
    )


@app.delete("/files")
async def clear_files():
    """Clear all uploaded files and reset the index."""
    global _chat_engine, _file_cache, _current_files
    
    _file_cache.clear()
    _chat_engine = None
    _current_files = []
    
    return {"status": "cleared", "message": "All files have been removed"}


def get_app() -> FastAPI:
    """Get the FastAPI application instance."""
    return app
