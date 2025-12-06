"""Configuration settings for the ICSI Chatbot."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "transcripts"
STORAGE_DIR = PROJECT_ROOT / "storage"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking settings - optimized for meeting transcripts
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval settings
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))

# Chat memory settings
CHAT_MEMORY_TOKEN_LIMIT = int(os.getenv("CHAT_MEMORY_TOKEN_LIMIT", "3000"))

# Metadata settings
NOTES_MAX_LENGTH = int(os.getenv("NOTES_MAX_LENGTH", "500"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# File upload settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "5"))
ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", ".mrt").split(",")
def validate_config() -> None:
    """Validate required configuration."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. "
            "Please set it in .env file or as environment variable."
        )
def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Content of the prompt file
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Please create it in the prompts/ directory."
        )
    return prompt_file.read_text(encoding="utf-8").strip()