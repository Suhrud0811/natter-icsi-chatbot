"""Command-line interface for chatbot - Thin client that calls the API."""

import sys
from pathlib import Path

import requests
from requests.exceptions import RequestException, ConnectionError, Timeout

from src.config import validate_config, CLI_API_URL, CLI_API_TIMEOUT
from src.logger import logger


def check_api_health() -> bool:
    """Check if API server is running and healthy.
    
    Returns:
        True if API is accessible, False otherwise
    """
    try:
        response = requests.get(
            f"{CLI_API_URL}/health",
            timeout=2
        )
        return response.status_code == 200
    except RequestException:
        return False


def upload_file(file_path: Path) -> dict:
    """Upload a file to the API server.
    
    Args:
        file_path: Path to the file to upload
        
    Returns:
        API response as dict
        
    Raises:
        RequestException: If upload fails
    """
    with open(file_path, 'rb') as f:
        files = {'files': (file_path.name, f, 'application/octet-stream')}
        response = requests.post(
            f"{CLI_API_URL}/upload",
            files=files,
            timeout=CLI_API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()


def send_chat_message(message: str) -> str:
    """Send a chat message to the API.
    
    Args:
        message: User's message
        
    Returns:
        AI response text
        
    Raises:
        RequestException: If chat request fails
    """
    response = requests.post(
        f"{CLI_API_URL}/chat",
        json={"message": message},
        timeout=CLI_API_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["response"]


def list_files() -> dict:
    """Get list of uploaded files from API.
    
    Returns:
        API response with files list
        
    Raises:
        RequestException: If request fails
    """
    response = requests.get(
        f"{CLI_API_URL}/files",
        timeout=CLI_API_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def clear_files() -> dict:
    """Clear all uploaded files via API.
    
    Returns:
        API response confirming deletion
        
    Raises:
        RequestException: If request fails
    """
    response = requests.delete(
        f"{CLI_API_URL}/files",
        timeout=CLI_API_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def run_cli():
    """Run the interactive CLI chatbot (thin client)."""
    logger.info("Starting CLI chatbot")
    print("=" * 60)
    print("Meeting Transcript Chatbot (API Client)")
    print("=" * 60)
    print()
    
    # Validate configuration
    try:
        validate_config()
        from src.config import ensure_directories
        ensure_directories()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Check API server health
    print(f"Connecting to API server at {CLI_API_URL}...")
    if not check_api_health():
        logger.error(f"Cannot connect to API server at {CLI_API_URL}")
        print()
        print(" Error: API server is not running or not accessible!")
        print()
        print("Please start the API server first:")
        print("  python main.py api")
        print()
        print("Or set CLI_API_URL to point to a running server:")
        print("  export CLI_API_URL=http://your-server:8000")
        sys.exit(1)
    
    print("✓ Connected to API server")
    print()
    print("Commands:")
    print("  upload <file.mrt>     - Upload a meeting transcript file")
    print("  files                 - List uploaded files")
    print("  clear                 - Clear all uploaded files")
    print("  quit or exit          - Exit the chatbot")
    print("-" * 60)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            # Handle quit
            if command in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            # Handle upload command
            if command == "upload":
                if len(parts) < 2:
                    print("Usage: upload <file.mrt>")
                    print("Example: upload data/transcripts/Bmr001.mrt\n")
                    continue
                
                file_path = Path(parts[1].strip())
                
                if not file_path.exists():
                    print(f"Error: File not found: {file_path}\n")
                    continue
                
                try:
                    print(f"Uploading '{file_path.name}'...")
                    logger.info(f"Uploading file via CLI: {file_path}")
                    result = upload_file(file_path)
                    
                    # Display results
                    if result.get("files_processed"):
                        print(f"✓ Processed: {', '.join(result['files_processed'])}")
                    if result.get("files_cached"):
                        print(f"⚡ Cached (already uploaded): {', '.join(result['files_cached'])}")
                    
                    print(f"Total files in system: {result['total_files']}")
                    print(f"Status: {result['status']}\n")
                    logger.debug(f"Upload successful: {result}")
                    
                except RequestException as e:
                    logger.error(f"Upload failed: {e}", exc_info=True)
                    print(f"Error uploading file: {e}\n")
                continue
            
            # Handle files command
            if command == "files":
                try:
                    result = list_files()
                    files = result.get("files", [])
                    
                    if files:
                        print(f"\nUploaded files ({result['count']}):")
                        for filename in files:
                            print(f"  - {filename}")
                        print()
                    else:
                        print("No files uploaded yet.\n")
                        
                except RequestException as e:
                    print(f"Error listing files: {e}\n")
                continue
            
            # Handle clear command
            if command == "clear":
                try:
                    result = clear_files()
                    print(f"✓ {result.get('message', 'All files cleared')}\n")
                except RequestException as e:
                    print(f"Error clearing files: {e}\n")
                continue
            
            # Handle chat (default)
            try:
                logger.debug(f"Sending chat message: {user_input[:50]}...")
                print("\nAssistant: ", end="", flush=True)
                response = send_chat_message(user_input)
                print(response)
                print()
                
            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                print("\n Error: Lost connection to API server\n")
            except Timeout as e:
                logger.error(f"Timeout error: {e}")
                print("\n Error: Request timed out\n")
            except RequestException as e:
                # Check if it's a "no files uploaded" error
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 503:
                        error_detail = e.response.json().get('detail', '')
                        if 'No files uploaded' in error_detail:
                            logger.warning("Chat attempted with no files uploaded")
                            print("\nPlease upload a file first using: upload <file.mrt>\n")
                        else:
                            logger.error(f"API error (503): {error_detail}")
                            print(f"\n Error: {error_detail}\n")
                    else:
                        logger.error(f"API error ({e.response.status_code}): {e}")
                        print(f"\n Error: {e}\n")
                else:
                    logger.error(f"Request error: {e}", exc_info=True)
                    print(f"\n Error: {e}\n")
            
        except KeyboardInterrupt:
            logger.info("CLI interrupted by user")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error in CLI: {e}", exc_info=True)
            print(f"\n Unexpected error: {e}\n")


if __name__ == "__main__":
    run_cli()
