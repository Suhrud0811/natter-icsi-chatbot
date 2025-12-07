"""Main entry point for ICSI Chatbot."""
import sys
import argparse
from src.config import API_HOST, API_PORT
from src.logger import logger
def main():
    parser = argparse.ArgumentParser(
        description="ICSI Meeting Corpus Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli          # Run interactive CLI chatbot
  python main.py api          # Start FastAPI server
  python main.py api --port 8080  # Start server on custom port
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run interactive CLI chatbot")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument(
        "--host",
        default=API_HOST,
        help=f"Host to bind to (default: {API_HOST})",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=API_PORT,
        help=f"Port to bind to (default: {API_PORT})",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    if args.command == "cli":
        logger.info("Starting ICSI Chatbot in CLI mode")
        from src.cli import run_cli
        run_cli()
    
    elif args.command == "api":
        logger.info(f"Starting ICSI Chatbot API server on {args.host}:{args.port}")
        import uvicorn
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    
    else:
        logger.warning("No command specified")
        parser.print_help()
        sys.exit(1)
if __name__ == "__main__":
    main()