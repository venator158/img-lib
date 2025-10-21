"""
Simple startup script for the Image Similarity Search application.
Run this to start the FastAPI server.
"""

import sys
import os
from pathlib import Path
import logging

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI application."""
    try:
        import uvicorn
        from backend.main import app
        
        logger.info("Starting Image Similarity Search API server...")
        logger.info("Frontend available at: http://localhost:8000/app")
        logger.info("API documentation at: http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
