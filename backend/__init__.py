"""
Image Similarity Search Backend Module

This module provides a complete image similarity search system with:
- FAISS-powered vector search
- Deep learning image embeddings (ResNet)
- PostgreSQL database integration
- Prototype-based filtering for efficient search
- REST API backend with FastAPI

Main components:
- vector_processor: Image embedding generation and FAISS operations
- database: PostgreSQL database management and operations  
- main: FastAPI web application and REST endpoints
- config: Configuration management and environment variables
"""

__version__ = "1.0.0"
__author__ = "Image Similarity Search Team"

from .vector_processor import VectorSearchEngine, ImageEmbedder, FaissIndexManager
from .database import ImageSimilarityService, DatabaseManager, DatabaseConfig
from .config import Config, config

__all__ = [
    'VectorSearchEngine',
    'ImageEmbedder', 
    'FaissIndexManager',
    'ImageSimilarityService',
    'DatabaseManager',
    'DatabaseConfig',
    'Config',
    'config'
]