"""
Configuration management for the image similarity search application.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

class Config:
    """Application configuration class."""
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "imsrc")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres123")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "resnet50")
    DEVICE: str = os.getenv("DEVICE", "auto")
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "flatl2")
    USE_PROTOTYPE_FILTERING: bool = os.getenv("USE_PROTOTYPE_FILTERING", "true").lower() == "true"
    
    # API Configuration
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB
    DEFAULT_SEARCH_LIMIT: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", 10))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """Get database configuration as dictionary."""
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "dbname": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD
        }
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return {
            "db_host": cls.DB_HOST,
            "db_port": cls.DB_PORT,
            "db_name": cls.DB_NAME,
            "db_user": cls.DB_USER,
            "db_password": cls.DB_PASSWORD,
            "model_name": cls.MODEL_NAME,
            "device": cls.DEVICE,
            "faiss_index_path": cls.FAISS_INDEX_PATH,
            "faiss_index_type": cls.FAISS_INDEX_TYPE,
            "use_prototype_filtering": cls.USE_PROTOTYPE_FILTERING,
            "max_upload_size": cls.MAX_UPLOAD_SIZE,
            "default_search_limit": cls.DEFAULT_SEARCH_LIMIT,
            "api_host": cls.API_HOST,
            "api_port": cls.API_PORT,
            "log_level": cls.LOG_LEVEL
        }

# Create global config instance
config = Config()