"""
Database utilities for managing images, vectors, and prototypes.
Handles PostgreSQL operations and vector synchronization.
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration.

    Values default to environment variables when not provided so that
    creating `DatabaseConfig()` picks up settings from `.env` or the
    environment (matching `backend/config.py`).
    """
    host: str = None
    port: int = None
    dbname: str = None
    user: str = None
    password: str = None

    def __post_init__(self):
        # Read from environment if values not explicitly provided
        self.host = self.host or os.getenv("DB_HOST", "localhost")
        self.port = int(self.port or os.getenv("DB_PORT", 5432))
        self.dbname = self.dbname or os.getenv("DB_NAME", "imsrc")
        self.user = self.user or os.getenv("DB_USER", "postgres")
        self.password = self.password or os.getenv("DB_PASSWORD", "postgres123")

    def get_connection_string(self) -> str:
        return f"host={self.host} port={self.port} dbname={self.dbname} user={self.user} password={self.password}"


class DatabaseManager:
    """
    Manages database connections and operations for the image similarity search system.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(self.config.get_connection_string())
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_script(self, script_path: str):
        """Execute SQL script from file."""
        with open(script_path, 'r') as f:
            script = f.read()
            
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(script)
            conn.commit()
            logger.info(f"Successfully executed script: {script_path}")


class ImageManager:
    """
    Manages image operations in the database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Get image data by ID."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT i.image_id, i.image_data, i.metadata, i.category_id, c.category_name
                    FROM images i
                    JOIN categories c ON i.category_id = c.category_id
                    WHERE i.image_id = %s
                """, (image_id,))
                return dict(cur.fetchone()) if cur.rowcount > 0 else None
    
    def get_images_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """Get all images for a specific category."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT i.image_id, i.image_data, i.metadata, i.category_id, c.category_name
                    FROM images i
                    JOIN categories c ON i.category_id = c.category_id
                    WHERE i.category_id = %s
                """, (category_id,))
                return [dict(row) for row in cur.fetchall()]
    
    def get_all_images(self) -> List[Dict[str, Any]]:
        """Get all images from the database."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT i.image_id, i.image_data, i.metadata, i.category_id, c.category_name
                    FROM images i
                    JOIN categories c ON i.category_id = c.category_id
                    ORDER BY i.image_id
                """)
                return [dict(row) for row in cur.fetchall()]
    
    def insert_image(self, image_data: bytes, category_id: int, metadata: Dict = None) -> int:
        """Insert a new image and return its ID."""
        metadata = metadata or {}
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO images (image_data, metadata, category_id)
                    VALUES (%s, %s::jsonb, %s)
                    RETURNING image_id;
                """, (psycopg2.Binary(image_data), json.dumps(metadata), category_id))
                
                image_id = cur.fetchone()[0]
                conn.commit()
                return image_id
    
    def get_images_without_vectors(self) -> List[Dict[str, Any]]:
        """Get images that don't have corresponding vectors."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                    i.image_id, 
                    i.image_data, 
                    i.metadata, 
                    i.category_id, 
                    c.category_name,
                    cat_stats.total_images_in_category
                FROM images i
                JOIN categories c ON i.category_id = c.category_id
                JOIN (
                    SELECT category_id, COUNT(*) AS total_images_in_category
                    FROM images
                    GROUP BY category_id
                ) AS cat_stats ON i.category_id = cat_stats.category_id
                WHERE i.image_id NOT IN (
                    SELECT image_id FROM vectors
                )
                ORDER BY i.image_id;
                """)
                return [dict(row) for row in cur.fetchall()]


class VectorManager:
    """
    Manages vector operations in the database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def insert_vector(self, embedding: np.ndarray, image_id: int, index_id: int = None) -> int:
        """Insert a vector embedding for an image."""
        # Convert numpy array to list for PostgreSQL vector type
        embedding_list = embedding.tolist()
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vectors (embedding, image_id, index_id)
                    VALUES (%s::vector, %s, %s)
                    RETURNING vector_id;
                """, (embedding_list, image_id, index_id))
                
                vector_id = cur.fetchone()[0]
                conn.commit()
                return vector_id
    
    def insert_vectors_batch(self, vectors_data: List[Tuple[np.ndarray, int, Optional[int]]]):
        """Insert multiple vectors in batch."""
        # Convert numpy arrays to lists
        data_to_insert = [
            (embedding.tolist(), image_id, index_id)
            for embedding, image_id, index_id in vectors_data
        ]
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO vectors (embedding, image_id, index_id)
                    VALUES (%s::vector, %s, %s)
                """, data_to_insert)
                conn.commit()
        
        logger.info(f"Inserted {len(vectors_data)} vectors")
    
    def get_vector_by_image_id(self, image_id: int) -> Optional[np.ndarray]:
        """Get vector for a specific image."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT embedding FROM vectors WHERE image_id = %s
                """, (image_id,))
                
                result = cur.fetchone()
                return np.array(result[0]) if result else None
    
    def get_vectors_by_category(self, category_id: int) -> List[Tuple[int, np.ndarray]]:
        """Get all vectors for images in a specific category."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT v.image_id, v.embedding
                    FROM vectors v
                    JOIN images i ON v.image_id = i.image_id
                    WHERE i.category_id = %s
                """, (category_id,))
                
                results = []
                for row in cur.fetchall():
                    image_id = row[0]
                    vector_data = row[1]
                    
                    # Parse vector from pgvector format
                    if isinstance(vector_data, str):
                        # Remove brackets and split by commas
                        vector_str = vector_data.strip('[]')
                        vector_values = [float(x.strip()) for x in vector_str.split(',')]
                        vector = np.array(vector_values, dtype=np.float32)
                    elif isinstance(vector_data, list):
                        vector = np.array(vector_data, dtype=np.float32)
                    else:
                        # Fallback for other formats
                        vector = np.array(vector_data, dtype=np.float32)
                    
                    results.append((image_id, vector))
                
                return results
    
    def get_all_vectors(self) -> List[Tuple[int, np.ndarray]]:
        """Get all vectors from the database."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT image_id, embedding FROM vectors ORDER BY image_id
                """)
                
                results = []
                for row in cur.fetchall():
                    image_id = row[0]
                    vector_data = row[1]
                    
                    # Parse vector from pgvector format
                    if isinstance(vector_data, str):
                        # Remove brackets and split by commas
                        vector_str = vector_data.strip('[]')
                        vector_values = [float(x.strip()) for x in vector_str.split(',')]
                        vector = np.array(vector_values, dtype=np.float32)
                    elif isinstance(vector_data, list):
                        vector = np.array(vector_data, dtype=np.float32)
                    else:
                        # Fallback for other formats
                        vector = np.array(vector_data, dtype=np.float32)
                    
                    results.append((image_id, vector))
                
                return results


class CategoryManager:
    """
    Manages category operations.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all categories."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM categories ORDER BY category_id")
                return [dict(row) for row in cur.fetchall()]
    
    def get_category_by_id(self, category_id: int) -> Optional[Dict[str, Any]]:
        """Get category by ID."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM categories WHERE category_id = %s", (category_id,))
                return dict(cur.fetchone()) if cur.rowcount > 0 else None
    
    def insert_category(self, category_name: str) -> int:
        """Insert a new category."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO categories (category_name) VALUES (%s)
                    RETURNING category_id;
                """, (category_name,))
                
                category_id = cur.fetchone()[0]
                conn.commit()
                return category_id
    
    def ensure_cifar10_categories(self):
        """Ensure CIFAR-10 categories exist in the database."""
        cifar10_categories = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                for i, category_name in enumerate(cifar10_categories, 1):
                    cur.execute("""
                        INSERT INTO categories (category_id, category_name)
                        VALUES (%s, %s)
                        ON CONFLICT (category_id) DO NOTHING;
                    """, (i, category_name))
                conn.commit()
        
        logger.info("CIFAR-10 categories ensured in database")


class PrototypeManager:
    """
    Manages prototype vectors for categories.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.vector_manager = VectorManager(db_manager)
    
    def create_prototype_for_category(self, category_id: int) -> np.ndarray:
        """Create prototype vector for a category by averaging all vectors in that category."""
        vectors_data = self.vector_manager.get_vectors_by_category(category_id)
        
        if not vectors_data:
            raise ValueError(f"No vectors found for category {category_id}")
        
        # Extract embeddings and compute mean
        embeddings = [embedding for _, embedding in vectors_data]
        
        if not embeddings:
            raise ValueError(f"No valid embeddings found for category {category_id}")
            
        # Stack arrays and compute mean
        embeddings_array = np.stack(embeddings)
        prototype_vector = np.mean(embeddings_array, axis=0).astype(np.float32)
        
        # Store prototype in database
        self.insert_prototype(category_id, prototype_vector)
        
        logger.info(f"Created prototype for category {category_id} from {len(embeddings)} vectors")
        return prototype_vector
    
    def insert_prototype(self, category_id: int, prototype_vector: np.ndarray):
        """Insert or update a prototype vector."""
        prototype_list = prototype_vector.tolist()
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO _category_prototypes (category_id, prototype_vector)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (category_id) 
                    DO UPDATE SET prototype_vector = EXCLUDED.prototype_vector;
                """, (category_id, prototype_list))
                conn.commit()
    
    def get_prototype(self, category_id: int) -> Optional[np.ndarray]:
        """Get prototype vector for a category."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT prototype_vector FROM _category_prototypes 
                    WHERE category_id = %s
                """, (category_id,))
                
                result = cur.fetchone()
                if not result:
                    return None
                
                vector_data = result[0]
                # Parse vector from pgvector format
                if isinstance(vector_data, str):
                    # Remove brackets and split by commas
                    vector_str = vector_data.strip('[]')
                    vector_values = [float(x.strip()) for x in vector_str.split(',')]
                    vector = np.array(vector_values, dtype=np.float32)
                elif isinstance(vector_data, list):
                    vector = np.array(vector_data, dtype=np.float32)
                else:
                    # Fallback for other formats
                    vector = np.array(vector_data, dtype=np.float32)
                
                return vector
    
    def get_all_prototypes(self) -> Dict[int, np.ndarray]:
        """Get all prototypes as a dictionary mapping category_id to prototype vector."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT category_id, prototype_vector FROM _category_prototypes
                """)
                
                prototypes = {}
                for row in cur.fetchall():
                    category_id = row[0]
                    vector_data = row[1]
                    
                    # Parse vector from pgvector format
                    if isinstance(vector_data, str):
                        # Remove brackets and split by commas
                        vector_str = vector_data.strip('[]')
                        vector_values = [float(x.strip()) for x in vector_str.split(',')]
                        vector = np.array(vector_values, dtype=np.float32)
                    elif isinstance(vector_data, list):
                        vector = np.array(vector_data, dtype=np.float32)
                    else:
                        # Fallback for other formats
                        vector = np.array(vector_data, dtype=np.float32)
                    
                    prototypes[category_id] = vector
                
                return prototypes
    
    def create_all_prototypes(self):
        """Create prototypes for all categories that have vectors."""
        category_manager = CategoryManager(self.db_manager)
        categories = category_manager.get_all_categories()
        
        for category in categories:
            category_id = category['category_id']
            try:
                self.create_prototype_for_category(category_id)
            except ValueError as e:
                logger.warning(f"Skipping category {category_id}: {e}")


class FaissIndexManager:
    """
    Manages FAISS index metadata in the database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def register_index(self, index_type: str, index_filepath: str) -> int:
        """Register a FAISS index in the database."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO faiss (index_type, index_filepath)
                    VALUES (%s, %s)
                    RETURNING index_id;
                """, (index_type, index_filepath))
                
                index_id = cur.fetchone()[0]
                conn.commit()
                return index_id
    
    def get_index_info(self, index_id: int) -> Optional[Dict[str, Any]]:
        """Get FAISS index information."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM faiss WHERE index_id = %s", (index_id,))
                return dict(cur.fetchone()) if cur.rowcount > 0 else None
    
    def get_latest_index(self) -> Optional[Dict[str, Any]]:
        """Get the most recently created index."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM faiss ORDER BY index_id DESC LIMIT 1
                """)
                return dict(cur.fetchone()) if cur.rowcount > 0 else None
    
    def update_vector_index_ids(self, index_id: int):
        """Update all vectors to reference a specific FAISS index."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE vectors SET index_id = %s WHERE index_id IS NULL
                """, (index_id,))
                conn.commit()
                
                logger.info(f"Updated vectors to reference index {index_id}")


# High-level service class that combines all managers
class ImageSimilarityService:
    """
    High-level service for image similarity search operations.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.db_manager = DatabaseManager(config)
        self.image_manager = ImageManager(self.db_manager)
        self.vector_manager = VectorManager(self.db_manager)
        self.category_manager = CategoryManager(self.db_manager)
        self.prototype_manager = PrototypeManager(self.db_manager)
        self.faiss_index_manager = FaissIndexManager(self.db_manager)
    
    def setup_database(self, schema_path: str = None):
        """Setup database schema."""
        if schema_path is None:
            # Use default schema path
            schema_path = Path(__file__).parent.parent / "db" / "create_table.sql"
        
        if os.path.exists(schema_path):
            self.db_manager.execute_script(str(schema_path))
        
        # Ensure CIFAR-10 categories exist
        self.category_manager.ensure_cifar10_categories()
    
    def get_image_with_vector(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Get image data along with its vector."""
        image_data = self.image_manager.get_image_by_id(image_id)
        if not image_data:
            return None
        
        vector = self.vector_manager.get_vector_by_image_id(image_id)
        image_data['vector'] = vector
        return image_data
    
    def search_similar_by_prototype(self, query_vector: np.ndarray, 
                                  top_categories: int = 3) -> List[int]:
        """
        Find most similar categories using prototypes.
        
        Args:
            query_vector: Query image embedding
            top_categories: Number of top categories to return
            
        Returns:
            List of category IDs sorted by similarity
        """
        prototypes = self.prototype_manager.get_all_prototypes()
        
        similarities = []
        for category_id, prototype_vector in prototypes.items():
            # Compute cosine similarity
            dot_product = np.dot(query_vector, prototype_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_prototype = np.linalg.norm(prototype_vector)
            
            if norm_query > 0 and norm_prototype > 0:
                similarity = dot_product / (norm_query * norm_prototype)
            else:
                similarity = 0.0
            
            similarities.append((category_id, similarity))
        
        # Sort by similarity (descending) and return top categories
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [category_id for category_id, _ in similarities[:top_categories]]