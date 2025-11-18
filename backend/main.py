"""
FastAPI backend for image similarity search application.
Provides REST API endpoints for image upload, similarity search, and prototype-based filtering.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import io
import base64
from PIL import Image
import logging
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
import psycopg2.extras

def log_sql_event(query_type: str, operation: str, **kwargs):
    """Log structured JSON event for SQL operations"""
    log_data = {
        "ts": time.time(),
        "module": "main_api",
        "query_type": query_type,
        "operation": operation,
        **kwargs
    }
    # Extract and display the SQL query prominently
    sql_query = kwargs.get('sql', 'N/A')
    print(f"[{query_type}] {sql_query}")
    print(f"SQL_EVENT: {json.dumps(log_data)}")

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Import our custom modules
from backend.vector_processor import VectorSearchEngine, ImageEmbedder
from backend.database import ImageSimilarityService, DatabaseConfig
from backend.prototype_worker import start_background_worker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Similarity Search API",
    description="FAISS-powered image similarity search with prototype filtering",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
vector_engine: Optional[VectorSearchEngine] = None
db_service: Optional[ImageSimilarityService] = None
config: Dict[str, Any] = {}
# Stop callable returned by the background worker starter
worker_stopper = None

# Pydantic models for API responses
class ImageInfo(BaseModel):
    image_id: int
    category_id: int
    category_name: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class SearchResult(BaseModel):
    query_info: Dict[str, Any]
    similar_images: List[ImageInfo]
    search_time: float
    used_prototype_filtering: bool
    prototype_categories: Optional[List[int]] = None

class PrototypeInfo(BaseModel):
    category_id: int
    category_name: str
    similarity_score: float

class PrototypeSearchResult(BaseModel):
    prototypes: List[PrototypeInfo]
    search_time: float

class UploadResponse(BaseModel):
    message: str
    image_id: int
    processing_info: Dict[str, Any]

class SystemStatus(BaseModel):
    status: str
    faiss_index_loaded: bool
    database_connected: bool
    total_images: int
    total_vectors: int
    categories_with_prototypes: int
    index_info: Optional[Dict[str, Any]]
    # Dashboard-specific fields
    users: int
    images: int
    categories: int
    active_sessions: int


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_engine, db_service, config
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize database service
        db_config = DatabaseConfig(
            host=config.get("db_host", "localhost"),
            port=config.get("db_port", 5432),
            dbname=config.get("db_name", "imsrc"),
            user=config.get("db_user", "postgres"),
            password=config.get("db_password", "14789")
        )
        
        db_service = ImageSimilarityService(db_config)
        
        # Initialize vector search engine
        model_name = config.get("model_name", "resnet50")
        vector_engine = VectorSearchEngine(model_name=model_name)
        
        # Try to load existing FAISS index
        index_path = config.get("faiss_index_path", "data/faiss_index.bin")
        if os.path.exists(index_path):
            vector_engine.load_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        else:
            logger.warning(f"FAISS index not found at {index_path}. Run initialization script first.")

        # Start background worker that processes prototype/vector queues
        try:
            interval = int(os.getenv('WORKER_INTERVAL', 30))
            # start_background_worker returns an async 'stop' coroutine when awaited
            global worker_stopper
            worker_stopper = await start_background_worker(db_service, vector_engine, config, interval=interval)
        except Exception as e:
            logger.warning(f"Failed to start prototype worker: {e}")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown hook to stop background worker cleanly."""
    global worker_stopper
    if worker_stopper is not None:
        try:
            await worker_stopper()
        except Exception as e:
            logger.warning(f"Error while stopping worker: {e}")


def load_config() -> Dict[str, Any]:
    """Load application configuration."""
    # In a real application, this would load from a config file
    return {
        "db_host": os.getenv("DB_HOST", "localhost"),
        "db_port": int(os.getenv("DB_PORT", 5432)),
        "db_name": os.getenv("DB_NAME", "imsrc"),
        "db_user": os.getenv("DB_USER", "postgres"),
        "db_password": os.getenv("DB_PASSWORD", "14789"),
        "model_name": os.getenv("MODEL_NAME", "resnet18"),
        "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin"),
        "max_upload_size": int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024)),  # 10MB
        "default_search_limit": int(os.getenv("DEFAULT_SEARCH_LIMIT", 10)),
        "use_prototype_filtering": os.getenv("USE_PROTOTYPE_FILTERING", "true").lower() == "true"
    }


# Utility functions
def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size and file.size > config.get("max_upload_size", 10 * 1024 * 1024):
        raise HTTPException(status_code=400, detail="File too large")


def format_image_info(image_data: Dict[str, Any], similarity_score: Optional[float] = None) -> ImageInfo:
    """Format image data for API response."""
    return ImageInfo(
        image_id=image_data["image_id"],
        category_id=image_data["category_id"],
        category_name=image_data["category_name"],
        metadata=image_data["metadata"],
        similarity_score=similarity_score
    )


# API Endpoints


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health information."""
    try:
        # Check database connection
        all_images = db_service.image_manager.get_all_images()
        all_vectors = db_service.vector_manager.get_all_vectors()
        all_prototypes = db_service.prototype_manager.get_all_prototypes()
        all_categories = db_service.category_manager.get_all_categories()
        
        # Get user count and active sessions
        user_count = 0
        active_sessions = 0
        try:
            with db_service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM users")
                    user_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM user_sessions WHERE created_at > NOW() - INTERVAL '24 hours'")
                    active_sessions = cur.fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get user stats: {e}")
        
        # Check FAISS index
        faiss_loaded = vector_engine and vector_engine.faiss_manager.index is not None
        
        # Get index info
        index_info = None
        if faiss_loaded:
            latest_index = db_service.faiss_index_manager.get_latest_index()
            if latest_index:
                index_info = {
                    "index_id": latest_index["index_id"],
                    "index_type": latest_index["index_type"],
                    "filepath": latest_index["index_filepath"],
                    "ntotal": vector_engine.faiss_manager.index.ntotal if vector_engine.faiss_manager.index else 0
                }
        
        return SystemStatus(
            status="healthy",
            faiss_index_loaded=faiss_loaded,
            database_connected=True,
            total_images=len(all_images),
            total_vectors=len(all_vectors),
            categories_with_prototypes=len(all_prototypes),
            index_info=index_info,
            # Dashboard fields
            users=user_count,
            images=len(all_images),
            categories=len(all_categories),
            active_sessions=active_sessions
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return SystemStatus(
            status="error",
            faiss_index_loaded=False,
            database_connected=False,
            total_images=0,
            total_vectors=0,
            categories_with_prototypes=0,
            index_info=None,
            users=0,
            images=0,
            categories=0,
            active_sessions=0
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    category_id: Optional[int] = Query(None, description="Category ID for the image")
):
    """Upload a new image and generate its embedding."""
    validate_image(file)
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Generate embedding
        embedding = vector_engine.embedder.embed_image(image_data)
        
        # If no category provided, try to predict it using prototypes
        if category_id is None:
            prototypes = db_service.prototype_manager.get_all_prototypes()
            if prototypes:
                similarities = []
                for cat_id, prototype_vec in prototypes.items():
                    similarity = np.dot(embedding, prototype_vec) / (
                        np.linalg.norm(embedding) * np.linalg.norm(prototype_vec)
                    )
                    similarities.append((cat_id, similarity))
                
                # Use category with highest similarity
                category_id = max(similarities, key=lambda x: x[1])[0]
            else:
                category_id = 1  # Default to first category
        
        # Store image in database
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "upload_source": "api"
        }
        
        image_id = db_service.image_manager.insert_image(image_data, category_id, metadata)
        
        # Store vector
        vector_id = db_service.vector_manager.insert_vector(embedding, image_id)
        
        # Add to FAISS index if available
        if vector_engine.faiss_manager.index is not None:
            vector_engine.add_to_index(image_id, image_data)
        
        return UploadResponse(
            message="Image uploaded successfully",
            image_id=image_id,
            processing_info={
                "vector_id": vector_id,
                "predicted_category": category_id,
                "embedding_dimension": len(embedding),
                "added_to_faiss": vector_engine.faiss_manager.index is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@app.post("/search")
async def search_similar_images(
    file: UploadFile = File(...),
    limit: int = Query(10, ge=1, le=50, description="Number of similar images to return"),
    use_prototypes: bool = Query(False, description="Use prototype filtering (slower but more accurate)"),
    prototype_limit: int = Query(3, ge=1, le=10, description="Number of top categories to consider")
):
    """Search for similar images using uploaded query image."""
    validate_image(file)
    
    if vector_engine.faiss_manager.index is None:
        raise HTTPException(status_code=503, detail="FAISS index not available. Run initialization first.")
    
    try:
        import time
        start_time = time.time()
        
        # Read and process query image
        query_data = await file.read()
        embedding_start = time.time()
        query_embedding = vector_engine.embedder.embed_image(query_data)
        embedding_time = time.time() - embedding_start
        
        logger.info(f"Query embedding generated in {embedding_time:.2f}s - shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        
        prototype_categories = None
        
        if use_prototypes and config.get("use_prototype_filtering", True):
            # First, find most similar categories using prototypes
            logger.info("Starting prototype similarity search...")
            prototype_categories = db_service.search_similar_by_prototype(
                query_embedding, top_categories=prototype_limit
            )
            logger.info(f"Found similar categories: {prototype_categories}")
            
            # Search in full index first, then filter by categories (use pre-computed embedding)
            all_results = vector_engine.search_similar_by_embedding(query_embedding, k=limit*5)  # Get more results to filter
            
            # Filter results by prototype categories
            results = []
            for image_id, similarity_score in all_results:
                image_info = db_service.image_manager.get_image_by_id(image_id)
                if image_info and image_info['category_id'] in prototype_categories:
                    results.append((image_id, similarity_score))
                    if len(results) >= limit:
                        break
        else:
            # Search in full FAISS index (fast path - use pre-computed embedding)
            logger.info("Using direct FAISS search...")
            search_start = time.time()
            results = vector_engine.search_similar_by_embedding(query_embedding, k=limit)
            search_time = time.time() - search_start
            logger.info(f"FAISS search completed in {search_time:.2f}s, found {len(results)} results")
            
            # Log similarity scores for debugging
            if results:
                scores = [score for _, score in results]
                logger.info(f"Similarity scores - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Avg: {np.mean(scores):.4f}")
        
        # Get detailed image information with category names
        similar_images = []
        for image_id, similarity_score in results:
            # Get image info with category name from database
            with db_service.db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT i.image_id, i.category_id, i.metadata, c.category_name 
                        FROM images i 
                        JOIN categories c ON i.category_id = c.category_id 
                        WHERE i.image_id = %s
                    """, (image_id,))
                    image_info = cur.fetchone()
                    
                    if image_info:
                        similar_images.append({
                            "image_id": image_info["image_id"],
                            "category_id": image_info["category_id"],
                            "category_name": image_info["category_name"],
                            "similarity_score": similarity_score,
                            "metadata": image_info["metadata"]
                        })
        
        search_time = time.time() - start_time
        
        # Return format expected by frontend
        return {
            "query_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "embedding_dimension": len(query_embedding),
                "method": "prototype" if use_prototypes else "similarity"
            },
            "results": similar_images,
            "search_time": search_time,
            "used_prototype_filtering": use_prototypes and prototype_categories is not None,
            "prototype_categories": prototype_categories
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")


@app.post("/search-prototypes")
async def search_prototypes(file: UploadFile = File(...)):
    """Search for similar category prototypes."""
    validate_image(file)
    
    try:
        import time
        start_time = time.time()
        
        # Read and process query image
        query_data = await file.read()
        query_embedding = vector_engine.embedder.embed_image(query_data)
        
        # Get all prototypes
        prototypes = db_service.prototype_manager.get_all_prototypes()
        categories = {cat["category_id"]: cat["category_name"] 
                     for cat in db_service.category_manager.get_all_categories()}
        
        # Calculate similarities
        prototype_results = []
        for category_id, prototype_vector in prototypes.items():
            similarity = np.dot(query_embedding, prototype_vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(prototype_vector)
            )
            
            prototype_results.append(PrototypeInfo(
                category_id=category_id,
                category_name=categories.get(category_id, f"Category {category_id}"),
                similarity_score=float(similarity)
            ))
        
        # Sort by similarity
        prototype_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        search_time = time.time() - start_time
        
        return PrototypeSearchResult(
            prototypes=prototype_results,
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"Error in prototype search: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing prototype search: {str(e)}")


@app.get("/images")
async def get_images_list(
    limit: int = Query(10, ge=1, le=100),
    category_id: Optional[int] = Query(None)
):
    """Get list of images with optional filtering."""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if category_id:
                    log_sql_event("SELECT", "images_by_category", category_id=category_id, limit=limit)
                    cur.execute("""
                        SELECT i.image_id, i.category_id, c.category_name
                        FROM images i 
                        JOIN categories c ON i.category_id = c.category_id
                        WHERE i.category_id = %s 
                        ORDER BY i.image_id
                        LIMIT %s
                    """, (category_id, limit))
                else:
                    log_sql_event("SELECT", "images_list", limit=limit)
                    cur.execute("""
                        SELECT i.image_id, i.category_id, c.category_name
                        FROM images i 
                        JOIN categories c ON i.category_id = c.category_id
                        ORDER BY i.image_id
                        LIMIT %s
                    """, (limit,))
                
                images = cur.fetchall()
                return {"images": [dict(img) for img in images]}
                
    except Exception as e:
        logger.error(f"Error getting images list: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")


@app.get("/images/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: int):
    """Get image thumbnail by ID."""
    return await get_image(image_id)  # For now, return full image as thumbnail


@app.get("/images/{image_id}")
async def get_image(image_id: int):
    """Get image by ID and return as image response."""
    try:
        # Get image data from database
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT image_data FROM images WHERE image_id = %s", (image_id,))
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Image not found")
                
                image_data = result[0]
        
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving image")


@app.get("/images/{image_id}/info", response_model=ImageInfo)
async def get_image_info(image_id: int):
    """Get image information by ID."""
    try:
        image_data = db_service.image_manager.get_image_by_id(image_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return format_image_info(image_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image info {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving image info")


@app.get("/categories")
async def get_categories():
    """Get all categories with image counts."""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("SELECT", "categories_with_counts")
                cur.execute("""
                    SELECT c.category_id, c.category_name, COUNT(i.image_id) as image_count
                    FROM categories c
                    LEFT JOIN images i ON c.category_id = i.category_id
                    GROUP BY c.category_id, c.category_name
                    ORDER BY c.category_name
                """)
                categories = [dict(row) for row in cur.fetchall()]
                
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error retrieving categories: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving categories")


@app.get("/categories/{category_id}/images")
async def get_category_images(
    category_id: int,
    limit: int = Query(50, ge=1, le=100, description="Number of images to return")
):
    """Get images for a specific category."""
    try:
        images = db_service.image_manager.get_images_by_category(category_id)
        
        # Limit results
        images = images[:limit]
        
        # Format response
        formatted_images = [format_image_info(img) for img in images]
        
        return {
            "category_id": category_id,
            "images": formatted_images,
            "total_count": len(formatted_images)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving images for category {category_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving category images")


# Administrative endpoints
@app.post("/admin/rebuild-index")
async def rebuild_faiss_index():
    """Rebuild FAISS index from database vectors."""
    try:
        # Get all images and vectors
        all_vectors = db_service.vector_manager.get_all_vectors()
        
        if not all_vectors:
            raise HTTPException(status_code=400, detail="No vectors found in database")
        
        # Prepare data for index building
        images_data = []
        for image_id, vector in all_vectors:
            image_info = db_service.image_manager.get_image_by_id(image_id)
            if image_info:
                images_data.append((image_id, image_info["image_data"]))
        
        # Rebuild index
        index_path = config.get("faiss_index_path", "data/faiss_index.bin")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        vector_engine.build_index(images_data, index_type='flatl2', save_path=index_path)
        
        # Register in database
        index_id = db_service.faiss_index_manager.register_index('flatl2', index_path)
        db_service.faiss_index_manager.update_vector_index_ids(index_id)
        
        return {
            "message": "FAISS index rebuilt successfully",
            "index_id": index_id,
            "total_vectors": len(all_vectors),
            "index_path": index_path
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding FAISS index: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


@app.post("/admin/rebuild-prototypes")
async def rebuild_prototypes():
    """Rebuild category prototypes from existing vectors."""
    try:
        db_service.prototype_manager.create_all_prototypes()
        
        prototypes = db_service.prototype_manager.get_all_prototypes()
        
        return {
            "message": "Prototypes rebuilt successfully",
            "categories_processed": len(prototypes),
            "prototype_categories": list(prototypes.keys())
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding prototypes: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding prototypes: {str(e)}")


@app.get("/admin/queues")
async def admin_get_queues(limit: int = 100):
    """Return current queued prototype and deletion entries (for operators)."""
    try:
        proto_rows = []
        del_rows = []

        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT category_id, last_updated FROM prototype_recompute_queue ORDER BY last_updated LIMIT %s", (limit,))
                proto_rows = cur.fetchall()
                cur.execute("SELECT image_id, queued_at FROM vector_deletion_queue ORDER BY queued_at LIMIT %s", (limit,))
                del_rows = cur.fetchall()

        def _fmt_proto(r):
            return {"category_id": r[0], "last_updated": (r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1]))}

        def _fmt_del(r):
            return {"image_id": r[0], "queued_at": (r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1]))}

        return {
            "prototype_recompute_queue": [_fmt_proto(r) for r in proto_rows],
            "vector_deletion_queue": [_fmt_del(r) for r in del_rows],
            "prototype_queue_count": len(proto_rows),
            "deletion_queue_count": len(del_rows)
        }

    except Exception as e:
        logger.exception(f"Error fetching admin queues: {e}")
        raise HTTPException(status_code=500, detail="Error fetching admin queues")


@app.post("/admin/process-queues")
async def admin_process_queues(process_prototypes: bool = True, process_deletions: bool = True):
    """Force immediate processing of queues (runs the same work as the background worker).

    This endpoint offloads heavy tasks to threads to avoid blocking the event loop.
    """
    result = {"processed_prototypes": [], "processed_deletions": 0}

    try:
        # Process prototype queue
        if process_prototypes:
            with db_service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT category_id FROM prototype_recompute_queue ORDER BY last_updated")
                    proto_rows = cur.fetchall()

            category_ids = [r[0] for r in proto_rows] if proto_rows else []

            for cat_id in category_ids:
                try:
                    await asyncio.to_thread(db_service.prototype_manager.create_prototype_for_category, cat_id)
                    # remove from queue
                    with db_service.db_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            log_sql_event("DELETE", "api_prototype_queue_cleanup", category_id=cat_id)
                            cur.execute("DELETE FROM prototype_recompute_queue WHERE category_id = %s", (cat_id,))
                            conn.commit()
                    result["processed_prototypes"].append(cat_id)
                except Exception as e:
                    logger.exception(f"Failed to process prototype for category {cat_id}: {e}")

        # Process deletion queue
        if process_deletions:
            with db_service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT image_id FROM vector_deletion_queue ORDER BY queued_at")
                    del_rows = cur.fetchall()

            to_delete_ids = [r[0] for r in del_rows] if del_rows else []

            if to_delete_ids:
                # Build FAISS index from current images (offload to thread)
                all_images = db_service.image_manager.get_all_images()
                images_data = []
                for img in all_images:
                    img_data = img['image_data']
                    if isinstance(img_data, memoryview):
                        img_data = img_data.tobytes()
                    images_data.append((img['image_id'], img_data))

                index_path = config.get('faiss_index_path', 'data/faiss_index.bin')
                index_type = config.get('faiss_index_type', 'flatl2')
                # Offload index build
                try:
                    await asyncio.to_thread(vector_engine.build_index, images_data, index_type, index_path)
                    index_id = db_service.faiss_index_manager.register_index(index_type, str(index_path))
                    db_service.faiss_index_manager.update_vector_index_ids(index_id)
                    # Clear deletion queue
                    with db_service.db_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            log_sql_event("DELETE", "api_vector_deletion_queue_clear")
                            cur.execute("DELETE FROM vector_deletion_queue")
                            rows_deleted = cur.rowcount
                            conn.commit()
                            log_sql_event("DELETE", "api_vector_deletion_queue_cleared", rows_deleted=rows_deleted)
                    result["processed_deletions"] = len(to_delete_ids)
                except Exception as e:
                    logger.exception(f"Failed to process vector deletions: {e}")

        return result

    except Exception as e:
        logger.exception(f"Error processing queues: {e}")
        raise HTTPException(status_code=500, detail="Error processing queues")


# Authentication and User Management Functions
async def hash_password(password: str) -> str:
    """Hash password using bcrypt-like method (simplified for demo)"""
    return hashlib.sha256(f"{password}salt".encode()).hexdigest()

async def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return await hash_password(password) == hashed

async def create_session(user_id: int) -> str:
    """Create user session"""
    session_id = secrets.token_urlsafe(32)
    with db_service.db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_sessions (session_id, user_id) VALUES (%s, %s)",
                (session_id, user_id)
            )
            conn.commit()
    return session_id

# Authentication Endpoints
@app.post("/auth/login")
async def login(request: dict):
    """Authenticate user and create session"""
    try:
        username = request.get('username')
        password = request.get('password')
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get user
                log_sql_event("SELECT", "user_authentication", username=username)
                cur.execute(
                    "SELECT user_id, username, password_hash, role, is_active FROM users WHERE username = %s",
                    (username,)
                )
                user = cur.fetchone()
                
                if not user or not user['is_active']:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Verify password (simplified for demo)
                expected_hash = await hash_password(password)
                if user['password_hash'] != expected_hash:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Create session
                session_id = await create_session(user['user_id'])
                
                return {
                    "session_id": session_id,
                    "username": user['username'],
                    "role": user['role'],
                    "message": "Login successful"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/logout")
async def logout(session_id: str):
    """Logout user and invalidate session"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                log_sql_event("DELETE", "user_logout", session_id=session_id)
                cur.execute("DELETE FROM user_sessions WHERE session_id = %s", (session_id,))
                conn.commit()
                
        return {"message": "Logout successful"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

# Admin API Endpoints
@app.get("/admin/users")
async def get_all_users():
    """Get all users for admin management"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("SELECT", "admin_get_users")
                cur.execute("""
                    SELECT user_id, username, email, role, created_at, last_login, is_active
                    FROM users ORDER BY created_at DESC
                """)
                users = cur.fetchall()
                return [dict(user) for user in users]
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Error fetching users")

@app.post("/admin/users")
async def create_user(request: dict):
    """Create new user with role"""
    try:
        password_hash = await hash_password(request['password'])
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                log_sql_event("INSERT", "admin_create_user", username=request['username'])
                cur.execute(
                    "SELECT create_user_with_role(%s, %s, %s, %s)",
                    (request['username'], request['email'], password_hash, request['role'])
                )
                user_id = cur.fetchone()[0]
                conn.commit()
                return {"message": "User created successfully", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.post("/admin/privileges")
async def manage_user_privileges(request: dict):
    """Grant or revoke user privileges"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                if request['action'] == 'grant':
                    log_sql_event("INSERT", "admin_grant_privilege", user_id=request['user_id'])
                    cur.execute(
                        "INSERT INTO user_privileges (user_id, resource, permission) VALUES (%s, %s, %s)",
                        (request['user_id'], request['resource'], request['permission'])
                    )
                elif request['action'] == 'revoke':
                    log_sql_event("DELETE", "admin_revoke_privilege", user_id=request['user_id'])
                    cur.execute(
                        "DELETE FROM user_privileges WHERE user_id = %s AND resource = %s AND permission = %s",
                        (request['user_id'], request['resource'], request['permission'])
                    )
                conn.commit()
                return {"message": f"Privilege {request['action']}ed successfully"}
    except Exception as e:
        logger.error(f"Error managing privileges: {e}")
        raise HTTPException(status_code=500, detail=f"Error managing privileges: {str(e)}")

@app.post("/admin/categories")
async def create_category_admin(request: dict):
    """Create new category via admin panel"""
    try:
        category_name = request.get('category_name', '').strip()
        if not category_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        # Check if category already exists
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT category_id FROM categories WHERE category_name = %s", (category_name,))
                if cur.fetchone():
                    raise HTTPException(status_code=400, detail="Category already exists")
                
                # Create new category
                cur.execute("INSERT INTO categories (category_name) VALUES (%s) RETURNING category_id", (category_name,))
                category_id = cur.fetchone()[0]
                conn.commit()
                
        return {"message": "Category created successfully", "category_id": category_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating category: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating category: {str(e)}")

@app.put("/admin/images/{image_id}")
async def update_image_admin(image_id: int, request: dict):
    """Update image metadata and category"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                updates = []
                params = []
                
                if 'category_id' in request:
                    updates.append("category_id = %s")
                    params.append(request['category_id'])
                
                if 'metadata' in request:
                    updates.append("metadata = %s")
                    params.append(json.dumps(request['metadata']))
                
                if updates:
                    params.append(image_id)
                    query = f"UPDATE images SET {', '.join(updates)} WHERE image_id = %s"
                    log_sql_event("UPDATE", "admin_update_image", image_id=image_id)
                    cur.execute(query, params)
                    conn.commit()
                    
                return {"message": "Image updated successfully"}
    except Exception as e:
        logger.error(f"Error updating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating image: {str(e)}")

@app.delete("/admin/images/{image_id}")
async def delete_image_admin(image_id: int):
    """Delete image and its vectors"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if image exists first
                cur.execute("SELECT image_id FROM images WHERE image_id = %s", (image_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Image not found")
                
                # Delete vector first (will trigger cleanup)
                log_sql_event("DELETE", "admin_delete_vector", image_id=image_id)
                cur.execute("DELETE FROM vectors WHERE image_id = %s", (image_id,))
                
                # Delete image
                log_sql_event("DELETE", "admin_delete_image", image_id=image_id)
                cur.execute("DELETE FROM images WHERE image_id = %s", (image_id,))
                conn.commit()
                
        return {"message": "Image deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")

@app.delete("/admin/categories/{category_id}")
async def delete_category_admin(category_id: int):
    """Delete category and all its images"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if category exists first
                cur.execute("SELECT category_id FROM categories WHERE category_id = %s", (category_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Category not found")
                
                # Delete associated vectors first
                log_sql_event("DELETE", "admin_delete_category_vectors", category_id=category_id)
                cur.execute("DELETE FROM vectors WHERE image_id IN (SELECT image_id FROM images WHERE category_id = %s)", (category_id,))
                
                # Delete associated images
                log_sql_event("DELETE", "admin_delete_category_images", category_id=category_id)
                cur.execute("DELETE FROM images WHERE category_id = %s", (category_id,))
                
                # Delete category
                log_sql_event("DELETE", "admin_delete_category", category_id=category_id)
                cur.execute("DELETE FROM categories WHERE category_id = %s", (category_id,))
                conn.commit()
                
        return {"message": "Category and all associated images deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting category: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting category: {str(e)}")

@app.post("/admin/batch-process")
async def batch_process_images(request: dict):
    """Execute batch operations on images"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("SELECT", "admin_batch_process", category_id=request['category_id'])
                cur.execute(
                    "SELECT * FROM batch_process_category(%s, %s, %s)",
                    (request['category_id'], request['operation'], request.get('batch_size', 100))
                )
                results = cur.fetchall()
                conn.commit()
                
        return {"results": [dict(row) for row in results]}
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")

# Advanced Query Endpoints
@app.post("/admin/queries/nested")
async def execute_nested_query(request: dict):
    """Execute nested query examples"""
    try:
        query_type = request['query_type']
        
        queries = {
            'top_categories': """
                SELECT i.*, c.category_name FROM images i
                JOIN categories c ON i.category_id = c.category_id
                WHERE i.category_id IN (
                    SELECT category_id FROM (
                        SELECT category_id, COUNT(*) as cnt 
                        FROM images GROUP BY category_id 
                        ORDER BY cnt DESC LIMIT 3
                    ) top_cats
                )
                ORDER BY i.category_id, i.image_id
            """,
            'above_avg_size': """
                SELECT i.image_id, i.category_id, LENGTH(i.image_data) as size_bytes
                FROM images i
                WHERE LENGTH(i.image_data) > (
                    SELECT AVG(LENGTH(image_data)) FROM images
                )
                ORDER BY size_bytes DESC
            """,
            'categories_with_prototypes': """
                SELECT c.*, 
                    (SELECT COUNT(*) FROM images WHERE category_id = c.category_id) as image_count
                FROM categories c
                WHERE c.category_id IN (
                    SELECT category_id FROM _category_prototypes
                )
                ORDER BY c.category_name
            """
        }
        
        if query_type not in queries:
            raise HTTPException(status_code=400, detail="Invalid query type")
            
        query = queries[query_type]
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                start_time = time.time()
                log_sql_event("SELECT", "admin_nested_query", query_name=query_type)
                cur.execute(query)
                results = cur.fetchall()
                execution_time = (time.time() - start_time) * 1000
                
        return {
            "query_type": query_type,
            "query": query.strip(),
            "execution_time": round(execution_time, 2),
            "data": [dict(row) for row in results]
        }
    except Exception as e:
        logger.error(f"Error executing nested query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing nested query: {str(e)}")

@app.post("/admin/queries/join")
async def execute_join_query(request: dict):
    """Execute join query examples"""
    try:
        query_type = request['query_type']
        
        queries = {
            'image_category_vector': """
                SELECT i.image_id, i.metadata, c.category_name, 
                       v.vector_id, 
                       CASE WHEN v.vector_id IS NOT NULL THEN 'Yes' ELSE 'No' END as has_vector
                FROM images i
                LEFT JOIN categories c ON i.category_id = c.category_id
                LEFT JOIN vectors v ON i.image_id = v.image_id
                ORDER BY i.image_id
                LIMIT 50
            """,
            'category_statistics': """
                SELECT c.category_name, 
                       COUNT(i.image_id) as image_count,
                       COUNT(v.vector_id) as vector_count,
                       AVG(LENGTH(i.image_data))::INT as avg_size_bytes,
                       CASE WHEN p.prototype_id IS NOT NULL THEN 'Yes' ELSE 'No' END as has_prototype
                FROM categories c
                LEFT JOIN images i ON c.category_id = i.category_id
                LEFT JOIN vectors v ON i.image_id = v.image_id
                LEFT JOIN _category_prototypes p ON c.category_id = p.category_id
                GROUP BY c.category_id, c.category_name, p.prototype_id
                ORDER BY image_count DESC
            """,
            'user_activity_summary': """
                SELECT u.username, u.role,
                       COUNT(DISTINCT s.session_id) as session_count,
                       COUNT(l.log_id) as activity_count,
                       MAX(s.created_at) as last_login
                FROM users u
                LEFT JOIN user_sessions s ON u.user_id = s.user_id
                LEFT JOIN user_activity_log l ON u.user_id = l.user_id
                WHERE u.is_active = true
                GROUP BY u.user_id, u.username, u.role
                ORDER BY activity_count DESC
            """
        }
        
        if query_type not in queries:
            raise HTTPException(status_code=400, detail="Invalid query type")
            
        query = queries[query_type]
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                start_time = time.time()
                log_sql_event("SELECT", "admin_join_query", query_name=query_type)
                cur.execute(query)
                results = cur.fetchall()
                execution_time = (time.time() - start_time) * 1000
                
        return {
            "query_type": query_type,
            "query": query.strip(),
            "execution_time": round(execution_time, 2),
            "data": [dict(row) for row in results]
        }
    except Exception as e:
        logger.error(f"Error executing join query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing join query: {str(e)}")

@app.post("/admin/queries/aggregate")
async def execute_aggregate_query(request: dict):
    """Execute aggregate query examples"""
    try:
        query_type = request['query_type']
        
        queries = {
            'category_summary': """
                SELECT c.category_name,
                       COUNT(i.image_id) as total_images,
                       MIN(LENGTH(i.image_data)) as min_size_bytes,
                       MAX(LENGTH(i.image_data)) as max_size_bytes,
                       AVG(LENGTH(i.image_data))::INT as avg_size_bytes,
                       SUM(LENGTH(i.image_data)) as total_size_bytes
                FROM categories c
                LEFT JOIN images i ON c.category_id = i.category_id
                GROUP BY c.category_id, c.category_name
                HAVING COUNT(i.image_id) > 0
                ORDER BY total_images DESC
            """,
            'size_statistics': """
                SELECT 
                    COUNT(*) as total_images,
                    MIN(LENGTH(image_data)) as smallest_bytes,
                    MAX(LENGTH(image_data)) as largest_bytes,
                    AVG(LENGTH(image_data))::INT as average_bytes,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(image_data))::INT as median_bytes,
                    SUM(LENGTH(image_data)) as total_storage_bytes
                FROM images
            """,
            'user_stats': """
                SELECT 
                    u.role,
                    COUNT(*) as user_count,
                    COUNT(CASE WHEN u.is_active THEN 1 END) as active_users,
                    AVG(EXTRACT(EPOCH FROM (now() - u.created_at))/86400)::INT as avg_days_since_creation,
                    COUNT(DISTINCT s.session_id) as total_sessions
                FROM users u
                LEFT JOIN user_sessions s ON u.user_id = s.user_id
                GROUP BY u.role
                ORDER BY user_count DESC
            """
        }
        
        if query_type not in queries:
            raise HTTPException(status_code=400, detail="Invalid query type")
            
        query = queries[query_type]
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                start_time = time.time()
                log_sql_event("SELECT", "admin_aggregate_query", query_name=query_type)
                cur.execute(query)
                results = cur.fetchall()
                execution_time = (time.time() - start_time) * 1000
                
        return {
            "query_type": query_type,
            "query": query.strip(),
            "execution_time": round(execution_time, 2),
            "data": [dict(row) for row in results]
        }
    except Exception as e:
        logger.error(f"Error executing aggregate query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing aggregate query: {str(e)}")

@app.post("/admin/queries/custom")
async def execute_custom_query(request: dict):
    """Execute custom SQL query (SELECT only)"""
    try:
        query = request['query'].strip()
        
        # Security check - only allow SELECT statements
        if not query.upper().startswith('SELECT'):
            raise HTTPException(status_code=400, detail="Only SELECT statements are allowed")
            
        # Additional security checks
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise HTTPException(status_code=400, detail=f"Keyword '{keyword}' is not allowed")
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                start_time = time.time()
                log_sql_event("SELECT", "admin_custom_query", query=query[:100])
                cur.execute(query)
                results = cur.fetchall()
                execution_time = (time.time() - start_time) * 1000
                
        return {
            "query": query,
            "execution_time": round(execution_time, 2),
            "data": [dict(row) for row in results]
        }
    except Exception as e:
        logger.error(f"Error executing custom query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing custom query: {str(e)}")

# Stored Procedures Endpoints
@app.post("/admin/procedures/execute")
async def execute_stored_procedure(request: dict):
    """Execute stored procedures and functions"""
    try:
        procedure_name = request['procedure_name']
        parameters = request.get('parameters', {})
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                start_time = time.time()
                log_sql_event("FUNCTION", "admin_execute_procedure", procedure=procedure_name)
                
                if procedure_name == 'get_category_statistics':
                    cur.execute("SELECT * FROM get_category_statistics()")
                elif procedure_name == 'get_database_health':
                    cur.execute("SELECT * FROM get_database_health()")
                elif procedure_name == 'cleanup_old_data':
                    days = parameters.get('days_old', 30)
                    cur.execute("SELECT * FROM cleanup_old_data(%s)", (days,))
                elif procedure_name == 'rebuild_database_indexes':
                    cur.execute("SELECT * FROM rebuild_database_indexes()")
                elif procedure_name == 'analyze_similarity_patterns':
                    category_id = parameters.get('category_id')
                    if category_id:
                        cur.execute("SELECT * FROM analyze_similarity_patterns(%s)", (category_id,))
                    else:
                        cur.execute("SELECT * FROM analyze_similarity_patterns()")
                elif procedure_name == 'generate_backup_metadata':
                    cur.execute("SELECT * FROM generate_backup_metadata()")
                else:
                    raise HTTPException(status_code=400, detail="Unknown procedure")
                    
                results = cur.fetchall()
                execution_time = (time.time() - start_time) * 1000
                conn.commit()
                
        return {
            "procedure_name": procedure_name,
            "execution_time": round(execution_time, 2),
            "data": [dict(row) for row in results]
        }
    except Exception as e:
        logger.error(f"Error executing procedure: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing procedure: {str(e)}")

# Trigger Management Endpoints
@app.get("/admin/triggers")
async def get_database_triggers():
    """Get all database triggers"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("SELECT", "admin_get_triggers")
                cur.execute("""
                    SELECT trigger_name, event_object_table as table_name, 
                           event_manipulation, action_statement
                    FROM information_schema.triggers 
                    WHERE trigger_schema = 'public'
                    ORDER BY trigger_name
                """)
                triggers = cur.fetchall()
                
        return {"triggers": [dict(trigger) for trigger in triggers]}
    except Exception as e:
        logger.error(f"Error fetching triggers: {e}")
        raise HTTPException(status_code=500, detail="Error fetching triggers")

@app.get("/admin/health")
async def get_database_health_admin():
    """Get database health metrics"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("SELECT", "admin_health_check")
                cur.execute("SELECT * FROM get_database_health()")
                health_metrics = cur.fetchall()
                
        return {"health_metrics": [dict(metric) for metric in health_metrics]}
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        raise HTTPException(status_code=500, detail="Error getting health metrics")

@app.post("/admin/cleanup")
async def cleanup_database():
    """Cleanup old sessions and temporary data"""
    try:
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                log_sql_event("FUNCTION", "admin_cleanup")
                cur.execute("SELECT * FROM cleanup_old_data(30)")
                result = cur.fetchone()
                conn.commit()
                
        return dict(result)
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
        raise HTTPException(status_code=500, detail="Error in cleanup")

@app.get("/admin/reports/full")
async def generate_full_report():
    """Generate comprehensive system report"""
    try:
        # Gather various metrics
        status = await get_system_status()
        
        with db_service.db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Category statistics
                cur.execute("SELECT * FROM get_category_statistics()")
                category_stats = cur.fetchall()
                
                # Health metrics
                cur.execute("SELECT * FROM get_database_health()")
                health_metrics = cur.fetchall()
                
                # User statistics
                cur.execute("""
                    SELECT role, COUNT(*) as count,
                           COUNT(CASE WHEN is_active THEN 1 END) as active_count
                    FROM users GROUP BY role
                """)
                user_stats = cur.fetchall()
                
        return {
            "generated_at": datetime.now().isoformat(),
            "sections": {
                "system_status": status.__dict__,
                "category_statistics": [dict(row) for row in category_stats],
                "health_metrics": [dict(row) for row in health_metrics],
                "user_statistics": [dict(row) for row in user_stats]
            }
        }
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Error generating report")

# Mount static files (for frontend)
static_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Root route - serve login page
@app.get("/")
async def root():
    """Serve the login page"""
    try:
        login_path = static_path / "login.html"
        if login_path.exists():
            with open(login_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return HTMLResponse(content)
        else:
            return HTMLResponse(f"<h1>Login page not found at {login_path}</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading page: {e}</h1>", status_code=500)

# Serve specific HTML files
@app.get("/login")
@app.get("/login.html")
async def login_page():
    """Serve the login page"""
    return await root()

@app.get("/admin-dashboard.html")
async def admin_dashboard():
    """Serve the simplified admin dashboard page"""
    dashboard_path = static_path / "admin-simple.html"
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>Admin dashboard not found</h1>", status_code=404)

@app.get("/admin-simple.html")
async def admin_simple():
    """Serve the simplified admin panel page"""
    dashboard_path = static_path / "admin-simple.html"
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>Admin panel not found</h1>", status_code=404)

@app.get("/user-browser.html")
async def user_browser():
    """Serve the user browser page"""
    browser_path = static_path / "user-browser.html"
    if browser_path.exists():
        with open(browser_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>User browser not found</h1>", status_code=404)

# Serve JavaScript files
@app.get("/admin-dashboard.js")
async def admin_js():
    """Serve the admin dashboard JavaScript"""
    js_path = static_path / "admin-simple.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/admin-simple.js")
async def admin_simple_js():
    """Serve the simplified admin JavaScript"""
    js_path = static_path / "admin-simple.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/app")
async def serve_frontend():
    """Redirect to login page for proper authentication flow"""
    return HTMLResponse("""
    <script>
        window.location.href = '/';
    </script>
    <p>Redirecting to login...</p>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)