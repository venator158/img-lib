"""
FastAPI backend for image similarity search application.
Provides REST API endpoints for image upload, similarity search, and prototype-based filtering.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import io
import base64
from PIL import Image
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Import our custom modules
from .vector_processor import VectorSearchEngine, ImageEmbedder
from .database import ImageSimilarityService, DatabaseConfig

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
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


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

@app.get("/")
async def root():
    """Root endpoint - redirect to frontend."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app")


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health information."""
    try:
        # Check database connection
        all_images = db_service.image_manager.get_all_images()
        all_vectors = db_service.vector_manager.get_all_vectors()
        all_prototypes = db_service.prototype_manager.get_all_prototypes()
        
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
            index_info=index_info
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
            index_info=None
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


@app.post("/search", response_model=SearchResult)
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
            
            # Search in full index first, then filter by categories
            all_results = vector_engine.search_similar(query_data, k=limit*5)  # Get more results to filter
            
            # Filter results by prototype categories
            results = []
            for image_id, similarity_score in all_results:
                image_info = db_service.image_manager.get_image_by_id(image_id)
                if image_info and image_info['category_id'] in prototype_categories:
                    results.append((image_id, similarity_score))
                    if len(results) >= limit:
                        break
        else:
            # Search in full FAISS index (fast path)
            logger.info("Using direct FAISS search...")
            search_start = time.time()
            results = vector_engine.search_similar(query_data, k=limit)
            search_time = time.time() - search_start
            logger.info(f"FAISS search completed in {search_time:.2f}s, found {len(results)} results")
        
        # Get detailed image information
        similar_images = []
        for image_id, similarity_score in results:
            image_info = db_service.image_manager.get_image_by_id(image_id)
            if image_info:
                similar_images.append(format_image_info(image_info, similarity_score))
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query_info={
                "filename": file.filename,
                "content_type": file.content_type,
                "embedding_dimension": len(query_embedding)
            },
            similar_images=similar_images,
            search_time=search_time,
            used_prototype_filtering=use_prototypes and prototype_categories is not None,
            prototype_categories=prototype_categories
        )
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")


@app.post("/search/prototypes", response_model=PrototypeSearchResult)
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


@app.get("/images/{image_id}")
async def get_image(image_id: int):
    """Get image by ID and return as image response."""
    try:
        image_data = db_service.image_manager.get_image_by_id(image_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return StreamingResponse(
            io.BytesIO(image_data["image_data"]),
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
    """Get all categories."""
    try:
        categories = db_service.category_manager.get_all_categories()
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


# Mount static files (for frontend)
static_path = Path(__file__).parent.parent / "frontend"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    @app.get("/app")
    async def serve_frontend():
        """Serve the frontend application."""
        index_path = static_path / "index.html"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=content)
        else:
            raise HTTPException(status_code=404, detail="Frontend not available")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)