"""
Initialization script for processing CIFAR-10 images, generating vectors, 
creating prototypes, and building FAISS indices.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List, Tuple
import time

# Add project root to path for proper package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database import ImageSimilarityService, DatabaseConfig
from backend.vector_processor import VectorSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database(service: ImageSimilarityService, schema_path: str = None):
    """Setup database schema and CIFAR-10 categories."""
    logger.info("Setting up database schema...")
    
    if schema_path is None:
        schema_path = Path(__file__).parent / "create_table.sql"
    
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found: {schema_path}")
        return False
    
    try:
        service.setup_database(str(schema_path))
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        # If tables already exist, that's okay
        if "already exists" in str(e).lower():
            logger.info("Database tables already exist, skipping schema creation")
            # Still ensure CIFAR-10 categories exist
            service.category_manager.ensure_cifar10_categories()
            return True
        else:
            logger.error(f"Database setup failed: {e}")
            return False

def generate_vectors_for_images(service: ImageSimilarityService, 
                              vector_engine: VectorSearchEngine,
                              batch_size: int = None):
    """Generate vectors for images that don't have them yet."""
    logger.info("Checking for images without vectors...")
    
    images_without_vectors = service.image_manager.get_images_without_vectors()
    
    if not images_without_vectors:
        logger.info("All images already have vectors")
        return True
    
    logger.info(f"Found {len(images_without_vectors)} images without vectors")
    
    # Auto-detect batch size if not specified
    if batch_size is None:
        # Check if GPU is available for larger batches
        import torch
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            if available_memory > 8 * 1024**3:  # > 8GB
                batch_size = 100
            elif available_memory > 4 * 1024**3:  # > 4GB  
                batch_size = 50
            else:
                batch_size = 25
            logger.info(f"GPU detected, using batch size: {batch_size}")
        else:
            batch_size = 10
            logger.info(f"CPU mode, using batch size: {batch_size}")
    
    try:
        import time
        start_time = time.time()
        total_processed = 0
        
        # Process in batches
        for i in range(0, len(images_without_vectors), batch_size):
            batch = images_without_vectors[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(images_without_vectors) + batch_size - 1)//batch_size
            
            batch_start = time.time()
            
            try:
                # Prepare batch data - handle memoryview objects
                image_data_list = []
                for img in batch:
                    img_data = img['image_data']
                    # Convert memoryview to bytes if needed
                    if isinstance(img_data, memoryview):
                        img_data = img_data.tobytes()
                    image_data_list.append(img_data)
                
                # Generate embeddings using optimized batch processing
                embeddings = vector_engine.embedder.embed_batch(image_data_list, batch_size=None)
                
                # Prepare vectors for database insertion
                vectors_data = [
                    (embeddings[j], batch[j]['image_id'], None)
                    for j in range(len(batch))
                ]
                
                # Insert into database
                service.vector_manager.insert_vectors_batch(vectors_data)
                
                # Update progress tracking
                total_processed += len(batch)
                batch_time = time.time() - batch_start
                elapsed_total = time.time() - start_time
                
                # Calculate rates and ETA
                images_per_second = total_processed / elapsed_total if elapsed_total > 0 else 0
                remaining_images = len(images_without_vectors) - total_processed
                eta_seconds = remaining_images / images_per_second if images_per_second > 0 else 0
                
                logger.info(f"Batch {batch_num}/{total_batches} completed in {batch_time:.1f}s | "
                           f"Processed: {total_processed}/{len(images_without_vectors)} | "
                           f"Rate: {images_per_second:.1f} images/sec | "
                           f"ETA: {eta_seconds/60:.1f}m")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"Successfully generated vectors for {len(images_without_vectors)} images")
        return True
        
    except Exception as e:
        logger.error(f"Error generating vectors: {e}")
        return False

def create_prototypes(service: ImageSimilarityService):
    """Create prototype vectors for all categories."""
    logger.info("Creating category prototypes...")
    
    try:
        service.prototype_manager.create_all_prototypes()
        
        prototypes = service.prototype_manager.get_all_prototypes()
        logger.info(f"Successfully created prototypes for {len(prototypes)} categories")
        return True
        
    except Exception as e:
        logger.error(f"Error creating prototypes: {e}")
        return False

def build_faiss_index(service: ImageSimilarityService,
                     vector_engine: VectorSearchEngine,
                     index_type: str = 'flatl2',
                     output_path: str = None):
    """Build FAISS index from database vectors."""
    logger.info("Building FAISS index...")
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "faiss_index.bin"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Get all images
        all_images = service.image_manager.get_all_images()
        
        if not all_images:
            logger.error("No images found in database")
            return False
        
        # Prepare image data for index building - handle memoryview objects
        images_data = []
        for img in all_images:
            img_data = img['image_data']
            # Convert memoryview to bytes if needed
            if isinstance(img_data, memoryview):
                img_data = img_data.tobytes()
            images_data.append((img['image_id'], img_data))
        
        logger.info(f"Building {index_type} index for {len(images_data)} images...")
        
        # Build index
        index = vector_engine.build_index(
            images_data,
            index_type=index_type,
            save_path=str(output_path)
        )
        
        # Register index in database
        index_id = service.faiss_index_manager.register_index(index_type, str(output_path))
        service.faiss_index_manager.update_vector_index_ids(index_id)
        
        logger.info(f"FAISS index built successfully: {output_path}")
        logger.info(f"Index ID: {index_id}, Total vectors: {index.ntotal}")
        return True
        
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        return False

def run_system_check(service: ImageSimilarityService, vector_engine: VectorSearchEngine):
    """Run system health check."""
    logger.info("Running system health check...")
    
    try:
        # Check database connections
        categories = service.category_manager.get_all_categories()
        images = service.image_manager.get_all_images()
        vectors = service.vector_manager.get_all_vectors()
        prototypes = service.prototype_manager.get_all_prototypes()
        
        logger.info(f"Database Status:")
        logger.info(f"  - Categories: {len(categories)}")
        logger.info(f"  - Images: {len(images)}")
        logger.info(f"  - Vectors: {len(vectors)}")
        logger.info(f"  - Prototypes: {len(prototypes)}")
        
        # Check FAISS index
        if vector_engine.faiss_manager.index is not None:
            logger.info(f"FAISS Index Status:")
            logger.info(f"  - Index type: {vector_engine.faiss_manager.index_type}")
            logger.info(f"  - Total vectors: {vector_engine.faiss_manager.index.ntotal}")
            logger.info(f"  - Dimension: {vector_engine.faiss_manager.embedding_dim}")
        else:
            logger.warning("FAISS index not loaded")
        
        # Test similarity search if possible
        if images and vector_engine.faiss_manager.index is not None:
            logger.info("Testing similarity search with first image...")
            test_image = images[0]
            results = vector_engine.search_similar(test_image['image_data'], k=5)
            logger.info(f"Search test successful: found {len(results)} similar images")
        
        logger.info("System health check completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Initialize Image Similarity Search System")
    parser.add_argument("--setup-db", action="store_true", help="Setup database schema")
    parser.add_argument("--generate-vectors", action="store_true", help="Generate vectors for images")
    parser.add_argument("--create-prototypes", action="store_true", help="Create category prototypes")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index")
    parser.add_argument("--index-type", default="flatl2", choices=["flatl2", "ivfflat", "hnsw"], 
                       help="Type of FAISS index to build")
    parser.add_argument("--index-path", help="Path to save FAISS index")
    parser.add_argument("--model", default="resnet18", choices=["resnet50", "resnet18"],
                       help="Model to use for embeddings")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing (auto-detect if not specified)")
    parser.add_argument("--all", action="store_true", help="Run all initialization steps")
    parser.add_argument("--check", action="store_true", help="Run system health check")
    parser.add_argument("--schema-path", help="Path to database schema file")
    
    args = parser.parse_args()
    
    # If no specific actions, show help
    if not any([args.setup_db, args.generate_vectors, args.create_prototypes, 
                args.build_index, args.all, args.check]):
        parser.print_help()
        return
    
    logger.info("=== Image Similarity Search System Initialization ===")
    
    # Initialize services
    try:
        logger.info("Initializing services...")
        
        # Database service
        db_config = DatabaseConfig()  # Uses defaults from config
        service = ImageSimilarityService(db_config)
        
        # Load model from environment or use argument
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        model_name = os.getenv('MODEL_NAME', args.model)
        device = os.getenv('DEVICE', 'auto')
        
        # Vector engine
        vector_engine = VectorSearchEngine(model_name=model_name, device=device)
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"Database: {db_config.dbname}@{db_config.host}:{db_config.port}")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return
    
    start_time = time.time()
    success_count = 0
    total_steps = 0
    
    # Run initialization steps
    if args.all or args.setup_db:
        total_steps += 1
        logger.info("\n--- Step: Setup Database ---")
        if setup_database(service, args.schema_path):
            success_count += 1
    
    if args.all or args.generate_vectors:
        total_steps += 1
        logger.info("\n--- Step: Generate Vectors ---")
        if generate_vectors_for_images(service, vector_engine, args.batch_size):
            success_count += 1
    
    if args.all or args.create_prototypes:
        total_steps += 1
        logger.info("\n--- Step: Create Prototypes ---")
        if create_prototypes(service):
            success_count += 1
    
    if args.all or args.build_index:
        total_steps += 1
        logger.info("\n--- Step: Build FAISS Index ---")
        if build_faiss_index(service, vector_engine, args.index_type, args.index_path):
            success_count += 1
    
    if args.check:
        logger.info("\n--- System Health Check ---")
        run_system_check(service, vector_engine)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n=== Initialization Complete ===")
    logger.info(f"Completed {success_count}/{total_steps} steps successfully")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    
    if success_count == total_steps and total_steps > 0:
        logger.info("✅ All initialization steps completed successfully!")
        logger.info("Your image similarity search system is ready to use.")
        logger.info("Start the API server with: python -m backend.main")
    else:
        logger.warning("⚠️ Some initialization steps failed. Check the logs above.")

if __name__ == "__main__":
    main()