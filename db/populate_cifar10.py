"""
Enhanced script to populate images from CIFAR-10 dataset into PostgreSQL database.
This version includes better error handling and progress tracking.
"""

import psycopg2
from tensorflow.keras.datasets import cifar10
from PIL import Image
import io
import json
import logging
from typing import Dict, Any, List
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile', 
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def generate_metadata(idx: int, split: str = 'train') -> Dict[str, Any]:
    """Generate metadata for an image."""
    return {
        "source": "cifar10",
        "image_index": idx,
        "split": split,
        "description": "32x32 color image from CIFAR-10 dataset",
        "format": "PNG"
    }

def ensure_categories_exist(cursor) -> bool:
    """Ensure CIFAR-10 categories exist in the database."""
    try:
        logger.info("Ensuring CIFAR-10 categories exist...")
        
        for class_id, class_name in CIFAR10_CLASSES.items():
            category_id = class_id + 1  # Database uses 1-based indexing
            
            cursor.execute("""
                INSERT INTO categories (category_id, category_name)
                VALUES (%s, %s)
                ON CONFLICT (category_id) DO UPDATE SET 
                category_name = EXCLUDED.category_name;
            """, (category_id, class_name))
        
        logger.info(f"Successfully ensured {len(CIFAR10_CLASSES)} categories exist")
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring categories: {e}")
        return False

def populate_images(num_images: int = 1000, 
                   use_test_set: bool = False,
                   db_config: Dict[str, str] = None) -> bool:
    """
    Populate database with CIFAR-10 images.
    
    Args:
        num_images: Number of images to populate (max 50000 for train, 10000 for test)
        use_test_set: Whether to use test set instead of training set
        db_config: Database configuration dictionary
    """
    
    # Default database configuration
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "imsrc",
            "user": "postgres",
            "password": "14789"
        }
    
    logger.info("Loading CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        if use_test_set:
            x_data, y_data = x_test, y_test
            split_name = "test"
            max_images = len(x_test)
        else:
            x_data, y_data = x_train, y_train
            split_name = "train"
            max_images = len(x_train)
            
        logger.info(f"Loaded {split_name} set with {max_images} images")
        
        # Limit number of images
        num_images = min(num_images, max_images)
        logger.info(f"Will populate {num_images} images from {split_name} set")
        
    except Exception as e:
        logger.error(f"Error loading CIFAR-10 dataset: {e}")
        return False
    
    # Connect to database
    try:
        conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['dbname']} user={db_config['user']} password={db_config['password']}"
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()
        logger.info("Connected to database successfully")
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    
    try:
        # Ensure categories exist
        if not ensure_categories_exist(cur):
            return False
        conn.commit()
        
        # Check for existing images to avoid duplicates
        cur.execute("""
            SELECT COUNT(*) FROM images 
            WHERE metadata->>'source' = 'cifar10' 
            AND metadata->>'split' = %s
        """, (split_name,))
        
        existing_count = cur.fetchone()[0]
        logger.info(f"Found {existing_count} existing CIFAR-10 {split_name} images")
        
        # Process images in batches
        batch_size = 100
        successful_inserts = 0
        failed_inserts = 0
        
        start_time = time.time()
        
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            batch_size_actual = batch_end - i
            
            logger.info(f"Processing batch {i//batch_size + 1}: images {i+1} to {batch_end}")
            
            batch_data = []
            
            # Prepare batch data
            for j in range(i, batch_end):
                try:
                    image_array = x_data[j]
                    category_id = int(y_data[j][0]) + 1  # Convert to 1-based indexing
                    
                    # Convert to PIL Image and then to PNG bytes
                    img = Image.fromarray(image_array)
                    byte_io = io.BytesIO()
                    img.save(byte_io, format='PNG')
                    img_bytes = byte_io.getvalue()
                    
                    metadata = json.dumps(generate_metadata(j, split_name))
                    
                    batch_data.append((
                        psycopg2.Binary(img_bytes),
                        metadata,
                        category_id
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error processing image {j}: {e}")
                    failed_inserts += 1
            
            # Insert batch
            if batch_data:
                try:
                    cur.executemany("""
                        INSERT INTO images (image_data, metadata, category_id)
                        VALUES (%s, %s::jsonb, %s)
                    """, batch_data)
                    
                    successful_inserts += len(batch_data)
                    conn.commit()
                    
                    # Progress update
                    progress = (batch_end / num_images) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / batch_end) * (num_images - batch_end) if batch_end > 0 else 0
                    
                    logger.info(f"Progress: {progress:.1f}% ({successful_inserts} images) - ETA: {eta:.1f}s")
                    
                except Exception as e:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                    conn.rollback()
                    failed_inserts += batch_size_actual
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n=== Population Complete ===")
        logger.info(f"Successfully inserted: {successful_inserts} images")
        logger.info(f"Failed insertions: {failed_inserts} images")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average: {successful_inserts/total_time:.1f} images/second")
        
        return successful_inserts > 0
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
        conn.rollback()
        return False
        
    finally:
        cur.close()
        conn.close()
        logger.info("Database connection closed")

def check_database_status(db_config: Dict[str, str] = None) -> Dict[str, Any]:
    """Check current database status."""
    
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "imsrc",
            "user": "postgres",
            "password": "14789"
        }
    
    try:
        conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['dbname']} user={db_config['user']} password={db_config['password']}"
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()
        
        # Get counts
        cur.execute("SELECT COUNT(*) FROM categories")
        categories_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM images")
        images_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM vectors")
        vectors_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM _category_prototypes")
        prototypes_count = cur.fetchone()[0]
        
        # Get category breakdown
        cur.execute("""
            SELECT c.category_name, COUNT(i.image_id) as image_count
            FROM categories c
            LEFT JOIN images i ON c.category_id = i.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY c.category_id
        """)
        category_breakdown = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "categories": categories_count,
            "images": images_count,
            "vectors": vectors_count,
            "prototypes": prototypes_count,
            "category_breakdown": category_breakdown,
            "connection": "successful"
        }
        
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        return {"connection": "failed", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Populate database with CIFAR-10 images")
    parser.add_argument("--num-images", type=int, default=1000, 
                       help="Number of images to populate (default: 1000)")
    parser.add_argument("--use-test", action="store_true",
                       help="Use test set instead of training set")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-name", default="imsrc", help="Database name")
    parser.add_argument("--db-user", default="postgres", help="Database user")
    parser.add_argument("--db-password", default="14789", help="Database password")
    parser.add_argument("--status", action="store_true", help="Check database status only")
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password
    }
    
    if args.status:
        logger.info("Checking database status...")
        status = check_database_status(db_config)
        
        if status.get("connection") == "successful":
            logger.info(f"Database Status:")
            logger.info(f"  Categories: {status['categories']}")
            logger.info(f"  Images: {status['images']}")
            logger.info(f"  Vectors: {status['vectors']}")
            logger.info(f"  Prototypes: {status['prototypes']}")
            logger.info(f"\nCategory Breakdown:")
            for cat_name, img_count in status['category_breakdown']:
                logger.info(f"  {cat_name}: {img_count} images")
        else:
            logger.error(f"Database connection failed: {status.get('error')}")
        
        return
    
    logger.info("=== CIFAR-10 Database Population ===")
    logger.info(f"Target images: {args.num_images}")
    logger.info(f"Dataset split: {'test' if args.use_test else 'train'}")
    logger.info(f"Database: {args.db_name}@{args.db_host}:{args.db_port}")
    
    success = populate_images(
        num_images=args.num_images,
        use_test_set=args.use_test,
        db_config=db_config
    )
    
    if success:
        logger.info("✅ Population completed successfully!")
        # Show final status
        status = check_database_status(db_config)
        if status.get("connection") == "successful":
            logger.info(f"Final database status: {status['images']} total images")
    else:
        logger.error("❌ Population failed!")

if __name__ == "__main__":
    main()