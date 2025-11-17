#!/usr/bin/env python3
"""
Test script to demonstrate SQL event logging for all query types.
Run this to see the structured JSON logs in the terminal.

This script demonstrates logging for:
7.1 AGGREGATE (GROUP BY/COUNT) - Category statistics and prototype computation
7.2 UPDATE operations - Vector index updates, prototype upserts  
7.3 DELETE operations - Queue cleanup, stored procedure calls
7.4 CORRELATED queries - Category-filtered joins
7.5 NESTED queries - NOT IN subqueries with aggregates
8.1 STORED PROCEDURES - delete_image_vectors procedure
8.2 TRIGGERS - All database triggers that fire on INSERT/UPDATE/DELETE

Output format: 
[QUERY_TYPE] SQL statement
SQL_EVENT: {detailed JSON with parameters}
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.database import ImageSimilarityService, DatabaseConfig
import logging

# Configure logging to show our SQL events
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("=== SQL Event Logging Demo ===")
    print("This will demonstrate logging for all SQL query types:")
    print("7.1 AGGREGATE (GROUP BY/COUNT)")
    print("7.2 UPDATE operations") 
    print("7.3 DELETE operations")
    print("7.4 CORRELATED queries")
    print("7.5 NESTED queries")
    print("8.1 STORED PROCEDURES")
    print("8.2 TRIGGERS")
    print()
    
    try:
        # Initialize service (uses environment variables or defaults)
        service = ImageSimilarityService()
        
        # NESTED query with GROUP BY aggregate
        images_without_vectors = service.image_manager.get_images_without_vectors()
        print(f"Found {len(images_without_vectors)} images without vectors")
        print()
        
        # CORRELATED query
        try:
            vectors = service.vector_manager.get_vectors_by_category(1)
            print(f"Found {len(vectors)} vectors for category 1")
        except Exception as e:
            print(f"Category query: {e}")
        print()
        
        # UPDATE operation
        try:
            service.faiss_index_manager.update_vector_index_ids(1)
            print("Vector index IDs updated")
        except Exception as e:
            print(f"Update operation: {e}")
        print()
        
        # AGGREGATE operation (prototype creation)
        try:
            prototype = service.prototype_manager.create_prototype_for_category(1)
            print("Prototype created successfully")
        except Exception as e:
            print(f"Prototype creation: {e}")
        print()
        
        # DELETE operations - simulate queue cleanup
        try:
            with service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # First add a test entry to delete
                    cur.execute("INSERT INTO prototype_recompute_queue (category_id) VALUES (999) ON CONFLICT DO NOTHING")
                    conn.commit()
                    
                    # Now demonstrate DELETE
                    from backend.database import log_sql_event
                    log_sql_event("DELETE", "test_prototype_queue_cleanup", 
                                 sql="DELETE FROM prototype_recompute_queue WHERE category_id = %s")
                    cur.execute("DELETE FROM prototype_recompute_queue WHERE category_id = %s", (999,))
                    deleted_rows = cur.rowcount
                    conn.commit()
                    print(f"Deleted {deleted_rows} rows from prototype queue")
                    
                    # Test vector deletion queue cleanup
                    log_sql_event("DELETE", "test_vector_deletion_queue_cleanup", 
                                 sql="DELETE FROM vector_deletion_queue WHERE queued_at < NOW() - INTERVAL '1 hour'")
                    cur.execute("DELETE FROM vector_deletion_queue WHERE queued_at < NOW() - INTERVAL '1 hour'")
                    deleted_rows = cur.rowcount  
                    conn.commit()
                    print(f"Cleaned up {deleted_rows} old vector deletion queue entries")
        except Exception as e:
            print(f"DELETE operations: {e}")
        print()
        
        # STORED PROCEDURE with DELETE (if an image exists)
        try:
            with service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if any images exist to delete
                    cur.execute("SELECT image_id FROM images LIMIT 1")
                    result = cur.fetchone()
                    if result:
                        image_id = result[0]
                        from backend.database import log_sql_event
                        log_sql_event("STORED_PROCEDURE", "test_delete_image_vectors_procedure", 
                                     sql=f"CALL delete_image_vectors({image_id})")
                        print(f"Would call: CALL delete_image_vectors({image_id})")
                        print("(Not actually executed to preserve data)")
                    else:
                        print("No images found to demonstrate delete_image_vectors procedure")
        except Exception as e:
            print(f"Stored procedure demo: {e}")
        print()
        
        # TRIGGER demonstration - operations that fire triggers
        try:
            with service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    from backend.database import log_sql_event
                    
                    # Demonstrate trigger on image insert (compute_hash_and_prevent_duplicate)
                    log_sql_event("TRIGGER", "image_insert_triggers", 
                                 sql="INSERT INTO images (image_data, metadata, category_id) - fires trg_compute_hash_and_prevent_duplicate, trg_images_update_count")
                    print("Image INSERT would fire triggers:")
                    print("  - trg_compute_hash_and_prevent_duplicate (BEFORE INSERT)")
                    print("  - trg_images_update_count (AFTER INSERT)")
                    
                    # Demonstrate trigger on vector operations
                    log_sql_event("TRIGGER", "vector_operations_triggers", 
                                 sql="INSERT/UPDATE/DELETE on vectors - fires trg_vectors_mark_prototype, trg_vectors_queue_deletion")
                    print("Vector operations fire triggers:")
                    print("  - trg_vectors_mark_prototype (AFTER INSERT/UPDATE/DELETE)")
                    print("  - trg_vectors_queue_deletion (AFTER DELETE)")
                    
                    # Demonstrate FAISS index trigger
                    log_sql_event("TRIGGER", "faiss_index_triggers", 
                                 sql="INSERT INTO faiss - fires trg_faiss_after_insert")
                    print("FAISS index INSERT fires trigger:")
                    print("  - trg_faiss_after_insert (AFTER INSERT)")
                    
        except Exception as e:
            print(f"Trigger demonstration: {e}")
        print()
        
        # Demonstrate actual stored procedure usage (if available)
        try:
            with service.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if stored procedure exists
                    cur.execute("""
                        SELECT routine_name FROM information_schema.routines 
                        WHERE routine_type = 'PROCEDURE' AND routine_name = 'delete_image_vectors'
                    """)
                    if cur.fetchone():
                        from backend.database import log_sql_event
                        log_sql_event("STORED_PROCEDURE", "check_procedure_exists", 
                                     sql="SELECT routine_name FROM information_schema.routines WHERE routine_type = 'PROCEDURE'")
                        print("Found delete_image_vectors stored procedure in database")
                    else:
                        print("delete_image_vectors stored procedure not found")
        except Exception as e:
            print(f"Procedure check: {e}")
        print()
        
        print("=== Check populate_cifar10.py for more examples ===")
        print("Run: python db/populate_cifar10.py --status")
        print("This will show AGGREGATE queries with GROUP BY")
        
    except Exception as e:
        print(f"Database connection error: {e}")
        print("Make sure your database is running and environment variables are set:")
        print("DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")

if __name__ == "__main__":
    main()