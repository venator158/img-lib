#!/usr/bin/env python3
"""
Debug script to check vector data format
"""

import sys
sys.path.append('backend')
from database import ImageSimilarityService, DatabaseConfig
import numpy as np

def main():
    # Initialize service
    config = DatabaseConfig()
    service = ImageSimilarityService(config)

    # Test getting vectors for category 1 (airplane)
    try:
        print("Testing vector retrieval...")
        vectors_data = service.vector_manager.get_vectors_by_category(1)
        print(f'Found {len(vectors_data)} vectors for category 1')
        
        if vectors_data:
            image_id, embedding = vectors_data[0]
            print(f'Image ID: {image_id}')
            print(f'Embedding type: {type(embedding)}')
            if hasattr(embedding, 'shape'):
                print(f'Embedding shape: {embedding.shape}')
            if hasattr(embedding, '__len__'):
                print(f'Embedding length: {len(embedding)}')
            if hasattr(embedding, '__getitem__'):
                print(f'First 5 values: {embedding[:5]}')
            else:
                print(f'Embedding value sample: {str(embedding)[:100]}...')
                
            # Try to create prototype manually
            print("\nTesting prototype creation...")
            embeddings = [emb for _, emb in vectors_data[:10]]  # Use first 10 vectors
            print(f"Processing {len(embeddings)} embeddings")
            
            # Check each embedding
            for i, emb in enumerate(embeddings[:3]):
                print(f"Embedding {i}: type={type(emb)}, shape={getattr(emb, 'shape', 'no shape')}")
            
            # Try to stack them
            try:
                embeddings_array = np.stack(embeddings[:10])
                print(f"Stacked array shape: {embeddings_array.shape}")
                prototype = np.mean(embeddings_array, axis=0)
                print(f"Prototype shape: {prototype.shape}")
                print("✅ Prototype creation successful!")
            except Exception as stack_error:
                print(f"❌ Stack error: {stack_error}")
                
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()