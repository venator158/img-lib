"""
Vector processing module for image similarity search.
Handles image embeddings, FAISS index creation, and similarity search operations.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Optional, Union
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path

class ImageEmbedder:
    """
    Handles image embedding generation using pre-trained models.
    """
    
    def __init__(self, model_name: str = 'resnet50', device: Optional[str] = None):
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = 'cpu'
                print("GPU not available, using CPU")
        else:
            self.device = device
        
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transform()
        
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Feature dimension: {self.feature_dim}")
        
    def _load_model(self) -> nn.Module:
        """Load pre-trained model and remove the final classification layer."""
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove the final fully connected layer to get feature embeddings
            model.fc = nn.Identity()
            feature_dim = 2048
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        model = model.to(self.device)
        model.eval()
        self.feature_dim = feature_dim
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def embed_image(self, image: Union[Image.Image, np.ndarray, bytes, memoryview]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image, numpy array, image bytes, or memoryview
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(image, (bytes, memoryview)):
            # Convert memoryview to bytes if needed
            if isinstance(image, memoryview):
                image = image.tobytes()
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)
            
        return embedding.cpu().numpy().flatten()
    
    def embed_batch(self, images: List[Union[Image.Image, np.ndarray, bytes, memoryview]], 
                   batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing (auto-detect based on device if None)
            
        Returns:
            numpy array of embeddings with shape (n_images, embedding_dim)
        """
        # Auto-detect optimal batch size based on device and available memory
        if batch_size is None:
            if self.device == 'cuda':
                # GPU: Use larger batches for better throughput
                available_memory = torch.cuda.get_device_properties(0).total_memory
                if available_memory > 8 * 1024**3:  # > 8GB
                    batch_size = 64
                elif available_memory > 4 * 1024**3:  # > 4GB  
                    batch_size = 32
                else:
                    batch_size = 16
            else:
                # CPU: Use smaller batches to avoid memory issues
                batch_size = 8
        
        embeddings = []
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []
            
            try:
                for image in batch:
                    if isinstance(image, (bytes, memoryview)):
                        # Convert memoryview to bytes if needed
                        if isinstance(image, memoryview):
                            image = image.tobytes()
                        image = Image.open(io.BytesIO(image))
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray(image.astype('uint8'))
                        
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    batch_tensors.append(self.transform(image))
                
                # Stack and move to device
                batch_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    # Enable mixed precision if available on GPU
                    if self.device == 'cuda' and torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            batch_embeddings = self.model(batch_tensor)
                    else:
                        batch_embeddings = self.model(batch_tensor)
                    
                    # Move back to CPU and convert to numpy
                    embeddings.append(batch_embeddings.cpu().numpy())
                
                # Clear intermediate tensors from GPU memory
                if self.device == 'cuda':
                    del batch_tensor, batch_embeddings
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory with batch size {len(batch)}, reducing batch size...")
                    # Clear cache and retry with smaller batch
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    # Process images one by one
                    for single_image in batch:
                        single_embedding = self.embed_image(single_image)
                        embeddings.append(single_embedding.reshape(1, -1))
                else:
                    raise e
        
        return np.vstack(embeddings)


class FaissIndexManager:
    """
    Manages FAISS indices for similarity search.
    """
    
    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        self.index = None
        self.index_type = None
        
    def create_index(self, index_type: str = 'flatl2', nlist: int = 100) -> faiss.Index:
        """
        Create a new FAISS index.
        
        Args:
            index_type: Type of index ('flatl2', 'ivfflat', 'hnsw')
            nlist: Number of clusters for IVF indices
            
        Returns:
            FAISS index
        """
        if index_type == 'flatl2':
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == 'ivfflat':
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.index = index
        self.index_type = index_type
        return index
    
    def add_vectors(self, vectors: np.ndarray, train_first: bool = True):
        """
        Add vectors to the FAISS index.
        
        Args:
            vectors: Numpy array of vectors
            train_first: Whether to train the index first (for IVF indices)
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
            
        vectors = vectors.astype('float32')
        
        # Train the index if needed (for IVF indices)
        if train_first and hasattr(self.index, 'train'):
            if not self.index.is_trained:
                self.index.train(vectors)
        
        self.index.add(vectors)
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not created or loaded.")
            
        query_vectors = query_vectors.astype('float32')
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
            
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def save_index(self, filepath: str):
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
            
        faiss.write_index(self.index, filepath)
        
        # Save metadata
        metadata = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'ntotal': self.index.ntotal
        }
        
        metadata_path = filepath + '.meta'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(filepath)
        
        # Load metadata
        metadata_path = filepath + '.meta'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.index_type = metadata.get('index_type')
                self.embedding_dim = metadata.get('embedding_dim', self.embedding_dim)


class VectorSearchEngine:
    """
    High-level interface for image similarity search combining embeddings and FAISS.
    """
    
    def __init__(self, model_name: str = 'resnet50', device: Optional[str] = None):
        self.embedder = ImageEmbedder(model_name, device)
        self.faiss_manager = FaissIndexManager(self.embedder.feature_dim)
        self.image_ids = []  # Keep track of image IDs corresponding to index positions
        
    def build_index(self, images: List[Tuple[int, Union[Image.Image, np.ndarray, bytes]]], 
                   index_type: str = 'flatl2', save_path: Optional[str] = None) -> faiss.Index:
        """
        Build FAISS index from images.
        
        Args:
            images: List of tuples (image_id, image_data)
            index_type: Type of FAISS index to create
            save_path: Optional path to save the index
            
        Returns:
            Built FAISS index
        """
        print(f"Generating embeddings for {len(images)} images...")
        
        # Extract image data and IDs
        image_data = [img[1] for img in images]
        self.image_ids = [img[0] for img in images]
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(image_data)
        
        print(f"Creating {index_type} index...")
        # Create and populate index
        self.faiss_manager.create_index(index_type)
        self.faiss_manager.add_vectors(embeddings)
        
        if save_path:
            print(f"Saving index to {save_path}")
            self.faiss_manager.save_index(save_path)
            
            # Save image ID mapping
            ids_path = save_path + '.ids'
            with open(ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        
        return self.faiss_manager.index
    
    def load_index(self, index_path: str):
        """Load pre-built index and image ID mapping."""
        print(f"Loading index from {index_path}")
        self.faiss_manager.load_index(index_path)
        
        # Load image ID mapping
        ids_path = index_path + '.ids'
        if os.path.exists(ids_path):
            with open(ids_path, 'rb') as f:
                self.image_ids = pickle.load(f)
    
    def search_similar(self, query_image: Union[Image.Image, np.ndarray, bytes, memoryview], 
                      k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar images.
        
        Args:
            query_image: Query image
            k: Number of similar images to return
            
        Returns:
            List of tuples (image_id, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_image(query_image)
        
        # Search in FAISS index
        distances, indices = self.faiss_manager.search(query_embedding, k)
        
        # Convert to image IDs and similarity scores
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.image_ids):
                image_id = self.image_ids[idx]
                # Convert L2 distance to similarity score using sigmoid normalization
                # Sigmoid function: 1 / (1 + exp(k * (distance - threshold)))
                # For ResNet embeddings, typical good matches have distances 400-600
                threshold = 500.0  # Distance threshold for 50% similarity
                k = 0.01  # Steepness parameter (smaller = gentler curve)
                similarity_score = 1.0 / (1.0 + np.exp(k * (distance - threshold)))
                results.append((image_id, similarity_score))
        
        return results
    
    def search_similar_by_embedding(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar images using a pre-computed query embedding.
        More efficient when you already have the embedding.
        
        Args:
            query_embedding: Pre-computed query image embedding
            k: Number of similar images to return
            
        Returns:
            List of tuples (image_id, similarity_score)
        """
        # Search in FAISS index
        distances, indices = self.faiss_manager.search(query_embedding, k)
        
        # Debug: Print raw distances
        if len(distances[0]) > 0:
            print(f"DEBUG: Raw distances - Min: {np.min(distances[0]):.2f}, Max: {np.max(distances[0]):.2f}, Mean: {np.mean(distances[0]):.2f}")
        
        # Convert to image IDs and similarity scores
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.image_ids):
                image_id = self.image_ids[idx]
                # Convert L2 distance to similarity score using sigmoid normalization
                # Sigmoid function: 1 / (1 + exp(k * (distance - threshold)))
                # For ResNet embeddings, typical good matches have distances 400-600
                threshold = 500.0  # Distance threshold for 50% similarity
                k = 0.01  # Steepness parameter (smaller = gentler curve)
                similarity_score = 1.0 / (1.0 + np.exp(k * (distance - threshold)))
                results.append((image_id, similarity_score))
        
        return results
    
    def add_to_index(self, image_id: int, image_data: Union[Image.Image, np.ndarray, bytes, memoryview]):
        """Add a new image to the existing index."""
        embedding = self.embedder.embed_image(image_data)
        embedding = embedding.reshape(1, -1).astype('float32')
        
        self.faiss_manager.index.add(embedding)
        self.image_ids.append(image_id)


# Utility functions for prototype management
def compute_prototype_vector(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute prototype vector as the mean of all embeddings in a category.
    
    Args:
        embeddings: Array of embeddings for images in the same category
        
    Returns:
        Prototype vector (mean of embeddings)
    """
    return np.mean(embeddings, axis=0)

def prototype_similarity(query_embedding: np.ndarray, prototypes: dict) -> dict:
    """
    Compute similarity between query and all prototypes.
    
    Args:
        query_embedding: Query image embedding
        prototypes: Dict mapping category_id to prototype vector
        
    Returns:
        Dict mapping category_id to similarity score
    """
    similarities = {}
    
    for category_id, prototype_vector in prototypes.items():
        # Compute cosine similarity
        dot_product = np.dot(query_embedding, prototype_vector)
        norm_query = np.linalg.norm(query_embedding)
        norm_prototype = np.linalg.norm(prototype_vector)
        
        if norm_query > 0 and norm_prototype > 0:
            similarity = dot_product / (norm_query * norm_prototype)
        else:
            similarity = 0.0
            
        similarities[category_id] = similarity
    
    return similarities