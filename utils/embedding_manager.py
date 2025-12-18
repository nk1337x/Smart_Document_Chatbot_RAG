"""
Embedding Manager Module
========================
Generates embeddings for text using HuggingFace Sentence Transformers.
All models are free and open-source.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    A class to manage text embeddings using HuggingFace Sentence Transformers.
    
    Available models (all free and open-source):
    - all-MiniLM-L6-v2: Fast and lightweight (384 dimensions)
    - all-mpnet-base-v2: Better quality (768 dimensions)
    - paraphrase-MiniLM-L6-v2: Good for paraphrases (384 dimensions)
    """
    
    # Model information for reference
    MODEL_INFO = {
        'all-MiniLM-L6-v2': {
            'dimensions': 384,
            'max_seq_length': 256,
            'description': 'Fast and lightweight, good balance of speed and quality'
        },
        'all-mpnet-base-v2': {
            'dimensions': 768,
            'max_seq_length': 384,
            'description': 'Higher quality embeddings, slower'
        },
        'paraphrase-MiniLM-L6-v2': {
            'dimensions': 384,
            'max_seq_length': 128,
            'description': 'Optimized for paraphrase detection'
        }
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager with a specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence transformer model.
        Downloads the model if not already cached.
        """
        print(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dim()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            return self.MODEL_INFO.get(self.model_name, {}).get('dimensions', 384)
        return self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                           batch_size: int = 32,
                           show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: A single text string or list of text strings
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if self.model is None:
            self._load_model()
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts
        texts = [t if t and t.strip() else " " for t in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1 for normalized embeddings)
        """
        # Since embeddings are normalized, dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         document_embeddings: np.ndarray,
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar documents to a query.
        
        Args:
            query_embedding: Embedding of the query (1D array)
            document_embeddings: Embeddings of documents (2D array)
            top_k: Number of top results to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        # Compute similarities
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return indices with scores
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        Convenience method for encoding queries.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding (1D numpy array)
        """
        return self.generate_embeddings([query])[0]
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.get_embedding_dim(),
            **self.MODEL_INFO.get(self.model_name, {})
        }
