"""
Vector Store Module
===================
Manages document storage and retrieval using FAISS vector database.
FAISS is free, open-source, and runs entirely locally.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path

# FAISS for efficient similarity search
import faiss


class VectorStore:
    """
    A vector store implementation using FAISS for efficient similarity search.
    
    Features:
    - Fast approximate nearest neighbor search
    - Support for various index types
    - Persistence (save/load to disk)
    - Metadata storage for documents
    """
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index to use
                - "flat": Exact search (slower but accurate)
                - "ivf": Inverted file index (faster for large datasets)
                - "hnsw": Hierarchical Navigable Small World (good balance)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []  # Store document chunks with metadata
        self.id_map = {}  # Map FAISS IDs to document indices
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the specified type."""
        
        if self.index_type == "flat":
            # Exact search using L2 distance
            # For cosine similarity with normalized vectors, L2 and cosine are equivalent
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine sim
            
        elif self.index_type == "ivf":
            # Inverted file index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            # nlist = number of clusters (adjust based on dataset size)
            nlist = 100
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
        elif self.index_type == "hnsw":
            # HNSW for good speed/accuracy tradeoff
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            embeddings: Numpy array of embeddings (shape: [num_docs, embedding_dim])
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # For IVF index, we need to train it first
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add embeddings to index
        start_idx = len(self.documents)
        self.index.add(embeddings)
        
        # Store documents and update ID map
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            self.id_map[idx] = idx
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for the most similar documents to a query.
        
        Args:
            query_embedding: The query embedding (1D array)
            top_k: Number of results to return
            
        Returns:
            List of document dictionaries with added 'score' field
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Adjust top_k if we have fewer documents
        top_k = min(top_k, len(self.documents))
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Valid index
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def search_with_threshold(self, query_embedding: np.ndarray,
                             top_k: int = 5,
                             threshold: float = 0.5) -> List[Dict]:
        """
        Search with a minimum similarity threshold.
        
        Args:
            query_embedding: The query embedding
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of documents above the threshold
        """
        results = self.search(query_embedding, top_k)
        return [doc for doc in results if doc.get('score', 0) >= threshold]
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents from the store."""
        self.documents = []
        self.id_map = {}
        self._initialize_index()
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_map': self.id_map,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded VectorStore instance
        """
        path = Path(path)
        
        # Load documents and metadata
        with open(path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        store = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type']
        )
        
        # Load FAISS index
        store.index = faiss.read_index(str(path / "index.faiss"))
        
        # Restore documents and ID map
        store.documents = data['documents']
        store.id_map = data['id_map']
        
        return store
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in the store."""
        return self.documents.copy()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'total_documents': len(self.documents),
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0
        }
