"""
Text Chunker Module
===================
Splits text into manageable chunks with overlap for better retrieval.
"""

from typing import List, Dict, Optional
import re


class TextChunker:
    """
    A class to split text into chunks suitable for embedding and retrieval.
    
    Features:
    - Configurable chunk size and overlap
    - Preserves sentence boundaries when possible
    - Maintains metadata for source tracking
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters (default: 500)
            chunk_overlap: Number of characters to overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split into chunks
            metadata: Optional metadata to attach to each chunk (e.g., source file)
            
        Returns:
            List of dictionaries, each containing:
            - 'text': The chunk text
            - 'metadata': Metadata including source and chunk index
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = {
                    **(metadata or {}),
                    'chunk_index': chunk_index,
                    'start_char': sum(len(' '.join(current_chunk[:i])) for i in range(len(current_chunk))),
                }
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                chunk_index += 1
                
                # Calculate overlap - keep some sentences for context
                overlap_length = 0
                overlap_sentences = []
                
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = {
                **(metadata or {}),
                'chunk_index': chunk_index,
            }
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while handling edge cases.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern to split on sentence boundaries
        # Handles: periods, question marks, exclamation marks
        # Avoids splitting on: abbreviations, decimals, etc.
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # First, split by newlines to preserve paragraph structure
        paragraphs = text.split('\n')
        
        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Split paragraph into sentences
            para_sentences = re.split(sentence_pattern, paragraph)
            
            for sent in para_sentences:
                sent = sent.strip()
                if sent:
                    sentences.append(sent)
        
        return sentences
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def chunk_by_tokens(self, text: str, metadata: Optional[Dict] = None,
                        tokens_per_chunk: int = 256) -> List[Dict]:
        """
        Alternative chunking method based on approximate token count.
        Assumes ~4 characters per token on average.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
            tokens_per_chunk: Target tokens per chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Approximate characters per chunk (4 chars ≈ 1 token)
        char_per_chunk = tokens_per_chunk * 4
        
        # Temporarily update chunk size
        original_size = self.chunk_size
        self.chunk_size = char_per_chunk
        
        chunks = self.chunk_text(text, metadata)
        
        # Restore original size
        self.chunk_size = original_size
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'total_characters': 0
            }
        
        lengths = [len(chunk['text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_characters': sum(lengths)
        }
