"""
Document Processor Module
=========================
Handles text extraction from various document formats (PDF, TXT).
Uses free and open-source libraries.
"""

import io
from typing import Optional
import PyPDF2
import pdfplumber


class DocumentProcessor:
    """
    A class to extract text from various document formats.
    
    Supports:
    - PDF files (using PyPDF2 and pdfplumber)
    - TXT files (plain text)
    """
    
    def __init__(self):
        """Initialize the document processor"""
        pass
    
    def extract_from_pdf(self, file) -> str:
        """
        Extract text from a PDF file.
        
        Uses pdfplumber as primary extractor (better for complex layouts)
        and falls back to PyPDF2 if needed.
        
        Args:
            file: A file-like object (from Streamlit uploader) or file path
            
        Returns:
            Extracted text as a string
        """
        text_content = []
        
        try:
            # Reset file pointer if it's a file-like object
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for reference
                        text_content.append(f"[Page {page_num}]\n{page_text}")
            
            if text_content:
                return "\n\n".join(text_content)
                
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            # Fall back to PyPDF2
            pass
        
        # Fallback: Use PyPDF2
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
            
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"[Page {page_num}]\n{page_text}")
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")
    
    def extract_from_txt(self, file) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file: A file-like object (from Streamlit uploader) or file path
            
        Returns:
            Text content as a string
        """
        try:
            # Reset file pointer if it's a file-like object
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Try to read as string
            if hasattr(file, 'read'):
                content = file.read()
                
                # If content is bytes, decode it
                if isinstance(content, bytes):
                    # Try UTF-8 first, then fall back to other encodings
                    for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                        try:
                            return content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    
                    # Last resort: decode with errors='replace'
                    return content.decode('utf-8', errors='replace')
                
                return content
            
            # If it's a file path
            with open(file, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            raise ValueError(f"Failed to extract text from TXT: {e}")
    
    def extract_text(self, file, file_type: str) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file: A file-like object or file path
            file_type: File extension (e.g., '.pdf', '.txt')
            
        Returns:
            Extracted text as a string
        """
        file_type = file_type.lower()
        
        if file_type == '.pdf':
            return self.extract_from_pdf(file)
        elif file_type == '.txt':
            return self.extract_from_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple consecutive newlines
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove multiple consecutive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
