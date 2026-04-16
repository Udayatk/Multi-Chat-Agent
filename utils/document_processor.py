"""Document processing module for PDF, TXT, and MD files."""

import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import PyPDF2
from io import BytesIO
import streamlit as st


class DocumentProcessor:
    """Process various document types and extract text"""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'PDF Document',
        '.txt': 'Text File',
        '.md': 'Markdown File'
    }
    
    def __init__(self):
        self.processed_documents = {}
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content to detect duplicates"""
        return hashlib.sha256(file_content).hexdigest()
    
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Extract text from PDF with metadata"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            pages_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'content': page_text,
                    'word_count': len(page_text.split())
                })
            
            full_text = " ".join([page['content'] for page in pages_text])
            
            return {
                'filename': filename,
                'file_type': 'pdf',
                'pages': len(pdf_reader.pages),
                'full_text': full_text,
                'pages_content': pages_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'processed_at': datetime.now(),
                'file_hash': self.get_file_hash(file_content),
                'source_type': 'document'
            }
        except Exception as e:
            st.error(f"Error processing PDF {filename}: {str(e)}")
            return None
    
    def extract_text_from_txt(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Extract text from plain text file"""
        try:
            text = file_content.decode('utf-8')
            
            return {
                'filename': filename,
                'file_type': 'txt',
                'full_text': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'processed_at': datetime.now(),
                'file_hash': self.get_file_hash(file_content),
                'source_type': 'document'
            }
        except Exception as e:
            st.error(f"Error processing TXT {filename}: {str(e)}")
            return None
    
    def process_document(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Process document based on file extension"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        processors = {
            '.pdf': self.extract_text_from_pdf,
            '.txt': self.extract_text_from_txt,
            '.md': self.extract_text_from_txt
        }
        
        if file_ext not in processors:
            st.error(f"Unsupported file format: {file_ext}")
            return None
        
        return processors[file_ext](file_content, filename)
