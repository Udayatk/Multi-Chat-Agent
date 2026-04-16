"""Semantic text chunking module using sentence-based splitting."""

from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SemanticChunker:
    """Semantic chunking using sentence-based splitting"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Semantically chunk text based on sentences"""
        # Clean text
        text = text.strip()
        if not text:
            return []
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'word_count': len(chunk_text.split()),
                    'source': source_info.get('filename', source_info.get('url', 'unknown')),
                    'source_type': source_info.get('source_type', 'document')
                })
                
                chunk_id += 1
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length < self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'word_count': len(chunk_text.split()),
                'source': source_info.get('filename', source_info.get('url', 'unknown')),
                'source_type': source_info.get('source_type', 'document')
            })
        
        return chunks
