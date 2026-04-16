"""Embedding generation module using Sentence Transformers."""

from typing import List
from sentence_transformers import SentenceTransformer
import streamlit as st


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers (local, reliable, no API calls)"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize embedding generator with sentence-transformers model (768 dimensions)"""
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            self.model = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformers"""
        if self.model is None:
            return []
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return []
