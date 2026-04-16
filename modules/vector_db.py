"""Vector database module for Pinecone integration."""

import hashlib
from typing import List, Dict, Any
import pinecone
import streamlit as st


class PineconeVectorDB:
    """Pinecone vector database for storing and retrieving embeddings"""
    
    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None):
        self.api_key = api_key
        self.environment = environment or 'us-east-1'
        self.index_name = index_name or 'rag-chatbot'
        self.index = None
        self.initialized = False
        
        if self.api_key:
            self.initialize()
    
    def initialize(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=self.api_key)
            
            # Check if index exists, if not create it
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                st.info(f"Creating Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
                )
                import time
                time.sleep(1)
            
            self.index = pc.Index(self.index_name)
            self.initialized = True
            st.success("✅ Connected to Pinecone vector database")
        except Exception as e:
            st.warning(f"⚠️ Pinecone connection failed: {str(e)}. Using in-memory vector storage instead.")
            self.initialized = False
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Store embeddings in Pinecone"""
        if not self.initialized or not self.index:
            st.warning("Pinecone not available. Using in-memory storage.")
            return False
        
        try:
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_id = hashlib.md5(chunk['text'].encode()).hexdigest()
                metadata = {
                    'text': chunk['text'][:1000],
                    'source': chunk['source'],
                    'chunk_id': chunk['chunk_id'],
                    'source_type': chunk['source_type']
                }
                vectors_to_upsert.append((vector_id, embedding, metadata))
            
            self.index.upsert(vectors=vectors_to_upsert)
            return True
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if not self.initialized or not self.index:
            return []
        
        try:
            results = self.index.query(vector=embedding, top_k=k, include_metadata=True)
            
            output = []
            for match in results['matches']:
                output.append({
                    'text': match['metadata'].get('text', ''),
                    'score': match['score'],
                    'source': match['metadata'].get('source', 'unknown'),
                    'chunk_id': match['metadata'].get('chunk_id', 0),
                    'source_type': match['metadata'].get('source_type', 'document')
                })
            
            return output
        except Exception as e:
            st.error(f"Error searching embeddings: {str(e)}")
            return []
