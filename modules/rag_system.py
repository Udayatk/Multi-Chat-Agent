"""Complete RAG system with semantic search and AI responses."""

import time
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st

from utils.document_processor import DocumentProcessor
from utils.web_scraper import WebScraper
from utils.semantic_chunker import SemanticChunker
from utils.embeddings import EmbeddingGenerator
from modules.vector_db import PineconeVectorDB
from modules.llm_providers import NvidiaAPI


class RAGSystem:
    """Complete RAG system with semantic search and AI responses"""
    
    def __init__(self, model_name: str = None, pinecone_api_key: str = None, nvidia_api_key: str = None):
        self.pinecone_api_key = pinecone_api_key
        self.chunker = SemanticChunker(chunk_size=1000, overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = PineconeVectorDB(api_key=pinecone_api_key)
        self.in_memory_chunks = []
        self.conversation_history = []
        
        # Initialize NVIDIA API
        if not nvidia_api_key:
            st.error("NVIDIA_API_KEY not found in environment variables.")
            raise RuntimeError("NVIDIA API key is required")
        
        self.llm = NvidiaAPI(api_key=nvidia_api_key)
        self.model_name = self.llm.model_name
    
    def process_document(self, file_content: bytes, filename: str) -> bool:
        """Process document and store chunks with embeddings"""
        processor = DocumentProcessor()
        doc_data = processor.process_document(file_content, filename)
        
        if not doc_data:
            return False
        
        # Create chunks
        chunks = self.chunker.chunk_text(
            doc_data['full_text'],
            {'filename': filename, 'source_type': 'document'}
        )
        
        if not chunks:
            st.error("No chunks created from document")
            return False
        
        # Generate embeddings
        embeddings = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_generator.generate_embedding(chunk['text'])
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
        progress_bar.empty()
        
        # Store in vector DB
        self.vector_db.store_embeddings(chunks, embeddings)
        
        # Also store in memory
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            self.in_memory_chunks.append(chunk)
        
        st.success(f"✅ Processed {filename} with {len(chunks)} semantic chunks")
        return True
    
    def process_website(self, url: str) -> bool:
        """Process website content and store chunks with embeddings"""
        scraper = WebScraper()
        web_data = scraper.extract_from_url(url)
        
        if not web_data:
            return False
        
        # Create chunks
        chunks = self.chunker.chunk_text(
            web_data['content'],
            {'url': url, 'title': web_data['title'], 'source_type': 'website'}
        )
        
        if not chunks:
            st.error("No chunks created from website")
            return False
        
        # Generate embeddings
        embeddings = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_generator.generate_embedding(chunk['text'])
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
        progress_bar.empty()
        
        # Store in vector DB
        self.vector_db.store_embeddings(chunks, embeddings)
        
        # Also store in memory
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            self.in_memory_chunks.append(chunk)
        
        st.success(f"✅ Processed {url} ({web_data['title']}) with {len(chunks)} semantic chunks")
        return True
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search in Pinecone if available
        results = self.vector_db.search(query_embedding, k=k)
        
        # If no results from Pinecone, search in memory
        if not results and self.in_memory_chunks:
            results = self._search_in_memory(query_embedding, k=k)
        
        return results
    
    def _search_in_memory(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Fallback: search in memory chunks"""
        if not query_embedding or not self.in_memory_chunks:
            return []
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            import math
            dot_product = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0
            return dot_product / (mag_a * mag_b)
        
        results = []
        for chunk in self.in_memory_chunks:
            if 'embedding' in chunk:
                score = cosine_similarity(query_embedding, chunk['embedding'])
                results.append({
                    'text': chunk['text'],
                    'score': score,
                    'source': chunk['source'],
                    'chunk_id': chunk['chunk_id'],
                    'source_type': chunk['source_type']
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def generate_response(self, query: str, context_limit: int = 3000) -> Dict[str, Any]:
        """Generate AI response based on semantic search"""
        
        # Semantic search
        search_results = self.semantic_search(query, k=5)
        
        if not search_results:
            return {
                'answer': "I couldn't find relevant information to answer your question. Please try uploading more documents or websites.",
                'sources': [],
                'used_embeddings': True
            }
        
        # Build context
        context_parts = []
        sources = []
        total_chars = 0
        
        for result in search_results:
            text = result['text']
            
            if total_chars + len(text) > context_limit:
                break
            
            source_label = result['source']
            if result['source_type'] == 'website':
                source_label = f"🌐 {result['source']}"
            else:
                source_label = f"📄 {result['source']}"
            
            context_parts.append(f"[{source_label}]\n{text}")
            
            sources.append({
                'source': result['source'],
                'chunk_id': result['chunk_id'],
                'similarity_score': result['score'],
                'source_type': result['source_type'],
                'preview': text[:150] + "..." if len(text) > 150 else text
            })
            
            total_chars += len(text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt for NVIDIA Gemma-4-31b-it
        prompt = f"""You are an expert AI assistant specialized in analyzing documents and answering questions based on provided context. 

Context from sources:
{context}

User Question: {query}

Instructions:
1. Provide a comprehensive, accurate answer based ONLY on the provided context
2. If the answer is not found in the context, clearly state: "I don't have this information in the provided sources"
3. Always cite the source when referencing specific information
4. If multiple sources provide relevant information, synthesize them appropriately
5. Be concise but thorough
6. Use markdown formatting for better readability

Answer:"""
        
        try:
            answer = self.llm.generate_response(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1024,
                stream=False
            )
            
            # Add to conversation history
            self.conversation_history.append({
                'question': query,
                'answer': answer,
                'sources': sources,
                'timestamp': datetime.now()
            })
            
            return {
                'answer': answer,
                'sources': sources,
                'used_embeddings': True,
                'search_results_count': len(search_results)
            }
        
        except Exception as e:
            error_msg = str(e)
            return {
                'answer': f"Error generating response: {error_msg}",
                'sources': [],
                'used_embeddings': True
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
