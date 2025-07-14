import streamlit as st
import os
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Document processing
import PyPDF2
from io import BytesIO

# Google Gemini
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Page configuration
st.set_page_config(
    page_title="Advanced Multi-PDF Chat AI Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleDocumentProcessor:
    """Simple document processor without embeddings"""
    
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
                'file_hash': self.get_file_hash(file_content)
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
                'file_hash': self.get_file_hash(file_content)
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Simple text chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                look_back = min(100, end - start)
                sentence_ends = ['.', '!', '?', '\n\n']
                
                for i in range(end - look_back, end):
                    if text[i] in sentence_ends and i < len(text) - 1:
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_id': len(chunks),
                    'word_count': len(chunk_text.split())
                })
            
            start = max(start + chunk_size - overlap, end)
        
        return chunks

class SimpleRAGEngine:
    """Simple RAG engine using text search instead of embeddings"""
    
    def __init__(self, model_name: str = "models/gemini-2.0-flash-001"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.documents = {}
        self.conversation_history = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search index"""
        for doc in documents:
            doc_id = doc['file_hash']
            self.documents[doc_id] = doc
    
    def keyword_search(self, query: str, k: int = 5, document_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        results = []
        query_words = query.lower().split()
        
        for doc_id, doc in self.documents.items():
            # Apply document filter
            if document_filter and doc['filename'] not in document_filter:
                continue
            
            # Search in chunks if available, otherwise in full text
            if 'chunks' in doc:
                for chunk in doc['chunks']:
                    text_lower = chunk['text'].lower()
                    score = sum(1 for word in query_words if word in text_lower)
                    
                    if score > 0:
                        results.append({
                            'text': chunk['text'],
                            'score': score / len(query_words),
                            'metadata': {
                                'filename': doc['filename'],
                                'file_type': doc['file_type'],
                                'chunk_id': chunk['chunk_id'],
                                'word_count': chunk['word_count']
                            }
                        })
            else:
                # Search in full text
                text_lower = doc['full_text'].lower()
                score = sum(1 for word in query_words if word in text_lower)
                
                if score > 0:
                    # Create chunks on the fly for display
                    preview = doc['full_text'][:500] + "..." if len(doc['full_text']) > 500 else doc['full_text']
                    results.append({
                        'text': preview,
                        'score': score / len(query_words),
                        'metadata': {
                            'filename': doc['filename'],
                            'file_type': doc['file_type'],
                            'chunk_id': 0,
                            'word_count': doc['word_count']
                        }
                    })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def generate_response(
        self, 
        query: str, 
        context_window: int = 3000,
        document_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate response using simple RAG"""
        
        # Search for relevant content
        search_results = self.keyword_search(
            query=query,
            k=5,
            document_filter=document_filter
        )
        
        if not search_results:
            return {
                'answer': "I couldn't find relevant information in the uploaded documents to answer your question.",
                'sources': [],
                'context_used': ""
            }
        
        # Build context from search results
        context_parts = []
        sources = []
        total_chars = 0
        
        for result in search_results:
            text = result['text']
            metadata = result['metadata']
            
            if total_chars + len(text) > context_window:
                break
            
            context_parts.append(f"[From {metadata['filename']}]: {text}")
            sources.append({
                'filename': metadata['filename'],
                'file_type': metadata['file_type'],
                'chunk_id': metadata.get('chunk_id', 0),
                'similarity_score': result['score'],
                'preview': text[:200] + "..." if len(text) > 200 else text
            })
            
            total_chars += len(text)
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""Based on the following context from uploaded documents, please answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Provide a comprehensive answer based only on the information in the context
- If the answer is not in the context, clearly state that
- Be specific and cite relevant details from the documents
- If multiple documents contain relevant information, synthesize the information appropriately

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            
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
                'context_used': context,
                'search_results_count': len(search_results)
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, there was an error generating the response: {str(e)}",
                'sources': [],
                'context_used': context
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.documents = {}
    st.session_state.document_processor = SimpleDocumentProcessor()
    st.session_state.rag_engine = SimpleRAGEngine()
    st.session_state.conversation_history = []
    st.session_state.current_session_start = datetime.now()

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to RAG system"""
    if not uploaded_files:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_docs = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Check if file already processed
        file_content = uploaded_file.read()
        file_hash = st.session_state.document_processor.get_file_hash(file_content)
        
        if file_hash in st.session_state.documents:
            st.info(f"File {uploaded_file.name} already processed (duplicate detected)")
            continue
        
        # Process document
        doc_data = st.session_state.document_processor.process_document(
            file_content, uploaded_file.name
        )
        
        if doc_data:
            # Chunk the document
            chunks = st.session_state.document_processor.chunk_text(
                doc_data['full_text'],
                chunk_size=1000,
                overlap=200
            )
            doc_data['chunks'] = chunks
            
            # Store in session state
            st.session_state.documents[file_hash] = doc_data
            processed_docs.append(doc_data)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Add to RAG engine
    if processed_docs:
        status_text.text("Indexing documents...")
        st.session_state.rag_engine.add_documents(processed_docs)
        status_text.text("✅ Processing complete!")
        time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()

def main():
    """Main application"""
    
    # Header
    st.title("🚀 Advanced Multi-PDF Chat AI Agent")
    st.markdown("*Powered by Google Gemini AI with Intelligent Document Search*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md'],
            help="Supported formats: PDF, TXT, MD"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Document Library
        if st.session_state.documents:
            st.subheader("📚 Document Library")
            for file_hash, doc in st.session_state.documents.items():
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Type:** {doc['file_type'].upper()}")
                    st.write(f"**Words:** {doc['word_count']:,}")
                    if 'pages' in doc:
                        st.write(f"**Pages:** {doc['pages']}")
                    if 'chunks' in doc:
                        st.write(f"**Chunks:** {len(doc['chunks'])}")
                    st.write(f"**Processed:** {doc['processed_at'].strftime('%Y-%m-%d %H:%M')}")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{file_hash}"):
                        del st.session_state.documents[file_hash]
                        st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "📊 Analytics"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        search_interface()
    
    with tab3:
        analytics_interface()

def chat_interface():
    """Main chat interface"""
    st.header("💬 Chat with Your Documents")
    
    if not st.session_state.documents:
        st.info("👆 Please upload and process some documents to start chatting!")
        return
    
    # Chat controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Document filter
        available_docs = [doc['filename'] for doc in st.session_state.documents.values()]
        selected_docs = st.multiselect(
            "Filter by documents (optional)",
            available_docs,
            help="Leave empty to search all documents"
        )
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.rag_engine.clear_conversation_history()
            st.rerun()
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source['filename']}** (Score: {source['similarity_score']:.3f})")
                            st.write(f"Preview: {source['preview']}")
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Get response from RAG engine
                response_data = st.session_state.rag_engine.generate_response(
                    query=prompt,
                    document_filter=selected_docs if selected_docs else None
                )
                
                response_time = time.time() - start_time
                
                # Display response
                st.markdown(response_data['answer'])
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant", 
            "content": response_data['answer'],
            "sources": response_data.get('sources', [])
        }
        st.session_state.conversation_history.append(assistant_message)

def search_interface():
    """Search interface"""
    st.header("🔍 Document Search")
    
    if not st.session_state.documents:
        st.warning("Please upload and process documents first.")
        return
    
    # Search input
    query = st.text_input("Enter your search query:")
    
    # Document filter
    available_docs = [doc['filename'] for doc in st.session_state.documents.values()]
    selected_docs = st.multiselect(
        "Filter by documents (optional)",
        available_docs
    )
    
    if query and st.button("🔍 Search"):
        results = st.session_state.rag_engine.keyword_search(
            query=query,
            k=10,
            document_filter=selected_docs if selected_docs else None
        )
        
        if not results:
            st.warning("No results found.")
        else:
            st.subheader(f"🎯 Found {len(results)} results")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1}: {result['metadata']['filename']} (Score: {result['score']:.3f})"):
                    st.write(f"**File:** {result['metadata']['filename']}")
                    st.write(f"**Type:** {result['metadata']['file_type']}")
                    st.write(f"**Content:**")
                    st.write(result['text'])

def analytics_interface():
    """Simple analytics interface"""
    st.header("📊 Analytics")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(st.session_state.documents))
    
    with col2:
        total_words = sum(doc['word_count'] for doc in st.session_state.documents.values())
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        st.metric("Conversations", len(st.session_state.conversation_history) // 2)  # Divide by 2 for user/assistant pairs
    
    # Document details
    if st.session_state.documents:
        st.subheader("📚 Document Details")
        
        doc_data = []
        for doc in st.session_state.documents.values():
            doc_data.append({
                'Filename': doc['filename'],
                'Type': doc['file_type'].upper(),
                'Words': doc['word_count'],
                'Pages': doc.get('pages', 1),
                'Chunks': len(doc.get('chunks', [])),
                'Processed': doc['processed_at'].strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
    
    # Session info
    st.subheader("🌍 Session Info")
    st.write(f"**Session Start:** {st.session_state.current_session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Google API Key:** {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Not set'}")

if __name__ == "__main__":
    main()
