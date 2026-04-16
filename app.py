"""
RAG-Based AI Chatbot - Web + PDF QA System
Streamlit UI for semantic search and document Q&A
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from modules.rag_system import RAGSystem

# Load environment variables (for local dev)
load_dotenv()

# Get API keys (prefer st.secrets for Streamlit Cloud)
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')

# Use st.secrets if running on Streamlit Cloud
if 'NVIDIA_API_KEY' in st.secrets:
    NVIDIA_API_KEY = st.secrets['NVIDIA_API_KEY']
    os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY
if 'PINECONE_API_KEY' in st.secrets:
    PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

if not NVIDIA_API_KEY:
    st.error("⚠️ NVIDIA_API_KEY not found in environment variables or Streamlit secrets. Please add it to your .env file or Streamlit secrets.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG-Based AI Chatbot (Web + PDF QA System)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem(
        pinecone_api_key=PINECONE_API_KEY,
        nvidia_api_key=NVIDIA_API_KEY
    )
    st.session_state.conversation_history = []
    st.session_state.processed_sources = []


def main():
    """Main application"""
    
    # Header
    st.title("📚 RAG-Based AI Chatbot (Web + PDF QA System)")
    current_model = st.session_state.rag_system.llm.model_name
    st.markdown(f"*Powered by NVIDIA {current_model} with Semantic Search & Pinecone Vector Database*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Content Management")
        
        # Choose input type
        input_type = st.radio(
            "Select Content Source:",
            ["Upload PDF/Text", "Add Website"],
            horizontal=False
        )
        
        if input_type == "Upload PDF/Text":
            # File upload
            uploaded_files = st.file_uploader(
                "📤 Upload Documents",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'md'],
                help="Supported formats: PDF, TXT, MD"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Documents", type="primary", use_container_width=False):
                    for uploaded_file in uploaded_files:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            file_content = uploaded_file.read()
                            if st.session_state.rag_system.process_document(file_content, uploaded_file.name):
                                st.session_state.processed_sources.append({
                                    'name': uploaded_file.name,
                                    'type': 'document',
                                    'added': datetime.now()
                                })
        
        else:  # Website
            # Website URL input
            website_url = st.text_input("🌐 Enter Website URL:", placeholder="https://example.com")
            
            if website_url:
                if st.button("🔄 Scrape & Process Website", type="primary", use_container_width=False):
                    with st.spinner(f"Processing {website_url}..."):
                        if st.session_state.rag_system.process_website(website_url):
                            st.session_state.processed_sources.append({
                                'name': website_url,
                                'type': 'website',
                                'added': datetime.now()
                            })
        
        st.divider()
        
        # Display processed sources
        if st.session_state.processed_sources:
            st.subheader("📚 Processed Sources")
            for source in st.session_state.processed_sources:
                icon = "🌐" if source['type'] == 'website' else "📄"
                st.write(f"{icon} {source['name']}")
            
            if st.button("🗑️ Clear All Sources", use_container_width=False):
                pinecone_key = os.getenv('PINECONE_API_KEY', '')
                nvidia_key = os.getenv('NVIDIA_API_KEY')
                # Use st.secrets if available
                if 'NVIDIA_API_KEY' in st.secrets:
                    nvidia_key = st.secrets['NVIDIA_API_KEY']
                if 'PINECONE_API_KEY' in st.secrets:
                    pinecone_key = st.secrets['PINECONE_API_KEY']
                st.session_state.rag_system = RAGSystem(pinecone_api_key=pinecone_key, nvidia_api_key=nvidia_key)
                st.session_state.processed_sources = []
                st.session_state.conversation_history = []
                st.rerun()
        
        st.divider()
        
        # System info
        st.subheader("⚙️ System Info")
        st.write(f"**Model:** {st.session_state.rag_system.llm.model_name}")
        st.write(f"**Embeddings:** Google Embedding-001")
        st.write(f"**Database:** {'Pinecone' if st.session_state.rag_system.vector_db.initialized else 'In-Memory'}")
        st.write(f"**Loaded Chunks:** {len(st.session_state.rag_system.in_memory_chunks)}")
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Conversation History"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        analytics_interface()


def chat_interface():
    """Chat interface"""
    st.header("💬 Chat with Your Sources")
    
    if not st.session_state.processed_sources:
        st.info("👆 Please upload documents or add websites to start chatting!")
    else:
        # Display conversation
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("📚 View Sources"):
                            for source in message["sources"]:
                                st.write(f"**{source['source']}** (Score: {source['similarity_score']:.3f})")
                                st.write(f"*{source['preview']}*")
                                st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response_data = st.session_state.rag_system.generate_response(query=prompt)
                    
                    st.markdown(response_data['answer'])
            
            # Add to history
            assistant_message = {
                "role": "assistant",
                "content": response_data['answer'],
                "sources": response_data.get('sources', [])
            }
            st.session_state.conversation_history.append(assistant_message)


def analytics_interface():
    """Conversation history and analytics"""
    st.header("📊 Conversation History")
    
    if st.session_state.conversation_history:
        # Show stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_messages = sum(1 for m in st.session_state.conversation_history if m["role"] == "user")
            st.metric("Total Questions", user_messages)
        
        with col2:
            st.metric("Loaded Chunks", len(st.session_state.rag_system.in_memory_chunks))
        
        with col3:
            st.metric("Sources", len(st.session_state.processed_sources))
        
        st.divider()
        
        # Create dataframe
        history_data = []
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                history_data.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Role': 'Question',
                    'Message': msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=False)
        
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state.conversation_history = []
            st.session_state.rag_system.clear_history()
            st.rerun()
    else:
        st.info("No conversation yet. Start chatting to see history.")


if __name__ == "__main__":
    main()
