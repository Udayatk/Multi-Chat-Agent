<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Multi-PDF Chat AI Agent Instructions

This is a Python application that creates an AI-powered chat interface for querying multiple PDF documents simultaneously.

## Key Components:
- **PDF Processing**: Use PyPDF2 or similar libraries for text extraction
- **Vector Embeddings**: Implement RAG (Retrieval Augmented Generation) using sentence-transformers or OpenAI embeddings
- **Vector Database**: Use ChromaDB or FAISS for efficient similarity search
- **Chat Interface**: Streamlit-based web interface for user interaction
- **AI Integration**: OpenAI GPT or similar models for conversational responses

## Development Guidelines:
- Follow clean code principles with proper error handling
- Implement chunking strategies for large documents
- Use environment variables for API keys and sensitive data
- Create modular components for PDF processing, embedding, and chat functionality
- Implement proper session management for multi-document conversations
- Add progress indicators for long-running operations like PDF processing
