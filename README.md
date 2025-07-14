# 🚀 Advanced Multi-PDF Chat AI Agent

An intelligent AI-powered chat application that allows you to upload multiple PDF documents and have conversations with their content using Google Gemini AI.

## ✨ Features

- 📄 **Multi-format Support**: Upload PDF, TXT, and Markdown files
- 🤖 **AI-Powered Chat**: Conversation interface powered by Google Gemini
- 🔍 **Intelligent Search**: Keyword-based document search with relevance scoring
- 📊 **Analytics Dashboard**: Document statistics and session analytics
- **Analytics Dashboard**: Track usage statistics and document analytics
- **Multi-Tab Interface**: Organized chat, search, and analytics views
- **Document Management**: Smart deduplication and document library
- **Conversation Memory**: Maintains context throughout your session
- **User-Friendly Interface**: Modern, responsive Streamlit web interface
- **Simple Setup**: Minimal dependencies and easy configuration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   copy .env.example .env
   ```
   Edit the `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload PDFs**: Use the sidebar to upload multiple PDF files
2. **Wait for Processing**: The app will automatically extract text from all PDFs
3. **Start Chatting**: Ask questions about your documents in the chat interface
4. **Explore**: Try different types of questions:
   - "What is the main topic of these documents?"
   - "Can you summarize the key points?"
   - "Find information about [specific topic]"

## 🛠️ Technical Details

### Architecture

- **Frontend**: Streamlit web application
- **PDF Processing**: PyPDF2 for text extraction
- **AI Model**: Google Gemini 2.0 Flash
- **Memory**: Session-based conversation history

### Dependencies

This project uses minimal dependencies:
- `streamlit` - Web interface
- `PyPDF2` - PDF text extraction
- `google-generativeai` - Google Gemini AI integration
- `python-dotenv` - Environment variable management

## 🔧 Configuration

Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 📝 File Structure

```
Multi Chat Agent/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (your API keys)
├── .env.example          # Environment template
└── README.md             # This file
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
- **gemini-pro**: Google's most capable model for text
- **gemini-pro-vision**: Gemini with vision capabilities

### File Structure

```
Multi-PDF-Chat-Agent/
├── app.py                 # Main Streamlit application
├── pdf_processor.py       # PDF text extraction and processing
├── chat_engine.py         # AI chat and retrieval logic
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .github/
│   └── copilot-instructions.md
└── README.md             # This file
```

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `CHUNK_SIZE`: Text chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between text chunks (default: 200)
- `TEMPERATURE`: AI response creativity (default: 0.7)

### Customization

You can modify the following parameters in the code:

- **Chunk Size**: Adjust `chunk_size` in `PDFProcessor` for different text splitting
- **Model Selection**: Change `model_name` in `ChatEngine` for different OpenAI models
- **Retrieval Settings**: Modify `search_kwargs` in `ChatEngine` for different search behavior

## 🔧 Troubleshooting

### Common Issues

1. **"No module named" errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API errors**: Verify your API key is correct and has credits
3. **PDF processing fails**: Ensure your PDFs contain extractable text (not just images)
4. **Memory issues**: For large PDFs, try reducing chunk size or processing fewer documents

### Performance Tips

- **Large PDFs**: Process documents in smaller batches
- **Memory Usage**: Restart the app periodically when processing many documents
- **Response Speed**: Use smaller embedding models for faster processing

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://python.langchain.com/)
- AI responses by [OpenAI](https://openai.com/)
- Vector embeddings by [Sentence Transformers](https://www.sbert.net/)
