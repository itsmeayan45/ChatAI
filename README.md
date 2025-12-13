# Conversational Q&A Chatbot with PDF

A Streamlit-based conversational AI chatbot that allows users to upload PDF documents and ask questions about their content using RAG (Retrieval Augmented Generation).

## Features

- **PDF Document Upload**: Upload any PDF file to analyze
- **Conversational Memory**: Maintains chat history across the conversation
- **Context-Aware Responses**: Uses chat history to understand follow-up questions
- **RAG Implementation**: Combines document retrieval with LLM responses
- **Session Management**: Support for multiple chat sessions with unique session IDs

## Technology Stack

- **Streamlit**: Web interface
- **LangChain**: RAG pipeline and conversation management
- **FAISS**: Vector store for document embeddings
- **OpenRouter API**: Access to free LLM models (tngtech/tng-r1t-chimera:free)
- **PyPDF**: PDF document loading

## How It Works

1. **PDF Processing**:

   - PDF is uploaded and temporarily saved
   - Document is loaded and split into chunks (1000 chars with 200 overlap)
   - Chunks are embedded and stored in FAISS vector store

2. **Query Processing**:

   - User question is contextualized using chat history
   - Relevant document chunks are retrieved
   - LLM generates answer based on retrieved context

3. **Conversation Flow**:
   - Chat history is maintained per session
   - Each question is processed with full conversation context
   - Responses are concise and context-aware

## Setup

1. **Install Dependencies**:

```bash
pip install streamlit langchain langchain-community langchain-openai faiss-cpu pypdf python-dotenv numpy
```

2. **Set Up API Key**:
   Create a `.env` file in the project directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

3. **Run the Application**:

```bash
streamlit run app.py
```

## Usage

1. Open the application in your browser (typically http://localhost:8501)
2. Enter a session ID (or use the default)
3. Upload a PDF file
4. Wait for the PDF to be processed
5. Start asking questions about the PDF content
6. The chatbot will respond based on the document content and conversation history

## Configuration

- **Model**: `tngtech/tng-r1t-chimera:free` (via OpenRouter)
- **Temperature**: 0.7
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Dimensions**: 384 (using SimpleEmbeddings)

## Notes

- Uses a simple random embedding implementation (SimpleEmbeddings) for demonstration purposes
- Free model accessed through OpenRouter API
- Temporary PDF files are stored locally during processing
- Chat history is stored in session state and persists during the session

## Limitations

- Uses random embeddings (not semantic) - for production use, implement proper embeddings
- Depends on free API availability
- Temporary files are created during PDF processing

## Author

Created as a mini project for Operating System Lab.
