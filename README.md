# Gen AI Projects

A comprehensive collection of AI-powered applications built with LangChain, Streamlit, and various free LLM APIs. This repository showcases practical implementations of chatbots, RAG systems, SQL agents, search engines, and text summarization tools.

## 🚀 Projects

### 1. Q&A Chatbot

**Location:** `Q&A_chatbot/`

A simple question-answering chatbot with a clean Streamlit interface.

**Features:**

- Direct conversational interface
- Powered by free OpenRouter models (Google Gemini Flash, Mistral)
- Clean and intuitive UI
- Configurable response parameters
- Multiple model implementations (OpenRouter, Ollama)

**Files:**

- `app.py` - Main Streamlit application
- `test_api.py` - API connection testing
- `find_free_models.py` - Discover available free models
- `ollamaapp.py` - Ollama integration variant
- `test_free_models.py` - Test free model availability
- `test_params.py` - Parameter configuration testing

**Run:**

```bash
cd Q&A_chatbot
streamlit run app.py
```

---

### 2. RAG Q&A Chatbot

**Location:** `rag_q&a_chatbot/`

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents.

**Features:**

- PDF document upload and processing
- FAISS vector store for efficient retrieval
- Custom embeddings implementation
- Context-aware responses from documents
- Question answering with source context

**Files:**

- `ragapp.py` - Main RAG application
- `pdfs/` - Directory for PDF documents

**Run:**

```bash
cd rag_q&a_chatbot
streamlit run ragapp.py
```

---

### 3. Conversational Q&A Chatbot with PDF

**Location:** `conversational_q&a_chatbot/`

An advanced conversational chatbot with PDF support and persistent chat history.

**Features:**

- Upload and query PDF documents
- Session-based chat history
- Conversational memory (remembers previous messages)
- RAG-based responses with context
- Streamlit Cloud ready
- Free AI models via OpenRouter

**Files:**

- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `README.md` - Project-specific documentation

**Run:**

```bash
cd conversational_q&a_chatbot
streamlit run app.py
```

---

### 4. Chat with SQL Database

**Location:** `chat_with_sqldb/`

A natural language interface for SQL databases powered by LangChain SQL agents and Groq's LLaMA model.

**Features:**

- 🗣️ Natural language to SQL query conversion
- 🗄️ Multiple database support (SQLite and MySQL)
- 🤖 AI-powered SQL agent using Groq LLaMA 3.3 70B
- 💬 Interactive chat interface with message history
- 🔍 Automatic query generation and execution

**Files:**

- `app.py` - Main Streamlit application
- `sqlite.py` - SQLite database utilities
- `requirements.txt` - Dependencies
- `README.md` - Detailed documentation
- `studentdb.db` - Sample SQLite database

**Run:**

```bash
cd chat_with_sqldb
streamlit run app.py
```

**Usage:**

1. Choose between SQLite (default) or MySQL
2. Enter your Groq API key
3. Ask questions about your database in plain English
4. View SQL queries and results

---

### 5. AI-Powered Search Engine

**Location:** `search_engine/`

An intelligent search application that uses LangChain agents to search across multiple sources (web, academic papers, and Wikipedia).

**Features:**

- 🔍 Multi-source search (DuckDuckGo, ArXiv, Wikipedia)
- 🤖 Intelligent agent-based query routing
- 🚀 Powered by Llama3-8B via Groq API
- 💬 Streaming responses for better UX
- 🎯 Context-aware tool selection
- 📚 Academic paper search via ArXiv

**Files:**

- `app.py` - Main Streamlit application
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `START_HERE.txt` - Quick start guide
- `tools_agents.ipynb` - Jupyter notebook with examples

**Run:**

```bash
cd search_engine
streamlit run app.py
```

**Tools Available:**

- **DuckDuckGo**: General web search
- **ArXiv**: Academic research papers
- **Wikipedia**: Encyclopedia knowledge

---

### 6. Text Summarization

**Location:** `summerize_text/`

A Jupyter notebook demonstrating text summarization capabilities using Groq's LLaMA model.

**Features:**

- Speech/text summarization
- Powered by Groq LLaMA 3.3 70B
- Expert system prompts for concise summaries
- Interactive Jupyter notebook environment

**Files:**

- `text_summerization.ipynb` - Main notebook with examples

**Run:**

```bash
cd summerize_text
jupyter notebook text_summerization.ipynb
```

---

## 🛠️ Tech Stack

**Core Technologies:**

- **Python 3.13+** - Programming language
- **Streamlit** - Web UI framework
- **LangChain** - LLM framework and chains
- **FAISS** - Vector database for embeddings
- **PyPDF** - PDF document processing
- **NumPy** - Numerical operations for embeddings
- **python-dotenv** - Environment variable management

**AI Models & APIs:**

- **OpenRouter API** - Free AI models (Gemini, Mistral)
- **Groq API** - Fast LLaMA model inference
- **SQLAlchemy** - Database connections and ORM

**Additional Tools:**

- **ArXiv API** - Academic paper search
- **Wikipedia API** - Knowledge base queries
- **DuckDuckGo Search** - Web search
- **Jupyter Notebook** - Interactive development

---

## 📋 Prerequisites

- **Python 3.8+** (Python 3.13 recommended)
- **API Keys:**
  - OpenRouter API key (free at [openrouter.ai](https://openrouter.ai/))
  - Groq API key (free at [console.groq.com](https://console.groq.com/keys))
- **Virtual environment** (recommended)

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/itsmeayan45/gen-ai.git
cd Gen-ai-projects
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv genvenv

# Activate (Windows)
genvenv\Scripts\activate

# Activate (Linux/Mac)
source genvenv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional
```

### 5. Run Any Project

Navigate to the project folder and run:

```bash
# Example: Run Q&A Chatbot
cd Q&A_chatbot
streamlit run app.py

# Example: Run Search Engine
cd search_engine
streamlit run app.py

# Example: Run SQL Chat
cd chat_with_sqldb
streamlit run app.py
```

---

## 🌐 Free AI Models Used

This repository uses completely free AI models:

**OpenRouter (No Cost):**

- `google/gemini-2.0-flash-exp:free`
- `mistralai/mistral-7b-instruct:free`
- `tngtech/tng-r1t-chimera:free`

**Groq (Free Tier):**

- `llama-3.3-70b-versatile`
- `llama3-8b-8192`

No API costs required for basic usage! 🎉

---

## 📁 Project Structure

```
Gen-ai-projects/
├── Q&A_chatbot/                    # Basic Q&A chatbot
│   ├── app.py                      # Main application
│   ├── ollamaapp.py               # Ollama variant
│   ├── test_api.py                # API testing
│   └── find_free_models.py        # Model discovery
│
├── rag_q&a_chatbot/               # RAG chatbot with PDF
│   ├── ragapp.py                  # Main application
│   └── pdfs/                      # PDF storage
│
├── conversational_q&a_chatbot/    # Advanced conversational chatbot
│   ├── app.py                     # Main application
│   ├── requirements.txt           # Dependencies
│   └── README.md                  # Documentation
│
├── chat_with_sqldb/               # SQL database chatbot
│   ├── app.py                     # Main application
│   ├── sqlite.py                  # SQLite utilities
│   ├── studentdb.db              # Sample database
│   └── README.md                  # Documentation
│
├── search_engine/                 # AI-powered search
│   ├── app.py                     # Main application
│   ├── tools_agents.ipynb        # Examples notebook
│   └── README.md                  # Documentation
│
├── summerize_text/                # Text summarization
│   └── text_summerization.ipynb  # Jupyter notebook
│
├── genvenv/                       # Virtual environment
├── app.py                         # Root chatbot app
├── requirements.txt               # Global dependencies
├── .env                           # API keys (not in git)
├── .gitignore                     # Git exclusions
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🚢 Deployment

### Streamlit Cloud Deployment

Most projects are ready for deployment on Streamlit Cloud:

1. **Push to GitHub**
2. **Connect to [share.streamlit.io](https://share.streamlit.io/)**
3. **Add API keys in Streamlit secrets:**
   ```toml
   OPENROUTER_API_KEY = "your_key_here"
   GROQ_API_KEY = "your_key_here"
   ```
4. **Deploy!**

For detailed deployment instructions, see individual project README files.

---

## 💡 Use Cases

- **Q&A Chatbots**: Customer support, FAQs, general inquiries
- **RAG Systems**: Document analysis, knowledge base queries, research assistance
- **SQL Chat**: Natural language database queries, data analysis
- **Search Engine**: Academic research, web search, knowledge discovery
- **Text Summarization**: Document summarization, content compression

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Fork this repository
- Create a feature branch
- Submit pull requests
- Report issues
- Suggest improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ayan**

- GitHub: [@itsmeayan45](https://github.com/itsmeayan45)

---

## 🙏 Acknowledgments

- **LangChain** - Amazing LLM framework
- **OpenRouter** - Free AI model access
- **Groq** - Fast LLM inference
- **Streamlit** - Easy-to-use web framework
- **Open Source Community** - For all the amazing tools and libraries


## ⭐ Star This Repository

If you find this project helpful, please give it a star! It helps others discover these resources.

---

**Happy Coding! 🚀**
