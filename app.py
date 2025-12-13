# by the way , i am GORIB , so i use opensource model to implement this , using openrouter api key to access free models
import streamlit as st
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
import numpy as np
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

class SimpleEmbeddings(Embeddings):
    def __init__(self, size=384):
        self.size = size
    
    def embed_documents(self, texts):
        return [np.random.rand(self.size).tolist() for _ in texts]
    
    def embed_query(self, text):
        return np.random.rand(self.size).tolist()

@st.cache_resource
def get_embeddings():
    return SimpleEmbeddings(size=384)

embeddings = get_embeddings()


st.title("ChatAI")
st.write("Upload a PDF and have a conversation about its content")


# Get API key from Streamlit secrets (for cloud) or environment (for local)
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    # Remove quotes if present
    api_key = api_key.strip('"').strip("'")
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip('"').strip("'")

if not api_key:
    st.error("Please set OPENROUTER_API_KEY in Streamlit secrets or .env file")
    st.info("For Streamlit Cloud: Go to App Settings → Secrets and add: OPENROUTER_API_KEY = 'your-key-here'")
    st.stop()

# Debug info (remove after testing)
st.sidebar.text(f"API Key loaded: {api_key[:10]}...")

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="google/gemini-flash-1.5:free",
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    )

llm = get_llm()



# Session ID
session_id = st.text_input("Session ID", value="default_session")


if 'store' not in st.session_state:
    st.session_state.store = {}


if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None
if 'pdf_hash' not in st.session_state:
    st.session_state.pdf_hash = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file:", type='pdf')

# Process uploaded file
if uploaded_file:
    
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(file_bytes).hexdigest()
    
    
    if st.session_state.pdf_hash != current_hash:
        with st.spinner("Processing PDF..."):
            # Save temporary PDF
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(file_bytes)
            
            # Load and split document
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Create vector store
            st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.pdf_hash = current_hash
            
            # Clean up temp file
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
            
            st.success("PDF processed successfully! You can now ask questions.")
            # Reset chain so it gets recreated with new vectorstore
            st.session_state.conversational_rag_chain = None
    
    # Create chains (only if not already created)
    if st.session_state.vectorstore is not None and st.session_state.conversational_rag_chain is None:
        retriever = st.session_state.vectorstore.as_retriever()
        
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
       
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\\n\\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
       
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
       
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        
        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Ask a question about the PDF:")
    
    if user_input and st.session_state.conversational_rag_chain:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get response
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
            
            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            with st.chat_message("assistant"):
                st.write(response['answer'])
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                st.error("❌ Authentication failed. Please check your API key in Streamlit secrets.")
            else:
                st.error(f"❌ Error: {error_msg}")
else:
    st.info("Please upload a PDF file to start chatting.")
# Adding a temp.pdf for checking , this is mini project of my collage in Operating System Lab
