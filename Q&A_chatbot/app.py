import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OPENAI"

## Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user Queries"),
        ("user","Question:{question}")
    ]
)
def generate_response(question,api_key,llm,temperature,max_tokens):
    llm=ChatOpenAI(
        model=llm,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## title of the app
st.title=("Q&A Chatbot")
api_key=os.getenv("OPENROUTER_API_KEY", "")
llm=st.sidebar.selectbox("Select Model (FREE - No charges)",[
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free"
],index=0)
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=2000,value=1000)
## main interface for user input
st.write("Ask any questions")
user_input=st.text_input("You:")
if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the query:")
