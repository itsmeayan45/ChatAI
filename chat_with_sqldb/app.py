import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from pydantic import SecretStr
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
load_dotenv()

st.set_page_config(page_title="chat With SQL DB")
st.title("chat with databases")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

# radio options
radio_opt=["Use SQlite 3 database -> studentdb.db","Connect to your SQL database"]
selected_opt=st.sidebar.radio(label="Choose the db you want to chat",options=radio_opt)
if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MYSQL host name")
    mysql_user=st.sidebar.text_input("MYSQL user")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MYSQL database")

else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="Groq api key",type="password")

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCALDB:
        db_file_path=(Path(__file__).parent/"studentdb.db").absolute()
        creator=lambda : sqlite3.connect(f"file:{db_file_path}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_db and mysql_password and mysql_user):
            st.error("Please provide all MYSQL connection details.")
            st.stop()
            return None
        
        # URL encode credentials to handle special characters
        encoded_user = quote_plus(mysql_user)
        encoded_password = quote_plus(mysql_password)
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{encoded_user}:{encoded_password}@{mysql_host}/{mysql_db}"))
    return None

# Check if all required inputs are provided before proceeding
if not api_key:
    st.info("Please add the Groq API key in the sidebar to continue")
    st.stop()

if db_uri==MYSQL:
    if not (mysql_host and mysql_user and mysql_password and mysql_db):
        st.info("Please provide all MySQL connection details in the sidebar")
        st.stop()

## LLM model
llm=ChatGroq(api_key=SecretStr(api_key),model="llama-3.3-70b-versatile",streaming=True)
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)

if db is None:
    st.error("Failed to configure database")
    st.stop()

## toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

agent=create_sql_agent(
    llm,
    toolkit,
    verbose=True,
    agent_type="zero-shot-react-description"
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state['messages']=[{'role':'assistant','content':'How can i help you?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query=st.chat_input(placeholder="Ask Anything from the database")

if user_query:
    st.session_state.messages.append({'role':'user','content':user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.invoke({"input": user_query},config={"callbacks":[streamlit_callback]})
        st.session_state.messages.append({'role':'assistant','content':response['output']})
        st.write(response['output'])