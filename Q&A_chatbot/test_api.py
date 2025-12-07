from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY", "")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please response to the user Queries"),
    ("user", "Question:{question}")
])

# Test with different configurations
print("\n=== Testing OpenRouter API ===\n")

try:
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        max_completion_tokens=1000
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    question = "Explain what machine learning is in detail"
    print(f"Question: {question}\n")
    
    answer = chain.invoke({'question': question})
    print(f"Answer:\n{answer}\n")
    print(f"Answer length: {len(answer)} characters")
    
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"Details: {str(e)}")
