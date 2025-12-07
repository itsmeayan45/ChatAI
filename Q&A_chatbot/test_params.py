from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", "")

# Test different parameter names
print("Testing parameter names...")

try:
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        model_kwargs={"max_tokens": 1000}
    )
    print("✓ max_tokens works")
    print("✓ max_tokens works")
    result = llm.invoke("Say hello")
    print(f"Result: {result.content}")
except Exception as e:
    print(f"✗ max_tokens failed: {e}")
