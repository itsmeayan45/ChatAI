from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", "")

# Test different free models
free_models = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "mistralai/mistral-7b-instruct:free"
]

print("Testing free models on OpenRouter...\n")

for model_name in free_models:
    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_completion_tokens=100
        )
        result = llm.invoke("Say hello in one sentence")
        print(f"✓ {model_name}")
        print(f"  Response: {result.content[:100]}...\n")
    except Exception as e:
        print(f"✗ {model_name}")
        print(f"  Error: {str(e)[:150]}...\n")
