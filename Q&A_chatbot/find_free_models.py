from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", "")

# Extended list of free models to test
free_models = [
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemini-flash-1.5:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
    "gryphe/mythomist-7b:free",
    "undi95/toppy-m-7b:free"
]

print("Testing extended list of free models...\n")
working_models = []

for model_name in free_models:
    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_completion_tokens=50
        )
        result = llm.invoke("Hi")
        working_models.append(model_name)
        print(f"✓ {model_name}")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            print(f"⚠ {model_name} (rate-limited)")
        else:
            print(f"✗ {model_name}")

print(f"\n{'='*60}")
print(f"Working models ({len(working_models)}):")
for model in working_models:
    print(f"  - {model}")
