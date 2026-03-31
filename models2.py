import os
import httpx
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"

response = httpx.get(url)

print(response.json())