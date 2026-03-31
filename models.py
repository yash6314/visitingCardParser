import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

for model in models:
    print("MODEL:", model.name)
    print("SUPPORTED METHODS:", model.supported_generation_methods)
    print("-----")