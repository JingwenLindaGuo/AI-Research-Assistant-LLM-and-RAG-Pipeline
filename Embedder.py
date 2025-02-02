from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OpenAI.api_key = os.getenv("API_KEY")
print(OpenAI.api_key)