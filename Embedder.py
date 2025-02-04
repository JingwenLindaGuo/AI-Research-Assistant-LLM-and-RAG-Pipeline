from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(data):
    response = client.embeddings.create(
        input=[d['text'] for d in data],
        model="text-embedding-3-small"
    )
    return response   