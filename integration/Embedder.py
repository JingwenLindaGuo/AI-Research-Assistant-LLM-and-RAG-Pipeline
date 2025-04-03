from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(data):

    text_inputs = [d['text'] for d in data] if isinstance(data, list) else [data]

    response = client.embeddings.create(
        input=text_inputs,
        model="text-embedding-3-small"
    )
    return [{"values": item.embedding} for item in response.data]  


