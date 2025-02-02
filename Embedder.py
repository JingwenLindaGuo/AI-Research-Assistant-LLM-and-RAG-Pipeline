from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


response = client.embeddings.create(
    input="Text to be embeded",
    model="text-embedding-3-small"
)

print(response.data[0].embedding)