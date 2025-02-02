from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Pinecone client with API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))