from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from .Embedder import embed
import random
import string

# Load environment variables
load_dotenv()

# Create Pinecone client with API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def create_index(dimension=1536, metric="cosine", cloud="aws", region="us-east-1"):
    
    # Generate random name for an index
    str_characters = string.ascii_letters + string.digits
    index_name = ''.join(random.choice(str_characters) for _ in range(12))

    # Check if the index already exists
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            ) 
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists, skipping creation.")
    
    # Return the name of the index created
    return index_name

def delete_index(index_name):
    try:
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully!")
    except Exception as e: # if the index does not exist catch the error
        print(f"Index '{index_name}' does not exist or was already deleted.")

def save_embeddings_to_index(index_name, data, namespace="default"):

    # Retrieve the index instance
    index = pc.Index(index_name)

    # Embed the data
    embeddings = embed(data)

    # Prepare vectors for upsert
    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {"text": d['text']}
        })
    
    # Upsert embeddings into the index
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    print(f"Upserted {len(vectors)} embeddings into index '{index_name}', namespace '{namespace}'.")