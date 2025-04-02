from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from Embedder import embed, batch_embedding
import random
import string
import uuid
import itertools

# Load environment variables
load_dotenv()

# Create Pinecone client with API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def create_index(dimension=1536, metric="cosine", cloud="aws", region="us-east-1"):
    
    # Generate random name for an index
    str_characters = string.ascii_lowercase + string.digits
    index_name = ''.join(random.choice(str_characters) for _ in range(10))

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

def chunks(iterable, batch_size=200):
    """Yield successive batch_size chunks from iterable."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def save_embeddings_to_index(index_name, text, namespace="default"):
    index = pc.Index(index_name)

    # Create a list of dictionaries with unique IDs for each text chunk
    data_with_ids = []
    for t in text:
        data_with_ids.append({
            "id": str(uuid.uuid4()),
            "text": t
        })

    # Get embeddings for all text chunks
    raw_embeddings = batch_embedding(data_with_ids)
    print(f"Generated {len(raw_embeddings)} embeddings")

    if not raw_embeddings:
         print("Embedding process returned no results. Skipping upsert.")
         return 

    if len(data_with_ids) != len(raw_embeddings):
        print(f"Mismatch between number of text items ({len(data_with_ids)}) and embeddings ({len(raw_embeddings)}). Skipping upsert.")
        return

    # Build a list of vector records in the format expected by Pinecone
    vectors = []
    for d, e in zip(data_with_ids, raw_embeddings):
        vectors.append({
            "id": d['id'],
            "values": e,  
            "metadata": {"text": d['text']}
        })
    print(f"Vectors prepared for upsert: {len(vectors)}")
    
    if not vectors:
         print("Warning: No vectors were prepared for upsert, even after processing embeddings.")
         return
    
    # Upsert vectors in batches (e.g., 200 records per batch)
    batch_size = 200  # You can adjust this up to 1000, ensuring you do not exceed 2MB per request
    for batch in chunks(vectors, batch_size=batch_size):
         try:
             index.upsert(
                 vectors=batch,
                 namespace=namespace
             )
             print(f"Upserted batch of {len(batch)} embeddings into index '{index_name}', namespace '{namespace}'.")
         except Exception as upsert_error:
             print(f"Error during Pinecone upsert for a batch: {upsert_error}")