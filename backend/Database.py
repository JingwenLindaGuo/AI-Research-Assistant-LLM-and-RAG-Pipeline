from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
# from integration.Embedder import embed # for script
from backend.Embedder import embed # for notebook
import random
import string
import uuid

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


def to_int(s: str) -> int:
    return int(s)

def save_embeddings_to_index(index_name, text, titles, authors, dates, sources, namespace="default"):
    # Retrieve the index instance
    index = pc.Index(index_name)

    # Convert text into a list of dicts with unique IDs
    data_with_ids = []

    # 注意：这里我们假设 text, titles, authors, dates, sources 都是列表
    for t, ti, a, d, s in zip(text, titles, authors, dates, sources):

        # merge each text chunk with its title and author
        t = f"'{t}'\ntext from {ti}\nauthored by {', '.join(a)}"

        data_with_ids.append({
            "id": str(uuid.uuid4()),  # generate a random UUID
            "text": t,
            "title": ti.lower(),
            "author": a,
            "date": to_int(d),
            "source": s
        })

    print('pinecone gonna store: ')
    for i in data_with_ids[:5]:
        print(i)
    # Embed the data
    embeddings = embed(data_with_ids)

    # Prepare vectors for upsert
    vectors = []
    for d, e in zip(data_with_ids, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {
                "text": d['text'],
                "title": d['title'],
                "author": d["author"], 
                "date": d["date"],
                "source": d['source']
            }
        })
    
    # Upsert embeddings into the index
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    print(f"Upserted {len(vectors)} embeddings into index '{index_name}', namespace '{namespace}'.")


