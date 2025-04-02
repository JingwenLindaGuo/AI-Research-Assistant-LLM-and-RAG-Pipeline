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

def batch_embedding(text_chunks): 

    embedding_list = []
    batch_size = 500 
    total_items = len(text_chunks)

    for i in range(0, total_items, batch_size):
        batch_data = text_chunks[i:min( i + batch_size, total_items)]
        batch_texts = [item['text'] for item in batch_data]

        print(f"Processing batch {i // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} (items {i+1}-{min(i + batch_size, total_items)})")

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts  
            )

            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            embedding_list.extend(batch_embeddings)

        except Exception as e:
            print(f"An error occurred processing batch starting at index {i}: {e}")
            break
    return embedding_list 