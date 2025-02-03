import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_5DwedT_D4bc61kqQVy1LCVf3mGuVPMYRRkLhHcS58kTuL4Pz2PB3gGRwBkunivicMq8yA8")
index = pc.Index("test")

# Initialize OpenAI Embeddings (matches what Member C is using)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the Pinecone vector database.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        List[Dict]: Retrieved chunks with metadata.
    """
    query_embedding = embeddings.embed_documents([query])[0]  # Convert query to vector

    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,  # Search using the query embedding
        top_k=top_k,  # Get the top-k most similar chunks
        include_metadata=True  # Include metadata (e.g., document title, source)
    )

    # Extract results and return formatted documents
    retrieved_chunks = []
    if "matches" in results and results["matches"]:  # Check if there are any results
        for match in results["matches"]:
            retrieved_chunks.append({
                "text": match["metadata"].get("text", "No text available"),  # Handle missing data
                "source": match["metadata"].get("source", "Unknown source")  # Handle missing metadata
            })
    else:
        print("⚠️ No relevant documents found in Pinecone.")

    return retrieved_chunks


# Test retrieval function
query = "What are the latest advancements in deep learning?"
results = retrieve_chunks(query, top_k=3)

for result in results:
    print(f"Chunk: {result['text']}")
    print(f"Source: {result['source']}")
    print("-" * 50)


def prepare_prompt(query: str, retrieved_chunks: list):
    """
    Prepare a prompt by formatting retrieved chunks.

    Args:
        query (str): User query.
        retrieved_chunks (list): List of retrieved document chunks.

    Returns:
        str: Formatted prompt to send to GPT.
    """
    if not retrieved_chunks:
        return f"Query: {query}\nAnswer: I'm sorry, but I couldn't find any relevant documents."

    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"
    return prompt


from langchain.llms import OpenAI

def answer_query(query: str, top_k: int = 3):
    """
    End-to-end RAG pipeline: Retrieve relevant chunks and generate an answer.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        str: Generated answer from GPT.
    """
    # Step 1: Retrieve relevant chunks from Pinecone
    retrieved_chunks = retrieve_chunks(query, top_k)

    # Step 2: Prepare the prompt
    prompt = prepare_prompt(query, retrieved_chunks)

    # Step 3: Generate answer using GPT-4o-mini
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
    return llm(prompt)


# Test full pipeline
query = "Summarize the latest research in natural language processing."
answer = answer_query(query, top_k=3)
print("Generated Answer:")
print(answer)
