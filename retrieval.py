import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="path_to_vector_store"  # Update with the path where embeddings are stored
))
collection = client.get_collection("research_papers")  # Ensure collection name matches what C created

def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the vector store.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        List[Dict]: Retrieved chunks with metadata.
    """
    # Embed the query
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Ensure compatibility with ChromaDB

    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results

query = "What are the latest advancements in deep learning?"
results = retrieve_chunks(query)
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"Document: {doc}")
    print(f"Source: {meta['source']}")

def prepare_prompt(query: str, chunks: list):
    """
    Prepare a prompt by concatenating retrieved chunks.
    """
    context = "\n\n".join([chunk for chunk in chunks])
    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"
    return prompt

from langchain.llms import OpenAI

def generate_answer(prompt: str):
    """
    Generate an answer using OpenAI GPT API.
    """
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
    return llm(prompt)

def answer_query(query: str, top_k: int = 3):
    """
    End-to-end pipeline for answering a user query.
    """
    results = retrieve_chunks(query, top_k)
    chunks = results["documents"]
    prompt = prepare_prompt(query, chunks)
    return generate_answer(prompt)
