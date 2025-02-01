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
