import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from Embedder import embed
from langchain_openai import ChatOpenAI


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "test"
index = pc.Index(index_name)  

def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the Pinecone vector database.
    """
    # Embed the query using Member C's embedder.py
    query_response = embed([{"text": query}])  # Call embed function
    query_embedding = query_response.data[0].embedding  # ✅ Correct way to extract embedding

    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,  
        top_k=top_k,  
        include_metadata=True,
        namespace="test"
    )

    # Extract results and return formatted documents
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            retrieved_chunks.append({
                "text": match["metadata"].get("text", "No text available"),
                "source": match["metadata"].get("source", "Unknown source")
            })
    else:
        print("⚠️ No relevant documents found in Pinecone.")

    return retrieved_chunks




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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # ✅ Correct
    return llm.invoke(prompt)  # ✅ Use `invoke()`, NOT `__call__`



# Test full pipeline
query = "Tell me about the history of Apple Inc."
answer = answer_query(query, top_k=3)
print("Generated Answer:")
print(answer)
