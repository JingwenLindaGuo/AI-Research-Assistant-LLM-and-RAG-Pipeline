import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from Embedder import embed
from langchain_openai import ChatOpenAI

# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Memory buffers to store past interactions and retrieved documents
conversation_history = []  # Stores the last few queries and responses
document_memory = []  # Stores relevant document chunks for future context

def retrieve_chunks(query: str, index_obj, top_k: int = 3):
    query_response = embed([{"text": query}])
    query_embedding = query_response[0]["values"]
    results = index_obj.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    )
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            chunk = {
                "text": match["metadata"].get("text", "No text available"),
                "source": match["metadata"].get("source", "Unknown source")
            }
            retrieved_chunks.append(chunk)
            document_memory.append(chunk)
    else:
        print("⚠️ No relevant documents found in Pinecone.")
    return retrieved_chunks




def prepare_prompt(query: str, retrieved_chunks: list):
    """
    Prepare a prompt by including both past conversation history and document memory.
    
    Args:
        query (str): The current user query.
        retrieved_chunks (list): The most relevant document chunks retrieved.

    Returns:
        str: The formatted prompt to pass to the LLM.
    """
    # Combine recent conversation history (last 5 exchanges)
    history_text = "\n".join(conversation_history[-5:])  # Keep the last 5 messages
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""

    # Combine relevant document memory (last 5 retrieved documents)
    memory_text = "\n\n".join([chunk["text"] for chunk in document_memory[-5:]])  # Keep last 5 documents
    document_part = f"Relevant documents:\n{memory_text}\n\n" if memory_text else ""

    # Construct the final prompt with context and the new query
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:"
    )

    return prompt

def answer_query(query: str, index_obj, top_k: int = 3):
    # Use index_obj for the query
    retrieved_chunks = retrieve_chunks(query, index_obj, top_k)
    prompt = prepare_prompt(query, retrieved_chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    answer = llm.invoke(prompt)
    conversation_history.append(f"Query: {query}\nAnswer: {answer}")
    return answer

def clear_memory():
    conversation_history.clear()
    document_memory.clear()
    print("Memory cleared.")