import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from Embedder import embed
from langchain_openai import ChatOpenAI


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "test"
index = pc.Index(index_name)  

# Memory buffer to store past interactions and retrieved documents
conversation_history = []
document_memory = []

def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the Pinecone vector database.
    Append results to document_memory for future use.
    """
    query_response = embed([{"text": query}])  
    query_embedding = query_response.data[0].embedding  

    results = index.query(
        vector=query_embedding,  
        top_k=top_k,  
        include_metadata=True,
        namespace="test"
    )

    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            chunk = {
                "text": match["metadata"].get("text", "No text available"),
                "source": match["metadata"].get("source", "Unknown source")
            }
            retrieved_chunks.append(chunk)
            document_memory.append(chunk)  # ✅ Append to document memory
    else:
        print("⚠️ No relevant documents found in Pinecone.")

    return retrieved_chunks


def prepare_prompt(query: str, retrieved_chunks: list):
    """
    Prepare a prompt by including both past conversation history and document memory.
    """
    # Combine conversation history
    history_text = "\n".join(conversation_history[-5:])  # Keep the last 5 messages
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""

    # Combine document memory
    memory_text = "\n\n".join([chunk["text"] for chunk in document_memory[-5:]])  # Keep last 5 documents
    document_part = f"Relevant documents:\n{memory_text}\n\n" if memory_text else ""

    # Create the final prompt
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:"
    )

    return prompt

def answer_query(query: str, top_k: int = 3):
    """
    End-to-end RAG pipeline with memory for conversation and documents.
    """
    # Retrieve relevant chunks and update document memory
    retrieved_chunks = retrieve_chunks(query, top_k)

    # Prepare the prompt with memory context
    prompt = prepare_prompt(query, retrieved_chunks)

    # Generate an answer using GPT-4o-mini
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    answer = llm.invoke(prompt)

    # Store the interaction in conversation memory
    conversation_history.append(f"Query: {query}\nAnswer: {answer}")

    return answer

while True:
    query = input("Enter your query: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = answer_query(query, top_k=3)
    print("Generated Answer:")
    print(answer)
