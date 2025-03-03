import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from Embedder import embed
from langchain_openai import ChatOpenAI

# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the Pinecone index name and connect to it
index_name = "test"
index = pc.Index(index_name)  

# Memory buffers to store past interactions and retrieved documents
conversation_history = []  # Stores the last few queries and responses
document_memory = []  # Stores relevant document chunks for future context

def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the Pinecone vector database.
    Append results to document_memory for future use.
    
    Args:
        query (str): The user's query.
        top_k (int): Number of relevant documents to retrieve.

    Returns:
        list: A list of retrieved document chunks with metadata.
    """
    # Generate embedding for the query using the embedding model
    query_response = embed([{"text": query}])  
    query_embedding = query_response.data[0].embedding  

    # Perform similarity search in the Pinecone vector database
    results = index.query(
        vector=query_embedding,  
        top_k=top_k,  
        include_metadata=True,
        namespace="test"
    )

    # Extract and store the results
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            chunk = {
                "text": match["metadata"].get("text", "No text available"),  # Retrieve document text
                "source": match["metadata"].get("source", "Unknown source")  # Retrieve document source
            }
            retrieved_chunks.append(chunk)
            document_memory.append(chunk)  # Append to document memory for context persistence
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

def answer_query(query: str, top_k: int = 3):
    """
    End-to-end RAG pipeline with memory for conversation and documents.
    
    Args:
        query (str): The user's query.
        top_k (int): Number of relevant document chunks to retrieve.

    Returns:
        str: The generated answer from the LLM.
    """
    # Step 1: Retrieve relevant chunks from the vector database
    retrieved_chunks = retrieve_chunks(query, top_k)

    # Step 2: Prepare the input prompt with memory of past interactions and documents
    prompt = prepare_prompt(query, retrieved_chunks)

    # Step 3: Generate an answer using the GPT-4o-mini model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    answer = llm.invoke(prompt)

    # Store the conversation in memory for future context
    conversation_history.append(f"Query: {query}\nAnswer: {answer}")

    return answer

# Interactive loop for dynamic user queries
while True:
    query = input("Enter your query: ")
    
    # Allow user to exit the application gracefully
    if query.lower() in ["exit", "quit"]:
        break
    
    # Get the generated answer from the RAG pipeline
    answer = answer_query(query, top_k=3)
    
    # Output the generated answer to the user
    print("Generated Answer:")
    print(answer)
