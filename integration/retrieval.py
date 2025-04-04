import os
#from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings # supposedly the new,correct import
from pinecone import Pinecone
# from integration.Embedder import embed # for script
from .Embedder import embed # for notebook
from langchain_community.chat_models import ChatOpenAI # supposedly the new,correct import
#from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI    

# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Memory buffers to store past interactions and retrieved documents
conversation_history = []  # Stores the last few queries and responses
document_memory = []  # Stores relevant document chunks for future context

def retrieve_chunks(query: str, index_obj, top_k: int = 100):
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
        print("⚠️ No relevant documents found in Pinecone, wait for 5 seconds and trying again...") 
        sleep(5)
        if tries == 0:
            tries +=1
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
                print("⚠️ Still! No relevant documents found in Pinecone.")
    return retrieved_chunks

def prepare_prompt(query: str, retrieved_chunks: list):
    # Retrieve conversation history
    history_text = "\n".join(conversation_history[-10:])
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    
    # Optionally filter by paper title if mentioned in the query
    paper_title = None
    # if "typiclust" in query.lower():
    #     paper_title = "TypiClust"  # or extract a more precise title if needed

    if paper_title:
        filtered_chunks = [chunk for chunk in document_memory if paper_title.lower() in chunk.get("paper_title", "").lower()]
    else:
        filtered_chunks = document_memory

    memory_text = "\n\n".join([chunk["text"] for chunk in filtered_chunks])
    document_part = f"Relevant documents:\n{memory_text}\n\n" if memory_text else ""
    
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:"
    )
    return prompt

def answer_query(query: str, index_obj, top_k: int = 100, use_memory: bool = False):
    if use_memory:
        # Combine previously stored document memory with fresh retrieval
        new_retrieval = retrieve_chunks(query, index_obj, top_k)
        retrieved_chunks = document_memory + new_retrieval
    else:
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
