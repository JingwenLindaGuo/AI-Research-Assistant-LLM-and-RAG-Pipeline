import os
os.environ["PINECONE_API_KEY"] = "pcsk_594Xgb_Qtwvagd2kRcdaJtLSdnXXrnn2RN81sRkGZyJYCqQmiEJY1iw6rDdNATRvSmfkCZ"
print("Pinecone API Key:", os.getenv("PINECONE_API_KEY"))
from Search_or_Discuss import classify_and_extract
from data_utils import fetch_papers, verify_results, chunk_papers
from Database import create_index, save_embeddings_to_index, delete_index
from retrieval import answer_query, clear_memory
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# Create the index once (assuming it should persist during the conversation)
index_name = create_index()
index_obj = pc.Index(index_name)  

def handleUserQuery(user_input, user_index):
    # Use the provided user_input instead of the external variable
    classification = classify_and_extract(user_input)

    if classification["action"] == "search":
        filters = classification['filters']
        print("\nRunning Search with filters:", filters)

        # Step 1: Fetch papers
        papers = fetch_papers(filters)

        # Step 2: Verify & Filter papers
        filtered_papers = verify_results(papers, filters)

        # Step 3: Chunk papers
        paper_chunks = chunk_papers(filtered_papers)

        # Print sample chunks and save embeddings (update as needed)
        for i, chunk in enumerate(paper_chunks[:3]):
            print(f"\nChunk {i+1}:\n{chunk.page_content[:300]}...")
            save_embeddings_to_index(index_name=user_index, text=[chunk.page_content])
    
    elif classification["action"] == "discuss":
        print("\nDiscussion Topic:", classification["question"])
        # Call the RAG pipeline to generate a discussion answer
        answer = answer_query(user_input, index_obj, top_k=3)
        print("Generated Answer:")
        print(answer)
    else:
        # Default behavior if classification doesn't match expected actions:
        print("Action not recognized. Please try again.")

# Use a loop to enable continuous conversation
while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() in ["exit", "quit"]:
        break
    handleUserQuery(user_query, index_name)
delete_index(index_name=index_name)
clear_memory()