from Search_or_Discuss import classify_and_extract
from data_utils import fetch_papers, verify_results, chunk_papers

user_query = input("Enter your query: ")
classification = classify_and_extract(user_query)

if classification["action"] == "search":
    filters = classification['filters']
    print("\nRunning Search with filters:", classification['filters'])

    # Step 1: Fetch papers
    papers = fetch_papers(filters)

    # Step 2: Verify & Filter papers
    filtered_papers = verify_results(papers, filters)

    # Step 3: Chunk papers
    paper_chunks = chunk_papers(filtered_papers)

    # Print sample chunks
    for i, chunk in enumerate(paper_chunks[:3]):
        print(f"\nChunk {i+1}:\n{chunk.page_content[:300]}...")

elif classification["action"] == "discuss":
    print("\nDiscussion Topic:", classification["question"])