from Embedding_and_Indexing.Search_or_Discuss import classify_and_extract
from Embedding_and_Indexing.Embedder import embed


def handleUserQuery(user_input):
    
    processed_input = classify_and_extract(user_input)

    if processed_input.get("action") == "search":

        filters = processed_input.get("filters",{})
        question = processed_input.get("question","")

        print("Performing search with the following filters:")
        print(filters)
        print("User's question about the search results:", question)

    elif processed_input.get("action") == "discuss":

        question = processed_input.get("question","")

        print("Discussing:")
        print(question)
    
    else:
        # Handle unknown action
        print("Unknown action:", processed_input.get("action"))
        return {"status": "error", "message": "Invalid action"}

input = "get me the latest papers on machine learning after the year 2023, without having image generation"

handleUserQuery(input)