from openai import OpenAI
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Setup clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define system message
SYSTEM_PROMPT = """\
You are a helpful assistant for academic research. 
Your job is to determine whether the user is requesting to:
1) Search for new research papers, or 
2) Discuss or analyze already-retrieved papers.

Output your answer in valid JSON, using the following schema:

If the user wants to SEARCH:
{
  "action": "search",
  "filters": {
    "query": "<Main search topic>",
    "author": "<Author name or null>",
    "title": "<Title keyword or null>",
    "category": "<Research category code or null>",
    "abstract": "<Keyword in abstract or null>",
    "journal_ref": "<Journal name or null>",
    "doi": "<Specific paper DOI or null>",
    "exclude_words": "<Words to exclude or null>",
    "start_date": "<YYYYMMDD or null>",
    "end_date": "<YYYYMMDD or null>",
    "max_results": "<Maximum results or 10>"
    "sort_by": "<'relevance' or 'newest'>"
  },
  "question": "<The user's request for retrieved data>"
}

If the user wants to DISCUSS:
{
  "action": "discuss",
  "question": "<the user question or statement>"
}

Rules:
- Only output valid JSON according to the schema.
- Do not include any additional keys or text outside the JSON.
"""
def classify_and_extract(user_input):
    try:
        # Send prompt to Chatgpt-4o-mini
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0.0 
        )

        # Get the reponse
        assistant_message = response.choices[0].message.content

        # Load response as json
        structured_output = json.loads(assistant_message)
        return structured_output

    except json.JSONDecodeError:
        # If the model didn't return valid JSON
        return {
            "action": "error",
            "error": "Invalid JSON returned by GPT-4o",
            "raw_response": assistant_message
        }