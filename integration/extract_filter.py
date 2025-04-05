from openai import OpenAI
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Setup clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """\
You are a helpful assistant for academic research. Your task is to determine whether the user's request falls under one of two categories:

1. "search" actions: These are requests that ONLY require a search for research papers (with an optional summarisation of the retrieved results). For example:
   - "search for paper about typiclust that is released before 2000"
   - "tell me about papers about machine learning before 2020" (implying search and summarisation).

2. "other" actions: These are requests that involve additional tasks beyond a pure search and summary. For instance, if the user asks "search for typiclust and tell me about typiclust algorithm", this request should be classified as "other" since it involves discussing the algorithm in detail.

Based on the detected category, output your answer in valid JSON using the following schema:

For "search" actions:
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
    "max_results": "<Maximum results or 1>",
    "sort_by": "<'relevance' or 'newest'>"
  },
  "question": "summarise the paper(s) retrieved"
}

For "other" actions:
{
  "action": "other",
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
    "max_results": "<Maximum results or 1>",
    "sort_by": "<'relevance' or 'newest'>"
  },
  "question": "<The user's request for retrieved data>"
}

Guidelines:
- Clearly differentiate between requests that are pure search (with optional summary) and those that require further discussion or analysis.
- If the user's request contains terms such as "discuss", "explain", or requests for detailed explanations of the paper's contents (e.g., specific algorithm details), classify it as "other".
- Always output valid JSON without any additional text.
- Do not include any extra keys or comments outside the JSON.

Examples:
- Input: "search for paper about typiclust that is released before 2000" → Output should have "action": "search".
- Input: "tell me about papers about machine learning before 2020" → Output should have "action": "search" and the question should instruct to summarise the retrieved papers.
- Input: "search for typiclust and tell me about typiclust algorithm" → Output should have "action": "other".
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
 