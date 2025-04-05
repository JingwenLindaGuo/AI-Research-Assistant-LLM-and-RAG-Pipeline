from openai import OpenAI
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Setup clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define system message
SYSTEM_PROMPT = SYSTEM_PROMPT = SYSTEM_PROMPT = """\
You are a helpful assistant that extracts structured search filters from user queries related to academic papers.

Your job is to convert natural language input into a valid JSON object using the following schema:

{
  "filters": {
    "query": "<Main search topic or null>",
    "author": "<Author name or null>",
    "title": "<Title keyword or null>",
    "category": "<arXiv subject code or null>",
    "abstract": "<Keyword in abstract or null>",
    "journal_ref": "<Journal name or null>",
    "doi": "<DOI or null>",
    "exclude_words": "<Words to exclude or null>",
    "start_date": "<YYYYMMDD or null>",
    "end_date": "<YYYYMMDD or null>",
    "max_results": "<Max number of papers to return, or '1'>",
    "sort_by": "<'relevance' or 'newest'>"
  },
  "question": "<The user's request restated clearly>"
}

## INSTRUCTIONS (apply carefully for each field):

- query: The **main topic** or subject the user is searching for. This should be broad (e.g., "deep learning").
- author: If the user names a person, extract it exactly. Example: "by Ilya Sutskever" → `"author": "Ilya Sutskever"`.
- title: If specific words must be in the title (e.g. "with 'attention' in the title"), extract just those keywords.
- category: Convert category names (e.g. "computation and language") to **arXiv subject codes** like `"cs.CL"`.
- abstract: If the user says a topic must be discussed in the abstract, include the keyword (e.g. "NLP").
- journal_ref: If the user mentions a publication venue (e.g., NeurIPS, Nature), extract it exactly.
- doi: If a DOI is mentioned (e.g., "10.1234/abcde"), include it exactly as given. Never guess.
- exclude_words: For negations like “not about X”, “excluding Y”, or “not on Z”, extract just those keywords.
- start_date / end_date: Convert any date-based constraints into `YYYYMMDD` format. 
    - "before 2020" → `"end_date": "20191231"`
    - "after 2015" → `"start_date": "20160101"`
    - "from 2015 to 2017" → `"start_date": "20150101", "end_date": "20171231"`
- max_results: Extract numerical quantities from phrases like "3 papers", "at most 5", etc. Default to "1" if not specified.
- sort_by: If the user requests "most recent" or "latest", set `"sort_by": "newest"`. Otherwise, default to `"relevance"`.

## Formatting rules:
- Return only valid JSON matching the schema.
- All field values must be strings or `null`.
- Do not add any keys outside of the schema.
- Do not explain or comment — return the JSON object only.
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
 