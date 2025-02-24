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
    "title": "<string or empty>",
    "author": "<string or empty>",
    "abstract_keywords": "<string or empty>",
    "comment": "<string or empty>",
    "journal_reference": "<string or empty>",
    "category": "<string or empty>",
    "release_date": "<string or empty>",
    "keywords_to_be_excluded": "<string or empty>"
  }
}

If the user wants to DISCUSS:
{
  "action": "discuss",
  "question": "<the user question or statement>"
}

Rules:
- Only output valid JSON according to the schema.
- Do not include any additional keys or text outside the JSON.
- For "keywords", if none are specified, output an empty list [].
"""
