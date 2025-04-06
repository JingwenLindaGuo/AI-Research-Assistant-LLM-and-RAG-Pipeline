from openai import OpenAI
from dotenv import load_dotenv
import json
import os

"""
Description:
This script extracts academic research filters from user's query, using GPT-4o-mini model. 
It performs the following tasks: 
- Defines a SYSTEM_PROMPT that classify user queries as "search" or "other" and extract structured filters in JSON format for academic research.
- Implements the classify_and_extract() function to send the query to the model and parse the response into a structured JSON format following a predefined schema.
"""


load_dotenv()
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
  "question": "What is paper title, author and date? Summarise <Main search topic> from the paper"
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

## INSTRUCTIONS (apply carefully for each field):

- query: The *main topic* or subject the user is searching for. This should be broad (e.g., "deep learning").
- author: If the user names a person, extract it exactly. Example: "by Ilya Sutskever" → "author": "Ilya Sutskever".
- title: If specific words must be in the title (e.g. "with 'attention' in the title"), extract just those keywords.
- category: Convert category names (e.g. "computation and language") to *arXiv subject codes* like "cs.CL".
- abstract: If the user says a topic must be discussed in the abstract, include the keyword (e.g. "NLP").
- journal_ref: If the user mentions a publication venue (e.g., NeurIPS, Nature), extract it exactly.
- doi: If a DOI is mentioned (e.g., "10.1234/abcde"), include it exactly as given. Never guess.
- exclude_words: For negations like “not about X”, “excluding Y”, or “not on Z”, extract just those keywords.
- start_date / end_date: Convert any date-based constraints into YYYYMMDD format.
    - "before 2020" → "end_date": "20191231"
    - "after 2015" → "start_date": "20160101"
    - "from 2015 to 2017" → "start_date": "20150101", "end_date": "20171231"
- max_results: Extract numerical quantities from phrases like "3 papers", "at most 5", etc. Default to "1" if not specified.
- sort_by: If the user requests "most recent" or "latest", set "sort_by": "newest". Otherwise, default to "relevance".


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
 