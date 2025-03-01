import arxiv
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass

@dataclass
class PaperFilter:
    """ Stores filtering criteria for ArXiv search. """
    query: str = "LLM"
    author: str = None
    title: str = None
    category: str = None
    exclude_words: str = None
    abstract: str = None
    journal_ref: str = None
    doi: str = None
    sort_by: str = "relevance"
    max_results: int = 10

def format_query(filters: PaperFilter) -> str:
    """ Builds an ArXiv API query string from PaperFilter object. """

    search_parts = [filters.query]  # Base query

    if filters.author:
        search_parts.append(f'au:"{filters.author}"')  
    if filters.title:
        search_parts.append(f'ti:"{filters.title}"')  
    if filters.category:
        search_parts.append(f'cat:"{filters.category}"')  
    if filters.abstract:
        search_parts.append(f'abs:"{filters.abstract}"')  
    if filters.journal_ref:
        search_parts.append(f'jr:"{filters.journal_ref}"')  
    if filters.doi:
        search_parts.append(f'doi:"{filters.doi}"')  
    if filters.exclude_words:
        search_parts.append(f'NOT "{filters.exclude_words}"')  

    query_str = " AND ".join(search_parts)  # Combine with AND
    print(f"\nFormatted Query: {query_str}\n" + "-" * 100)

    return query_str

def fetch_papers(filters: PaperFilter):
    """ Fetches academic papers from ArXiv based on filters. """
    
    search_query = format_query(filters)  # Build query

    # Create ArXiv search object
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=filters.max_results,
        sort_by=arxiv.SortCriterion.Relevance if filters.sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))  # Convert generator to list
    return results

def verify_results(results, filters: PaperFilter):
    """ Verifies if fetched papers match the filters and prints details. """
    
    verified_results = []  # Store matching papers

    for result in results:
        paper_title = result.title
        paper_author_list = [author.name for author in result.authors]
        paper_abstract = result.summary
        paper_category = result.primary_category

        # Print paper details
        print(f"\nTitle: {paper_title}")
        print(f"- Author(s): {', '.join(paper_author_list)}")
        print(f"- Category: {paper_category}")
        print(f"- Abstract (first 200 chars): {paper_abstract[:200]}...")

        # Check filter matches
        match_results = {
            "Author Match": filters.author is None or any(filters.author.lower() in a.lower() for a in paper_author_list),
            "Title Match": filters.title is None or filters.title.lower() in paper_title.lower(),
            "Category Match": filters.category is None or filters.category in paper_category,
            "Abstract Match": filters.abstract is None or filters.abstract.lower() in paper_abstract.lower(),
            "Exclusion (NOT)": filters.exclude_words is None or filters.exclude_words.lower() not in paper_title.lower()
        }
        
        # Print filter match results
        print("\nFilter Matches:")
        for filter_name, is_match in match_results.items():
            print(f"- {filter_name}: {'✅' if is_match else '❌'}")

        print("-" * 100)

        # Add paper if it meets all criteria
        if all(match_results.values()):
            verified_results.append(result)

    print(f"\nFinal Papers Matching All Filters: {len(verified_results)}\n")
    return verified_results

def chunk_papers(results):
    """ Downloads, extracts, and chunks text from ArXiv PDFs. """
    
    all_chunks = []

    for result in results:

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            try:
                result.download_pdf(filename=temp_pdf.name)

                loader = PyPDFLoader(temp_pdf.name)
                papers = loader.load()

                full_text = " ".join(p.page_content for p in papers)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                paper_chunks = text_splitter.create_documents([full_text])

                all_chunks.extend(paper_chunks)
                print(f"Extracted {len(paper_chunks)} chunks from '{result.title}'")

            except (FileNotFoundError, ConnectionResetError) as e:
                print(f"Error: {e}")
                time.sleep(5)

    print(f"\nTotal Chunks Extracted: {len(all_chunks)}\n")
    return all_chunks