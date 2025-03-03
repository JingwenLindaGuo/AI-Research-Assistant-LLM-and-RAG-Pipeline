import arxiv
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fetch_papers(query="LLM", max_results=5, 
                 start_date=None, end_date=None, 
                 author=None, title=None, 
                 category=None, exclude_words=None, 
                 abstract=None, journal_ref=None, 
                 doi=None, sort_by="relevance"):

    # Build search query with optional filters
    search_query = query
    if author:
        search_query += f' au:"{author}"'
    if title:
        search_query += f' ti:"{title}"'
    if category:
        search_query += f' cat:"{category}"'  
    if abstract:
        search_query += f' abs:"{abstract}"' 
    if journal_ref:
        search_query += f' jr:"{journal_ref}"' 
    if doi:
        search_query += f' doi:"{doi}"' 
    if start_date or end_date:
        date_filter = f"submittedDate:[{start_date or '*'} TO {end_date or '*'}]"
        search_query += f" {date_filter}"  
    if exclude_words:
        search_query += f' NOT "{exclude_words}"' 

    # Initialize ArXiv client
    client = arxiv.Client()

    # Create ArXiv search object
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
    )

    all_chunks = []  # To store extracted text chunks

    # Iterate through the retrieved research papers
    for result in client.results(search):
        print(f"Processing: {result.title} (Published: {result.published.date()})")

        # Create a temporary file to store each downloaded PDF
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            try:
                # Download the paper and store it temporarily
                result.download_pdf(filename=temp_pdf.name)
                print(f"-> Downloaded: {result.title}")

                # Load the PDF file and combine text from all pages into a single string
                loader = PyPDFLoader(temp_pdf.name)
                paper = loader.load()
                full_text = " ".join(p.page_content for p in paper)

                # Extract text chunks and add to the main list
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                paper_chunks = text_splitter.create_documents([full_text])
                all_chunks.extend(paper_chunks)

            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred while downloading or processing PDF:", e)
                time.sleep(5)

    print(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks