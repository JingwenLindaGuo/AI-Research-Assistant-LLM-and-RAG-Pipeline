from langchain_community.document_loaders import PyPDFLoader
from data_utils import chunk_papers
from Database import save_embeddings_to_index
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=500, overlap=100):
    """
    Accepts LangChain Documents (e.g., from PyPDFLoader),
    and returns text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks

def ingest_uploaded_pdf(file_path: str, user_index: str):
    """
    Loads, chunks, and stores an uploaded PDF.
    """
    print(f"Loading uploaded PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print("Chunking uploaded document...")
    chunks = chunk_documents(documents)  # reuse the same chunking logic

    print(f"Storing {len(chunks)} chunks to index: {user_index}")
    for chunk in chunks:
        save_embeddings_to_index(index_name=user_index, text=[chunk.page_content])
    
    return len(chunks)
