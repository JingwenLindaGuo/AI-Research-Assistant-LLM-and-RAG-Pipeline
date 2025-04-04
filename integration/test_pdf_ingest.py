from pdf_ingest import ingest_uploaded_pdf

pdf_path = "testPaper.pdf"
user_index = "uploadpdftest2"  # or whatever index you’ve created in Pinecone

num_chunks = ingest_uploaded_pdf(pdf_path, user_index)

print(f"✅ Stored {num_chunks} chunks.")

