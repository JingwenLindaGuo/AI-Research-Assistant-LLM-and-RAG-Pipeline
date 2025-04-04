from dataprep.data_prep import *

standard_filters = {
    "query": "Deep Learning",
    "author": "Ilya Sutskever",
    "exclude_words": "Reinforcement Learning",
    "end_date": "20171230",
    "max_results": 3,
    "sort_by": "relevance"
    }  

def test_query_formatting():
    query = format_query(standard_filters)
    expected = 'Deep Learning AND au:"Ilya Sutskever" AND NOT "Reinforcement Learning" AND submittedDate:[* TO 20171230]'
    assert query == expected

# Test format_query with minimal filters
def test_small_query_formatting():
    filters_min = {"query": "AI"}
    query = format_query(filters_min)
    assert query == "AI"

# Test format_query with all fields
def test_full_query_formatting():    
    filters_full = {
        "query": "transformer",
        "author": "Vaswani",
        "title": "Attention",
        "category": "cs.CL",
        "abstract": "NLP",
        "journal_ref": "NeurIPS",
        "doi": "10.1234/abcde",
        "exclude_words": "vision",
        "start_date": "20170101",
        "end_date": "20201231"
    }
    query = format_query(filters_full)
    assert 'transformer' in query
    assert 'au:"Vaswani"' in query
    assert 'ti:"Attention"' in query
    assert 'cat:"cs.CL"' in query
    assert 'abs:"NLP"' in query
    assert 'jr:"NeurIPS"' in query
    assert 'doi:"10.1234/abcde"' in query
    assert 'NOT "vision"' in query
    assert 'submittedDate:[20170101 TO 20201231]' in query

# Test fetch_papers to ensure that papers matching the query are fetched
def test_paper_fetching():
    papers = fetch_papers(standard_filters)

    for paper in papers:
        paper_title = paper.title
        paper_author_list = [author.name for author in paper.authors]
        paper_date = paper.published.strftime("%Y%m%d%H%M")

        assert any(standard_filters.get('author').lower() in a.lower() for a in paper_author_list)
        assert standard_filters.get('exclude_words') is None or standard_filters.get('exclude_words', '').lower() not in paper_title.lower()
        assert (
            standard_filters.get('start_date') is None or str(paper_date) >= str(standard_filters.get('start_date', ""))
        ) and (
            standard_filters.get('end_date') is None or str(paper_date) <= str(standard_filters.get('end_date', "999999999999"))
        )
    assert len(papers) == standard_filters.get('max_results')

# Test fetch_papers with invalid query to ensure empty result
def test_paper_fetching_no_results():
    bad_filters = {
        "query": "asdkfjasdlfj",  
        "max_results": 2,
        "sort_by": "relevance"
    }
    papers = fetch_papers(bad_filters)
    assert len(papers) == 0

def test_paper_chunking():
    chunk_size = 500

    papers = fetch_papers(standard_filters)
    paper_chunks = chunk_papers(papers)

    assert all(0 < len(chunk.page_content) <= chunk_size for chunk in paper_chunks)

# Test chunk_papers skips non-PDF papers
def test_chunk_papers_skips_non_pdf():
    class DummyPaper:
        title = "Fake Paper"
        pdf_url = None 

    chunks = chunk_papers([DummyPaper()])
    assert chunks == [] 