import os
#from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings # supposedly the new,correct import
from pinecone import Pinecone
# from integration.Embedder import embed # for script
from Embedder import embed # for notebook
from langchain_community.chat_models import ChatOpenAI # supposedly the new,correct import
#from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI    
from time import sleep
# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Memory buffers to store past interactions and retrieved documents
conversation_history = []  # Stores the last few queries and responses
document_memory = []  # Stores relevant document chunks for future context

def parse_date(date_str: str) -> int:
    """
    turn YYYYMMDD or YYYYMMDDHHmm string into integer 
    """
    return int(date_str)


def get_embedding_count(index_obj, namespace="default"):
    """
    Returns the number of embeddings stored in the given namespace of the Pinecone index.
    
    参数:
    - index_obj: Pinecone Index object
    - namespace: 指定的 namespace，默认为 "default"
    
    返回:
    - int: 指定 namespace 中存储的 embedding 数量
    """
    stats = index_obj.describe_index_stats()
    # 获取指定 namespace 的统计数据
    namespace_stats = stats.get("namespaces", {}).get(namespace, {})
    return namespace_stats.get("vector_count", 0)


def retrieve_chunks(classification, index_obj, top_k: int = 100):
    tries = 0
    query = classification['question']
    query_response = embed([{"text": query}])
    query_embedding = query_response[0]["values"]

    # filter_condition = {}

    # # 2.1 如果paper_author_list不是null，添加精确匹配
    # author_filter = classification["filters"].get("paper_author_list")
    # if author_filter and author_filter != "null":
    #     # 假设metadata中存储的key是"author"，且存储形式为列表
    #     # 如果传入的author_filter是列表，则使用$in操作符进行匹配
    #     if isinstance(author_filter, list):
    #         filter_condition["author"] = {"$in": author_filter}
    #     else:
    #         filter_condition["author"] = author_filter

    # # 2.2 如果paper_title不是null，添加精确匹配
    # title_filter = classification["filters"].get("paper_title")
    # if title_filter and title_filter.lower() != "null":
    #     # 假设metadata中存储的key是"title"
    #     filter_condition["title"] = title_filter

    # # 2.3 如果start_date和end_date不是null，则进行日期区间过滤
    # start_date = classification["filters"].get("start_date")
    # end_date = classification["filters"].get("end_date")

    # # 确保metadata中存储的"date"是通过paper_date = result.published.strftime("%Y%m%d%H%M")转换成的数值格式
    # date_filter = {}
    # if start_date and start_date.lower() != "null":
    #     date_filter["$gte"] = parse_date(start_date)
    # if end_date and end_date.lower() != "null":
    #     date_filter["$lte"] = parse_date(end_date)

    # if date_filter:
    #     # 假设metadata中存储的key是"date"
    #     filter_condition["date"] = date_filter


    results = index_obj.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
        # filter=filter_condition if filter_condition else None
    )

    print("Number of embeddings in pinecone now: ", get_embedding_count(index_obj))
    
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            metadata = match["metadata"]
            chunk = {
                "text": metadata.get("text", "No text available"),
                "title": metadata.get("title", "Unknown title"),
                "author": metadata.get("author", "Unknown author"),
                "date": metadata.get("date", "Unknown date"),
                "source": metadata.get("source", "Unknown source")
            }
            retrieved_chunks.append(chunk)
            document_memory.append(chunk)

    
    else:
        print("⚠️ No relevant documents found in Pinecone, wait for 5 seconds and trying again...") 
        sleep(5)
        
        if tries == 0:
            tries +=1
            results = index_obj.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default",
            filter=filter_condition if filter_condition else None
        )

        retrieved_chunks = []
        if "matches" in results and results["matches"]:
            for match in results["matches"]:
                metadata = match["metadata"]
                chunk = {
                    "text": metadata.get("text", "No text available"),
                    "title": metadata.get("title", "Unknown title"),
                    "author": metadata.get("author", "Unknown author"),
                    "date": metadata.get("date", "Unknown date"),
                    "source": metadata.get("source", "Unknown source")
                }
                retrieved_chunks.append(chunk)
                document_memory.append(chunk)
                print("After retrying, relevant documents are found")
        else:
            print("⚠️ Still! No relevant documents found in Pinecone.") 

    print("Retrieved chunks: ")  
    print(retrieve_chunks)
    return retrieved_chunks

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def truncate_prompt(conversation_part: str, document_part: str, query: str, max_tokens: int = 128000) -> str:
    """
    Constructs a prompt from conversation history, document content, and query.
    
    If the conversation history plus query exceed max_tokens,
    the conversation history will be dropped, preserving only the query.
    
    If the document part exceeds the allowed token count, the last part of the document is preserved.
    """
    # Construct the non-document part: conversation history + "Query: {query}\nAnswer:"
    non_doc_text = conversation_part + f"Query: {query}\nAnswer:"
    non_doc_tokens = tokenizer.encode(non_doc_text)
    
    # If non-document part exceeds max_tokens, drop conversation history and use only the query prompt.
    query_only_text = f"Query: {query}\nAnswer:"
    query_only_tokens = tokenizer.encode(query_only_text)
    if len(non_doc_tokens) > max_tokens:
        print("Warning: Conversation and query part exceed max tokens! Dropping conversation history.")
        non_doc_text = query_only_text
        non_doc_tokens = query_only_tokens

    non_doc_count = len(non_doc_tokens)
    
    # Calculate the allowed tokens for the document part.
    allowed_doc_tokens = max_tokens - non_doc_count

    # Encode the document part tokens.
    doc_tokens = tokenizer.encode(document_part)
    doc_count = len(doc_tokens)
    
    if allowed_doc_tokens < 0:
        # This should not happen, but if it does, return a truncated version of the non-document part.
        print("Warning: Non-document part exceeds max tokens even after dropping conversation!")
        truncated_non_doc = tokenizer.decode(non_doc_tokens[:max_tokens])
        return truncated_non_doc
    else:
        if doc_count > allowed_doc_tokens:
            print('Warning: Document part exceeds max token, document length: ', doc_count, ', allowed document tokens: ', allowed_doc_tokens, 'truncating document part')
            # Instead of keeping the first tokens, keep the last allowed_doc_tokens tokens from the document.
            truncated_doc_tokens = doc_tokens[-allowed_doc_tokens:]
            truncated_doc = tokenizer.decode(truncated_doc_tokens)
        else:
            truncated_doc = document_part

    # Combine the non-document part with the (possibly truncated) document part.
    prompt = non_doc_text + "\n" + truncated_doc + "\n"
    return prompt



def prepare_prompt(classification, retrieved_chunks: list):
    # Retrieve conversation history
    history_text = "\n".join(conversation_history[-10:])
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    query = classification['question']

    memory_text = "\n\n".join([
        f"Paper Title: {chunk.get('title', 'Unknown title')}\n"
        f"Author: {chunk.get('author', 'Unknown author')}\n"
        f"Date: {chunk.get('date', 'Unknown date')}\n"
        f"Text Chunk: {chunk.get('text', 'No text available')}"
        for chunk in retrieved_chunks
    ])
    document_part = f"Documents Provided:\n{memory_text}\n\n" if memory_text else ""
    
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:"
    )
    print("\nPrompt passed to LLM:")
    print(prompt)
    truncate_prompt(conversation_part, document_part, query)

    tokens = tokenizer.encode(prompt) 
    
    return prompt

def answer_query(classification, index_obj, top_k: int = 100, use_memory: bool = False):
    query = classification["question"]
    # if use_memory:
    #     # Combine previously stored document memory with fresh retrieval
    #     new_retrieval = retrieve_chunks(query, index_obj, top_k)
    #     retrieved_chunks = document_memory + new_retrieval
    # else:
    retrieved_chunks = retrieve_chunks(classification, index_obj, top_k)
    
    prompt = prepare_prompt(classification, retrieved_chunks) 
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature = 0.7)
    answer = llm.invoke(prompt)
    conversation_history.append(f"Query: {query}\nAnswer: {answer}")
    return answer

def clear_memory():
    conversation_history.clear()
    document_memory.clear()
    print("Memory cleared.")
