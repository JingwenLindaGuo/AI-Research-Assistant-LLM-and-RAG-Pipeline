# %% [markdown]  我要用conda建立一个3.9或者3.10的env
# Setup Clearml 
import torch 

import os
from clearml import Dataset, Task
 
#comment out to dataset
cache_dir = "./datasets"  # 你可以改成更合适的路径
dataset_path =  cache_dir 
dataset_id="ea34bc9235bd4472b50a0a86d07b6e50"

if not os.path.exists(dataset_path):
    print("🔍 Local dataset not found, downloading...")
    dataset_path = Dataset.get(dataset_id=dataset_id).get_local_copy()
    os.makedirs(cache_dir, exist_ok=True)
    os.system(f'cp -r {dataset_path} {cache_dir}')  # 复制数据到缓存
else:
    print("✅ Using cached dataset")

data_path = os.path.join(dataset_path, "msmarco")  # 确保路径正确
#comment out

# %% [markdown]
# Load Embedding Model

# %%
from openai import OpenAI
from dotenv import load_dotenv
import os
 
load_dotenv()
 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openai_small(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding  

def openai_large(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding   


# %%
from sentence_transformers import SentenceTransformer
 
hf_mini_model = SentenceTransformer("all-MiniLM-L6-v2")   
hf_mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def hf_mini(text):
    return hf_mini_model.encode(text).tolist()

def hf_mpnet(text):
    return hf_mpnet_model.encode(text).tolist()

# %% [markdown]
# Define cosine similarity

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]
 

# %%
# from datasets import load_dataset
 
# dataset = load_dataset("stsb_multi_mt", name="en", split="test")
 
# print(dataset[0])

 
 

# %%
import random 
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer 

# download MSMARCO dataset
from clearml import Dataset

##uncomment to download the dataset for testing
# url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

# # download and unzip
# util.download_and_unzip(url, data_path) 

data_loader = GenericDataLoader(data_path)
corpus, queries, qrels = data_loader.load(split="test")

# sample 50 instances only
num_samples = 5
selected_query_ids = random.sample(list(queries.keys()), min(num_samples, len(queries)))
print('Number of samples:', min(num_samples, len(queries)))

msmarco_subset = []
for query_id in selected_query_ids:
    query_text = queries[query_id]
    if query_id in qrels:  # 只处理有 ground truth 的 queries
        relevant_doc = random.choice(list(qrels[query_id]))  # 选择一个 ground truth 文档
        non_relevant_docs = random.sample(
            [doc_id for doc_id in corpus.keys() if doc_id not in qrels[query_id]], 4  # 确保选择非相关文档
        )

        candidate_doc_ids = [relevant_doc] + non_relevant_docs  # 1 个 relevant + 4 个 non-relevant
        candidate_docs = [corpus[doc_id]["text"] for doc_id in candidate_doc_ids]

        msmarco_subset.append({
            "query": query_text,
            "relevant_doc": corpus[relevant_doc]["text"],  # 直接存文本
            "candidate_docs": candidate_docs
        })

# 打印 query、relevant doc、candidate docs 的前 100 个字符
for i, sample in enumerate(msmarco_subset):
    print(f"Sample {i+1}:")
    print(f"- Query: {sample['query'][:100]}...")  # 取 query 前 100 个字符
    print(f"- Relevant doc: {sample['relevant_doc'][:100]}...")  # 取 relevant doc 前 100 个字符
    print(f"- Number of candidate docs: {len(sample['candidate_docs'])}")

    for j, doc in enumerate(sample["candidate_docs"]):
        print(f"  - Candidate doc {j+1}: {doc[:100]}...")  # 取每个候选文档前 100 个字符
    print()


# %%
def evaluate_retrieval(dataset, embed_function):
    retrieved_ranks = []
    
    for example in dataset:
        query = example["query"]
        relevant_docs = set(example["relevant_docs"])  # 转换为集合，加快查询
        candidate_docs = example["candidate_docs"]

        print(f"- Query: {example['query'][:100]}...")  # 取 query 前 100 个字符
        print(f"- Relevant doc: {example['relevant_doc'][:100]}...")  # 取 relevant doc 前 100 个字符
        print(f"- Number of candidate docs: {len(example['candidate_docs'])}")

        for j, doc in enumerate(example["candidate_docs"]):
            print(f"  - Candidate doc {j+1}: {doc[:100]}...")  # 取每个候选文档前 100 个字符
        print()

        if not candidate_docs:
            print(f"⚠️ Query '{query}' has no candidate documents!")
            continue  # 跳过这个样本，避免空列表计算相似度

        query_embedding = embed_function(query)
        candidate_embeddings = [embed_function(doc) for doc in candidate_docs]

        similarities = [cosine_sim(query_embedding, emb) for emb in candidate_embeddings]
        ranked_docs = sorted(zip(candidate_docs, similarities), key=lambda x: x[1], reverse=True)

        ground_truth_rank = [i for i, (doc, _) in enumerate(ranked_docs) if doc in relevant_docs]
        
        if not ground_truth_rank:
            print(f"⚠️ No relevant docs found for query: {query}")
            print(f"Candidate docs: {candidate_docs}")
            print(f"Relevant docs: {relevant_docs}")
        else:
            retrieved_ranks.append(ground_truth_rank[0])

    if not retrieved_ranks:
        print("⚠️ Warning: No relevant documents found for any queries!")
        return float('nan')  # 返回 NaN 代替 0，避免误导性结果

    mrr = sum(1.0 / (rank + 1) for rank in retrieved_ranks) / len(retrieved_ranks)
    return mrr


# # comment
# openai_small_mrr = evaluate_retrieval(msmarco_subset, openai_small)
# openai_large_mrr = evaluate_retrieval(msmarco_subset, openai_large)

# print(f"OpenAI Small Model MRR: {openai_small_mrr:.4f}")
# print(f"OpenAI Large Model MRR: {openai_large_mrr:.4f}")

# hf_mini_mrr = evaluate_retrieval(msmarco_subset, hf_mini)
# hf_mpnet_mrr = evaluate_retrieval(msmarco_subset, hf_mpnet)

# print(f"OpenAI Small Model MRR: {hf_mini_mrr:.4f}")
# print(f"OpenAI Large Model MRR: {hf_mpnet_mrr:.4f}")


# %%


# %%

# cache_dir = "./datasets"  # 你可以改成更合适的路径
# dataset_path =  cache_dir 
# dataset_id="ea34bc9235bd4472b50a0a86d07b6e50"

# if not os.path.exists(dataset_path):
#     print("🔍 Local dataset not found, downloading...")
#     dataset_path = Dataset.get(dataset_id=dataset_id).get_local_copy()
#     os.makedirs(cache_dir, exist_ok=True)
#     os.system(f'cp -r {dataset_path} {cache_dir}')  # 复制数据到缓存
# else:
#     print("✅ Using cached dataset")

# data_path = os.path.join(dataset_path, "msmarco")  # 确保路径正确

task = Task.init(project_name="Embedding model evaluation", task_name="Compare Embedding Models - Retrieval3")
task.execute_remotely(queue_name="default", exit_process=True)

task.connect({
    "dataset": msmarco_subset,
    "num_samples": len(msmarco_subset),
})

print("ClearML initialised")


if torch.cuda.is_available():
    print("✅ GPU is available! Using:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not detected, running on CPU")

print('Evaluating openAI models')
openai_small_mrr = evaluate_retrieval(msmarco_subset, openai_small)
openai_large_mrr = evaluate_retrieval(msmarco_subset, openai_large)
print('Evaluating hf models')
hf_mini_mrr = evaluate_retrieval(msmarco_subset, hf_mini)
hf_mpnet_mrr = evaluate_retrieval(msmarco_subset, hf_mpnet)

task.get_logger().report_scalar("Retrieval MRR", "OpenAI Small Model", openai_small_mrr, iteration=1)
task.get_logger().report_scalar("Retrieval MRR", "OpenAI Large Model", openai_large_mrr, iteration=1)
task.get_logger().report_scalar("Retrieval MRR", "HF mini Model", hf_mini_mrr, iteration=1)
task.get_logger().report_scalar("Retrieval MRR", "HF mpnet Model", hf_mpnet_mrr, iteration=1)

print(f"OpenAI Small Model MRR: {openai_small_mrr:.4f}")
print(f"OpenAI Large Model MRR: {openai_large_mrr:.4f}")
print(f"OpenAI Small Model MRR: {hf_mini_mrr:.4f}")
print(f"OpenAI Large Model MRR: {hf_mpnet_mrr:.4f}")

task.close()

# %%



