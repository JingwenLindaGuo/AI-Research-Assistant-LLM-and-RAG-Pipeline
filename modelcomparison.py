# %% [markdown]  我要用conda建立一个3.9或者3.10的env
# Setup Clearml

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

# %% [markdown]
# install MTEB

# %%
# from mteb import MTEB

# %% [markdown]
# Semantic Textual Similarity

# %%
from datasets import load_dataset
 
dataset = load_dataset("stsb_multi_mt", name="en", split="test")
 
print(dataset[0])


# %%
import random
# randomly sample 50 from the dataset
random.seed(42)   
small_benchmark = dataset.shuffle(seed=42).select(range(50))


# %%
# def evaluate_on_mteb(dataset, embed_function):
#     similarities = []
#     human_scores = []

#     for example in dataset:
#         sent1, sent2, score = example["sentence1"], example["sentence2"], example["similarity_score"]
        
#         emb1 = embed_function(sent1)
#         emb2 = embed_function(sent2)

#         model_similarity = cosine_sim(emb1, emb2)

#         similarities.append(model_similarity)
#         human_scores.append(score / 5.0)   

#     from scipy.stats import pearsonr
#     correlation, _ = pearsonr(similarities, human_scores)
    
#     return correlation
 
# hf_mini_correlation = evaluate_on_mteb(small_benchmark, hf_mini)
# hf_mpnet_correlation = evaluate_on_mteb(small_benchmark, hf_mpnet) 
# openai_small_correlation = evaluate_on_mteb(small_benchmark, openai_small)
# openai_large_correlation = evaluate_on_mteb(small_benchmark, openai_large)

# print(f"OpenAI Small Model Pearson Correlation: {openai_small_correlation:.4f}")
# print(f"OpenAI Large Model Pearson Correlation: {openai_large_correlation:.4f}")
# print(f"hf_mini_correlation Pearson Correlation: {hf_mini_correlation:.4f}")
# print(f"hf_mpnet_correlation Pearson Correlation: {hf_mpnet_correlation:.4f}")

# %%
openai_small_correlation = 0.8673
openai_large_correlation = 0.8668
hf_mini_correlation = 0.7993
hf_mpnet_correlation = 0.8423

# %%
from clearml import Task 
task = Task.init(project_name="Embedding model evaluation", task_name="Compare Embedding Models - Semantic Similarity2")
 
task.get_logger().report_scalar("Semantic Meaning Pearson Correlation", "OpenAI Small Model", openai_small_correlation, iteration=1)
task.get_logger().report_scalar("Semantic Meaning Pearson Correlation", "OpenAI Large Model", openai_large_correlation, iteration=1)
task.get_logger().report_scalar("Semantic Meaning Pearson Correlation", "HF mini Model", hf_mini_correlation, iteration=1) 
task.get_logger().report_scalar("Semantic Meaning Pearson Correlation", "HF mpnet Model", hf_mpnet_correlation, iteration=1)

task.close()


# %% [markdown]
# Retrieval Evaluation

# %%
import random 
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer 

# download MSMARCO dataset
dataset = "msmarco"
data_path = f"datasets/{dataset}"   

##uncomment to download the dataset for testing
# url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

# # download and unzip
# util.download_and_unzip(url, data_path)
 
data_loader = GenericDataLoader(data_path)
corpus, queries, qrels = data_loader.load(split="test")

# sample 50 instances only
num_samples = 50   
selected_query_ids = random.sample(list(queries.keys()), min(num_samples, len(queries)))

msmarco_subset = []
for query_id in selected_query_ids:
    query_text = queries[query_id]
    if query_id in qrels:  # only handle queries that have ground truth
        relevant_docs = [doc_id for doc_id in qrels[query_id]]  # ground truth document id
        candidate_doc_ids = relevant_docs + random.sample(list(corpus.keys()), 10)  # 10 random incorrect options
        candidate_docs = [corpus[doc_id]["text"] for doc_id in candidate_doc_ids]  
        msmarco_subset.append({
            "query": query_text,
            "relevant_docs": relevant_docs,
            "candidate_docs": candidate_docs
        })
 


# %%
def evaluate_retrieval(dataset, embed_function):
    retrieved_ranks = []
    for example in dataset:
        query = example["query"]
        relevant_docs = example["relevant_docs"]
        candidate_docs = example["candidate_docs"]

        query_embedding = embed_function(query)
        candidate_embeddings = [embed_function(doc) for doc in candidate_docs]

        similarities = [cosine_sim(query_embedding, emb) for emb in candidate_embeddings]
        ranked_docs = sorted(zip(candidate_docs, similarities), key=lambda x: x[1], reverse=True)

        ground_truth_rank = [i for i, (doc, _) in enumerate(ranked_docs) if doc in relevant_docs]
        if ground_truth_rank:
            retrieved_ranks.append(ground_truth_rank[0])

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
from clearml import Task 
import torch
task = Task.init(project_name="Embedding model evaluation", task_name="Compare Embedding Models - Retrieval3") 
task.execute_remotely(queue_name="default", exit_process=True)

task.connect({
    "dataset": "MSMARCO",
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

task.close()

print(f"OpenAI Small Model MRR: {openai_small_mrr:.4f}")
print(f"OpenAI Large Model MRR: {openai_large_mrr:.4f}")
print(f"OpenAI Small Model MRR: {hf_mini_mrr:.4f}")
print(f"OpenAI Large Model MRR: {hf_mpnet_mrr:.4f}")

 

# %%



