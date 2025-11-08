import numpy as np
import pandas as pd
from openai import OpenAI
from utils import read_secret_key, load_faiss_index
from config import CONFIG


def compute_retrieval_score(distances, retrieved_clusters, cv_cluster):
    sims = 1 / (1 + np.array(distances))
    weights = np.array([1.0 if c == cv_cluster else 0.25 for c in retrieved_clusters])
    return float(np.mean(sims * weights))


def evaluate_all_clusters():
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    cv_path = "data/subset/selected_cvs.parquet"
    jobs_path = "data/subset/selected_job_descriptions.parquet"

    df_cvs = pd.read_parquet(cv_path)
    df_jobs = pd.read_parquet(jobs_path)
    index = load_faiss_index(CONFIG["faiss_index_path"])

    results = []
    k = 3

    for _, row in df_cvs.iterrows():
        cv_text = row["cv_standard"]
        cv_cluster = row["cluster_id"]

        resp = client.embeddings.create(model=CONFIG["embedding_model"], input=[cv_text])
        query_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

        distances, indices = index.search(query_vec, k)
        distances, indices = distances[0], indices[0]

        retrieved = df_jobs.iloc[indices]
        retrieved_clusters = retrieved["cluster_id"].tolist()
        score = compute_retrieval_score(distances, retrieved_clusters, cv_cluster)

        same_cluster_ratio = sum(1 for c in retrieved_clusters if c == cv_cluster) / k

        results.append({
            "cluster_id": cv_cluster,
            "same_cluster_ratio": same_cluster_ratio,
            "retrieval_score": score
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv("data/evaluation_results_embed.csv", index=False)

    summary = df_res[["same_cluster_ratio", "retrieval_score"]].describe().round(3)
    print("\n=== Evaluation Summary ===")
    print(summary)
    print("\nSaved evaluation_results.csv")



def test_single_job_retrieval(_):
    pass  # conserved for compatibility with main.py
