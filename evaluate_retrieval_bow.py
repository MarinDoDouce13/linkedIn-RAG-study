import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_retrieval_score(similarities, retrieved_clusters, cv_cluster):
    weights = np.array([1.0 if c == cv_cluster else 0.25 for c in retrieved_clusters])
    return float(np.mean(similarities * weights))


def evaluate_all_clusters_bow():
    cv_path = "data/subset/selected_cvs.parquet"
    jobs_path = "data/subset/selected_job_descriptions.parquet"

    df_cvs = pd.read_parquet(cv_path)
    df_jobs = pd.read_parquet(jobs_path)

    print(f"Loaded {len(df_cvs)} CVs and {len(df_jobs)} job descriptions for BoW retrieval.")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        lowercase=True,
    )

    job_texts = df_jobs["description"].fillna("").tolist()
    job_matrix = vectorizer.fit_transform(job_texts)

    results = []
    k = 3

    for _, row in df_cvs.iterrows():
        cv_text = row["cv_standard"]
        cv_cluster = row["cluster_id"]

        cv_vec = vectorizer.transform([cv_text])
        sims = cosine_similarity(cv_vec, job_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:k]
        top_sims = sims[top_indices]
        retrieved_clusters = df_jobs.iloc[top_indices]["cluster_id"].tolist()

        score = compute_retrieval_score(top_sims, retrieved_clusters, cv_cluster)
        same_cluster_ratio = sum(1 for c in retrieved_clusters if c == cv_cluster) / k

        results.append({
            "cluster_id": cv_cluster,
            "same_cluster_ratio": same_cluster_ratio,
            "retrieval_score": score
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv("data/evaluation_results_bow.csv", index=False)

    summary = df_res[["same_cluster_ratio", "retrieval_score"]].describe().round(3)
    print("\n=== BoW Retrieval Summary ===")
    print(summary)
    print("\nSaved evaluation_results_bow.csv")

