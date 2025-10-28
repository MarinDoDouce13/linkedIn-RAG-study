# evaluate_retrieval.py
import numpy as np
import pandas as pd
from openai import OpenAI
from utils import read_secret_key, load_faiss_index
from config import CONFIG

def test_single_job_retrieval(job_id_to_test):
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    df = pd.read_csv("sample_df_with_cv.csv", sep=",", low_memory=False)
    df_meta = pd.read_parquet("data/sqlite/job_offers.parquet")
    index = load_faiss_index(CONFIG["faiss_index_path"])

    df["job_id"] = df["job_id"].astype(str)
    df_meta["job_id"] = df_meta["job_id"].astype(str)
    job_id_to_test = str(job_id_to_test)

    row = df[df["job_id"] == job_id_to_test]
    if row.empty:
        raise ValueError(f"Job ID {job_id_to_test} not found.")

    cv_text = row["generated_cv"].values[0]
    if not isinstance(cv_text, str) or not cv_text.strip():
        raise ValueError(f"No CV text found for job {job_id_to_test}.")

    resp = client.embeddings.create(model=CONFIG["embedding_model"], input=[cv_text])
    query_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    k = 10
    distances, indices = index.search(query_vec, k)
    retrieved = df_meta.iloc[indices[0]][["job_id", "title_translated"]].copy()
    retrieved["distance"] = distances[0]

    retrieved_ids = retrieved["job_id"].tolist()

    if job_id_to_test in retrieved_ids:
        rank = retrieved_ids.index(job_id_to_test) + 1
        sim_score = 1 / (1 + retrieved.loc[retrieved["job_id"] == job_id_to_test, "distance"].values[0])
        print(f"Job {job_id_to_test} retrieved at rank {rank}/{k} (similarity score: {sim_score:.4f})")
    else:
        print(f"Job {job_id_to_test} not found in top {k} results.")

    print("\nTop retrieved job IDs and distances:")
    for i, row in retrieved.iterrows():
        print(f"{i+1}. {row['job_id']} | dist={row['distance']:.4f} | title={row['title_translated']}")
