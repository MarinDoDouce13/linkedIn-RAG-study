import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from config import CONFIG
from utils import read_secret_key, save_faiss_index


def build_embeddings():
    clusters_path = Path("data/cluster_reps_checkpoint_final_20251106_163826.parquet")
    jobs_path = Path("data/sampled_engineers_with_clusters_20251105_175242.parquet")

    df_clusters = pd.read_parquet(clusters_path)
    df_jobs = pd.read_parquet(jobs_path)

    selected_clusters = [
        86, 265, 46, 179, 146, 127, 230, 120, 10, 92,
        150, 281, 34, 287, 224, 79, 47, 6, 234, 247,
        126, 241, 285, 103, 77, 276, 279, 111, 62, 205
    ]
    print(f"Selected clusters: {selected_clusters}")

    cv_subset = df_clusters[df_clusters["cluster_id"].isin(selected_clusters)][
        ["cluster_id", "cv_standard"]
    ]
    jobs_subset = df_jobs[df_jobs["cluster_label"].isin(selected_clusters)][
        ["cluster_label", "description"]
    ].rename(columns={"cluster_label": "cluster_id"})

    output_dir = Path("data/subset")
    output_dir.mkdir(exist_ok=True)

    cv_subset.to_parquet(output_dir / "selected_cvs.parquet", index=False)
    jobs_subset.to_parquet(output_dir / "selected_job_descriptions.parquet", index=False)

    print(f"{len(cv_subset)} CVs saved.")
    print(f"{len(jobs_subset)} job descriptions saved.")

    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    texts = jobs_subset["description"].fillna("").tolist()
    vectors = []

    for i in range(0, len(texts), CONFIG["embedding_batch_size"]):
        batch = texts[i : i + CONFIG["embedding_batch_size"]]
        resp = client.embeddings.create(model=CONFIG["embedding_model"], input=batch)
        batch_vecs = [r.embedding for r in resp.data]
        vectors.extend(batch_vecs)

    embeddings = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_faiss_index(index, CONFIG["faiss_index_path"])

    jobs_subset.to_parquet("data/sqlite/job_offers.parquet", index=False)
    print(f"Indexed {len(texts)} job descriptions into FAISS.")
