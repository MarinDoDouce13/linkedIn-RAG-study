# ingest.py
import os
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from config import CONFIG
from utils import read_secret_key, save_faiss_index

def build_embeddings():
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    csv_path = "sample_df_with_cv.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    print(f"Loaded dataset with shape {df.shape}")

    df["text_for_embedding"] = (
        df["title_translated"].fillna("") + " " + df["description"].fillna("")
    )

    texts = df["text_for_embedding"].tolist()
    print(f"Embedding {len(texts)} job offers...")

    vectors = []
    for i in range(0, len(texts), CONFIG["embedding_batch_size"]):
        batch = texts[i:i + CONFIG["embedding_batch_size"]]
        response = client.embeddings.create(model=CONFIG["embedding_model"], input=batch)
        batch_vectors = [r.embedding for r in response.data]
        vectors.extend(batch_vectors)

    embeddings = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_faiss_index(index, CONFIG["faiss_index_path"])

    df[["job_id", "title_translated", "description"]].to_parquet(
        "data/sqlite/job_offers.parquet"
    )

    print(f"Indexed {len(texts)} job offers into FAISS and saved metadata.")
