# retriever.py
import numpy as np
import pandas as pd
from openai import OpenAI
from config import CONFIG
from utils import read_secret_key, load_faiss_index

def retrieve_similar_offers(query_text):
    # Load OpenAI client
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    # Load FAISS index + metadata
    index = load_faiss_index(CONFIG["faiss_index_path"])
    df = pd.read_parquet("data/sqlite/job_offers.parquet")

    # Embed the query
    resp = client.embeddings.create(model=CONFIG["embedding_model"], input=[query_text])
    query_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # Search in FAISS
    distances, indices = index.search(query_vec, CONFIG["retrieval_top_k"])

    # Retrieve corresponding job offers
    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        title = str(row["title_translated"])
        desc = str(row["description"])
        job_id = row["job_id"]
        results.append(f"[Job {job_id}] {title}\n\n{desc}")

    return results
