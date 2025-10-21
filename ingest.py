# ingest.py
import os
import numpy as np
import faiss
from openai import OpenAI
from config import CONFIG
from utils import load_texts, read_secret_key, save_faiss_index

def build_embeddings():
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)
    texts = load_texts(CONFIG["job_offers_dir"])

    vectors = []
    for i in range(0, len(texts), CONFIG["embedding_batch_size"]):
        batch = texts[i:i + CONFIG["embedding_batch_size"]]
        resp = client.embeddings.create(model=CONFIG["embedding_model"], input=batch)
        batch_vectors = [r.embedding for r in resp.data]
        vectors.extend(batch_vectors)

    embeddings = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_faiss_index(index, CONFIG["faiss_index_path"])

    print(f"Indexed {len(texts)} job offers into FAISS.")
