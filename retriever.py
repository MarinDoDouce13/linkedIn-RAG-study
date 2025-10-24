# retriever.py
import numpy as np
from openai import OpenAI
from config import CONFIG
from utils import load_faiss_index, read_secret_key, load_texts

def retrieve_similar_offers(query_text):
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    resp = client.embeddings.create(model=CONFIG["embedding_model"], input=[query_text])
    query_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    index = load_faiss_index(CONFIG["faiss_index_path"])
    distances, indices = index.search(query_vec, CONFIG["retrieval_top_k"])

    offers = load_texts(CONFIG["job_offers_dir"])
    retrieved = [(offers[i], float(distances[0][k])) for k, i in enumerate(indices[0]) if i < len(offers)]

    if CONFIG["use_rerank"]:
        retrieved = sorted(retrieved, key=lambda x: x[1] * CONFIG["rerank_weight"])

    return [doc for doc, _ in retrieved]
