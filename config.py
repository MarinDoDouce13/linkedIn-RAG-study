# config.py

CONFIG = {
    # Models
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o-mini",

    # Embedding parameters
    "embedding_batch_size": 32,

    # Retrieval
    "retrieval_top_k": 5,
    "use_rerank": True,
    "rerank_weight": 0.8,

    # LLM generation
    "llm_temperature": 0.3,
    "llm_max_tokens": 600,

    # Paths
    "job_offers_dir": "data/job_offers",
    "faiss_index_path": "data/embeddings.faiss",
    "sqlite_path": "data/sqlite",
    "openai_key_path": "secrets/openai_key.txt"
}
