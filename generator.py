# generator.py
import os
from openai import OpenAI
from config import CONFIG
from utils import read_secret_key

def generate_response(cv_text, retrieved_docs):
    key = read_secret_key(CONFIG["openai_key_path"])
    client = OpenAI(api_key=key)

    context = "\n\n".join(retrieved_docs)
    prompt = (
        f"Profil du candidat :\n{cv_text}\n\n"
        f"Offres d'emploi correspondantes :\n{context}\n\n"
        "Analyse les correspondances les plus pertinentes entre ce profil et les offres."
    )

    resp = client.chat.completions.create(
        model=CONFIG["llm_model"],
        temperature=CONFIG["llm_temperature"],
        max_tokens=CONFIG["llm_max_tokens"],
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content
