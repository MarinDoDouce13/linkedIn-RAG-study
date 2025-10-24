# utils.py
import os
import faiss
import numpy as np

def read_secret_key(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_texts(directory):
    texts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
    return texts

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)
