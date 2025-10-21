# main.py
from ingest import build_embeddings
from retriever import retrieve_similar_offers
from generator import generate_response

def main():
    build_embeddings()  # à exécuter une seule fois

    cv_text = input("Entrez le texte du CV ou profil: ")
    retrieved_docs = retrieve_similar_offers(cv_text)
    result = generate_response(cv_text, retrieved_docs)

    print("\n=== Résultat ===\n")
    print(result)

if __name__ == "__main__":
    main()
