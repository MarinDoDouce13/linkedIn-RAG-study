# main.py
import argparse
from ingest import build_embeddings
from retriever import retrieve_similar_offers
from generator import generate_response
from evaluate_retrieval import test_single_job_retrieval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", type=int, help="Job ID to test retrieval performance")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip embedding rebuild")
    args = parser.parse_args()

    if not args.skip_ingest:
        build_embeddings()

    if args.test_id:
        test_single_job_retrieval(args.test_id)
        return

    cv_text = input("Entrez le texte du CV ou profil: ")
    retrieved_docs = retrieve_similar_offers(cv_text)
    result = generate_response(cv_text, retrieved_docs)

    print("\n=== Résultat ===\n")
    print(result)

if __name__ == "__main__":
    main()
