import argparse
from ingest import build_embeddings
from evaluate_retrieval import evaluate_all_clusters
from evaluate_retrieval_bow import evaluate_all_clusters_bow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-embeddings", action="store_true", help="Force rebuilding embeddings")
    parser.add_argument("--method", type=str, default="bow", choices=["embedding", "bow"])
    args = parser.parse_args()

    if args.rebuild_embeddings:
        build_embeddings()
    else:
        print("Skipping embedding rebuild (use --rebuild-embeddings to force it)")

    if args.method == "embedding":
        evaluate_all_clusters()
    else:
        evaluate_all_clusters_bow()



if __name__ == "__main__":
    main()
