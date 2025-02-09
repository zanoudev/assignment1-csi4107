import json
import os
from preprocessing import load_corpus
from indexing import build_inverted_index, compute_bm25
from retrieval import rank_documents

# Get absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # One level up
CORPUS_FILE = os.path.join(BASE_DIR, "scifact", "corpus.jsonl")
QUERY_FILE = os.path.join(BASE_DIR, "scifact", "queries.jsonl")
RESULTS_FILE = os.path.join(BASE_DIR, "Results.txt")

def process_queries(query_file, index, doc_freq, corpus_size):
    """Processes queries and retrieves ranked results."""
    results = []
    with open(query_file, 'r', encoding="utf-8") as f:
        for line in f:
            query_data = json.loads(line)

            if '_id' not in query_data:
                print(f"Skipping query with missing '_id': {query_data}")
                continue  

            query_id = query_data['_id']

            # Only odd-numbered queries are used
            if int(query_id) % 2 == 0:
                continue  

            query_text = query_data.get('text', '')
            ranked_docs = rank_documents(query_text, index, doc_freq, corpus_size)

            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_name")

    return results

def save_results(results, output_file):
    """Saves ranked results in the correct format."""
    with open(output_file, 'w', encoding="utf-8") as f:
        for line in results:
            query_id, q0, doc_id, rank, score, run_name = line.split()
            score = f"{float(score):.4f}"  # Ensure correct floating-point precision
            f.write(f"{query_id} {q0} {doc_id} {rank} {score} {run_name}\n")

if __name__ == "__main__":
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_FILE)

    print("Building index...")
    index, doc_freq, doc_lengths, avg_doc_length = build_inverted_index(corpus)
    bm25_index = compute_bm25(index, doc_freq, len(corpus), avg_doc_length, doc_lengths)

    print("Processing queries...")
    results = process_queries(QUERY_FILE, bm25_index, doc_freq, len(corpus))

    print("Saving results...")
    save_results(results, RESULTS_FILE)

    print("Done! Results saved to", RESULTS_FILE)
