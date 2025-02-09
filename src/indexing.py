import math
from collections import defaultdict, Counter

def build_inverted_index(corpus):
    """Creates an inverted index and document frequency table."""
    index = defaultdict(list)
    doc_freq = Counter()
    doc_lengths = {}

    for doc_id, terms in corpus.items():
        term_counts = Counter(terms)
        doc_lengths[doc_id] = sum(term_counts.values())  # Store document length
        for term, count in term_counts.items():
            index[term].append((doc_id, count))
            doc_freq[term] += 1

    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)  # Compute average doc length
    return index, doc_freq, doc_lengths, avg_doc_length

def compute_bm25(index, doc_freq, corpus_size, avg_doc_length, doc_lengths, k1=1.3, b=0.6):
    """Computes BM25 scores for the index."""
    bm25_index = {}

    for term, postings in index.items():
        idf = math.log((corpus_size - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)

        term_weights = []
        for doc_id, tf in postings:
            doc_length = doc_lengths.get(doc_id, avg_doc_length)
            tf_weight = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_length / avg_doc_length)) + tf)
            term_weights.append((doc_id, tf_weight * idf))

        bm25_index[term] = term_weights

    return bm25_index
