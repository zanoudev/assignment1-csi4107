import numpy as np
from collections import defaultdict
from preprocessing import preprocess_text
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

def expand_query(query, doc_freq, corpus_size, expansion_threshold=0.1):
    """Expands a query using WordNet, but only for very rare terms."""
    expanded_terms = set(query)
    for term in query:
        if doc_freq.get(term, 0) / corpus_size <= expansion_threshold:  # Expand only rare words
            synsets = wordnet.synsets(term)
            if synsets:
                expanded_terms.add(synsets[0].lemmas()[0].name().lower())  # Only take the first synonym
    return list(expanded_terms)




def rank_documents(query, index, doc_freq, corpus_size):
    """Ranks documents using expanded queries with normalized term weighting."""
    query_terms = expand_query(preprocess_text(query), doc_freq, corpus_size)
    
    query_vector = {
        term: ((1 + np.log(1 + query_terms.count(term))) * np.log((corpus_size + 1) / (1 + doc_freq.get(term, 1))))
        for term in query_terms if term in index
    }

    doc_scores = defaultdict(float)
    
    for term, postings in index.items():
        if term in query_vector:
            for doc_id, weight in postings:
                doc_scores[doc_id] += weight * query_vector[term]

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:100]  # Return top 100 // without normalizing scores


