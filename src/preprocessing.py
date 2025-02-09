import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load stopwords and stemmer
STOPWORDS = set(stopwords.words('english')) - {"not", "no", "against", "without"}  # keep negations
ps = PorterStemmer()


# debug lines ----------------------------------------------------
# print("nltk path: ")
# print(nltk.__file__)
# nltk.data.path.append("C:/Users/zanou/AppData/Roaming/nltk_data")

from nltk.util import ngrams

def preprocess_text(text):
    """Tokenizes, removes stopwords, stems words."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word not in STOPWORDS]
    return tokens  # No bi-grams


def load_corpus(file_path):
    """Loads the corpus and preprocesses full document text."""
    corpus = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            if '_id' not in doc:
                continue  

            doc_id = doc['_id']
            title = doc.get("title", "")
            abstract = " ".join(doc.get("abstract", []))  # Convert list to string if needed

            full_text = title + " " + abstract
            corpus[doc_id] = preprocess_text(full_text)  # Use full text

    return corpus


