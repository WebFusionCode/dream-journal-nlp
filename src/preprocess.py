import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Ensure NLTK data is available (for Streamlit Cloud / servers)
for pkg in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean and normalize dream text."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return " ".join(text.split())

def preprocess_text(text: str):
    """Tokenize, remove stopwords, and lemmatize."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stopwords and len(t) > 1
    ]
    return tokens
