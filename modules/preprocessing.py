"""
NLP Preprocessing Module (Member 2 Responsibility)
Handles tokenization, lowercase conversion, stop-word filtering,
lemmatization, keyword extraction, and synonym expansion.
"""
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

# English stopwords fallback set if nltk download isn't loaded
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren'
}

def get_stop_words():
    """Returns stopwords set."""
    try:
        return set(stopwords.words('english'))
    except Exception:
        return DEFAULT_STOPWORDS

def clean_text(text):
    """
    Standard text normalization:
    - Lowercase conversion
    - Strip punctuation
    - Clean whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace slashes and hyphens with spaces for compound words (e.g. "unseen / new" -> "unseen new")
    text = re.sub(r'[/\\_\-]', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text, remove_stopwords=True):
    """
    Tokenizes text, strips non-alphanumeric, and filters stopwords.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    stop_words = get_stop_words() if remove_stopwords else set()
    filtered_tokens = [tok for tok in tokens if tok.isalnum() and tok not in stop_words and len(tok) > 1]
    return filtered_tokens

def extract_keywords(text, top_n=10):
    """
    Extracts frequency-based keywords from text.
    """
    tokens = tokenize_and_filter(text, remove_stopwords=True)
    freq = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:top_n]]

def get_synonyms(word):
    """
    Fetches WordNet synonyms for a given word.
    """
    synonyms = {word.lower()}
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name.isalpha():
                    synonyms.add(name)
    except Exception:
        pass
    return synonyms
