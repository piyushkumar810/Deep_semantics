"""
Semantic Comparison Module (Member 2 Responsibility)
Calculates:
1. TF-IDF Cosine Similarity (Scikit-Learn)
2. Dense Semantic Embeddings Similarity (Sentence Transformers)
3. Combined overall similarity score

Note: Similarity score supports the evaluation context,
but does NOT independently determine marks.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
from modules.preprocessing import clean_text

# Global model cache for SentenceTransformer
_SENTENCE_MODEL = None

def get_sentence_model():
    """
    Loads and caches SentenceTransformer model using local offline files.
    Falls back gracefully if torch/transformers are unavailable.
    """
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            if os.path.exists(config.SENTENCE_MODEL_NAME):
                _SENTENCE_MODEL = SentenceTransformer(config.SENTENCE_MODEL_NAME, local_files_only=True)
            else:
                _SENTENCE_MODEL = SentenceTransformer(config.SENTENCE_MODEL_NAME)
        except Exception as e:
            print(f"Warning: SentenceTransformer could not be loaded: {e}")
            _SENTENCE_MODEL = None
    return _SENTENCE_MODEL

def compute_tfidf_similarity(text1, text2):
    """
    Computes cosine similarity between two texts using TF-IDF.
    Returns float in range [0.0, 1.0].
    """
    c1 = clean_text(text1)
    c2 = clean_text(text2)
    if not c1 or not c2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([c1, c2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        print(f"TF-IDF similarity calculation error: {e}")
        return 0.0

def compute_embedding_similarity(text1, text2):
    """
    Computes cosine similarity using Sentence Transformers dense embeddings.
    Returns float in range [0.0, 1.0].
    """
    if not text1 or not text2:
        return 0.0

    model = get_sentence_model()
    if model is None:
        # Fallback to TF-IDF if model is missing
        return compute_tfidf_similarity(text1, text2)

    try:
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        e1, e2 = embeddings[0], embeddings[1]
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        sim = float(np.dot(e1, e2) / (norm1 * norm2))
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        print(f"Embedding similarity calculation error: {e}")
        return compute_tfidf_similarity(text1, text2)

def compare_answers(student_answer, reference_answer):
    """
    Performs comprehensive multi-layer similarity comparison.
    Returns dictionary with:
    - tfidf_similarity (0-100%)
    - embedding_similarity (0-100%)
    - overall_similarity (0-100%)
    - match_level ('High', 'Moderate', 'Low', 'Very Low')
    """
    tfidf_sim = compute_tfidf_similarity(student_answer, reference_answer)
    embed_sim = compute_embedding_similarity(student_answer, reference_answer)

    # Weighted blend: 65% dense semantic embedding + 35% lexical TF-IDF
    if get_sentence_model() is not None:
        overall_sim = (0.65 * embed_sim) + (0.35 * tfidf_sim)
    else:
        overall_sim = tfidf_sim

    overall_pct = round(overall_sim * 100, 1)

    if overall_pct >= 70:
        match_level = "High Semantic Alignment"
    elif overall_pct >= 45:
        match_level = "Moderate Semantic Alignment"
    elif overall_pct >= 25:
        match_level = "Partial / Low Semantic Alignment"
    else:
        match_level = "Very Low / Off-Topic Alignment"

    return {
        "tfidf_similarity": round(tfidf_sim * 100, 1),
        "embedding_similarity": round(embed_sim * 100, 1),
        "overall_similarity": overall_pct,
        "match_level": match_level
    }
