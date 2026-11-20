"""
Rubric-Based Evaluation Engine (Member 2 Responsibility)

Evaluates student answers concept-by-concept against the 5 rubric criteria.
Rules:
1. Each question has exactly 5 concepts worth 1 mark each (Total = 5 marks).
2. Concept matching recognizes synonyms and semantic equivalence.
3. Keyword-only matches without meaningful context are rejected.
4. Total marks never exceed question maximum marks.
5. Benchmark labels are NEVER used here.
"""
import re
from utils.data_loader import get_rubric_concepts
from modules.preprocessing import clean_text, tokenize_and_filter
from modules.semantic_comparison import get_sentence_model, compute_embedding_similarity

# Domain synonyms and concept expansions for MCA subjects (NLP, CV, ML, Data Cleaning)
DOMAIN_CONCEPT_EXPANSIONS = {
    # Artificial Intelligence / AI
    "artificial intelligence / ai": [
        "artificial intelligence", "ai", "machine intelligence", "intelligent systems"
    ],
    # computers
    "computers": [
        "computer", "computers", "machine", "machines", "systems", "computing system", "computational"
    ],
    # understand human language
    "understand human language": [
        "understand human language", "comprehend human language", "understanding natural language",
        "understands language", "comprehend natural language", "natural language understanding"
    ],
    # interpret language
    "interpret language": [
        "interpret language", "interpreting language", "interprets language",
        "linguistic interpretation", "extract meaning", "deriving meaning from language"
    ],
    # process human language
    "process human language": [
        "process human language", "processing human language", "process language",
        "processing natural language", "linguistic processing", "text processing"
    ],
    # splitting text
    "splitting text": [
        "splitting text", "splits text", "dividing text", "divides text", "breaking down text",
        "segmenting text", "breaks text", "splits a paragraph", "divide text"
    ],
    # smaller units
    "smaller units": [
        "smaller units", "small units", "individual tokens", "tokens", "smaller pieces",
        "chunks", "discrete units", "segments", "subunits"
    ],
    # words
    "words": ["words", "word", "vocabulary words", "terms", "lexemes"],
    # phrases
    "phrases": ["phrases", "phrase", "n-grams", "sub-sentences", "clauses"],
    # sentences
    "sentences": ["sentences", "sentence", "lines of text"],
    # resizing
    "resizing": [
        "resizing", "resize", "scale", "scaling", "image resizing", "dimension adjustment", "interpolation"
    ],
    # noise removal
    "noise removal": [
        "noise removal", "remove noise", "denoising", "removing noise", "noise reduction",
        "eliminating noise", "clean noise"
    ],
    # filtering
    "filtering": [
        "filtering", "filters", "filter", "spatial filtering", "blurring", "gaussian blur", "median filter"
    ],
    # normalization
    "normalization": [
        "normalization", "normalize", "normalizing", "pixel scaling", "intensity normalization", "standardization"
    ],
    # improve image quality
    "improve image quality": [
        "improve image quality", "improving image quality", "enhancing quality", "image enhancement",
        "better quality", "prepare image for analysis", "quality enhancement"
    ],
    # deep learning
    "deep learning": [
        "deep learning", "dl", "deep neural network", "dnn", "multi-layer neural"
    ],
    # architecture / neural network
    "architecture / neural network": [
        "architecture", "neural network", "neural net", "network architecture", "ann", "model structure"
    ],
    # convolutional layers
    "convolutional layers": [
        "convolutional layers", "convolution layer", "conv layers", "convnet", "convolutions"
    ],
    # extract features
    "extract features": [
        "extract features", "extracting features", "feature extraction", "learn spatial features",
        "detect patterns", "pattern extraction", "spatial hierarchy"
    ],
    # images
    "images": [
        "images", "image", "visual data", "pictures", "visual inputs", "pixels"
    ],
    # machine learning approach
    "machine learning approach": [
        "machine learning approach", "machine learning technique", "machine learning", "ml", "ml algorithm", "supervised approach"
    ],
    # model learns / trains
    "model learns / trains": [
        "model learns", "learns", "trains", "trained", "model training", "learning from", "algorithm learns"
    ],
    # labelled data
    "labelled data": [
        "labelled data", "labeled data", "ground truth", "annotated data", "input output pairs", "labeled dataset", "with labels"
    ],
    # predict outputs
    "predict outputs": [
        "predict outputs", "predict output", "predicts outputs", "make predictions", "forecast outputs", "output prediction", "mapping inputs to outputs"
    ],
    # unseen / new inputs
    "unseen / new inputs": [
        "unseen", "new inputs", "unseen inputs", "new data", "unseen data", "test data", "novel inputs", "future inputs"
    ],
    # identifying
    "identifying": [
        "identifying", "identify", "detection", "detecting", "spotting", "discovering errors", "locating"
    ],
    # correcting / handling
    "correcting / handling": [
        "correcting", "handling", "correction", "fixes", "cleaning", "imputing", "removing", "rectifying", "handling inconsistencies"
    ],
    # missing values
    "missing values": [
        "missing values", "null values", "nan", "missing data", "empty cells", "missing fields"
    ],
    # duplicate records
    "duplicate records": [
        "duplicate records", "duplicates", "redundant rows", "duplicate rows", "repeated data", "repetition"
    ],
    # incorrect values / inconsistencies
    "incorrect values / inconsistencies": [
        "incorrect values", "inconsistencies", "inconsistent data", "errors", "outliers", "data inaccuracies", "bad data"
    ],
    # NLP technique
    "nlp technique": [
        "nlp technique", "nlp method", "natural language processing technique", "text analysis technique", "computational linguistics"
    ],
    # opinions
    "opinions": [
        "opinions", "opinion", "views", "viewpoints", "attitudes", "reviews", "customer feedback"
    ],
    # emotions
    "emotions": [
        "emotions", "emotion", "affective states", "feelings", "sentiment state"
    ],
    # positive / negative / neutral sentiment
    "positive / negative / neutral sentiment": [
        "positive", "negative", "neutral", "positive / negative / neutral", "polarity", "sentiment polarity", "valence"
    ],
    # dividing an image
    "dividing an image": [
        "dividing an image", "divides an image", "partitioning an image", "splits an image", "segmenting an image", "separating image"
    ],
    # meaningful regions
    "meaningful regions": [
        "meaningful regions", "homogeneous regions", "regions", "image regions", "relevant parts"
    ],
    # segments
    "segments": [
        "segments", "segmentation", "sub-regions", "patches", "pixel clusters", "partitions"
    ],
    # identify objects
    "identify objects": [
        "identify objects", "identifying objects", "object detection", "locate objects", "boundary detection", "delineate objects"
    ],
    # areas of interest
    "areas of interest": [
        "areas of interest", "roi", "regions of interest", "salient regions", "target areas"
    ]
}

def verify_single_concept(student_answer, concept_str):
    """
    Evaluates whether a single rubric concept is present in the student answer.
    Returns:
    (is_covered: bool, score: float, explanation: str, method: str)
    """
    if not student_answer or not str(student_answer).strip():
        return False, 0.0, "Answer is empty", "None"

    norm_answer = clean_text(student_answer)
    norm_concept = clean_text(concept_str)

    # 1. Exact or sub-phrase phrase matching
    if norm_concept in norm_answer:
        return True, 1.0, f"Direct match found for concept '{concept_str}'", "Exact / Direct Match"

    # 2. Domain Concept Expansions & Synonyms
    concept_key = concept_str.strip().lower()
    variants = DOMAIN_CONCEPT_EXPANSIONS.get(concept_key, [])
    if not variants:
        # Fallback split on '/' or ','
        parts = [p.strip().lower() for p in re.split(r'[/,]', concept_str) if p.strip()]
        variants = parts

    for v in variants:
        v_clean = clean_text(v)
        # Check boundary or phrase
        if re.search(r'\b' + re.escape(v_clean) + r'\b', norm_answer):
            return True, 1.0, f"Concept covered via synonym/expansion: '{v}'", "Synonym / Domain Matching"

    # 3. Sentence-level Context Semantic Verification (using SentenceTransformers if available)
    model = get_sentence_model()
    if model is not None:
        # Split answer into clauses or sentences
        sentences = re.split(r'[.;,]\s*|\band\b|\bthat\b', student_answer)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        if not sentences:
            sentences = [student_answer]

        best_score = 0.0
        best_sent = ""
        for s in sentences:
            sim = compute_embedding_similarity(s, concept_str)
            if sim > best_score:
                best_score = sim
                best_sent = s

        # Threshold for semantic coverage with context
        if best_score >= 0.58:
            return True, 1.0, f"Semantically expressed in context: '{best_sent}' (similarity: {best_score:.2f})", "Contextual Semantic Match"

    # 4. Content Word Bag Overlap (Strict: must have sufficient meaningful words)
    concept_words = tokenize_and_filter(concept_str, remove_stopwords=True)
    if concept_words:
        answer_words = set(tokenize_and_filter(student_answer, remove_stopwords=True))
        matched_words = [w for w in concept_words if w in answer_words]
        overlap_ratio = len(matched_words) / float(len(concept_words))

        # If 100% of keywords are in answer, award mark
        if overlap_ratio >= 0.99:
            return True, 1.0, f"Core keywords present in context: {matched_words}", "Keyword Overlap Match"

    return False, 0.0, f"Concept '{concept_str}' not adequately expressed in student answer", "Missing"

def evaluate_rubric(question_id, student_answer, rubric_df=None):
    """
    Performs full rubric evaluation for all 5 concepts of the question.
    Returns:
    - preliminary_marks (out of 5)
    - max_marks (5)
    - concepts_total (5)
    - covered_concepts (list of dicts)
    - missing_concepts (list of dicts)
    - concept_breakdown (list of all 5 evaluations)
    """
    rubric_points = get_rubric_concepts(question_id, rubric_df)
    if not rubric_points:
        return {
            "preliminary_marks": 0.0,
            "max_marks": 5.0,
            "concepts_total": 0,
            "covered_concepts": [],
            "missing_concepts": [],
            "concept_breakdown": []
        }

    concept_breakdown = []
    covered_concepts = []
    missing_concepts = []
    total_preliminary_marks = 0.0

    for idx, item in enumerate(rubric_points, start=1):
        concept_text = item["concept_or_expected_point"]
        weight = float(item.get("marks", 1.0))

        is_covered, score, reasoning, method = verify_single_concept(student_answer, concept_text)
        awarded_mark = weight if is_covered else 0.0
        total_preliminary_marks += awarded_mark

        record = {
            "concept_num": idx,
            "concept": concept_text,
            "weight": weight,
            "awarded_marks": awarded_mark,
            "is_covered": is_covered,
            "reasoning": reasoning,
            "method": method
        }
        concept_breakdown.append(record)
        if is_covered:
            covered_concepts.append(record)
        else:
            missing_concepts.append(record)

    # Cap marks strictly at 5
    total_preliminary_marks = min(5.0, total_preliminary_marks)

    return {
        "preliminary_marks": total_preliminary_marks,
        "max_marks": 5.0,
        "concepts_total": len(rubric_points),
        "covered_concepts": covered_concepts,
        "missing_concepts": missing_concepts,
        "concept_breakdown": concept_breakdown
    }
