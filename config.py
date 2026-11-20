"""
Configuration settings for Deep Semantics.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")
RUBRIC_CSV = os.path.join(DATA_DIR, "marking_rubric.csv")
STUDENT_ANSWERS_CSV = os.path.join(DATA_DIR, "student_answers.csv")
ANSWER_IMAGES_DIR = os.path.join(DATA_DIR, "answer_images")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EVALUATION_RESULTS_CSV = os.path.join(REPORTS_DIR, "evaluation_results.csv")

# Ensure required directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Semantic Model Configuration
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
SENTENCE_MODEL_NAME = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "all-MiniLM-L6-v2"

# EasyOCR Model Storage
LOCAL_EASYOCR_DIR = os.path.join(BASE_DIR, "models", "easyocr")
os.makedirs(LOCAL_EASYOCR_DIR, exist_ok=True)

MAX_MARKS_DEFAULT = 5
CONCEPT_MARKS_DEFAULT = 1.0

# Thresholds
CONCEPT_SIMILARITY_THRESHOLD = 0.52
HIGH_CONFIDENCE_SIMILARITY_THRESHOLD = 0.65
