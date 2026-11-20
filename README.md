# Deep Semantics — Intelligent Student Assignment Evaluation Assistant

> **"Beyond Keywords. Understanding Meaning."**  
> An AI-powered assignment evaluation system that reads student answer-sheet images using OCR, matches questions against faculty reference answers, evaluates concept coverage against a 5-point rubric, computes preliminary marks, and assists faculty members in evaluation.

---

## 🌟 Key Features

1. **Computer Vision & OCR Preprocessing**:
   - OpenCV grayscale conversion, bilateral filtering, and adaptive Otsu thresholding.
   - High-accuracy EasyOCR text extraction for student IDs, question numbers, and answers.
2. **Automated Question Identification**:
   - Automatically detects Question IDs (`Q1`–`Q8`) from OCR text with manual faculty override.
3. **Multi-Layer Semantic Comparison**:
   - TF-IDF Cosine Similarity (Scikit-Learn).
   - Dense contextual semantic embeddings (`all-MiniLM-L6-v2` Sentence Transformers).
   - Note: Similarity provides alignment context but does not independently determine marks.
4. **Rubric-Based Concept Scoring Engine**:
   - 5 essential concepts per question, 1 mark per concept (Max: 5 marks).
   - Checks semantic equivalence and domain-specific terminology without being fooled by keyword stuffing.
5. **Actionable Feedback Generation**:
   - Explains covered concepts, missing concepts, and improvement advice.
6. **Faculty Decision Support**:
   - Faculty can review OCR text, adjust preliminary marks, and approve final grades.
7. **24-Sample Benchmark Validation Suite**:
   - Batch testing on all 24 answer-sheet images.
   - Reports Mean Absolute Error (MAE), exact mark agreement, and exports CSV reports.

---

## 📁 Project Structure

```
deep-semantics/
├── app.py                     # Main Streamlit web application
├── config.py                  # Project paths, thresholds, and model settings
├── requirements.txt           # Python library dependencies
├── README.md                  # Project documentation & execution guide
├── data/                      # Dataset folder
│   ├── questions.csv          # 8 questions, subjects, reference answers
│   ├── marking_rubric.csv     # 40 concepts (5 per question, 1 mark each)
│   ├── student_answers.csv    # 24 benchmark ground-truth samples
│   └── answer_images/         # 24 printed student answer sheets
├── models/                    # Offline model weights
│   ├── all-MiniLM-L6-v2/      # Cached SentenceTransformer model
│   ├── easyocr/               # EasyOCR CRAFT & English recognition weights
│   └── easyocr_network/       # EasyOCR user network cache
├── modules/                   # Core pipeline modules
│   ├── ocr.py                 # Member 1: OpenCV image prep & EasyOCR
│   ├── preprocessing.py       # Member 2: NLTK tokenization & cleaning
│   ├── question_identifier.py # Question catalog matcher
│   ├── semantic_comparison.py # Member 2: TF-IDF & Sentence Transformer similarity
│   ├── rubric_evaluation.py   # Member 2: 5-concept rubric scoring engine
│   ├── feedback.py            # Member 3: Actionable feedback generator
│   └── validation.py          # Member 3: 24-sample benchmark validator
├── reports/                   # Generated evaluation reports
│   └── evaluation_results.csv # Full benchmark testing report
└── utils/
    └── data_loader.py         # CSV loader utilities
```

---

## 👥 Team Work Division — 3 Members

### Member 1 — Computer Vision and OCR
- **Responsibilities**:
  - Image handling and OpenCV preprocessing (`cv2.bilateralFilter`, Gaussian blur, Otsu thresholding).
  - EasyOCR extraction pipeline (`modules/ocr.py`).
  - Parsing student ID, question ID, and answer text with regex normalization (`MCAOO1` $\rightarrow$ `MCA001`).
- **Demo Role**: Explain image-to-text pipeline, show sample image upload, and demonstrate OCR verification.

### Member 2 — NLP, Semantic Comparison, and Rubric Evaluation
- **Responsibilities**:
  - Text normalization, tokenization, and stop-word filtering (`modules/preprocessing.py`).
  - Semantic embeddings and TF-IDF cosine comparison (`modules/semantic_comparison.py`).
  - 5-point concept verification engine with domain synonym expansion (`modules/rubric_evaluation.py`).
- **Demo Role**: Explain why keyword matching fails, demonstrate semantic alignment, and walk through concept-by-concept scoring.

### Member 3 — Streamlit Dashboard, Feedback, and Validation
- **Responsibilities**:
  - Streamlit dashboard user interface (`app.py`).
  - Intelligent feedback generation (`modules/feedback.py`).
  - Faculty review, mark modification, and approval workflow.
  - Full 24-sample benchmark validation suite, MAE calculation, and CSV reporting (`modules/validation.py`).
- **Demo Role**: Present Streamlit interface, show faculty oversight workflow, and reveal benchmark validation results.

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.10 or 3.11 installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 10-Step Hackathon Demonstration Workflow

1. **Launch**: Run `streamlit run app.py`.
2. **Select Image**: Under *Single Answer Evaluation*, choose `MCA001_Q1.png` or upload any answer-sheet image.
3. **Verify OCR**: View the extracted Student ID (`MCA001`), Question ID (`Q1`), and Student Answer text.
4. **Identify Question**: See the system match `Q1` to *Natural Language Processing*.
5. **View Reference**: Review faculty reference answer and subject context.
6. **Semantic Analysis**: Inspect the blended similarity score (Sentence Transformer + TF-IDF).
7. **Rubric Scoring**: Inspect the 5-point concept breakdown (1 mark per verified concept).
8. **Feedback**: Review covered concepts, missing concepts, and suggestions.
9. **Faculty Review**: Adjust marks using the slider and approve the evaluation.
10. **Benchmark Validation**: Switch to *24-Answer Benchmark Validation* and click **Run Full Benchmark Evaluation** to see the overall MAE and interactive error distribution across all 24 answers.
