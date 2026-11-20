"""
Question Identifier Module
Matches extracted text or OCR metadata with question catalogue.
"""
import re
from utils.data_loader import load_questions

def identify_question(ocr_info, questions_df=None):
    """
    Identifies the question record from OCR extraction results.
    1. First checks ocr_info['question_id'] (e.g. 'Q1', 'Q2').
    2. Then searches raw text or question_text for question ID patterns.
    3. If question ID is missing or ambiguous, compares extracted question_text
       against available questions using keyword/text overlap.
    4. Returns dict with question details or None if manual selection required.
    """
    if questions_df is None:
        questions_df = load_questions()

    target_qid = None

    # Step 1: Check direct question_id in ocr_info
    if ocr_info and ocr_info.get("question_id"):
        cand = str(ocr_info["question_id"]).strip().upper()
        if cand in questions_df['question_id'].values:
            target_qid = cand

    # Step 2: Search in raw_full_text or question_text if still none
    if not target_qid and ocr_info:
        combined_text = f"{ocr_info.get('question_text', '')} {ocr_info.get('raw_full_text', '')}"
        m = re.search(r'\b(Q[1-8])\b', combined_text, re.IGNORECASE)
        if m:
            cand = m.group(1).upper()
            if cand in questions_df['question_id'].values:
                target_qid = cand

    # Step 3: Text match on question title
    if not target_qid and ocr_info and ocr_info.get("question_text"):
        q_text = ocr_info["question_text"].lower()
        best_match = None
        best_score = 0
        for _, row in questions_df.iterrows():
            ref_q = str(row['question']).lower()
            # Count common words
            words_cand = set(re.findall(r'\w+', q_text))
            words_ref = set(re.findall(r'\w+', ref_q))
            overlap = len(words_cand & words_ref)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_match = row['question_id']
        if best_match:
            target_qid = best_match

    if target_qid:
        row = questions_df[questions_df['question_id'] == target_qid].iloc[0]
        return {
            "status": "success",
            "question_id": row['question_id'],
            "subject": row['subject'],
            "question": row['question'],
            "maximum_marks": int(row['maximum_marks']),
            "reference_answer": row['reference_answer']
        }

    return {
        "status": "manual_selection_required",
        "question_id": None,
        "subject": None,
        "question": None,
        "maximum_marks": 5,
        "reference_answer": None
    }
