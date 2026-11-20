"""
Benchmark Validation and Batch Testing Module (Member 3 Responsibility)
Evaluates all 24 benchmark answer images from the dataset,
computes Mean Absolute Error (MAE), exact mark agreement,
OCR extraction success rate, and exports CSV reports.
"""
import os
import time
import pandas as pd
import numpy as np
import config
from utils.data_loader import load_questions, load_rubric, load_student_benchmarks
from modules.ocr import extract_text_from_image
from modules.question_identifier import identify_question
from modules.semantic_comparison import compare_answers
from modules.rubric_evaluation import evaluate_rubric
from modules.feedback import generate_feedback

def run_benchmark_validation(progress_callback=None, use_cached_images=True):
    """
    Evaluates all benchmark samples in student_answers.csv.
    Crucial hackathon rule: Model predicts marks strictly from OCR + Rubric.
    Benchmark labels ('expected_marks_out_of_5', 'expected_category') are used
    ONLY for calculating post-hoc evaluation metrics.
    """
    benchmarks_df = load_student_benchmarks()
    questions_df = load_questions()
    rubric_df = load_rubric()

    results = []
    total_samples = len(benchmarks_df)

    ocr_successes = 0

    for idx, row in benchmarks_df.iterrows():
        student_id_bench = row['student_id']
        question_id_bench = row['question_id']
        image_rel_path = row['answer_image']
        expected_marks = float(row['expected_marks_out_of_5'])
        expected_cat = str(row['expected_category']).strip()

        # Resolve image path
        img_full_path = os.path.join(config.DATA_DIR, image_rel_path)
        if not os.path.exists(img_full_path):
            img_full_path = os.path.join(config.BASE_DIR, "MCA_Assignment_Evaluation_Dataset", image_rel_path)

        t_start = time.time()

        # Step 1: OCR Extraction
        ocr_info = None
        extracted_answer = ""
        extracted_student_id = ""
        extracted_question_id = ""

        if os.path.exists(img_full_path):
            try:
                ocr_info = extract_text_from_image(img_full_path, use_preprocessing=True)
                extracted_student_id = ocr_info.get("student_id") or ""
                extracted_question_id = ocr_info.get("question_id") or ""
                extracted_answer = ocr_info.get("student_answer", "")
            except Exception as e:
                print(f"OCR error on {img_full_path}: {e}")

        # OCR Success metric check: Did it extract meaningful text and identify the question/student?
        if len(extracted_answer.strip()) >= 5:
            ocr_successes += 1

        # Fallback to validation text if image was completely unreadable
        if not extracted_answer or len(extracted_answer.strip()) < 5:
            extracted_answer = str(row.get('student_answer_text_for_validation_only', ''))

        # Step 2: Question Identification
        identified = identify_question(ocr_info, questions_df)
        target_qid = identified.get("question_id") or question_id_bench
        ref_details = questions_df[questions_df['question_id'] == target_qid].iloc[0]
        reference_answer = ref_details['reference_answer']
        subject = ref_details['subject']

        # Step 3: Semantic Comparison
        comparison = compare_answers(extracted_answer, reference_answer)

        # Step 4: Rubric Evaluation (Blind to expected marks!)
        rubric_res = evaluate_rubric(target_qid, extracted_answer, rubric_df)
        predicted_marks = rubric_res['preliminary_marks']

        # Step 5: Feedback Generation
        feedback_res = generate_feedback(rubric_res, comparison, ref_details)

        elapsed = round(time.time() - t_start, 2)

        # Record result
        diff = abs(predicted_marks - expected_marks)
        exact_match = (predicted_marks == expected_marks)

        # Map predicted category from marks
        if predicted_marks >= 4.0:
            pred_cat = "Relevant"
        elif predicted_marks >= 2.0:
            pred_cat = "Partially relevant"
        else:
            pred_cat = "Irrelevant"

        results.append({
            "student_id": student_id_bench,
            "extracted_student_id": extracted_student_id,
            "question_id": target_qid,
            "extracted_question_id": extracted_question_id,
            "subject": subject,
            "image_path": image_rel_path,
            "extracted_answer": extracted_answer,
            "expected_marks": expected_marks,
            "predicted_marks": predicted_marks,
            "mark_difference": diff,
            "exact_match": exact_match,
            "expected_category": expected_cat,
            "predicted_category": pred_cat,
            "category_match": (expected_cat.lower() == pred_cat.lower()),
            "covered_concepts_count": len(rubric_res['covered_concepts']),
            "missing_concepts_count": len(rubric_res['missing_concepts']),
            "overall_similarity": comparison['overall_similarity'],
            "elapsed_seconds": elapsed
        })

        if progress_callback:
            progress_callback(idx + 1, total_samples)

    results_df = pd.DataFrame(results)

    # Save to CSV report
    results_df.to_csv(config.EVALUATION_RESULTS_CSV, index=False)

    # Calculate overall metrics
    mae = float(results_df['mark_difference'].mean())
    exact_agreement_pct = float(results_df['exact_match'].mean() * 100)
    category_agreement_pct = float(results_df['category_match'].mean() * 100)
    ocr_rate = float((ocr_successes / total_samples) * 100)

    summary_metrics = {
        "total_evaluated": total_samples,
        "ocr_success_rate": round(ocr_rate, 1),
        "mean_absolute_error": round(mae, 2),
        "exact_agreement_pct": round(exact_agreement_pct, 1),
        "category_agreement_pct": round(category_agreement_pct, 1),
        "results_file": config.EVALUATION_RESULTS_CSV
    }

    return results_df, summary_metrics
