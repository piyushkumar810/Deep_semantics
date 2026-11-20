"""
Data loader utility for Deep Semantics.
"""
import os
import pandas as pd
import config

def load_questions(filepath=config.QUESTIONS_CSV):
    """
    Loads questions.csv containing:
    ['question_id', 'subject', 'question', 'maximum_marks', 'reference_answer']
    """
    if not os.path.exists(filepath):
        # Fallback to dataset folder if needed
        fallback = os.path.join(config.BASE_DIR, "MCA_Assignment_Evaluation_Dataset", "questions.csv")
        if os.path.exists(fallback):
            filepath = fallback
        else:
            raise FileNotFoundError(f"Questions file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    # Ensure column strings stripped
    df['question_id'] = df['question_id'].astype(str).str.strip()
    return df

def load_rubric(filepath=config.RUBRIC_CSV):
    """
    Loads marking_rubric.csv containing:
    ['question_id', 'concept_or_expected_point', 'marks']
    """
    if not os.path.exists(filepath):
        fallback = os.path.join(config.BASE_DIR, "MCA_Assignment_Evaluation_Dataset", "marking_rubric.csv")
        if os.path.exists(fallback):
            filepath = fallback
        else:
            raise FileNotFoundError(f"Rubric file not found at: {filepath}")
            
    df = pd.read_csv(filepath)
    df['question_id'] = df['question_id'].astype(str).str.strip()
    df['concept_or_expected_point'] = df['concept_or_expected_point'].astype(str).str.strip()
    return df

def load_student_benchmarks(filepath=config.STUDENT_ANSWERS_CSV):
    """
    Loads student_answers.csv for validation & testing ONLY:
    ['student_id', 'question_id', 'answer_image', 'student_answer_text_for_validation_only',
     'expected_marks_out_of_5', 'expected_category']
    """
    if not os.path.exists(filepath):
        fallback = os.path.join(config.BASE_DIR, "MCA_Assignment_Evaluation_Dataset", "student_answers.csv")
        if os.path.exists(fallback):
            filepath = fallback
        else:
            raise FileNotFoundError(f"Student benchmarks file not found at: {filepath}")
            
    df = pd.read_csv(filepath)
    df['student_id'] = df['student_id'].astype(str).str.strip()
    df['question_id'] = df['question_id'].astype(str).str.strip()
    return df

def get_question_details(question_id, questions_df=None):
    """
    Fetches reference question record by question_id.
    """
    if questions_df is None:
        questions_df = load_questions()
    
    matched = questions_df[questions_df['question_id'].str.upper() == str(question_id).strip().upper()]
    if not matched.empty:
        return matched.iloc[0].to_dict()
    return None

def get_rubric_concepts(question_id, rubric_df=None):
    """
    Returns the list of rubric concepts for a given question_id.
    """
    if rubric_df is None:
        rubric_df = load_rubric()
        
    matched = rubric_df[rubric_df['question_id'].str.upper() == str(question_id).strip().upper()]
    return matched.to_dict(orient='records')
