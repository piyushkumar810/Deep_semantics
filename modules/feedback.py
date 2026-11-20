"""
Feedback Generation Module (Member 3 Responsibility)
Generates comprehensive, constructive, and actionable feedback
for students and faculty members.
"""

def generate_feedback(evaluation_result, comparison_result, question_details=None):
    """
    Constructs structured feedback from evaluation and comparison results.
    Returns:
    - summary: Overall verdict
    - covered_summary: List of well-covered points
    - missing_summary: List of gaps
    - suggestions: Constructive improvement advice
    - faculty_recommendation: Note for teacher review
    """
    marks = evaluation_result.get("preliminary_marks", 0.0)
    covered = evaluation_result.get("covered_concepts", [])
    missing = evaluation_result.get("missing_concepts", [])
    overall_sim = comparison_result.get("overall_similarity", 0.0)

    # 1. Overall assessment
    if marks >= 4.0:
        verdict = "Excellent understanding demonstrated. The answer thoroughly addresses the core concepts."
        grade_category = "Relevant / High Quality"
    elif marks >= 2.5:
        verdict = "Good foundational knowledge, but key technical concepts or details are omitted."
        grade_category = "Partially Relevant / Needs Completion"
    elif marks > 0:
        verdict = "Limited coverage. The response touches on minor aspects but misses the primary definitions."
        grade_category = "Partially Relevant / Weak"
    else:
        verdict = "Answer does not address the required concepts, or discusses an unrelated topic."
        grade_category = "Irrelevant / Off-Topic"

    # 2. Covered concepts list
    covered_summary = [c["concept"] for c in covered]

    # 3. Missing concepts list
    missing_summary = [m["concept"] for m in missing]

    # 4. Actionable improvement suggestions
    suggestions = []
    if missing_summary:
        suggestions.append(f"To achieve full marks, include clear explanations for: {', '.join(missing_summary)}.")

    if marks < 3.0 and overall_sim < 30.0:
        suggestions.append("Ensure your response directly answers the specific prompt rather than discussing general domain terms.")

    if len(covered_summary) > 0 and len(missing_summary) > 0:
        suggestions.append("Expand on how these mechanisms interact (e.g., connect foundational terms with operational processes).")

    if not suggestions:
        suggestions.append("Great job! Keep maintaining this level of depth and terminology precision in technical examinations.")

    # 5. Faculty review recommendation
    if marks == 0.0:
        rec = "⚠️ Manual Review Recommended: Answer scored 0/5. Verify if the OCR text accurately captured handwriting or unusual formatting."
    elif marks in [2.0, 3.0]:
        rec = "ℹ️ Standard Review: The student provided a partial answer. Check if credit should be given for partially expressed ideas."
    else:
        rec = "✅ High Agreement: Response matches required rubric criteria closely. Safe for quick faculty approval."

    return {
        "verdict": verdict,
        "grade_category": grade_category,
        "preliminary_marks": marks,
        "overall_similarity": overall_sim,
        "covered_concepts": covered_summary,
        "missing_concepts": missing_summary,
        "improvement_suggestions": suggestions,
        "faculty_recommendation": rec
    }
