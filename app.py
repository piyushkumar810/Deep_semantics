"""
Streamlit Web Application for Deep Semantics:
Intelligent Student Assignment Evaluation Assistant
"""
import os
import io
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import config
from utils.data_loader import (
    load_questions,
    load_rubric,
    load_student_benchmarks,
    get_question_details,
    get_rubric_concepts
)
from modules.ocr import extract_text_from_image, preprocess_image
from modules.question_identifier import identify_question
from modules.semantic_comparison import compare_answers
from modules.rubric_evaluation import evaluate_rubric
from modules.feedback import generate_feedback
from modules.validation import run_benchmark_validation

# Page configuration
st.set_page_config(
    page_title="Deep Semantics — Intelligent Assignment Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .tagline {
        font-size: 1.1rem;
        color: #546E7A;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .concept-covered {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 6px;
        border-left: 4px solid #4CAF50;
    }
    .concept-missing {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 6px;
        border-left: 4px solid #E53935;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def get_cached_data():
    questions = load_questions()
    rubric = load_rubric()
    benchmarks = load_student_benchmarks()
    return questions, rubric, benchmarks

questions_df, rubric_df, benchmarks_df = get_cached_data()

# App Header
st.markdown('<p class="main-title">🎓 Deep Semantics</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline"><b>Beyond Keywords. Understanding Meaning.</b> — AI-Powered Assignment Evaluation Assistant for Faculty</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Evaluation Controls")
    app_mode = st.radio(
        "Select Operation Mode:",
        ["📝 Single Answer Evaluation", "📊 24-Answer Benchmark Validation", "📚 Dataset & Rubric Explorer"]
    )
    st.markdown("---")
    st.subheader("💡 System Architecture")
    st.caption("**1. Computer Vision & OCR:** OpenCV preprocessing + EasyOCR engine.")
    st.caption("**2. NLP & Rubric Engine:** Tokenization, stopword removal, TF-IDF cosine, and Sentence Transformers embeddings.")
    st.caption("**3. Faculty Decision Support:** Concept-wise preliminary scoring (out of 5) with full manual override.")
    st.markdown("---")
    st.info("⚠️ **Faculty Oversight Notice:** All AI marks are preliminary recommendations and subject to faculty verification.")

# MODE 1: Single Answer Evaluation
if app_mode == "📝 Single Answer Evaluation":
    st.subheader("Step 1: Answer Sheet Input")

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        input_source = st.radio(
            "Select Image Source:",
            ["📁 Choose Sample from Dataset", "📤 Upload Custom Image"],
            horizontal=True
        )

    # Image selection
    selected_image_path = None
    uploaded_file = None

    if input_source == "📁 Choose Sample from Dataset":
        with col_input2:
            img_files = sorted(os.listdir(config.ANSWER_IMAGES_DIR))
            chosen_file = st.selectbox("Select Sample Answer Sheet:", img_files)
            selected_image_path = os.path.join(config.ANSWER_IMAGES_DIR, chosen_file)
    else:
        with col_input2:
            uploaded_file = st.file_uploader("Upload Student Answer Sheet (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    target_image_source = uploaded_file if uploaded_file is not None else selected_image_path

    if target_image_source:
        col_img, col_ocr = st.columns([1, 1.2])

        with col_img:
            st.markdown("##### 📄 Answer-Sheet Preview")
            if isinstance(target_image_source, str):
                st.image(target_image_source, use_container_width=True, caption=os.path.basename(target_image_source))
            else:
                st.image(target_image_source, use_container_width=True, caption="Uploaded Answer Sheet")

        # Session state for OCR and Evaluation
        if "last_processed_img" not in st.session_state or st.session_state.last_processed_img != str(target_image_source):
            with st.spinner("🔍 Running OpenCV Preprocessing and EasyOCR Text Extraction..."):
                ocr_data = extract_text_from_image(target_image_source, use_preprocessing=True)
                st.session_state.ocr_data = ocr_data
                st.session_state.last_processed_img = str(target_image_source)
                st.session_state.faculty_marks_override = None

        ocr_data = st.session_state.get("ocr_data", {})

        with col_ocr:
            st.markdown("##### 🔍 Step 2: Extracted OCR Verification")
            st.caption("Faculty can review and correct any OCR mistakes before final scoring.")

            ocr_student_id = st.text_input("Extracted Student ID:", value=ocr_data.get("student_id") or "MCA001")
            
            # Question identification
            all_qids = list(questions_df['question_id'].unique())
            detected_qid = ocr_data.get("question_id") or "Q1"
            qid_index = all_qids.index(detected_qid) if detected_qid in all_qids else 0

            selected_qid = st.selectbox(
                "Identified Question ID (Select manually if needed):",
                all_qids,
                index=qid_index
            )

            ocr_answer_text = st.text_area(
                "Extracted Student Answer:",
                value=ocr_data.get("student_answer") or "",
                height=130
            )

        st.markdown("---")

        # Step 3 & 4: Reference Fetch & Evaluation
        st.subheader("Step 3 & 4: Reference Matching & Rubric Evaluation")

        ref_details = get_question_details(selected_qid, questions_df)
        subject_name = ref_details.get("subject", "MCA")
        question_title = ref_details.get("question", "")
        reference_answer = ref_details.get("reference_answer", "")

        with st.expander(f"📖 Question Reference: **{selected_qid} — {subject_name}**", expanded=True):
            st.markdown(f"**Prompt:** {question_title}")
            st.markdown(f"**Faculty Reference Answer:** *{reference_answer}*")

        # Run Semantic Comparison & Rubric Scoring
        comparison_res = compare_answers(ocr_answer_text, reference_answer)
        rubric_res = evaluate_rubric(selected_qid, ocr_answer_text, rubric_df)
        feedback_res = generate_feedback(rubric_res, comparison_res, ref_details)

        # Dashboard Top Metrics
        preliminary_score = rubric_res['preliminary_marks']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric(
                label="Preliminary Score",
                value=f"{preliminary_score} / 5.0 Marks",
                delta=f"{len(rubric_res['covered_concepts'])}/5 Concepts"
            )

        with col_m2:
            st.metric(
                label="Overall Semantic Alignment",
                value=f"{comparison_res['overall_similarity']}%",
                help="Blended score: Dense Sentence-Transformer + TF-IDF Cosine Similarity"
            )

        with col_m3:
            st.metric(
                label="TF-IDF Similarity",
                value=f"{comparison_res['tfidf_similarity']}%"
            )

        with col_m4:
            st.metric(
                label="Evaluation Verdict",
                value=feedback_res['grade_category']
            )

        # Concept Breakdown & Visualization
        col_table, col_chart = st.columns([1.3, 1])

        with col_table:
            st.markdown("##### 📋 Concept-wise Rubric Breakdown (1 Mark Each)")
            breakdown_list = []
            for c in rubric_res['concept_breakdown']:
                status = "✅ Covered (+1.0)" if c['is_covered'] else "❌ Missing (0.0)"
                breakdown_list.append({
                    "Concept #": f"Point {c['concept_num']}",
                    "Rubric Concept": c['concept'],
                    "Status": status,
                    "Detection Method": c['method'],
                    "Verification Details": c['reasoning']
                })
            breakdown_df = pd.DataFrame(breakdown_list)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        with col_chart:
            st.markdown("##### 📊 Coverage Analysis")
            fig = go.Figure(go.Pie(
                labels=["Covered Concepts", "Missing Concepts"],
                values=[len(rubric_res['covered_concepts']), len(rubric_res['missing_concepts'])],
                marker_colors=["#4CAF50", "#EF5350"],
                hole=0.5
            ))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=240,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feedback Section
        st.markdown("##### 📝 Intelligent Feedback & Recommendations")
        col_fb1, col_fb2 = st.columns([1, 1])

        with col_fb1:
            st.markdown("**Concepts Correctly Expressed:**")
            if rubric_res['covered_concepts']:
                for c in rubric_res['covered_concepts']:
                    st.markdown(f"<div class='concept-covered'>✓ {c['concept']}</div>", unsafe_allow_html=True)
            else:
                st.caption("No rubric concepts were verified in this answer.")

            st.markdown("<br>**Missing / Incomplete Concepts:**", unsafe_allow_html=True)
            if rubric_res['missing_concepts']:
                for m in rubric_res['missing_concepts']:
                    st.markdown(f"<div class='concept-missing'>✗ {m['concept']}</div>", unsafe_allow_html=True)
            else:
                st.caption("All 5 concepts have been satisfactorily covered.")

        with col_fb2:
            st.info(f"**Faculty Review Advice:**\n\n{feedback_res['faculty_recommendation']}")
            st.markdown("**Actionable Suggestions for Student:**")
            for s in feedback_res['improvement_suggestions']:
                st.markdown(f"- {s}")

        st.markdown("---")

        # Step 9: Faculty Review & Approval
        st.subheader("Step 5: Faculty Review & Mark Modification")
        col_fac1, col_fac2 = st.columns([1, 1])

        with col_fac1:
            final_approved_marks = st.slider(
                "Modify or Confirm Marks (out of 5):",
                min_value=0.0,
                max_value=5.0,
                value=float(preliminary_score),
                step=0.5
            )
            faculty_notes = st.text_input("Faculty Remarks / Notes:", value="Verified concept coverage.")

        with col_fac2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Approve & Save Evaluation", use_container_width=True):
                # Append or record evaluation
                record = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "student_id": ocr_student_id,
                    "question_id": selected_qid,
                    "subject": subject_name,
                    "preliminary_marks": preliminary_score,
                    "approved_marks": final_approved_marks,
                    "covered_count": len(rubric_res['covered_concepts']),
                    "missing_count": len(rubric_res['missing_concepts']),
                    "overall_similarity": comparison_res['overall_similarity'],
                    "faculty_remarks": faculty_notes
                }
                st.success(f"🎉 Evaluation Approved for Student **{ocr_student_id}** on **{selected_qid}** with **{final_approved_marks}/5.0 Marks**!")
                st.json(record)


# MODE 2: 24-Answer Benchmark Validation
elif app_mode == "📊 24-Answer Benchmark Validation":
    st.subheader("Benchmark Validation Suite (All 24 Student Answers)")
    st.markdown("""
    Evaluate the full batch of 24 benchmark answer-sheet images against the rubric without using benchmark labels during evaluation.
    Compare predictions against ground-truth labels to measure **Mean Absolute Error (MAE)** and **Exact Mark Agreement**.
    """)

    if st.button("🚀 Run Full Benchmark Evaluation", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / float(total))
            status_text.text(f"Evaluating answer sheet {current} of {total}...")

        t_start = time.time()
        results_df, metrics = run_benchmark_validation(progress_callback=update_progress)
        elapsed_total = round(time.time() - t_start, 2)
        status_text.success(f"✅ Completed evaluation of all {metrics['total_evaluated']} samples in {elapsed_total}s!")

        # Metric cards
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("Total Answers Evaluated", f"{metrics['total_evaluated']} / 24")
        with col_b2:
            st.metric("Mean Absolute Error (MAE)", f"{metrics['mean_absolute_error']} Marks")
        with col_b3:
            st.metric("Exact Score Agreement", f"{metrics['exact_agreement_pct']}%")
        with col_b4:
            st.metric("Category Agreement", f"{metrics['category_agreement_pct']}%")

        # Interactive Charts
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("##### 🎯 Expected vs. Predicted Marks Comparison")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[f"{r['student_id']}_{r['question_id']}" for _, r in results_df.iterrows()],
                y=results_df['expected_marks'],
                name='Expected Benchmark Marks',
                marker_color='#1E88E5'
            ))
            fig_bar.add_trace(go.Bar(
                x=[f"{r['student_id']}_{r['question_id']}" for _, r in results_df.iterrows()],
                y=results_df['predicted_marks'],
                name='AI Predicted Marks',
                marker_color='#43A047'
            ))
            fig_bar.update_layout(barmode='group', xaxis_tickangle=-45, height=350, margin=dict(l=10, r=10, t=20, b=50))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            st.markdown("##### 📈 Marks Error Distribution (MAE)")
            fig_hist = px.histogram(
                results_df,
                x="mark_difference",
                nbins=6,
                color_discrete_sequence=['#FB8C00'],
                labels={"mark_difference": "Absolute Error (|Predicted - Expected|)"}
            )
            fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=50))
            st.plotly_chart(fig_hist, use_container_width=True)

        # Full results table
        st.markdown("##### 📄 Comprehensive Benchmark Validation Report")
        display_df = results_df[[
            'student_id', 'question_id', 'subject', 'expected_marks', 'predicted_marks',
            'mark_difference', 'exact_match', 'expected_category', 'predicted_category',
            'overall_similarity'
        ]]
        st.dataframe(display_df, use_container_width=True)

        # Download CSV
        csv_bytes = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Benchmark Results Report (CSV)",
            data=csv_bytes,
            file_name="evaluation_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# MODE 3: Dataset & Rubric Explorer
elif app_mode == "📚 Dataset & Rubric Explorer":
    st.subheader("📚 Dataset & Rubric Explorer")

    tab1, tab2, tab3 = st.tabs(["Questions Catalogue", "Marking Rubrics (5 Concepts)", "Student Answers Ground Truth"])

    with tab1:
        st.markdown("##### 8 Reference Questions from MCA Syllabus")
        st.dataframe(questions_df, use_container_width=True)

    with tab2:
        st.markdown("##### 40 Rubric Concepts (5 Concepts per Question, 1 Mark Each)")
        qid_filter = st.selectbox("Filter Rubric by Question ID:", ["All"] + list(questions_df['question_id'].unique()))
        if qid_filter != "All":
            filtered_rubric = rubric_df[rubric_df['question_id'] == qid_filter]
        else:
            filtered_rubric = rubric_df
        st.dataframe(filtered_rubric, use_container_width=True)

    with tab3:
        st.markdown("##### 24 Benchmark Samples for Validation")
        st.dataframe(benchmarks_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Deep Semantics Mini-Hackathon Project | Built with Python, Streamlit, OpenCV, EasyOCR, Sentence Transformers, and Scikit-Learn.")
