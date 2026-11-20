MCA Intelligent Assignment Evaluation - Mini Hackathon Dataset

Contents
- questions.csv: questions and faculty reference answers
- marking_rubric.csv: five concept points per question, one mark each
- student_answers.csv: 24 sample answers (3 per question), image path, validation answer text, and expected benchmark marks
- answer_images/: 24 corresponding answer-sheet images (clean printed text for OCR practice)

Suggested 2-hour task
1. Load a question and reference answer from questions.csv.
2. Upload a corresponding image from answer_images/.
3. Extract text using OCR (EasyOCR or Tesseract).
4. Compare extracted text against the reference answer and rubric concepts.
5. Calculate preliminary marks out of 5 and generate feedback.
6. Test against student_answers.csv.

Important
- The expected_marks and expected_category columns are benchmark labels for testing only. Do not use them as model inputs.
- The benchmark marks are illustrative, based on concept coverage, and are not a universal grading standard.
- TF-IDF/cosine similarity alone does not establish correctness. Use rubric/concept matching and allow faculty review.
- Images are printed-text samples to make OCR testing reliable within two hours. Handwriting recognition can be added as an extension.
