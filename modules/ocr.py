"""
Image Preprocessing and OCR Extraction Module (Member 1 Responsibility)
"""
import re
import cv2
import numpy as np
from PIL import Image

# Global EasyOCR reader cache
_EASYOCR_READER = None

def get_easyocr_reader():
    """
    Initializes and caches the EasyOCR Reader instance.
    Uses local project models and user_network directories to ensure access.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import os
            import easyocr
            import config
            model_storage = config.LOCAL_EASYOCR_DIR
            user_network = os.path.join(config.BASE_DIR, "models", "easyocr_network")
            os.makedirs(model_storage, exist_ok=True)
            os.makedirs(user_network, exist_ok=True)
            
            # Initialize for English with CPU using local directories
            _EASYOCR_READER = easyocr.Reader(
                ['en'],
                gpu=False,
                verbose=False,
                model_storage_directory=model_storage,
                user_network_directory=user_network,
                download_enabled=False
            )
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            _EASYOCR_READER = None
    return _EASYOCR_READER

def preprocess_image(image_input):
    """
    Preprocesses the answer-sheet image using OpenCV to improve OCR clarity:
    1. Converts PIL / file path / numpy array to OpenCV BGR
    2. Converts to Grayscale
    3. Noise removal using Gaussian blur
    4. Adaptive thresholding / Otsu thresholding for crisp text contrast
    5. Light dilation/denoising
    """
    # Load image depending on input type
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        # Streamlit UploadedFile
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        image_input.seek(0)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not read or decode image.")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Rescale slightly if image is too small (e.g. width < 1000)
    h, w = gray.shape[:2]
    if w < 1000:
        scale = 1200 / float(w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. Denoise with bilateral filter to preserve edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(denoised, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, gray, thresh

def extract_text_from_image(image_input, use_preprocessing=True):
    """
    Extracts structured fields from an answer-sheet image:
    - student_id
    - question_id
    - question_text
    - student_answer
    - raw_full_text
    - line_count
    """
    orig_img, gray_img, processed_img = preprocess_image(image_input)
    # Printed text images are clean; grayscale yields superior OCR accuracy
    target_img = gray_img if use_preprocessing else orig_img

    extracted_lines = []
    reader = get_easyocr_reader()

    if reader is not None:
        try:
            results = reader.readtext(target_img)
            for bbox, text, conf in results:
                if text and text.strip():
                    extracted_lines.append(text.strip())
        except Exception as e:
            print(f"EasyOCR extraction exception: {e}")

    # Fallback to pytesseract if EasyOCR didn't succeed
    if not extracted_lines:
        try:
            import pytesseract
            raw = pytesseract.image_to_string(gray_img)
            extracted_lines = [line.strip() for line in raw.split('\n') if line.strip()]
        except Exception as e:
            print(f"Tesseract extraction error/not available: {e}")

    raw_full_text = "\n".join(extracted_lines)

    # Parse metadata and answer sections
    parsed_info = parse_ocr_answer_sheet(extracted_lines, raw_full_text)
    parsed_info["raw_full_text"] = raw_full_text
    parsed_info["line_count"] = len(extracted_lines)

    return parsed_info

def parse_ocr_answer_sheet(lines, raw_full_text):
    """
    Parses Student ID, Question ID, Question text, and Student Answer from text lines.
    Handles typical OCR nuances like 'MCAOO1' -> 'MCA001', 'O1' -> 'Q1'.
    """
    student_id = None
    question_id = None
    question_text = ""
    student_answer = ""

    # 1. Regex search for Student ID
    # Patterns: 'Student ID: MCA001', 'Student ID: MCAOO1', 'MCAOO1'
    sid_match = re.search(r'Student\s*ID\s*[:=\-]?\s*([A-Za-z0-9]+)', raw_full_text, re.IGNORECASE)
    if sid_match:
        cand = sid_match.group(1).upper()
        cand = cand.replace('O', '0').replace('I', '1')
        student_id = cand
    else:
        mca_match = re.search(r'\b(MCA\s*[0-9OI]{3,})\b', raw_full_text, re.IGNORECASE)
        if mca_match:
            cand = mca_match.group(1).replace(" ", "").upper().replace('O', '0').replace('I', '1')
            student_id = cand

    # 2. Regex search for Question ID
    qid_match = re.search(r'Question\s*(?:ID|Number|No)?\s*[:=\-]?\s*([Qq][0-9])\b', raw_full_text, re.IGNORECASE)
    if qid_match:
        question_id = qid_match.group(1).upper()
    else:
        q_standalone = re.search(r'\b(Q[1-8])\b', raw_full_text, re.IGNORECASE)
        if q_standalone:
            question_id = q_standalone.group(1).upper()

    # 3. Line-by-line parsing for Question Text and Student Answer
    in_question_section = False
    in_answer_section = False
    question_parts = []
    answer_parts = []

    for line in lines:
        cleaned_line = line.strip()
        lower_line = cleaned_line.lower()

        # Stop markers
        if "end of answer" in lower_line:
            in_answer_section = False
            continue

        # Check for answer section header: MUST NOT match 'STUDENT ANSWER SHEET'
        if re.search(r'\bstudent\s*answer\s*[:=\-]?', lower_line) and "sheet" not in lower_line:
            in_question_section = False
            in_answer_section = True
            after_header = re.split(r'student\s*answer\s*[:=\-]?', cleaned_line, flags=re.IGNORECASE)
            if len(after_header) > 1 and after_header[1].strip():
                ans_text = after_header[1].strip()
                if "end of answer" not in ans_text.lower():
                    answer_parts.append(ans_text)
            continue

        # Check for question header: "Question: Q1" or "Question:"
        if lower_line.startswith("question:") or lower_line == "question":
            in_question_section = True
            after_q = re.split(r'question\s*[:=\-]?', cleaned_line, flags=re.IGNORECASE)
            remainder = after_q[1].strip() if len(after_q) > 1 else ""
            if remainder:
                if re.match(r'^[Qq][0-9]$', remainder):
                    if not question_id:
                        question_id = remainder.upper()
                else:
                    question_parts.append(remainder)
            continue

        if in_answer_section:
            answer_parts.append(cleaned_line)
        elif in_question_section:
            # Avoid student id / headers
            if not any(hdr in lower_line for hdr in ['student id', 'mca department', 'student answer sheet']):
                question_parts.append(cleaned_line)

    if answer_parts:
        student_answer = " ".join(answer_parts).strip()
    else:
        # Fallback regex if header-based parse failed: find text after Student Answer (excluding Sheet)
        match_ans = re.search(r'student\s*answer\s*[:=\-](.*?)(?:end of answer|$)', raw_full_text, re.IGNORECASE | re.DOTALL)
        if match_ans:
            student_answer = match_ans.group(1).strip()
        else:
            student_answer = raw_full_text

    if question_parts:
        question_text = " ".join(question_parts).strip()

    # Clean up student answer
    student_answer = re.sub(r'[\r\n]+', ' ', student_answer).strip()
    student_answer = re.sub(r'\s+', ' ', student_answer)
    student_answer = re.sub(r'[:=\-]+$', '', student_answer).strip()

    return {
        "student_id": student_id,
        "question_id": question_id,
        "question_text": question_text,
        "student_answer": student_answer
    }
