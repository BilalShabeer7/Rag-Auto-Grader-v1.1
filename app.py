"""
RAG-Based Student Work Auto-Grader
Teacher-Friendly Rubric Input (PDF / DOCX / TXT / Paste)
NO JSON exposure
"""

import streamlit as st
import re
from typing import Dict, List, Optional
from PyPDF2 import PdfReader
import docx2txt
import numpy as np

# ======================================================
# FILE READERS
# ======================================================

def read_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )

    if uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        return docx2txt.process(uploaded_file)

    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")

    return ""


# ======================================================
# RUBRIC PARSER (CRITICAL FIX)
# ======================================================

def parse_rubric(text: str) -> Optional[Dict]:
    """
    Parses IELTS-style rubric tables safely.
    Expected delimiter: |
    """

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None

    header = re.split(r"\s*\|\s*", lines[0].lower())
    if not header or "band" not in header[0]:
        return None

    bands = []

    for line in lines[1:]:
        parts = re.split(r"\s*\|\s*", line)
        if len(parts) < 5:
            continue

        try:
            band = int(parts[0])
        except ValueError:
            continue

        bands.append({
            "band": band,
            "task": parts[1],
            "coherence": parts[2],
            "lexical": parts[3],
            "grammar": parts[4]
        })

    if not bands:
        return None

    return {
        "type": "ielts_band_rubric",
        "bands": sorted(bands, key=lambda x: x["band"], reverse=True),
        "criteria_weights": {
            "task": 0.4,
            "coherence": 0.2,
            "lexical": 0.2,
            "grammar": 0.2
        }
    }


# ======================================================
# SEMANTIC SCORING (PLACEHOLDER – RAG READY)
# ======================================================

def semantic_similarity(a: str, b: str) -> float:
    """
    Lightweight similarity approximation.
    Replace with embeddings / RAG later.
    """
    if not a or not b:
        return 0.0

    a_words = set(a.lower().split())
    b_words = set(b.lower().split())

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    return intersection / union if union else 0.0


# ======================================================
# GRADING ENGINE
# ======================================================

def grade_with_rubric(
    rubric: Dict,
    model_answer: str,
    student_answer: str
) -> Dict:

    sim = semantic_similarity(model_answer, student_answer)
    length = len(student_answer.split())

    # Heuristic + semantic fusion
    base_score = (
        0.6 * sim +
        0.4 * min(1.0, length / 250)
    )

    # Map score to band
    for band in rubric["bands"]:
        threshold = band["band"] / 9
        if base_score >= threshold:
            return {
                "band": band["band"],
                "reasoning": band
            }

    return {
        "band": 4,
        "reasoning": rubric["bands"][-1]
    }


def default_grade(student_answer: str) -> Dict:
    length = len(student_answer.split())

    if length > 280:
        band = 7
    elif length > 200:
        band = 6
    elif length > 120:
        band = 5
    else:
        band = 4

    return {
        "band": band,
        "reasoning": "Default grading applied (no rubric)."
    }


# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config("RAG Auto-Grader", layout="wide")
st.title("📝 RAG-Based Student Work Auto-Grader")

# ------------------ Rubric ------------------
st.header("📊 Rubric (Teacher-Friendly)")

rubric_file = st.file_uploader(
    "Upload rubric (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"]
)

rubric_paste = st.text_area(
    "Or paste rubric table",
    height=200,
    placeholder="Band | Task Response | Coherence & Cohesion | Lexical Resource | Grammar"
)

rubric_text = rubric_paste.strip()
if not rubric_text:
    rubric_text = read_file(rubric_file)

rubric = None
if rubric_text:
    rubric = parse_rubric(rubric_text)
    if rubric:
        st.success("✅ Rubric parsed successfully")
        st.dataframe(rubric["bands"])
    else:
        st.warning("⚠️ Rubric detected but could not be parsed. Default grading will be used.")

# ------------------ Answers ------------------
st.header("📘 Model Answer")

model_file = st.file_uploader(
    "Upload model answer (optional)",
    type=["pdf", "docx", "txt"],
    key="model"
)

model_answer = st.text_area(
    "Or paste model answer",
    height=180
)

if not model_answer:
    model_answer = read_file(model_file)

st.header("🧑‍🎓 Student Answer")

student_file = st.file_uploader(
    "Upload student answer",
    type=["pdf", "docx", "txt"],
    key="student"
)

student_answer = st.text_area(
    "Or paste student answer",
    height=180
)

if not student_answer:
    student_answer = read_file(student_file)

# ------------------ Grade ------------------
if st.button("🚀 Grade Answer"):

    if not student_answer.strip():
        st.error("Student answer is required.")
    else:
        if rubric:
            result = grade_with_rubric(rubric, model_answer, student_answer)
        else:
            result = default_grade(student_answer)

        st.subheader("✅ Result")
        st.metric("Band Score", result["band"])

        if isinstance(result["reasoning"], dict):
            st.markdown("### 📌 Rubric Match")
            st.json(result["reasoning"], expanded=False)
        else:
            st.info(result["reasoning"])
# ======================================================
# FEEDBACK GENERATION
# ======================================================

def generate_feedback(
    rubric: Optional[Dict],
    band: int,
    student_answer: str
) -> str:
    """
    Generates human-readable feedback using rubric descriptors.
    """

    if not rubric:
        return (
            "Your response was evaluated using default criteria. "
            "To improve, focus on expanding ideas, improving coherence, "
            "and reducing grammatical errors."
        )

    band_info = next(
        (b for b in rubric["bands"] if b["band"] == band),
        None
    )

    if not band_info:
        return "Feedback could not be generated for this band."

    feedback = f"""
**Task Response:** {band_info['task']}

**Coherence & Cohesion:** {band_info['coherence']}

**Lexical Resource:** {band_info['lexical']}

**Grammar:** {band_info['grammar']}

**Improvement Tip:**  
To move to the next band, aim to expand your ideas further, use more precise vocabulary, 
and improve sentence variety and grammatical accuracy.
"""
    return feedback.strip()


# ======================================================
# DISPLAY FINAL FEEDBACK
# ======================================================

if "result" in locals() and result:

    st.markdown("---")
    st.subheader("🧠 Detailed Feedback")

    detailed_feedback = generate_feedback(
        rubric if rubric else None,
        result["band"],
        student_answer
    )

    st.markdown(detailed_feedback)


# ======================================================
# OPTIONAL: DOWNLOAD REPORT
# ======================================================

def build_report(band: int, feedback: str) -> str:
    return f"""
IELTS Writing Evaluation Report
-------------------------------
Final Band: {band}

Feedback:
{feedback}
"""


if "result" in locals() and result:
    report_text = build_report(result["band"], detailed_feedback)

    st.download_button(
        label="📥 Download Feedback Report",
        data=report_text,
        file_name="grading_report.txt",
        mime="text/plain"
    )


# ======================================================
# APP FOOTER
# ======================================================

st.markdown(
    """
    <hr>
    <center>
    <small>
    AI-Assisted Auto-Grader • Teacher-Friendly • No JSON • RAG-Ready
    </small>
    </center>
    """,
    unsafe_allow_html=True
)
