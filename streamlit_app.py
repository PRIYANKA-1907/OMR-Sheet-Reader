import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.omr.omr_utils import preprocess_image, classify_bubbles, map_to_answers, calculate_score

# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(page_title="OMR Sheet Reader", layout="wide")
st.title("OMR Sheet Reader - Extract Answers from Sheet")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload OMR Sheet Image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded OMR Sheet", use_container_width=True)

    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # ------------------------------
    # Preprocess Image
    # ------------------------------
    processed_img, thresh = preprocess_image(image_cv)
    st.image(thresh, caption="Thresholded Image", use_container_width=True)

    # ------------------------------
    # Detect & Classify Bubbles
    # ------------------------------
    detected_bubbles = classify_bubbles(thresh, questions=40, options=4)  # Adjust questions/options as per sheet

    # ------------------------------
    # Map Bubbles to Answers
    # ------------------------------
    layout = {"Math": 20, "Science": 20}  # Adjust according to your sheet
    student_answers = map_to_answers(detected_bubbles, layout)

    # ------------------------------
    # Display Extracted Answers
    # ------------------------------
    st.subheader("Extracted Answers from OMR Sheet")
    for subject, answers in student_answers.items():
        st.write(f"**{subject}**: {answers}")

    # ------------------------------
    # Calculate Score
    # ------------------------------
    answer_key = {
        "Math": ['1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4'],
        "Science": ['2','1','3','4','2','1','3','4','2','1','3','4','2','1','3','4','2','1','3','4']
    }
    total_score = calculate_score(student_answers, answer_key)
    st.write(f"**Total Score: {total_score}**")
