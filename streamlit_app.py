import streamlit as st
from PIL import Image
from src.omr.omr_utils import preprocess_omr, detect_bubbles, classify_bubbles, map_to_answers

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
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)

    # ------------------------------
    # Preprocess Image
    # ------------------------------
    processed_img = preprocess_omr(image)

    # ------------------------------
    # Detect & Classify Bubbles
    # ------------------------------
    bubbles = detect_bubbles(processed_img)
    classified_bubbles = classify_bubbles(processed_img, bubbles)

    # ------------------------------
    # Map Bubbles to Answers
    # ------------------------------
    student_answers = map_to_answers(classified_bubbles)

    # ------------------------------
    # Display Extracted Answers
    # ------------------------------
    st.subheader("Extracted Answers from OMR Sheet")
    for subject, answers in student_answers.items():
        st.write(f"**{subject}**: {answers}")
