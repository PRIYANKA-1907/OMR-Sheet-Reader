# src/omr/omr_utils.py

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess an OMR image.
    Accepts:
        - image: str (file path) or NumPy array (OpenCV image)
    Returns:
        - original image (BGR)
        - thresholded image
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image  # Already a NumPy array (OpenCV image)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )
    return img, thresh


def detect_bubbles(thresh_img, questions=20, options=4):
    """
    Detect filled bubbles in thresholded image.
    Returns a list of selected options as strings ('1', '2', '3', '4').
    """
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    bubble_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 2000:  # Adjust based on bubble size
            bubble_contours.append(c)

    # Sort top-to-bottom, left-to-right
    bubble_contours = sorted(
        bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
    )
    
    answers = []
    for q in range(questions):
        start = q * options
        end = start + options
        question_bubbles = bubble_contours[start:end]
        
        filled = None
        max_fill = 0
        for i, c in enumerate(question_bubbles):
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh_img, thresh_img, mask=mask))
            if total > max_fill:
                max_fill = total
                filled = i
        
        answers.append(str(filled + 1))  # '1', '2', '3', or '4'
    
    return answers


def classify_bubbles(thresh_img, questions=20, options=4):
    """
    Wrapper to detect bubbles, same as detect_bubbles.
    """
    return detect_bubbles(thresh_img, questions, options)


def map_to_answers(detected_bubbles, layout):
    """
    Map detected bubble answers to subjects based on layout.
    layout: dict with subject name and number of questions
    Example:
        layout = {"Math": 20, "Science": 20}
    """
    answers = {}
    idx = 0
    for subject, q_count in layout.items():
        answers[subject] = detected_bubbles[idx: idx + q_count]
        idx += q_count
    return answers


def calculate_score(extracted_answers, answer_key):
    """
    Compare extracted answers with the answer key.
    Returns total score and per-subject score.
    """
    score = 0
    for sub, key in answer_key.items():
        ans = extracted_answers.get(sub, [])
        sub_score = sum([1 for a, k in zip(ans, key) if a == k])
        score += sub_score
        print(f"{sub} Score: {sub_score}/{len(key)}")
    print(f"Total Score: {score}")
    return score
