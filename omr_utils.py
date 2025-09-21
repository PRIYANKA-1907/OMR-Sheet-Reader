import cv2
import numpy as np

def preprocess_image(image):
    """Convert to grayscale and threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, thresh

def classify_bubbles(thresh_image, questions=40, options=4):
    """Dummy bubble detection: return random answers for testing"""
    # Replace with actual detection logic
    detected = []
    for _ in range(questions):
        detected.append(np.random.randint(1, options+1))
    return detected

def map_to_answers(detected_bubbles, layout):
    """Map detected bubbles to subjects"""
    student_answers = {}
    idx = 0
    for subject, q_count in layout.items():
        student_answers[subject] = [str(b) for b in detected_bubbles[idx:idx+q_count]]
        idx += q_count
    return student_answers

def calculate_score(student_answers, answer_key):
    """Simple scoring"""
    total = 0
    for subject, key in answer_key.items():
        answers = student_answers.get(subject, [])
        for a,b in zip(answers, key):
            if a == b:
                total += 1
    return total
