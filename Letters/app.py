import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("arasl_alphabet_model_2.pkl")

# =========================
# MEDIAPIPE HANDS
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# =========================
# PREDICTION FUNCTION
# =========================

def predict(image):

    if image is None:
        return "No image uploaded"

    # Gradio gives RGB image
    image_rgb = image.copy()

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "No hand detected"

    hand_landmarks = results.multi_hand_landmarks[0]

    features = []

    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return f"Predicted Letter: {prediction}"

# =========================
# GRADIO INTERFACE
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Textbox(label="Prediction"),
    title="Arabic Sign Language Detection",
    description="Upload an image containing an Arabic sign language letter."
)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    demo.launch()