import streamlit as st
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image

model = joblib.load("arasl_alphabet_model_2.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

st.title("Arabic Sign Language Detection")
st.write("Upload an image containing an Arabic sign language letter.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image")

    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        st.error("No hand detected")
    else:
        hand_landmarks = results.multi_hand_landmarks[0]
        features = []

        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]

        st.success(f"Predicted Letter: {prediction}")