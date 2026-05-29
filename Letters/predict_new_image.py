import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load(r"C:\Users\Abdelrhman Yasser\Downloads\ArASL\arasl_alphabet_model_2.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

img_path = r"C:\Users\Abdelrhman Yasser\Downloads\3.png"
original = cv2.imread(img_path)

if original is None:
    print("Image not found.")
    exit()

# Different preprocessing attempts
variants = []

# original resized
img1 = cv2.resize(original, (700,700))
variants.append(img1)

# equalized grayscale
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
variants.append(img2)

# flipped
img3 = cv2.flip(img1, 1)
variants.append(img3)

# flipped equalized
img4 = cv2.flip(img2, 1)
variants.append(img4)

detected = False

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1) as hands:

    for image in variants:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features).reshape(1, -1)

        pred = model.predict(features)[0]

        print("Predicted Arabic Letter:", pred)

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        detected = True
        break

if not detected:
    print("No hand detected after all attempts.")