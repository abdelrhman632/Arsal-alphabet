import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands

IMAGE_FOLDER = r"C:\Users\Abdelrhman Yasser\Downloads\ArASL\ArASLDatabase54KFinal"

data = []

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.1) as hands:

    for label in os.listdir(IMAGE_FOLDER):

        class_path = os.path.join(IMAGE_FOLDER, label)

        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {label}")

        for img_file in os.listdir(class_path):

            img_path = os.path.join(class_path, img_file)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = results.multi_hand_landmarks[0]

            row = []

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            data.append(row)

# Build dataframe
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

df = pd.DataFrame(data, columns=columns)

df.to_csv("arasl_landmarks.csv", index=False)

print("Finished extracting landmarks.")
print("Dataset shape:", df.shape)