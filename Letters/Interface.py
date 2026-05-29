import cv2
import pickle
import mediapipe as mp
import numpy as np

# =========================
# MODE SELECTION
# =========================

print("\n==== Sign Language System ====")
print("1 -> Alphabet Letters")
print("2 -> Gestures")

choice = input("Select Mode: ")

if choice == "1":
    MODEL_PATH = r"C:\Users\Abdelrhman Yasser\Desktop\Uni\Term 8\NN\ArASL\arasl_alphabet_model_2.pkl"
    MODE_NAME = "Letters"

# elif choice == "2":
#     MODEL_PATH = "gesture_model.pkl"   # future model
#     MODE_NAME = "Gestures"

else:
    print("Invalid Choice")
    exit()



with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"{MODE_NAME} model loaded successfully.")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils



cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb)

    prediction_text = "No Hand Detected"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )



            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Convert to numpy
            features = np.array(landmarks).reshape(1, -1)



            prediction = model.predict(features)

            prediction_text = str(prediction[0])


    cv2.putText(
        frame,
        f"{MODE_NAME}: {prediction_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ArASL System", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()