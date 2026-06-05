from pathlib import Path
import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent

MAX_FRAMES = 60
FEATURES_PER_FRAME = 134

HOLISTIC_MODEL_PATH = BASE_DIR / "holistic_landmarker.task"
LSTM_MODEL_PATH = BASE_DIR / "arabic_sign_lstm_model.h5"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


def get_landmarks(landmarks):
    if not landmarks:
        return None

    if isinstance(landmarks, list):
        if len(landmarks) == 0:
            return None

        if isinstance(landmarks[0], list):
            return landmarks[0]

        return landmarks

    return None


def extract_features(results):
    row = []

    face = get_landmarks(results.face_landmarks)
    pose = get_landmarks(results.pose_landmarks)
    left_hand = get_landmarks(results.left_hand_landmarks)
    right_hand = get_landmarks(results.right_hand_landmarks)

    # Face: forehead and chin
    if face and len(face) > 152:
        row += [
            face[10].x, face[10].y, face[10].z,
            face[152].x, face[152].y, face[152].z,
        ]
    else:
        row += [0.0] * 6

    # Torso: midpoint between shoulders
    if pose and len(pose) > 12:
        mid_x = (pose[11].x + pose[12].x) / 2
        mid_y = (pose[11].y + pose[12].y) / 2
        row += [mid_x, mid_y]
    else:
        row += [0.0] * 2

    # Left hand: 21 landmarks × x,y,z
    if left_hand and len(left_hand) == 21:
        for lm in left_hand:
            row += [lm.x, lm.y, lm.z]
    else:
        row += [0.0] * 63

    # Right hand: 21 landmarks × x,y,z
    if right_hand and len(right_hand) == 21:
        for lm in right_hand:
            row += [lm.x, lm.y, lm.z]
    else:
        row += [0.0] * 63

    return row


def load_detector():
    if not HOLISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HOLISTIC_MODEL_PATH.name}. "
            "Place holistic_landmarker.task inside the Gestures folder."
        )

    base_options = python.BaseOptions(
        model_asset_path=str(HOLISTIC_MODEL_PATH)
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.4,
        min_face_detection_confidence=0.4,
        min_hand_landmarks_confidence=0.4,
    )

    return vision.HolisticLandmarker.create_from_options(options)


def predict_video(video_path, show_video=False):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LSTM_MODEL_PATH.name}. "
            "Place arabic_sign_lstm_model.h5 inside the Gestures folder."
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LABEL_ENCODER_PATH.name}. "
            "Place label_encoder.pkl inside the Gestures folder."
        )

    model = load_model(LSTM_MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    detector = load_detector()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    sequence = []

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        try:
            results = detector.detect_for_video(mp_image, timestamp_ms)
        except RuntimeError as e:
            print(f"MediaPipe failed on frame: {e}")
            continue

        left_hand = get_landmarks(results.left_hand_landmarks)
        right_hand = get_landmarks(results.right_hand_landmarks)

        # Only collect frames where at least one hand is detected
        if left_hand or right_hand:
            features = extract_features(results)

            if len(features) == FEATURES_PER_FRAME:
                sequence.append(features)
            else:
                print("Wrong feature length:", len(features))

        if show_video:
            # Optional drawing for visualization
            pose = get_landmarks(results.pose_landmarks)

            if pose:
                for idx in [11, 12]:
                    lm = pose[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

            face = get_landmarks(results.face_landmarks)

            if face:
                for idx in [10, 152]:
                    lm = face[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

            for hand_data in [results.left_hand_landmarks, results.right_hand_landmarks]:
                hand = get_landmarks(hand_data)

                if hand:
                    for idx in range(21):
                        lm = hand[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            cv2.imshow("Gesture Prediction Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    detector.close()

    if show_video:
        cv2.destroyAllWindows()

    if len(sequence) == 0:
        return "No hand frames detected. Cannot predict.", 0.0

    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_FRAMES,
        dtype="float32",
        padding="post",
        truncating="post",
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(prediction[0][predicted_index])

    return predicted_label, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict Arabic sign language gesture from a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the video while processing."
    )

    args = parser.parse_args()

    label, confidence = predict_video(
        video_path=args.video,
        show_video=args.show,
    )

    print("Predicted sign:", label)
    print("Confidence:", f"{confidence:.2%}")


if __name__ == "__main__":
    main()