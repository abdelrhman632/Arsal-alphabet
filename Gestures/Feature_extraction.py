import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# --- CONFIGURATION ---
DATASET_PATH = r"Main Code\Dataset\Videos\train"
MODEL_PATH = r"C:\Users\Mazen\Desktop\Obsidian_synthing\UNI\NN project\Main Code\holistic_landmarker.task"
OUTPUT_CSV = "sign_data.csv"

# --- CSV HEADER ---
header = [
    'label', 'video_id', 'frame_id',
    'forehead_x', 'forehead_y', 'forehead_z',
    'chin_x', 'chin_y', 'chin_z',
    'torso_x', 'torso_y'
]

for hand in ['lh', 'rh']:
    for i in range(21):
        header += [f'{hand}_{i}_x', f'{hand}_{i}_y', f'{hand}_{i}_z']

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,

)

detector = vision.HolisticLandmarker.create_from_options(options)

# --- GLOBAL TIMESTAMP (NEVER RESETS) ---
global_timestamp_ms = 0

video_id_counter = 0

with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n📂 CATEGORY: {folder_name}")

        for video_file in os.listdir(folder_path):
            if not video_file.lower().endswith('.mp4'):
                continue

            video_full_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_full_path)

            if not cap.isOpened():
                print(f"❌ Failed to open {video_file}")
                continue

            video_id_counter += 1
            frame_id = 0

            # FPS-based timing (more accurate than fixed 33ms)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            frame_time_ms = int(1000 / fps)

            print(f"  🎥 Processing: {video_file} (FPS: {fps:.2f})")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # --- Resize for consistency ---
                frame = cv2.resize(frame, (1920, 1080))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # --- GLOBAL TIMESTAMP ---
                timestamp_ms = global_timestamp_ms
                global_timestamp_ms += frame_time_ms

                results = detector.detect_for_video(mp_image, timestamp_ms)

                current_row = [folder_name, video_id_counter, frame_id]

                # ---------------- FACE (2 points) ----------------
                face = results.face_landmarks
                if face and len(face) > 152:
                    current_row += [
                        face[10].x, face[10].y, face[10].z,
                        face[152].x, face[152].y, face[152].z
                    ]
                else:
                    current_row += [0.0] * 6

                # ---------------- TORSO ----------------
                pose = results.pose_landmarks
                if pose and len(pose) > 12:
                    mid_x = (pose[11].x + pose[12].x) / 2
                    mid_y = (pose[11].y + pose[12].y) / 2
                    current_row += [mid_x, mid_y]
                else:
                    current_row += [0.0] * 2

                # ---------------- HANDS ----------------
                hands_detected = False

                for hand_landmarks in [
                    results.left_hand_landmarks,
                    results.right_hand_landmarks
                ]:
                    if hand_landmarks and len(hand_landmarks) == 21:
                        hands_detected = True
                        for lm in hand_landmarks:
                            current_row += [lm.x, lm.y, lm.z]
                    else:
                        current_row += [0.0] * 63

                # ---------------- OPTIONAL FILTER ----------------
                # Skip frames with no hands (recommended for sign language)
                if not hands_detected:
                    frame_id += 1
                    continue

                # ---------------- SANITY CHECK ----------------
                if len(current_row) != len(header):
                    print(f"⚠️ Length mismatch at frame {frame_id}: {len(current_row)} vs {len(header)}")

                if frame_id % 30 == 0:
                    print(
                        f"    Frame {frame_id} | "
                        f"Face: {len(face) if face else 0} | "
                        f"Pose: {len(pose) if pose else 0} | "
                        f"LH: {len(results.left_hand_landmarks) if results.left_hand_landmarks else 0} | "
                        f"RH: {len(results.right_hand_landmarks) if results.right_hand_landmarks else 0}"
                    )

                writer.writerow(current_row)
                frame_id += 1

            cap.release()

detector.close()

print("\n✅ Processing complete. CSV saved.")