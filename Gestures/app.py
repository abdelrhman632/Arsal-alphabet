from pathlib import Path

import gradio as gr

from predict_video import predict_video


def classify_video(video_file):
    if video_file is None:
        return "Please upload a video."

    label, confidence = predict_video(
        video_path=Path(video_file),
        show_video=False
    )

    if confidence == 0.0:
        return label

    return f"Predicted Gesture: {label}\nConfidence: {confidence:.2%}"


app = gr.Interface(
    fn=classify_video,
    inputs=gr.Video(label="Upload a sign language gesture video"),
    outputs=gr.Textbox(label="Prediction"),
    title="Arabic Sign Language Gesture Detection",
    description=(
        "Upload a short Arabic sign language video. "
        "The system extracts landmarks and predicts the gesture using the trained LSTM model."
    ),
)


if __name__ == "__main__":
    app.launch()