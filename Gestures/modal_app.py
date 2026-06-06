import modal

app = modal.App("arsal-gestures")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "ffmpeg",
    )
    .pip_install_from_requirements("Gestures/requirements.txt")
    .add_local_file("Gestures/app.py", remote_path="/root/Gestures/app.py")
    .add_local_file("Gestures/predict_video.py", remote_path="/root/Gestures/predict_video.py")
    .add_local_file(
        "Gestures/arabic_sign_lstm_model.h5",
        remote_path="/root/Gestures/arabic_sign_lstm_model.h5",
    )
    .add_local_file(
        "Gestures/holistic_landmarker.task",
        remote_path="/root/Gestures/holistic_landmarker.task",
    )
    .add_local_file(
        "Gestures/label_encoder.pkl",
        remote_path="/root/Gestures/label_encoder.pkl",
    )
)


@app.function(image=image, gpu=None, timeout=600)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def gradio_app():
    import os
    import sys
    from fastapi import FastAPI
    import gradio as gr

    gestures_dir = "/root/Gestures"
    sys.path.insert(0, gestures_dir)
    os.chdir(gestures_dir)

    from app import app as gradio_demo

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, gradio_demo, path="/")