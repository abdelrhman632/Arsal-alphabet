import modal

app = modal.App("arsal-alphabet-letters")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libegl1", "libgles2")
    .pip_install_from_requirements("Letters/requirements.txt")
    .add_local_file("Letters/app.py", remote_path="/root/Letters/app.py")
    .add_local_file(
        "Letters/arasl_alphabet_model_2.pkl",
        remote_path="/root/Letters/arasl_alphabet_model_2.pkl",
    )
)


@app.function(image=image, gpu=None, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def gradio_app():
    import os
    import sys
    from fastapi import FastAPI
    import gradio as gr

    letters_dir = "/root/Letters"
    sys.path.insert(0, letters_dir)
    os.chdir(letters_dir)

    from app import demo

    api = FastAPI()
    return gr.mount_gradio_app(api, demo, path="")