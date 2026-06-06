import modal

app = modal.App("arsal-alphabet-letters")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("Letters/requirements.txt")
)


@app.function(
    image=image,
    gpu=None,
    timeout=600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def gradio_app():
    import os
    import sys

    letters_dir = "/root/Letters"
    if letters_dir not in sys.path:
        sys.path.insert(0, letters_dir)

    os.chdir(letters_dir)

    from app import demo

    return demo.app
