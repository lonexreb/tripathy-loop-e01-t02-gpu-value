"""Optional Modal boundary. Install the `modal` extra before deployment."""

import modal

APP_NAME = "tripathy-loop-e01-t02-gpu-value"
app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(".")


@app.function(image=image, timeout=300)
def smoke() -> dict[str, str]:
    return {"app": APP_NAME, "status": "ok"}
