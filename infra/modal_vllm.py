"""OpenAI-compatible LLM endpoint for the AI Scientist, served by vLLM on a Modal GPU.

Deploy:   modal secret create ai-scientist-vllm VLLM_API_KEY=<random token>   (once)
          modal deploy infra/modal_vllm.py
Test:     modal run infra/modal_vllm.py
The deploy prints the URL (https://<workspace>--ai-scientist-llm-serve.modal.run); put it in .env as
MODAL_LLM_URL together with MODAL_LLM_API_KEY=<the same token>, then run the RSI loop with --driver modal.
"""
import os
import subprocess

import modal

MODEL_NAME = os.environ.get("AI_SCIENTIST_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
SERVED_NAME = "llm"
GPU = os.environ.get("AI_SCIENTIST_GPU", "H100")
MAX_MODEL_LEN = int(os.environ.get("AI_SCIENTIST_MAX_LEN", "65536"))
PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "AI_SCIENTIST_MODEL": MODEL_NAME, "AI_SCIENTIST_MAX_LEN": str(MAX_MODEL_LEN)})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("ai-scientist-llm")


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("ai-scientist-vllm")],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve():
    cmd = ["vllm", "serve", MODEL_NAME, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", str(PORT),
           "--api-key", os.environ["VLLM_API_KEY"], "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", "0.92", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def smoke_test():
    import time
    import urllib.request

    url = serve.get_web_url()
    print("endpoint:", url)
    for _ in range(80):
        try:
            with urllib.request.urlopen(url + "/health", timeout=30) as r:
                if r.status == 200:
                    print("healthy")
                    return
        except Exception:
            time.sleep(15)
    raise SystemExit("server did not become healthy")
