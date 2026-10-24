"""Self-hosted DeepSeek V4 Flash on Modal: an OpenAI-compatible vLLM endpoint for the experiments.

One-time setup (from the repo root, with the Modal CLI logged in):

    modal secret create groupcollusion-llm-key LLM_API_KEY=<random string>   # same value as MODAL_LLM_KEY in .env
    modal run modal_app/deepseek.py::download                                # ~160 GB of weights into a Volume
    modal deploy modal_app/deepseek.py                                       # prints the endpoint URL

Then put MODAL_LLM_URL=<that URL>/v1 and MODAL_LLM_KEY=<the key> in .env; runs with llm.provider=vllm (the
default) go there. The server scales to zero after SCALEDOWN_S idle seconds, so an idle endpoint costs nothing,
but the first call after that waits for a cold start (loading the weights takes several minutes).

Settings follow the vLLM recipe for DeepSeek-V4-Flash (https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash):
data parallel + expert parallel over 4 GPUs, deepseek_v4 tokenizer mode and reasoning parser. The checkpoint stores
MoE experts in FP4 and most other weights in FP8, so Blackwell (B200) GPUs run it natively.
"""
import os
import subprocess

import modal

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"  # pinned so every run uses the same weights
SERVED_NAME = "deepseek/deepseek-v4-flash"  # the name the configs already use on OpenRouter
N_GPUS = 4
GPU = f"B200:{N_GPUS}"  # about $25/h while a container is up (B200 $6.25/h on Modal, Oct 2026)
MAX_CONTAINERS = 2  # each container is one full copy of the model; raise for big sweeps
SCALEDOWN_S = 600  # idle seconds before a container shuts down
PORT = 8000

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0")
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")  # Modal looks for `python` to detect the version
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_USE_V1": "1",
        # loading 160 GB from the Volume takes longer than vLLM's default 600 s engine start limit
        "VLLM_ENGINE_READY_TIMEOUT_S": "2700",
    })
)
hf_cache = modal.Volume.from_name("groupcollusion-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("groupcollusion-vllm-cache", create_if_missing=True)
volumes = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}

app = modal.App("groupcollusion-deepseek")


@app.function(image=image, volumes=volumes, cpu=8, memory=32768, timeout=4 * 3600)
def download():
    """Fetch the pinned checkpoint into the shared Volume (CPU only; run once)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=REVISION, max_workers=32)
    hf_cache.commit()
    print("weights at", path)


@app.function(
    image=image,
    gpu=GPU,
    cpu=32,
    memory=256 * 1024,  # RAM for eager safetensors reads (one shard file at a time per rank)
    volumes=volumes,
    secrets=[modal.Secret.from_name("groupcollusion-llm-key")],
    timeout=24 * 3600,
    scaledown_window=SCALEDOWN_S,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=1000, target_inputs=500)  # 1000 is Modal's cap; vLLM batches internally
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--revision", REVISION,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["LLM_API_KEY"],
        "--tokenizer-mode", "deepseek_v4",
        "--reasoning-parser", "deepseek_v4",
        "--data-parallel-size", str(N_GPUS),
        "--enable-expert-parallel",
        "--kv-cache-dtype", "fp8",
        "--max-model-len", "65536",  # prompts are a few thousand tokens; completions are capped at 32k
        "--gpu-memory-utilization", "0.9",
        "--safetensors-load-strategy", "eager",  # whole-file reads; much faster than mmap on a network Volume
    ]
    print(" ".join(c if c != os.environ["LLM_API_KEY"] else "<key>" for c in cmd), flush=True)
    subprocess.Popen(cmd)
