# Dockerfile for training on an Advantech AGX Thor (aarch64, Jetson Thor Blackwell).
#
# Build on the Thor itself:
#   docker build -f docker/Dockerfile.thor -t p2p:thor .
# Run:
#   docker run --gpus all --rm -it \
#       -v $(pwd):/workspace \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       p2p:thor \
#       bash

# Pick up NVIDIA's Jetson Thor base image with CUDA + cuDNN preinstalled.
# Version tags follow the L4T / JetPack release cadence; pin to whatever's
# actually deployed on the Thor before building.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r36.3.0-pth2.4-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv, which works fine on aarch64.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

# Copy only metadata first so dependency installs are cached across code edits.
COPY pyproject.toml /workspace/
COPY .python-version /workspace/
RUN uv venv --python "$(cat .python-version)" /workspace/.venv

# Thor-specific extra: we skip bitsandbytes (no aarch64 wheels), skip
# Unsloth (x86-only dep tree), rely on stock transformers + peft + trl.
RUN . /workspace/.venv/bin/activate \
    && uv pip install -e ".[dev,search,gpu-thor]"

# Copy the rest of the source.
COPY . /workspace/

# Pre-warm the HF cache with the base model on build. Toggle via --build-arg HF_PREFETCH=0
ARG HF_PREFETCH=0
RUN if [ "$HF_PREFETCH" = "1" ]; then \
        . /workspace/.venv/bin/activate && \
        python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
                   AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct'); \
                   AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')"; \
    fi

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["source .venv/bin/activate && exec bash"]
