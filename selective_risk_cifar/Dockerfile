ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
    make \
    ripgrep \
    rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The base image already includes CUDA-enabled torch. Do not let pip overwrite it.
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio)([<>= ]|$)' /tmp/requirements.txt > /tmp/requirements-docker.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements-docker.txt pytest

CMD ["bash"]
