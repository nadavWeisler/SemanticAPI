#!/bin/bash
# Build script for Render deployments.
#
# Models can be sourced in two ways (controlled by USE_HF_DOWNLOAD):
#
#   USE_HF_DOWNLOAD=1  – download via Hugging Face Hub (huggingface_hub).
#                        Set HF_TOKEN for private / gated repositories.
#                        Override individual repos with HF_REPO_<LANG> /
#                        HF_FILE_<LANG> environment variables.
#
#   USE_HF_DOWNLOAD=0  – download the original FastText .vec files from
#   (default)             dl.fbaipublicfiles.com using wget (legacy).

set -euo pipefail

mkdir -p embeddings

if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
    echo "Downloading models from Hugging Face Hub…"
    pip install --quiet huggingface_hub
    python download_models.py
else
    echo "Downloading models from dl.fbaipublicfiles.com…"
    wget -q -O embeddings/cc.he.300.vec.gz \
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz
    gunzip embeddings/cc.he.300.vec.gz

    wget -q -O embeddings/cc.en.300.vec.gz \
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
    gunzip embeddings/cc.en.300.vec.gz
fi