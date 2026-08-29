# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      gnupg \
      ninja-build \
      nlohmann-json3-dev \
      patchelf \
      pkg-config \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      "libnvinfer-dev=11.1.0.106-1+cuda13.3" \
      "libnvinfer-headers-dev=11.1.0.106-1+cuda13.3" \
      "libnvinfer-headers-plugin-dev=11.1.0.106-1+cuda13.3" \
      "libnvinfer-safe-headers-dev=11.1.0.106-1+cuda13.3" \
      "libnvinfer-plugin11=11.1.0.106-1+cuda13.3" \
      "libnvinfer11=11.1.0.106-1+cuda13.3" \
      "libnvonnxparsers-dev=11.1.0.106-1+cuda13.3" \
      "libnvonnxparsers11=11.1.0.106-1+cuda13.3" \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python3.12 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip \
    && pip install \
      "torch==2.12.0+cu130" \
      "torchvision==0.27.0+cu130" \
      "torchaudio==2.11.0+cu130" \
      --index-url https://download.pytorch.org/whl/cu130 \
    && pip install \
      "accelerate" \
      "apache-tvm-ffi==0.1.12" \
      "build>=1.2" \
      "chronos-forecasting>=2.2.2" \
      "clang-format==22.1.8" \
      "conan-py-build==0.4.3" \
      "cuda-python==13.0.3" \
      "diffusers" \
      "ftfy" \
      "huggingface_hub>=0.23" \
      "librosa" \
      "lizard==1.21.2" \
      "ml_dtypes>=0.4" \
      "nvidia-modelopt==0.44.0" \
      "numpy>=1.24,<2.5" \
      "onnx>=1.16" \
      "onnxscript>=0.2" \
      "Pillow" \
      "protobuf" \
      "pytest<9" \
      "PyYAML>=6.0" \
      "ruff==0.16.4" \
      "safetensors>=0.4" \
      "scipy" \
      "sentencepiece>=0.1.99" \
      "setuptools>=80,<82" \
      "soundfile" \
      "tensorrt==11.1.0.106" \
      "timm>=1.0" \
      "tokenizers" \
      "transformers==5.2.0" \
    && pip install "nemo_toolkit[tts]==2.7.0" \
    && pip install --no-deps \
      "git+https://github.com/NVIDIA/NeMo.git@c9040511b" \
    && pip install --upgrade "transformers==5.2.0" \
    && pip install --force-reinstall \
      "torch==2.12.0+cu130" \
      "torchvision==0.27.0+cu130" \
      "torchaudio==2.11.0+cu130" \
      --index-url https://download.pytorch.org/whl/cu130 \
    && pip install "setuptools>=80,<82"

ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH=$TRT_LIB_DIR:/usr/local/cuda/lib64
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13

RUN python3.12 -c \
      "import importlib.metadata as m, tensorrt, torch, transformers, tvm_ffi; assert tensorrt.__version__ == '11.1.0.106'; assert torch.__version__ == '2.12.0+cu130'; assert transformers.__version__ == '5.2.0'; assert m.version('nvidia-modelopt') == '0.44.0'; assert m.version('conan-py-build') == '0.4.3'; assert m.version('apache-tvm-ffi') == '0.1.12'; assert 80 <= int(m.version('setuptools').split('.', 1)[0]) < 82" \
    && python3.12 -c \
      "from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt" \
    && test -f "$TRT_INC_DIR/NvInferVersion.h" \
    && test -f "$TRT_LIB_DIR/libnvinfer.so.11" \
    && test -f "$TRT_LIB_DIR/libnvonnxparser.so.11"

WORKDIR /workspace/tensorrt-model-connect
CMD ["bash"]
