FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# システムのアップデートと必要なパッケージのインストール
RUN apt-get update && apt-get install -yq \
    wget \
    build-essential \
    g++ \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libgtk2.0-dev \
    git \
    cuda-command-line-tools-12-2

# 環境変数の設定
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN export TF_ENABLE_ONEDNN_OPTS=0

# NVIDIA コンテナランタイムフックを有効にする
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install python
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip setuptools
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --upgrade nvidia-tensorrt

WORKDIR /work
