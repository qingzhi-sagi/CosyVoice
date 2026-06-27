FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y wget build-essential git git-lfs sox libsox-dev ffmpeg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update -y && \
    apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get clean && \
    git lfs install

ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=/usr/local/cuda-12.1/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

COPY requirements.txt /opt/CosyVoice/

RUN echo "setuptools<70" > /tmp/constraints.txt && \
    PIP_CONSTRAINT=/tmp/constraints.txt pip3 install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host=mirrors.aliyun.com \
    --no-cache-dir

COPY . /opt/CosyVoice/

RUN cd runtime/python/grpc && \
    python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto

ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS"

EXPOSE 50000

CMD ["python3", "runtime/python/fastapi/server.py", "--port", "50000", "--model_dir", "iic/CosyVoice2-0.5B"]