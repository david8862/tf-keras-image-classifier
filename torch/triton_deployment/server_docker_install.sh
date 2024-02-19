#!/bin/bash
#
# Reference doc:
# https://blog.csdn.net/sgyuanshi/article/details/123536579
# https://cloud.tencent.com/developer/article/2346623
# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html
# https://zhuanlan.zhihu.com/p/574146311
# https://zhuanlan.zhihu.com/p/361934132

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <model_repository_path>"
    exit 1
fi
MODEL_REPO_PATH=$1


# install CUDA/CuDNN/TensorRT

# clone triton inference server repo
git clone -b r23.12 https://github.com/triton-inference-server/server.git

# download demon models
pushd server/docs/examples
./fetch_models.sh
popd

# install docker ce
apt update
apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/ $(lsb_release -cs) stable"
apt update
apt install docker-ce docker-ce-cli containerd.io


# install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt update
apt install nvidia-docker2
systemctl restart docker

# get & run triton server docker image
docker pull nvcr.io/nvidia/tritonserver:23.12-py3
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$MODEL_REPO_PATH:/models nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models --grpc-use-ssl=false

# check if server is ready
curl -v localhost:8000/v2/health/ready


# another way to run triton server
# 1. launch interactive docker image to enter CLI
# 2. install needed packages & tools
# 3. start tritonserver
# 4. use perf_analyzer to check model inference performance
docker run --gpus=1 -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$MODEL_REPO_PATH:/models nvcr.io/nvidia/tritonserver:23.12-py3 bash
$ pip install --upgrade pip
$ pip install numpy pillow triton-model-analyzer
$ tritonserver --model-repository=/models &> /tmp/server.log &
$ perf_analyzer -u localhost:8001 -i grpc -m <model name> --percentile=95 --concurrency-range 1:2
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 2 concurrent requests
  Using synchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 2200
    Throughput: 122.211 infer/sec
    p50 latency: 8175 usec
    p90 latency: 8243 usec
    p95 latency: 8277 usec
    p99 latency: 8445 usec
    Avg gRPC time: 8171 usec ((un)marshal request/response 45 usec + response wait 8126 usec)
  Server:
    Inference count: 2200
    Execution count: 2200
    Successful request count: 2200
    Avg request latency: 7833 usec (overhead 30 usec + queue 77 usec + compute input 36 usec + compute infer 7683 usec + compute output 5 usec)

Request concurrency: 2
  Client:
    Request count: 2304
    Throughput: 127.991 infer/sec
    p50 latency: 15617 usec
    p90 latency: 15685 usec
    p95 latency: 15720 usec
    p99 latency: 15801 usec
    Avg gRPC time: 15609 usec ((un)marshal request/response 51 usec + response wait 15558 usec)
  Server:
    Inference count: 2304
    Execution count: 2304
    Successful request count: 2304
    Avg request latency: 15233 usec (overhead 29 usec + queue 7430 usec + compute input 46 usec + compute infer 7721 usec + compute output 5 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 122.211 infer/sec, latency 8277 usec
Concurrency: 2, throughput: 127.991 infer/sec, latency 15720 usec


# NOTE: perf_analyzer could also be installed and run remotely on client side,
# but may need to use special version and solve some lib dependency. e.g on
# Ubuntu 22.04, try following steps:
apt install libb64-dev libcudart11.0
pip install triton-model-analyzer==1.10.0

# install openssl 1.1
wget https://www.openssl.org/source/openssl-1.1.1g.tar.gz
tar -xvf openssl-1.1.1g.tar.gz
cd openssl-1.1.1g
./config shared --openssldir=/usr/local/openssl --prefix=/usr/local/openssl
make && make install

LD_LIBRARY_PATH=/usr/local/openssl/lib:$LD_LIBRARY_PATH perf_analyzer -u <triton server ip>:8001 -i grpc -m <model name> --percentile=95 --concurrency-range 1:2

