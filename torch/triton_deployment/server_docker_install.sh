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
