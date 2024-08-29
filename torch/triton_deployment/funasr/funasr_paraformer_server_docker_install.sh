#!/bin/bash
#
# Reference doc:
# https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu

# install CUDA/CuDNN/TensorRT

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


# prepare FunASR repo
git clone https://github.com/modelscope/FunASR.git
cd FunASR/runtime/triton_gpu/

# update Dockerfile to fix minor issue
sed -i "/FROM nvcr.io\/nvidia\/tritonserver:23.01-py3/c FROM nvcr.io\/nvidia\/tritonserver:23.12-py3" runtime/triton_gpu/Dockerfile/Dockerfile.server
sed -i "/RUN pip3 install torch torchaudio/c RUN pip3 install torch==1.13.0 torchaudio" runtime/triton_gpu/Dockerfile/Dockerfile.server

# build docker image with Dockerfile/Dockerfile.server
docker build . -f Dockerfile/Dockerfile.server -t triton-paraformer:23.12

# prepare git-lfs
apt install git-lfs
git-lfs install


############################################################################################################################################
# online (streaming) part
#
# get online pretrained model, according to https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README_paraformer_online.md
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx.git

# put pretrained onnx model to model repo, the model repo should be like:
#
# ├── README.md
# └── model_repo_paraformer_large_online
#     ├── cif_search
#     │   ├── 1
#     │   │   └── model.py
#     │   └── config.pbtxt
#     ├── decoder
#     │   ├── 1
#     │   │   └── decoder.onnx
#     │   └── config.pbtxt
#     ├── encoder
#     │   ├── 1
#     │   │   └── model.onnx
#     │   └── config.pbtxt
#     ├── feature_extractor
#     │   ├── 1
#     │   │   └── model.py
#     │   ├── config.pbtxt
#     │   └── config.yaml
#     ├── lfr_cmvn_pe
#     │   ├── 1
#     │   │   └── lfr_cmvn_pe.onnx
#     │   ├── am.mvn
#     │   ├── config.pbtxt
#     │   └── export_lfr_cmvn_pe_onnx.py
#     └── streaming_paraformer
#         ├── 1
#         └── config.pbtxt
#
mkdir model_repo_paraformer_large_online/decoder/1/
cp -drf speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/decoder_quant.onnx model_repo_paraformer_large_online/decoder/1/decoder.onnx

mkdir model_repo_paraformer_large_online/encoder/1/
cp -drf speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/model_quant.onnx model_repo_paraformer_large_online/encoder/1/model.onnx

mkdir model_repo_paraformer_large_online/lfr_cmvn_pe/1/
cp -drf speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/am.mvn model_repo_paraformer_large_online/feature_extractor/am.mvn

cd model_repo_paraformer_large_online/lfr_cmvn_pe/
python export_lfr_cmvn_pe_onnx.py

mkdir model_repo_paraformer_large_online/streaming_paraformer/1/


# launch docker container
docker run --gpus all -it --rm --name "paraformer_stream_model_triton_server" -v ./:/workspace/ --shm-size=1g --net=host --ulimit memlock=-1 triton-paraformer:23.12

# launch triton service in docker container
cd /workspace
tritonserver --model-repository model_repo_paraformer_large_online \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000

# "Ctrl+p" -> "Ctrl+q" to quit from docker CLI, then check if server is ready
curl -v localhost:8000/v2/health/ready

# use python client to verify if service is ready
python funasr_client.py --server_addr=192.168.51.8 --model_name=streaming_paraformer --audio_path=test_audio.wav --streaming

############################################################################################################################################



############################################################################################################################################
# offline part
#

# install funasr package
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple funasr
pip install -U modelscope huggingface_hub

python
>>> from funasr import AutoModel
>>> model = AutoModel(model="paraformer", device="cpu")
>>> res = model.export(quantize=False)
...
output dir: /root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
>>>

ls -lh /root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
总计 1.7G
-rw-r--r-- 1 root root  11K  8月 29 14:43 am.mvn
-rw-r--r-- 1 root root  472  8月 29 14:43 configuration.json
-rw-r--r-- 1 root root 2.5K  8月 29 14:43 config.yaml
drwxr-xr-x 2 root root 4.0K  8月 29 14:43 example
drwxr-xr-x 2 root root 4.0K  8月 29 14:46 fig
-rw-r--r-- 1 root root 825M  8月 29 14:48 model.onnx
-rw-r--r-- 1 root root 840M  8月 29 14:46 model.pt
-rw-r--r-- 1 root root  20K  8月 29 14:46 README.md
-rw-r--r-- 1 root root 8.0M  8月 29 14:46 seg_dict
-rw-r--r-- 1 root root  92K  8月 29 14:46 tokens.json




# get offline pretrained model, according to https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README_paraformer_offline.md
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git

pretrained_model_dir=$(pwd)/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

cp $pretrained_model_dir/am.mvn ./model_repo_paraformer_large_offline/feature_extractor/
cp $pretrained_model_dir/config.yaml ./model_repo_paraformer_large_offline/feature_extractor/

# Refer here to get model.onnx (https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/export/README.md)
cp <exported_onnx_dir>/model.onnx ./model_repo_paraformer_large_offline/encoder/1/


