#!/bin/bash
#
# Reference doc:
# https://github.com/wenet-e2e/wenet/blob/main/runtime/gpu/README.md


# Download pretrained model "wenetspeech_u2pp_conformer_exp.tar.gz" from wenet model zoo with web browser. link:
# https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md
# https://wenet.org.cn/downloads?models=wenet&version=wenetspeech_u2pp_conformer_exp.tar.gz


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

# install other needed packages
apt-get install cmake libprotobuf-dev protobuf-compiler
pip install pybind11[global]
pip install Pillow pyyaml tqdm onnxruntime-gpu #onnxmltools


# prepare wenet repo
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
mkdir -p runtime/gpu/triton_samples

# update some wenet code & config to fix minor issue
sed -i "/from whisper.tokenizer import LANGUAGES as WhiserLanguages/c #from whisper.tokenizer import LANGUAGES as WhiserLanguages" wenet/utils/common.py
sed -i "/WHISPER_LANGS = tuple(WhiserLanguages.keys())/c #WHISPER_LANGS = tuple(WhiserLanguages.keys())" wenet/utils/common.py

# manually change "configs['cmvn']['cmvn_conf']" in export_onnx_gpu.py to "configs['cmvn_conf']"
sed -i "/from wenet.transformer.ctc import CTC/i\sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))"  wenet/bin/export_onnx_gpu.py

sed -i "/FROM nvcr.io\/nvidia\/tritonserver:23.01-py3/c FROM nvcr.io\/nvidia\/tritonserver:23.12-py3" runtime/gpu/Dockerfile/Dockerfile.server
sed -i "/RUN pip3 install torch torchaudio/c RUN pip3 install torch==1.13.0 torchaudio" runtime/gpu/Dockerfile/Dockerfile.server


# convert pretrained model to onnx format for Triton deployment
mv wenetspeech_u2pp_conformer_exp.tar.gz runtime/gpu/triton_samples
cd runtime/gpu/triton_samples
tar xzvf wenetspeech_u2pp_conformer_exp.tar.gz
cd ../../../
MODEL_DIR=$(realpath $PWD)/runtime/gpu/triton_samples/20220506_u2pp_conformer_exp_wenetspeech
ONNX_MODEL_DIR=$MODEL_DIR/export_onnx/
python wenet/bin/export_onnx_gpu.py --config=$MODEL_DIR/train.yaml --checkpoint=$MODEL_DIR/final.pt --cmvn_file=$MODEL_DIR/global_cmvn --ctc_weight=0.5 --output_onnx_dir=$ONNX_MODEL_DIR --fp16
cp $MODEL_DIR/units.txt $ONNX_MODEL_DIR/words.txt
cp $MODEL_DIR/train.yaml $ONNX_MODEL_DIR/

# streaming inference convert
STREAM_ONNX_MODEL_DIR=$MODEL_DIR/export_onnx_stream/
python wenet/bin/export_onnx_gpu.py --config=$MODEL_DIR/train.yaml --checkpoint=$MODEL_DIR/final.pt --cmvn_file=$MODEL_DIR/global_cmvn  --ctc_weight=0.1 --reverse_weight=0.4 --output_onnx_dir=$STREAM_ONNX_MODEL_DIR --fp16 --streaming
cp $MODEL_DIR/units.txt $STREAM_ONNX_MODEL_DIR/words.txt
cp $MODEL_DIR/train.yaml $STREAM_ONNX_MODEL_DIR/


# build server docker image
cd runtime/gpu/
docker pull nvcr.io/nvidia/tritonserver:23.12-py3
docker build . -f Dockerfile/Dockerfile.server -t wenet_server:latest
cd ../../

# launch offline model
docker run --gpus '"device=0"' -it --name "wenet_offline_model" -v $(realpath $PWD)/runtime/gpu/model_repo:/ws/model_repo -v $ONNX_MODEL_DIR:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh
# launch streaming model
docker run --gpus '"device=0"' -it --name "wenet_stream_model" -v $(realpath $PWD)/model_repo_stateful:/ws/model_repo -v $STREAM_ONNX_MODEL_DIR:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh

# check if server is ready
curl -v localhost:8000/v2/health/ready

