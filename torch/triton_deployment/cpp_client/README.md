## C++ client app for Triton inference service

Here are some C++ client app samples for Triton inference service, which could be built with following steps:

1. Prepare related package

Install deb packages
```
# apt-get install zlib1g-dev curl libssl-dev libcurl4-openssl-dev
```

Build & install rapidjson
```
# git clone --recurse-submodules https://github.com/Tencent/rapidjson.git
# cd rapidjson
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
        && make -j4
# make install
```

Build & install protobuf & grpc
```
# git clone --recurse-submodules https://github.com/grpc/grpc.git
# cd grpc/third_party/protobuf
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
        && make -j4
# make install

# cd ../../../
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
        && make -j4
# make install
```


2. Build triton client library
```
# git clone https://github.com/triton-inference-server/client.git
# cd client/src/c++/
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_ZLIB=OFF -DTRITON_ENABLE_EXAMPLES=OFF ..
# make -j4
```


3. Build grpc/http demo client application
```
# cd grpc/
# mkdir build && cd build
# cmake -DTRITON_CLIENT_ROOT_PATH=<Path_to_Triton_client>/src/c++ [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make

# cd ../../http/
# mkdir build && cd build
# cmake -DTRITON_CLIENT_ROOT_PATH=<Path_to_Triton_client>/src/c++ [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```


4. On server side, prepare imagenet pretrained onnx model file and run triton service
```
# cd ../
# python imagenet_model_dump.py
# ./server_docker_install.sh <model_repository_path>
```


5. Run application to do inference with triton model service
```
# cd grpc/build/
# ./classifier_grpc_client -h
Usage: classifier_grpc_client
--server_addr, -a: localhost
--server_port, -p: 8001
--model_name, -m: classifier_onnx
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./classifier_grpc_client -a 0.0.0.0 -p 8001 -m classifier_onnx -l ../../../../../configs/imagenet_2012_classes.txt -i ../../../../../example/cat.jpg -c 5 -w 2 -v 0
num_classes: 1000
input tensor info: name image_input, type FP32, shape_size 4, layout NCHW, batch 1, height 224, width 224, channels 3
origin image size: width:480, height:360, channel:3
output tensor info: name scores, type FP32, shape_size 2, batch 1, classes 1000
model invoke average time: 81.5562ms
classifier_postprocess time: 0.182ms
Inferenced class:
Persian_cat: 0.205846

# ./pipeline_grpc_client -h
Usage: classifier_grpc_client
--server_addr, -a: localhost
--server_port, -p: 8001
--model_name, -m: classifier_pipeline
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./pipeline_grpc_client -a 0.0.0.0 -p 8001 -m classifier_pipeline -l ../../../../../configs/imagenet_2012_classes.txt -i ../../../../../example/cat.jpg -c 5 -w 2 -v 0
num_classes: 1000
input tensor info: name input, type UINT8, shape (-1,-1,)
origin image size: width:480, height:360, channel:3
output tensor info: name output, type FP32, shape (-1,1,1000,), batch 1, classes 1000
model invoke average time: 46.7676ms
classifier_postprocess time: 0.175ms
Inferenced class:
lynx: 0.441083


# cd ../../http/build/
# ./classifier_http_client -h
Usage: classifier_http_client
--server_addr, -a: localhost
--server_port, -p: 8000
--model_name, -m: classifier_onnx
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./classifier_http_client -a 0.0.0.0 -p 8000 -m classifier_onnx -l ../../../../../configs/imagenet_2012_classes.txt -i ../../../../../example/cat.jpg -c 5 -w 2 -v 0
num_classes: 1000
input tensor info: name image_input, type FP32, shape_size 4, layout NCHW, batch 1, height 224, width 224, channels 3
origin image size: width:480, height:360, channel:3
output tensor info: name scores, type FP32, shape_size 2, batch 1, classes 1000
model invoke average time: 82.0538ms
classifier_postprocess time: 0.177ms
Inferenced class:
Persian_cat: 0.205846

# ./pipeline_http_client -h
Usage: pipeline_http_client
--server_addr, -a: localhost
--server_port, -p: 8000
--model_name, -m: classifier_pipeline
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./pipeline_http_client -a 0.0.0.0 -p 8000 -m classifier_pipeline -l ../../../../../configs/imagenet_2012_classes.txt -i ../../../../../example/cat.jpg -c 5 -w 2 -v 0
num_classes: 1000
input tensor info: name input, type UINT8, shape (-1,-1,)
origin image size: width:480, height:360, channel:3
output tensor info: name output, type FP32, shape (-1,1,1000,), batch 1, classes 1000
model invoke average time: 45.5586ms
classifier_postprocess time: 0.096ms
Inferenced class:
lynx: 0.441083
```

