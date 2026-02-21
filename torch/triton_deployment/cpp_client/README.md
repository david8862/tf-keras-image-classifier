## C++ client app for Triton inference service

Here are some C++ client app samples for Triton inference service, which could be built with following steps:

1. Update cmake to latest version (e.g. 4.2.3)

```
# apt remove --purge cmake
# wget https://github.com/Kitware/CMake/releases/download/v4.2.3/cmake-4.2.3-linux-x86_64.sh
# chmod +x cmake-4.2.3-linux-x86_64.sh
# mkdir /opt/cmake
# ./cmake-4.2.3-linux-x86_64.sh --prefix=/opt/cmake --skip-license
# ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
# cmake --version
```

2. Prepare related package

Install deb packages
```
# apt install zlib1g-dev curl libssl-dev libcurl4-openssl-dev
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

Build & install protobuf & grpc (**Abandoned**)
```
# git clone --recurse-submodules https://github.com/grpc/grpc.git
# cd grpc/third_party/protobuf
# mkdir build && cd build
# cmake -Dprotobuf_BUILD_TESTS=OFF [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
        && make -j4
# make install

# cd ../../../
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
        && make -j4
# make install
```


3. Build triton client library
```
# git clone --recursive https://github.com/triton-inference-server/client.git
# cd client/src/c++/
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=OFF
        -DTRITON_ENABLE_PERF_ANALYZER=OFF -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF -DTRITON_ENABLE_PYTHON_HTTP=OFF -DTRITON_ENABLE_PYTHON_GRPC=OFF
        -DTRITON_ENABLE_JAVA_HTTP=OFF -DTRITON_ENABLE_ZLIB=OFF -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_EXAMPLES=ON
        -DTRITON_ENABLE_TESTS=OFF ..
# make -j4
```


4. Build grpc/http demo client application
```
# cd http/
# mkdir build && cd build
# cmake -DTRITON_CLIENT_ROOT_PATH=<Path_to_Triton_client>/ [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make

# cd ../../grpc/
# mkdir build && cd build
# cmake -DTRITON_CLIENT_ROOT_PATH=<Path_to_Triton_client>/ [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

5. On server side, Refer to [README.md](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/triton_deployment/README.md) to run triton service


6. Run application to do inference with triton model service
```
# cd http/build/
# ./classifier_http_client -h
Usage: classifier_http_client
--server_addr, -a: triton server address. default: 'localhost'
--server_port, -p: triton server port. default: '8000'
--model_name, -m: model name for inference. default: 'classifier_onnx'
--image, -i: image file to inference. default: './dog.jpg'
--classes, -l: path to class name definitions. default: './classes.txt'
--top_k, -k: show top k classes result. default: '1'
--input_mean, -b: input mean. default: '127.5'
--input_std, -s: input standard deviation. default: '127.5'
--count, -c: loop model run for certain times. default: '1'
--warmup_runs, -w: number of warmup runs. default: '2'
--verbose, -v: [0|1] print more information. default: '0'

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
--server_addr, -a: triton server address. default: 'localhost'
--server_port, -p: triton server port. default: '8000'
--model_name, -m: model name for inference. default: 'classifier_pipeline'
--image, -i: image file to inference. default: './dog.jpg'
--classes, -l: path to class name definitions. default: './classes.txt'
--top_k, -k: show top k classes result. default: '1'
--count, -c: loop model run for certain times. default: '1'
--warmup_runs, -w: number of warmup runs. default: '2'
--verbose, -v: [0|1] print more information. default: '0'

# ./pipeline_http_client -a 0.0.0.0 -p 8000 -m classifier_pipeline -l ../../../../../configs/imagenet_2012_classes.txt -i ../../../../../example/cat.jpg -c 5 -w 2 -v 0
num_classes: 1000
input tensor info: name input, type UINT8, shape (-1,-1,)
origin image size: width:480, height:360, channel:3
output tensor info: name output, type FP32, shape (-1,1,1000,), batch 1, classes 1000
model invoke average time: 45.5586ms
classifier_postprocess time: 0.096ms
Inferenced class:
lynx: 0.441083


# cd ../../grpc/build/
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
```

### Reference Doc
https://blog.csdn.net/qq_38196982/article/details/127394044
