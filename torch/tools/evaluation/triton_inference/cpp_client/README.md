## C++ client (X86/ARM) app for Triton inference service

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

# cd ../..
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
# cmake -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_ZLIB=OFF -DTRITON_ENABLE_EXAMPLES=OFF ..
# make -j4
```


3. Build demo client application
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
