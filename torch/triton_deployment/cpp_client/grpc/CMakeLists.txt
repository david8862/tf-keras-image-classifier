cmake_minimum_required(VERSION 3.5)
project(grpc_client LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++17 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

#### get protobuf & grpc
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

#set(TRITON_CLIENT_ROOT_PATH /mnt/e/Downloads/triton_client/)
link_directories("${TRITON_CLIENT_ROOT_PATH}/build/install/lib/")

set(CLASSIFIER_GRPC_CLIENT_SRC
        classifier_grpc_client.cpp)
add_executable(classifier_grpc_client ${CLASSIFIER_GRPC_CLIENT_SRC})

target_include_directories(classifier_grpc_client PRIVATE ${TRITON_CLIENT_ROOT_PATH}/src/c++/library/ ${TRITON_CLIENT_ROOT_PATH}/build/_deps/repo-common-build/protobuf/)
target_link_libraries(classifier_grpc_client PRIVATE libgrpcclient.so
                                             PRIVATE gRPC::grpc++
                                             PRIVATE gRPC::grpc
                                             PUBLIC protobuf::libprotobuf)
#target_link_libraries(classifier_grpc_client PRIVATE libgrpcclient_static.a
                                             #PRIVATE gRPC::grpc++
                                             #PRIVATE gRPC::grpc
                                             #PUBLIC protobuf::libprotobuf)

set(PIPELINE_GRPC_CLIENT_SRC
        pipeline_grpc_client.cpp)
add_executable(pipeline_grpc_client ${PIPELINE_GRPC_CLIENT_SRC})

target_include_directories(pipeline_grpc_client PRIVATE ${TRITON_CLIENT_ROOT_PATH}/src/c++/library/ ${TRITON_CLIENT_ROOT_PATH}/build/_deps/repo-common-build/protobuf/)
target_link_libraries(pipeline_grpc_client PRIVATE libgrpcclient.so
                                             PRIVATE gRPC::grpc++
                                             PRIVATE gRPC::grpc
                                             PUBLIC protobuf::libprotobuf)
#target_link_libraries(pipeline_grpc_client PRIVATE libgrpcclient_static.a
                                             #PRIVATE gRPC::grpc++
                                             #PRIVATE gRPC::grpc
                                             #PUBLIC protobuf::libprotobuf)
