cmake_minimum_required(VERSION 3.5)
project(classifier_http_client LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++17 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

#### get libcurl, rapidjson & zlib
find_package(CURL REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
#find_package(ZLIB REQUIRED)
message(STATUS "Using curl ${CURL_VERSION_STRING}")
#message(STATUS "Using zlib ${ZLIB_VERSION_STRING}")


set(CLASSIFIER_HTTP_CLIENT_SRC
        classifier_http_client.cpp)

#set(TRITON_CLIENT_ROOT_PATH /mnt/e/Downloads/triton_client/src/c++)
link_directories("${TRITON_CLIENT_ROOT_PATH}/build/library/")

add_executable(classifier_http_client ${CLASSIFIER_HTTP_CLIENT_SRC})

target_include_directories(classifier_http_client PUBLIC ${RapidJSON_INCLUDE_DIRS}
                           PRIVATE ${TRITON_CLIENT_ROOT_PATH}/library/ ${TRITON_CLIENT_ROOT_PATH}/build/_deps/repo-common-build/protobuf/)

target_link_libraries(classifier_http_client PRIVATE libhttpclient.so)
#target_link_libraries(classifier_http_client PRIVATE libhttpclient_static.a
                                             #PUBLIC CURL::libcurl)
                                             #PRIVATE ZLIB::ZLIB)
