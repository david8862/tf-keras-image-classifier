cmake_minimum_required(VERSION 3.5)
project(classifier LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(CLASSIFIER_SRC
        classifier.cpp)

set(TENGINE_ROOT_PATH /mnt/d/Downloads/Tengine/)
link_directories("${TENGINE_ROOT_PATH}/build/install/lib/")

add_executable(classifier ${CLASSIFIER_SRC})

target_include_directories(classifier PRIVATE ${TENGINE_ROOT_PATH}/build/install/include/)
target_link_libraries(classifier PRIVATE -ltengine-lite -lstdc++ -lpthread -ldl)
