cmake_minimum_required(VERSION 3.17)
project(halloumi LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADER_FILES src/*.hpp src/*.cuh)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(halloumi ${SRC_FILES} ${HEADER_FILES})

set_target_properties(halloumi PROPERTIES
    CUDA_ARCHITECTURES "90"
    CUDA_SEPARABLE_COMPILATION ON
    nlohmann_json_DIR "~/.local/share/cmake/nlohmann_json"
)

find_package(nlohmann_json 3.11 REQUIRED)
target_link_libraries(halloumi PRIVATE nlohmann_json::nlohmann_json)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

