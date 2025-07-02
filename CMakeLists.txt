cmake_minimum_required(VERSION 3.17)
project(halloumi LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADER_FILES include/*.h include/*.cuh)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3  # Use latest stable tag
)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(halloumi ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(halloumi PRIVATE nlohmann_json::nlohmann_json)

set_target_properties(halloumi PROPERTIES
    CUDA_ARCHITECTURES "90"
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

