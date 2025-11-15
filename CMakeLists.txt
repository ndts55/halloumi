cmake_minimum_required(VERSION 3.17)
project(halloumi LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --generate-line-info")

file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADERS src/*.hpp src/*.cuh)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(halloumi ${SOURCES} ${HEADERS})

set_target_properties(halloumi PROPERTIES
    CUDA_ARCHITECTURES "90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(halloumi PRIVATE nlohmann_json::nlohmann_json spdlog)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

