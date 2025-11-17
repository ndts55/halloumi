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

# Get GoogleTest
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --generate-line-info")

file(GLOB_RECURSE SOURCES src/*.cu src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.cuh)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(halloumi ${SOURCES} ${HEADERS})

set_target_properties(halloumi PROPERTIES
    CUDA_ARCHITECTURES "90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(halloumi PRIVATE nlohmann_json::nlohmann_json)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# Test Setup
file(GLOB TEST_SOURCES tests/*.cu)
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cu$")
add_executable(tests ${TEST_SOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(tests PRIVATE GTest::gtest_main nlohmann_json::nlohmann_json)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(tests)

