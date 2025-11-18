#!/bin/bash

build_dir=$(pwd)/build
source_dir=$(pwd)
cmake -B $build_dir \
    -S $source_dir \
    -DCMAKE_CUDA_ARCHITECTURES="90" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"
cmake --build build -j --target tests
./build/tests