#!/bin/bash

function build() {
    local build_dir=$(pwd)/build
    local install_dir=$(pwd)/install
    local source_dir=$(pwd)
    cmake -B $build_dir \
        -S $source_dir \
        -DCMAKE_CUDA_ARCHITECTURES="90" \
        -DCMAKE_BUILD_TYPE=Debug \
        -Dnlohmann_json_DIR="~/.local/share/cmake/nlohmann_json" \
        "$@"
    cmake --build build -j
    # cmake --install $install_dir
        #-DCMAKE_BUILD_TYPE=Release \
}

build "$@"
