#!/bin/bash

function build() {
    local build_dir=$(pwd)/build
    local install_dir=$(pwd)/install
    local source_dir=$(pwd)
    cmake -B $build_dir \
        -S $source_dir \
        -DCMAKE_CUDA_ARCHITECTURES="90" \
        -DCMAKE_BUILD_TYPE=Release \
        "$@"
    cmake --build build -j
}

build "$@"
