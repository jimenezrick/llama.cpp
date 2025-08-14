#!/bin/bash

set -eu

CUDAENV=~/bin/cuda.env
source $CUDAENV

cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j8

# Download model at: ~/.cache/llama.cpp
# ./build/bin/llama-cli -hf ggml-org/gemma-3n-E4B-it-GGUF:Q8_0
./build/bin/llama-cli -m models/ggml-org_gemma-3n-E4B-it-GGUF_gemma-3n-E4B-it-Q8_0.gguf -ngl 9999 --color --multiline-input
