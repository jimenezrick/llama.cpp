#!/bin/bash

set -eu

: ${MODEL:=models/ggml-org_gemma-3n-E4B-it-GGUF_gemma-3n-E4B-it-Q8_0.gguf}
: ${CONTEXT:=0} # Model default
: ${GPU_LAYERS:=9999}

CUDAENV=~/bin/cuda.env
source $CUDAENV

cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j8

# - Download model at: ~/.cache/llama.cpp
# ./build/bin/llama-cli -hf ggml-org/gemma-3n-E4B-it-GGUF:Q8_0
#
# (More models at: https://huggingface.co/ggml-org)

# - Chat:
# ./build/bin/llama-cli -m models/ggml-org_gemma-3n-E4B-it-GGUF_gemma-3n-E4B-it-Q8_0.gguf -ngl 9999 --color --multiline-input
# - Server:
./build/bin/llama-server --model $MODEL  --gpu-layers $GPU_LAYERS --ctx-size $CONTEXT
