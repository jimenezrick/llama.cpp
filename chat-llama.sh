#!/bin/bash

set -e

CUDAENV=/mnt/scratch-nvme/ricardo/CUDA12/cuda.env
source $CUDAENV

env LLAMA_CUDA=1 make -j

# MODEL="${MODEL:-./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf}" # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL="${MODEL:-./models/Llama-3-8B-Instruct-32k-v0.1.Q5_K_M.gguf}" # https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-32k-v0.1-GGUF
# PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-./prompts/code.txt}
# PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-./prompts/chat.txt}
PROMPT_TEMPLATE=${1:-./prompts/mistral-code.txt}
export USER_NAME="${USER_NAME:-User}"
export AI_NAME="${AI_NAME:-System}"

# Adjust to the number of CPU cores you want to use.
N_THREAD="${N_THREAD:-8}"
# Number of tokens to predict (made it larger than default because we want a long interaction)
# N_PREDICTS="${N_PREDICTS:-2048}"

# Note: you can also override the generation options by specifying them on the command line:
# For example, override the context size by doing: ./chatLLaMa --ctx_size 1024
# GEN_OPTIONS="-ngl 9999"
# GEN_OPTIONS="--ctx-size $((26 * 1024))"
GEN_OPTIONS="${GEN_OPTIONS:---ctx-size $((22 * 1024)) -ngl 9999}"

PROMPT_FILE=$(mktemp -t llamacpp_prompt.XXXXXXX.txt)
bash $PROMPT_TEMPLATE >$PROMPT_FILE
echo Prompt file: $PROMPT_FILE

# shellcheck disable=SC2086 # Intended splitting of GEN_OPTIONS
./main $GEN_OPTIONS \
  --model "$MODEL" \
  --threads "$N_THREAD" \
  --color --interactive \
  --file ${PROMPT_FILE} \
  --reverse-prompt "${USER_NAME}:" \
  --in-prefix '[INST] ' \
  --in-suffix ' [/INST]' \
  --prompt-cache prompt.cache
  # "$@"
  # --n_predict "$N_PREDICTS" \
  # --in-prefix '<|eot_id|><|start_header_id|>user<|end_header_id|> ' \
  # --in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|> ' \
