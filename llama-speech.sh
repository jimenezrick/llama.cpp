#!/bin/bash

set -eu

DIR=$(cd $(dirname $0) && pwd )

CUDAENV=~/bin/cuda.env
source $CUDAENV

(
	cd $DIR

	# Download model at: ~/.cache/llama.cpp
	./build/bin/llama-tts --tts-oute-default -f $1 -ngl 9999
	aplay output.wav
)
