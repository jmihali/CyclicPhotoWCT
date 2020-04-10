#!/bin/sh

export ANACONDA=/scratch_net/biwidl217/conda/
export CUDA_PATH=/usr/bin/nvidia-smi??
export PATH=${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${CUDA_PATH}/include

