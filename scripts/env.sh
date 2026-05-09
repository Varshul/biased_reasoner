#!/usr/bin/env bash
# Source this file before running any training scripts:
#   source scripts/env.sh
#
# Or add to your ~/.zshrc:
#   source ~/Desktop/claude/biased_reasoner/scripts/env.sh

# Pip-installed nvidia packages don't auto-register their .so paths.
# These directories contain libcusparseLt, libcudnn, libcublas, etc. needed by torch 2.7+.
_NVIDIA_PATHS=(
  /usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/curand/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib
  /usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib
  /usr/local/lib/python3.10/dist-packages/cusparselt/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cublas/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cudnn/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cufft/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/curand/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cusolver/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/cusparse/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/nccl/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib
  /home/tomahawk/.local/lib/python3.10/site-packages/nvidia/nvtx/lib
)

_NEW_PATHS=$(IFS=:; echo "${_NVIDIA_PATHS[*]}")
export LD_LIBRARY_PATH="${_NEW_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _NVIDIA_PATHS _NEW_PATHS
