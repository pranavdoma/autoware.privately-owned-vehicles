#!/bin/bash
podman run -it --rm \
    --device nvidia.com/gpu=all \
    -p 6080:6080 \
    -e DISPLAY=:99 \
    -e ONNXRUNTIME_ROOT=/onnxruntime \
    -e LD_LIBRARY_PATH=/onnxruntime/lib:/cuda12-lib:/tensorrt10-lib:/cudnn-lib \
    -v /root/onnxruntime-linux-x64-gpu-1.22.0:/onnxruntime:z \
    -v /root/cuda12-lib:/cuda12-lib:z \
    -v /root/tensorrt-libs:/tensorrt10-lib:z \
    -v /root/cudnn-lib:/cudnn-lib:z \
    -v /root/trt_cache:/autoware/trt_cache:z \
    -v "$PWD"/model-weights:/autoware/model-weights:z \
    -v "$PWD"/launch:/autoware/launch:z \
    -v "$PWD"/../Test:/autoware/test:z \
    ghcr.io/autowarefoundation/visionpilot:latest \
    /autoware/launch/run_objectFinder.sh
