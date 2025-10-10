#!/bin/bash
# ROS2 STCN Tracker启动脚本
# 解决库冲突问题

# Set display for GUI applications
export DISPLAY=:0

# 优先使用系统库，避免conda库冲突
# Skip LD_PRELOAD on ARM64 - library path is different
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5

# CUDA相关设置，避免重复注册
export CUDA_MODULE_LOADING=LAZY
export TF_CPP_MIN_LOG_LEVEL=3

# 禁用TensorFlow/TensorRT/ONNX相关
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

# 禁用OpenCV的OpenCL加速，避免冲突
export OPENCV_OPENCL_RUNTIME=""

# PyTorch设置 - 内存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Ultralytics/YOLO设置 - 强制CPU模式
export YOLO_VERBOSE=False

# 禁用不必要的CUDA库
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

echo "Starting STCN Real-time Tracker..."
echo "Make sure ROS2 topics are publishing:"
echo "  - /camera_01/color/image_raw"
echo "  - /camera_02/color/image_raw"
echo ""

# 运行tracker
cd /home/choon/Documents/human-tracking
python3 -u realtime_stcn_tracking.py 2>&1
