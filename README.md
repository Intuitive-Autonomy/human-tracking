# Human Tracking and Pointcloud Reconstruction

Real-time human tracking and 3D pointcloud reconstruction system using STCN (Space-Time Correspondence Networks) and YOLO for dual-camera RGB-D setup in ROS2.

## Overview

This project combines video object segmentation with depth-based pointcloud reconstruction to track humans in real-time from dual RGB-D cameras. The system uses:

- **STCN**: Space-Time Correspondence Networks for efficient video object segmentation
- **YOLO**: YOLOv8 for human detection and initial mask generation
- **Depth Clustering**: Region growing in depth space to complete incomplete RGB masks
- **Ground Removal**: Height-based filtering to remove ground plane points
- **ROS2 Integration**: Full ROS2 support for camera transforms, image streaming, and pointcloud publishing

## System Architecture

```
┌─────────────┐     ┌─────────────┐
│  Camera 01  │     │  Camera 02  │
│  RGB + D    │     │  RGB + D    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               │
       ┌───────▼────────┐
       │  Image Stitch  │
       │  & Transform   │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ YOLO Detection │ (Initial + Periodic)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ STCN Tracking  │ (Memory Bank)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Depth Cluster  │ (Region Growing)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  Pointcloud    │
       │ Reconstruction │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Ground Removal │ (Height Filter)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │    Publish     │
       │ /human_pc      │
       └────────────────┘
```

## Key Features

- **Real-time Tracking**: 20-30 FPS tracking with FP16 optimization for Jetson platforms
- **Robust Segmentation**: Combines YOLO detection with STCN tracking for reliable human segmentation
- **Depth Enhancement**: Uses depth clustering to recover incomplete RGB masks
- **Periodic Refinement**: YOLO runs every 2 seconds to prevent drift and update memory
- **3D Reconstruction**: Generates complete human pointclouds from dual RGB-D cameras
- **ROS2 Native**: Full ROS2 support with proper QoS profiles and lifecycle management

## Components

### 1. camera_tf_publisher.py

Publishes static transforms from camera frames to base_link.

**Purpose**: Defines the spatial relationship between the two cameras and the robot base frame.

**Camera Configuration**:
- Global pitch offset: -15°
- Camera 01: -45° pitch, z=68.5cm (looking downward)
- Camera 02: +15° pitch, z=71.5cm (looking slightly upward)

**Usage** (ROS1):
```bash
python camera_tf_publisher.py
```

**Published Transforms**:
- `base_link` → `ob_camera_01_link`
- `base_link` → `ob_camera_02_link`

### 2. realtime_stcn_tracking.py

Main real-time human tracking and pointcloud reconstruction node.

**Purpose**:
- Subscribes to dual RGB-D camera streams
- Stitches and warps images from both cameras
- Tracks humans using YOLO + STCN
- Applies depth clustering to complete masks
- Reconstructs and publishes human pointcloud

**Key Features**:
- **Initial Detection**: YOLO detects human on first frame
- **STCN Tracking**: Efficient memory-based tracking (20-30 FPS)
- **Periodic YOLO**: Runs every 60 frames (~2 sec) to prevent drift
- **Depth Clustering**: Grows RGB mask using depth similarity
- **Ground Removal**: Filters points below 10cm height
- **Memory Management**: Optimized for Jetson (8GB GPU limit)

**Subscribed Topics**:
```
/ob_camera_01/color/image_raw      (sensor_msgs/Image)
/ob_camera_01/depth/image_raw      (sensor_msgs/Image)
/ob_camera_01/depth/camera_info    (sensor_msgs/CameraInfo)
/ob_camera_02/color/image_raw      (sensor_msgs/Image)
/ob_camera_02/depth/image_raw      (sensor_msgs/Image)
/ob_camera_02/depth/camera_info    (sensor_msgs/CameraInfo)
```

**Published Topics**:
```
/human_pointcloud                  (sensor_msgs/PointCloud2)
/ob_camera_01/human_mask          (sensor_msgs/Image) - debug
/ob_camera_02/human_mask          (sensor_msgs/Image) - debug
```

**Usage** (ROS2):
```bash
python3 realtime_stcn_tracking.py
```

**Configuration Parameters**:
```python
# Tracking
resolution = 480                    # STCN input resolution
max_memory_frames = 15              # Memory bank size
mem_every = 60                      # Update memory every 60 frames
yolo_period_frames = 60             # Run YOLO every 60 frames (~2s)

# Depth clustering
depth_tolerance_mm = 150            # Depth similarity threshold
min_cluster_size = 500              # Minimum cluster pixels

# Ground removal
ground_height_threshold = 0.10      # Remove points below 10cm
```

### 3. yolo_detector.py

Isolated YOLO subprocess for human detection.

**Purpose**: Runs YOLO in a separate process to avoid CUDA context conflicts with STCN.

**Communication**:
- Uses stdin/stdout with pickle serialization
- Receives: RGB frame (numpy array)
- Returns: Binary mask, number of detections, inference time

**Strategy**:
- Detects all humans in the frame
- Keeps the person closest to the horizontal center line
- Returns binary mask (0/255)

**Usage**:
This is automatically spawned by `realtime_stcn_tracking.py`. Can also be tested standalone:
```bash
python3 yolo_detector.py
```

### 4. depth_to_pointcloud.py

Depth to pointcloud converter for testing/verification.

**Purpose**: Simple utility to convert raw depth images to pointclouds without tracking.

**Subscribed Topics** (ROS1):
```
/ob_camera_01/depth/image_raw      (sensor_msgs/Image)
/ob_camera_01/depth/camera_info    (sensor_msgs/CameraInfo)
/ob_camera_02/depth/image_raw      (sensor_msgs/Image)
/ob_camera_02/depth/camera_info    (sensor_msgs/CameraInfo)
```

**Published Topics** (ROS1):
```
/ob_camera_01/pointcloud           (sensor_msgs/PointCloud2)
/ob_camera_02/pointcloud           (sensor_msgs/PointCloud2)
```

**Usage** (ROS1):
```bash
python depth_to_pointcloud.py
```

**Parameters**:
```bash
rosrun human_tracking depth_to_pointcloud.py _min_depth:=0.3 _max_depth:=5.0 _max_points:=5000
```

### 5. save_tracked_masks.py

Offline batch processing for dataset annotation.

**Purpose**:
- Process pre-recorded sessions from CSV file
- Run YOLO + STCN tracking on image sequences
- Save tracked masks for each frame
- Apply inverse perspective transform to masks

**Input**:
- CSV file (`camera_matches.csv`) with columns:
  - `session_id`: Unique session identifier
  - `camera_0_rgb_filename`: Path to camera 01 RGB image
  - `camera_1_rgb_filename`: Path to camera 02 RGB image
  - `frame_id`: Frame number

**Output**:
- Saves binary masks to `masks/<session_id>/<frame_id>.png`
- Masks are unwarped and split into top/bottom camera views

**Usage**:
```bash
python3 save_tracked_masks.py
```

**Workflow**:
1. Scans all sessions in CSV
2. For each session:
   - Loads RGB frames
   - Runs YOLO on first frame (keep_middle strategy)
   - Initializes STCN with detected mask
   - Tracks through all frames
   - Saves masks with inverse perspective transform

## Installation

### Prerequisites

- ROS2 (Humble or later) for real-time tracking
- ROS1 (Noetic) for camera_tf_publisher and depth_to_pointcloud utilities
- CUDA-capable GPU (tested on Jetson platforms)
- Python 3.8+

### Python Dependencies

```bash
pip install torch torchvision opencv-python
pip install ultralytics  # YOLOv8
pip install progressbar2 pillow
```

### ROS2 Dependencies

```bash
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs-py
```

### Model Weights

1. **STCN Model**: Download pretrained weights
```bash
python download_model.py
```
This downloads `stcn.pth` to `saves/` directory.

2. **YOLO Model**: Auto-downloaded on first run
```bash
# yolov8n-seg.pt will be downloaded automatically
```

## Quick Start

### Real-time Tracking (ROS2)

1. **Start camera drivers** (assuming OrbbecSDK):
```bash
ros2 launch orbbec_camera dual_camera.launch.py
```

2. **Publish camera transforms** (ROS1 for compatibility):
```bash
# Terminal 1
roscore
# Terminal 2
python camera_tf_publisher.py
```

3. **Run tracking node**:
```bash
python3 realtime_stcn_tracking.py
```

4. **Visualize in RViz2**:
```bash
rviz2
# Add PointCloud2 display
# Topic: /human_pointcloud
# Fixed Frame: base_link
```

### Offline Dataset Processing

```bash
python3 save_tracked_masks.py
```

Masks will be saved to `masks/<session_id>/` directory.

## Algorithm Details

### Tracking Pipeline

1. **Image Preprocessing**:
   - Downsample 1280×720 → 640×360 (0.5× scale)
   - Apply perspective transform (25° angle correction)
   - Stitch top and bottom camera views

2. **Initial Detection**:
   - Run YOLO on first frame
   - Select person closest to center
   - Initialize STCN memory bank

3. **Frame-by-Frame Tracking**:
   - STCN propagates mask using memory bank
   - Every 60 frames: run YOLO and merge with tracking
   - Update memory bank incrementally

4. **Depth Clustering**:
   - Extract depth from RGB mask region
   - Compute median depth (robust to outliers)
   - Grow region to pixels within ±150mm depth
   - Use morphological operations to clean mask

5. **Pointcloud Generation**:
   - Un-warp masks back to original camera views
   - Convert depth + mask → 3D points (camera frame)
   - Transform to base_link using TF
   - Combine and downsample to 5000 points

6. **Ground Removal**:
   - Filter points with Z < 0.10m (10cm height)
   - Publish clean human pointcloud

### Memory Management (Jetson Optimization)

- **FP16 Inference**: Converts STCN to half precision
- **Memory Bank Limit**: Max 15 frames in memory
- **Spatial Downsampling**: Process every other pixel for pointclouds
- **GPU Memory Limit**: 8GB allocation limit
- **Periodic Cleanup**: Cache cleared every 100 frames

## Troubleshooting

### YOLO Subprocess Fails
```
[ERROR] YOLO subprocess timeout
```
**Solution**: Ensure YOLO weights are downloaded and CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Tracking Lost
```
[WARN] Poor tracking detected - Quality: 0.2
```
**Solution**: System automatically runs YOLO re-detection after 3 poor frames.

### Low FPS
```
[INFO] [STCN Tracking] Frame 100: 150.234 ms
```
**Solutions**:
- Reduce `resolution` parameter (default 480)
- Reduce `max_memory_frames` (default 15)
- Enable mixed precision (FP16)
- Increase `mem_every` to update memory less frequently

### No Pointcloud Published
- Check camera_info topics are publishing intrinsics
- Verify TF transforms are available (`ros2 run tf2_tools view_frames`)
- Check depth images are valid (not all zeros)

## Performance Benchmarks

**Platform**: Jetson AGX Orin (32GB)

| Component | Time (ms) | FPS |
|-----------|-----------|-----|
| YOLO Detection | 35-45 | - |
| STCN Tracking (FP16) | 40-50 | 20-25 |
| Depth Clustering | <5 | - |
| Pointcloud Gen | 10-15 | - |
| **Total (with YOLO)** | ~100 | 10 |
| **Total (tracking only)** | ~60 | 16-17 |

## Citation

This project is based on STCN. Please cite:

```bibtex
@inproceedings{cheng2021stcn,
  title={Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={NeurIPS},
  year={2021}
}
```

## License

See original STCN repository for license information.

## Contact

For questions about this implementation, please open an issue on GitHub.
