#!/usr/bin/env python3
"""
Real-time STCN Tracking ROS2 Node
订阅两个摄像头话题，拼接后实时跟踪人体mask并可视化

Subscribes to:
  - /camera_01/color/image_raw
  - /camera_02/color/image_raw

Workflow:
1) 收集图像帧，从第一帧开始用YOLO检测人体mask (keep_middle策略)
2) 找到第一个有效mask后，初始化STCN跟踪器
3) 后续帧使用STCN实时跟踪并可视化结果

Usage:
Run LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5 python3 realtime_stcn_tracking.py on x86 computer
Run LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtiff.so.5 python3 realtime_stcn_tracking.py on arm64 computer
"""

# Set environment variables BEFORE importing any libraries
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from collections import deque
import threading
import subprocess
import pickle
import time
import struct
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# Import PyTorch components first
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model.eval_network import STCN
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot
from model.aggregate import aggregate


class RealtimeSTCNTracker(Node):
    def __init__(self):
        super().__init__('realtime_stcn_tracker')

        # Configuration - Optimized for Jetson
        self.conf_threshold = 0.3
        self.resolution = 360
        self.angle_degrees = 15
        self.input_scale = 0.5
        self.max_memory_frames = 15
        self.mem_every = 3
        self.top_k = 3

        # Tracking quality thresholds - Only run YOLO when tracking is lost
        self.min_mask_area = 500  # Minimum pixels for valid tracking
        self.tracking_lost_threshold = 3  # Consecutive frames with poor tracking
        self.poor_tracking_count = 0  # Counter for poor tracking frames

        # Periodic YOLO execution for mask refinement
        self.yolo_period_frames = 15  # Run YOLO every 60 frames (~2 seconds at 30fps)
        self.last_yolo_frame = -1000  # Last frame where YOLO was run

        # Improved tracking: mask history and quality tracking
        self.last_valid_mask = None  # Fallback mask for failed frames
        self.last_valid_mask_area = 0
        self.mask_quality_history = deque(maxlen=10)  # Track last 10 frames' quality
        self.mask_area_history = deque(maxlen=10)  # Track mask area history for trend detection

        # Depth-based region growing parameters
        self.depth_tolerance_mm = 600  # Allow depth variation for clustering
        self.min_cluster_size = 500  # Minimum pixels for a valid cluster
        self.morph_kernel_size = 3  # Kernel size for morphological operations

        # Ground plane removal parameters
        self.ground_removal_enabled = True  # Enable ground plane removal
        self.ground_height_threshold = 0.10  # Remove points below 10cm height (in meters)
        self.min_points_above_ground = 100  # Minimum points required after ground removal

        # Seam-based downsampling parameters
        self.seam_downsample_enabled = True  # Enable distance-based downsampling near seam
        self.seam_min_keep_prob = 0.2  # Minimum keep probability near seam (20%)
        self.seam_max_keep_prob = 1.0  # Maximum keep probability far from seam (100%)

        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Image buffers
        self.camera_01_img = None
        self.camera_02_img = None
        self.camera_01_depth = None
        self.camera_02_depth = None
        self.last_timestamp = None

        # Image timestamps (store original timestamps from camera messages)
        self.camera_01_timestamp = None
        self.camera_02_timestamp = None

        # Camera intrinsics - will be populated from camera_info topics
        self.cam01_intrinsics = None
        self.cam02_intrinsics = None

        # TF buffer for transforming pointclouds to base_link
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CUDA stream for async operations on Jetson
        self.cuda_stream = torch.cuda.Stream()
        self.use_fp16 = True  # Enable FP16 for Jetson optimization

        # Tracking state
        self.tracking_initialized = False
        self.first_mask = None
        self.processor = None
        self.frame_buffer = deque(maxlen=self.max_memory_frames)
        self.current_frame_idx = 0
        self.mask_frame_idx = -1

        # Transform for STCN
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(self.resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
        ])

        # Limit GPU memory usage to avoid conflicts with other processes
        if torch.cuda.is_available():
            # Get total GPU memory and calculate fraction for 8GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_fraction = min(8.0 / total_memory, 1.0)  # 8GB or max available
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
            torch.cuda.empty_cache()
            self.get_logger().info("GPU memory limited to 8GB")

        # Use subprocess for YOLO to isolate CUDA context
        self.get_logger().info("YOLO will run in isolated subprocess...")
        self.yolo_process = None
        self.yolo_ready = False

        self.get_logger().info("Loading STCN model...")
        torch.autograd.set_grad_enabled(False)
        self.stcn_model = STCN().cuda().eval()
        model_path = "saves/stcn.pth"
        prop_saved = torch.load(model_path, weights_only=False)

        # Handle stage 0 model compatibility
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)

        self.stcn_model.load_state_dict(prop_saved)

        # Convert to FP16 for Jetson optimization
        if self.use_fp16:
            self.stcn_model = self.stcn_model.half()
            self.get_logger().info("STCN model converted to FP16 for Jetson acceleration")

        self.get_logger().info("Models loaded successfully!")

        # Subscribers - Camera Info
        self.sub_info01 = self.create_subscription(
            CameraInfo, '/camera_01/depth/camera_info', self.callback_info01, 1)
        self.sub_info02 = self.create_subscription(
            CameraInfo, '/camera_02/depth/camera_info', self.callback_info02, 1)

        # Subscribers - Color images
        self.sub_cam01 = self.create_subscription(
            Image, '/camera_01/color/image_raw', self.callback_cam01_color, 1)
        self.sub_cam02 = self.create_subscription(
            Image, '/camera_02/color/image_raw', self.callback_cam02_color, 1)

        # Subscribers - Depth images
        self.sub_cam01_depth = self.create_subscription(
            Image, '/camera_01/depth/image_raw', self.callback_cam01_depth, 1)
        self.sub_cam02_depth = self.create_subscription(
            Image, '/camera_02/depth/image_raw', self.callback_cam02_depth, 1)

        # Publisher - Combined human pointcloud in base_link frame
        self.pub_human_pc = self.create_publisher(
            PointCloud2, '/human_pointcloud', 1)

        # Publishers - Masks (for debugging)
        self.pub_mask01 = self.create_publisher(
            Image, '/camera_01/human_mask', 1)
        self.pub_mask02 = self.create_publisher(
            Image, '/camera_02/human_mask', 1)


        # Processing flag to prevent concurrent processing
        self.processing = False

        self.get_logger().info("ROS2 node initialized. Waiting for image messages...")

    def callback_cam01_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Rotate camera_01 by 180 degrees
            img = cv2.rotate(img, cv2.ROTATE_180)
            with self.lock:
                self.camera_01_img = img
                self.camera_01_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error("Failed to convert camera_01 color image: %s" % str(e))

    def callback_cam02_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_02_img = img
                self.camera_02_timestamp = msg.header.stamp

            # Trigger processing when new frame arrives (only if not already processing)
            if not self.processing and self.camera_01_img is not None:
                self.processing = True
                try:
                    self.process_frame()
                finally:
                    self.processing = False
        except Exception as e:
            self.get_logger().error("Failed to convert camera_02 color image: %s" % str(e))

    def callback_cam01_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Rotate camera_01 depth by 180 degrees
            depth = cv2.rotate(depth, cv2.ROTATE_180)
            with self.lock:
                self.camera_01_depth = depth
        except Exception as e:
            self.get_logger().error("Failed to convert camera_01 depth image: %s" % str(e))

    def callback_cam02_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.camera_02_depth = depth
        except Exception as e:
            self.get_logger().error("Failed to convert camera_02 depth image: %s" % str(e))

    def callback_info01(self, msg):
        """Callback for camera 01 camera_info"""
        if self.cam01_intrinsics is None:
            # Extract intrinsics from camera info k matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            # In ROS2, k is a tuple/array
            k = msg.k
            self.cam01_intrinsics = {
                'fx': k[0],
                'fy': k[4],
                'cx': k[2],
                'cy': k[5]
            }
            self.get_logger().info("Camera 01 intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
                         (self.cam01_intrinsics['fx'], self.cam01_intrinsics['fy'],
                          self.cam01_intrinsics['cx'], self.cam01_intrinsics['cy']))

    def callback_info02(self, msg):
        """Callback for camera 02 camera_info"""
        if self.cam02_intrinsics is None:
            # Extract intrinsics from camera info k matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            # In ROS2, k is a tuple/array
            k = msg.k
            self.cam02_intrinsics = {
                'fx': k[0],
                'fy': k[4],
                'cx': k[2],
                'cy': k[5]
            }
            self.get_logger().info("Camera 02 intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
                         (self.cam02_intrinsics['fx'], self.cam02_intrinsics['fy'],
                          self.cam02_intrinsics['cx'], self.cam02_intrinsics['cy']))

    def stitch_images(self, img_top, img_bottom):
        """上下摄像头透视变换后拼接"""
        # Warp top camera
        h0, w0 = img_top.shape[:2]
        src0 = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])
        offset0 = int(w0 * self.angle_degrees / 90)
        dst0 = np.float32([[0, 0], [w0, 0], [w0 - offset0, h0], [offset0, h0]])
        M0 = cv2.getPerspectiveTransform(src0, dst0)
        warped_top = cv2.warpPerspective(img_top, M0, (w0, h0))

        # Warp bottom camera
        h1, w1 = img_bottom.shape[:2]
        src1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        offset1 = int(w1 * self.angle_degrees / 90)
        dst1 = np.float32([[offset1, 0], [w1 - offset1, 0], [w1, h1], [0, h1]])
        M1 = cv2.getPerspectiveTransform(src1, dst1)
        warped_bottom = cv2.warpPerspective(img_bottom, M1, (w1, h1))

        # Resize to common width and stack
        w_common = min(warped_top.shape[1], warped_bottom.shape[1])
        scale_top = w_common / warped_top.shape[1]
        scale_bottom = w_common / warped_bottom.shape[1]

        top_h = int(round(warped_top.shape[0] * scale_top))
        bot_h = int(round(warped_bottom.shape[0] * scale_bottom))

        top_resized = cv2.resize(warped_top, (w_common, top_h), interpolation=cv2.INTER_LINEAR)
        bot_resized = cv2.resize(warped_bottom, (w_common, bot_h), interpolation=cv2.INTER_LINEAR)

        stitched = np.vstack([top_resized, bot_resized])
        return stitched

    def _start_yolo_subprocess(self):
        """启动YOLO子进程"""
        if self.yolo_process is None:
            self.get_logger().info("Starting YOLO subprocess...")
            try:
                # Use the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.yolo_process = subprocess.Popen(
                    ['python3', 'yolo_detector.py'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # Discard stderr to avoid pipe interference
                    bufsize=0,
                    cwd=script_dir
                )
                # Wait for READY signal from stdout
                import select
                ready = select.select([self.yolo_process.stdout], [], [], 15.0)
                if ready[0]:
                    line = self.yolo_process.stdout.readline().decode().strip()
                    if "READY" in line:
                        self.yolo_ready = True
                        self.get_logger().info("YOLO subprocess ready!")
                    else:
                        self.get_logger().error("YOLO subprocess unexpected output: %s" % line)
                else:
                    self.get_logger().error("YOLO subprocess timeout (15s)")
                    if self.yolo_process:
                        self.yolo_process.kill()
                        self.yolo_process = None
            except Exception as e:
                self.get_logger().error("Failed to start YOLO subprocess: %s" % str(e))
                import traceback
                self.get_logger().error(traceback.format_exc())
                if self.yolo_process:
                    self.yolo_process.kill()
                self.yolo_process = None

    def extract_mask_keep_middle(self, frame_bgr):
        """使用YOLO提取最靠近中线的人体mask (通过子进程)"""
        # Start subprocess if needed
        if not self.yolo_ready:
            self._start_yolo_subprocess()

        if not self.yolo_ready or self.yolo_process is None:
            return None, 0, 0.0

        try:
            start_time = time.time()

            # Send frame to subprocess
            frame_data = pickle.dumps(frame_bgr, protocol=pickle.HIGHEST_PROTOCOL)
            size = len(frame_data).to_bytes(4, 'little')
            self.yolo_process.stdin.write(size)
            self.yolo_process.stdin.write(frame_data)
            self.yolo_process.stdin.flush()

            # Read result - read in chunks to avoid truncation
            size_bytes = self.yolo_process.stdout.read(4)
            if len(size_bytes) != 4:
                self.get_logger().error("Failed to read result size from YOLO subprocess")
                raise RuntimeError("Invalid size bytes")

            result_size = int.from_bytes(size_bytes, 'little')

            # Read result data in chunks
            result_data = b''
            remaining = result_size
            while remaining > 0:
                chunk = self.yolo_process.stdout.read(min(remaining, 65536))
                if not chunk:
                    break
                result_data += chunk
                remaining -= len(chunk)

            if len(result_data) != result_size:
                self.get_logger().error("Incomplete result: got %d, expected %d" % (len(result_data), result_size))
                raise RuntimeError("Incomplete data")

            result = pickle.loads(result_data)

            detection_time = time.time() - start_time

            return result['mask'], result['num_det'], detection_time

        except Exception as e:
            self.get_logger().error("YOLO subprocess communication error: %s" % str(e))
            self.yolo_ready = False
            if self.yolo_process:
                self.yolo_process.kill()
                self.yolo_process = None
            return None, 0, 0.0

    def frame_to_tensor(self, frame_bgr):
        """Convert BGR frame to STCN input tensor"""
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.im_transform(frame_pil)

        # Convert to FP16 if enabled for Jetson
        if self.use_fp16:
            frame_tensor = frame_tensor.half()

        return frame_tensor, frame_rgb

    def initialize_tracking(self, first_mask, orig_size):
        """Initialize STCN tracker with first mask"""
        # Prepare mask
        first_mask_bin = (first_mask > 0).astype(np.uint8) * 1
        labels = np.unique(first_mask_bin)
        labels = labels[labels != 0]

        if len(labels) == 0:
            raise ValueError("Mask is empty!")

        mask_onehot = all_to_onehot(first_mask_bin, labels)
        mask_tensor = torch.from_numpy(mask_onehot).float()

        # Store for tracking
        self.num_objects = len(labels)
        self.first_mask_tensor = mask_tensor
        self.orig_size = orig_size

        # Initialize memory banks with size limit
        from inference_memory_bank import MemoryBank
        self.mem_banks = {}
        self.max_memory_frames = 10  # 减少到10帧以节省显存
        for oi in range(1, self.num_objects + 1):
            bank = MemoryBank(k=1, top_k=self.top_k)
            # Initialize temp attributes that aren't set in __init__
            bank.temp_k = None
            bank.temp_v = None
            self.mem_banks[oi] = bank

        # Encode first frame with mask
        frame_tensor = self.frame_buffer[self.mask_frame_idx]['tensor']
        frame_cuda = frame_tensor.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, C, H, W)

        # Pad frame
        from util.tensor_util import pad_divide_by
        frame_cuda, self.pad = pad_divide_by(frame_cuda, 16)

        # Get spatial size after padding
        Hp, Wp = frame_cuda.shape[-2:]

        # Resize mask to padded size
        mask_tensor_resized = transforms.Resize(
            (Hp, Wp), interpolation=InterpolationMode.NEAREST
        )(mask_tensor).unsqueeze(1).cuda()

        with_bg_msk = torch.cat([1 - torch.sum(mask_tensor_resized, dim=0, keepdim=True), mask_tensor_resized], 0)

        # Encode key and value for first frame
        with torch.amp.autocast('cuda', enabled=True):
            key_k, _, qf16, _, _ = self.stcn_model.encode_key(frame_cuda[:,0])
            key_v = self.stcn_model.encode_value(frame_cuda[:,0], qf16, with_bg_msk[1:])
            key_k = key_k.unsqueeze(2)

            # Add to memory banks
            for i in range(self.num_objects):
                self.mem_banks[i+1].add_memory(key_k, key_v[i:i+1])

        self.last_mem_ti = self.mask_frame_idx
        self.tracking_initialized = True

        # Record first mask as valid fallback
        self.last_valid_mask = first_mask.copy()
        self.last_valid_mask_area = np.count_nonzero(first_mask)

    def track_frame(self, frame_tensor, orig_size):
        """Track single frame using STCN"""
        start_time = time.time()

        # Only clear CUDA cache occasionally to avoid latency spikes
        # Cache clearing is expensive on Jetson (~100ms), so do it sparingly
        if self.current_frame_idx % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Prepare frame
        frame_cuda = frame_tensor.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, C, H, W)

        # Pad frame - must use same padding as initialization
        from util.tensor_util import pad_divide_by
        frame_cuda, current_pad = pad_divide_by(frame_cuda, 16)

        # Use stored pad from initialization, but verify it matches
        if not hasattr(self, 'pad') or self.pad is None:
            self.pad = current_pad

        with torch.amp.autocast('cuda', enabled=True):
            # Encode query frame
            k16, qv16, qf16, qf8, qf4 = self.stcn_model.encode_key(frame_cuda[:,0])

            # Segment using memory banks
            out_mask = torch.cat([
                self.stcn_model.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
                for oi in range(1, self.num_objects + 1)
            ], 0)

            # Aggregate masks (returns shape: (k+1, H, W))
            out_mask = aggregate(out_mask, keep_bg=True)

            # Check if we need to add this frame to memory
            is_mem_frame = (abs(self.current_frame_idx - self.last_mem_ti) >= self.mem_every)

            if is_mem_frame:
                # Improved: Only add to memory if tracking quality is good
                # This prevents pollution from poor tracking frames
                mask_area = torch.sum(out_mask[1:]).item()  # Sum all object channels

                # Only add to memory if mask area is reasonable
                # Avoid adding frames where tracking drifted or is very uncertain
                if mask_area > self.min_mask_area * 0.7:  # Allow 70% of threshold
                    # Encode value and add to memory
                    prev_value = self.stcn_model.encode_value(frame_cuda[:,0], qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)

                    for i in range(self.num_objects):
                        bank = self.mem_banks[i+1]

                        # Limit memory bank size to prevent OOM
                        if bank.mem_k is not None:
                            # mem_k shape: (1, C, T*H*W), we track number of frames
                            chunk_size = prev_key.flatten(start_dim=2).shape[2]
                            current_frames = bank.mem_k.shape[2] // chunk_size

                            if current_frames >= self.max_memory_frames:
                                # Remove oldest frames - use .contiguous() and explicitly delete old tensors
                                old_k = bank.mem_k
                                old_v = bank.mem_v

                                # Keep only recent frames, create contiguous tensor
                                bank.mem_k = bank.mem_k[:, :, chunk_size:].contiguous()
                                bank.mem_v = bank.mem_v[:, :, chunk_size:].contiguous()

                                # Explicitly delete old tensors to free memory
                                del old_k, old_v

                        bank.add_memory(prev_key, prev_value[i:i+1])

                    self.last_mem_ti = self.current_frame_idx

        # Get final mask (foreground only, sum all object channels)
        # out_mask shape: (k+1, 1, H, W), we want objects only (index 1 onwards)
        mask_prob = out_mask[1:].sum(0)  # Shape: (1, H, W) - note the extra dimension

        # Convert to 4D tensor: (1, 1, H, W)
        mask_prob = mask_prob.unsqueeze(0)  # Now (1, 1, H, W)

        # Unpad - handle padding correctly
        # pad format: (left, right, top, bottom)
        lw, rw, lh, rh = self.pad

        # Only slice if there's actually padding
        if lh > 0 or rh > 0:
            if rh > 0:
                mask_prob = mask_prob[:, :, lh:-rh, :]
            else:
                mask_prob = mask_prob[:, :, lh:, :]

        if lw > 0 or rw > 0:
            if rw > 0:
                mask_prob = mask_prob[:, :, :, lw:-rw]
            else:
                mask_prob = mask_prob[:, :, :, lw:]

        # Resize to original size
        H, W = orig_size
        mask_prob = F.interpolate(mask_prob, size=(H, W), mode='bilinear', align_corners=False)

        # Convert to binary mask
        mask = (mask_prob[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

        tracking_time = time.time() - start_time

        return mask, tracking_time

    def check_yolo_mask_coverage(self, tracking_mask, yolo_mask, threshold=0.8):
        """
        Check if tracking_mask is mostly contained in yolo_mask

        Args:
            tracking_mask: Current STCN tracking mask
            yolo_mask: YOLO detection mask
            threshold: Percentage threshold (default 80%)

        Returns:
            (is_covered, coverage_ratio)
        """
        if tracking_mask is None or yolo_mask is None:
            return False, 0.0

        # Convert to binary
        track_bin = (tracking_mask > 128).astype(np.uint8)
        yolo_bin = (yolo_mask > 128).astype(np.uint8)

        track_pixels = np.count_nonzero(track_bin)
        if track_pixels == 0:
            return False, 0.0

        # Count overlap: tracking pixels that are in YOLO mask
        overlap = np.count_nonzero(track_bin & yolo_bin)
        coverage_ratio = overlap / track_pixels

        is_covered = coverage_ratio >= threshold
        return is_covered, coverage_ratio

    def compute_mask_increment(self, tracking_mask, yolo_mask):
        """
        Compute mask increment: YOLO mask - tracking mask
        Returns pixels in YOLO mask but not in tracking mask

        Args:
            tracking_mask: Current STCN tracking mask
            yolo_mask: YOLO detection mask

        Returns:
            increment_mask: Mask containing only new pixels from YOLO
        """
        track_bin = (tracking_mask > 128).astype(np.uint8)
        yolo_bin = (yolo_mask > 128).astype(np.uint8)

        # Increment = YOLO - tracking (pixels in YOLO but not in tracking)
        increment = (yolo_bin & ~track_bin).astype(np.uint8) * 255

        return increment

    def update_memory_bank_with_increment(self, frame_tensor, increment_mask, orig_size):
        """
        Update memory bank using mask increment instead of full re-initialization
        This adds new information to memory without disrupting existing tracking

        Args:
            frame_tensor: Frame tensor for encoding
            increment_mask: Mask increment (new pixels from YOLO)
            orig_size: Original image size
        """
        try:
            if increment_mask is None:
                return

            increment_pixels = np.count_nonzero(increment_mask > 128)
            if increment_pixels < 100:  # Too small to be meaningful
                self.get_logger().warning("Increment mask too small (%d pixels), skipping" % increment_pixels)
                return

            # Prepare frame for encoding
            frame_cuda = frame_tensor.unsqueeze(0).unsqueeze(0).cuda()

            # Pad frame
            from util.tensor_util import pad_divide_by
            frame_cuda, pad = pad_divide_by(frame_cuda, 16)

            # Get spatial size after padding
            Hp, Wp = frame_cuda.shape[-2:]

            # Resize increment mask to match frame size
            mask_resized = cv2.resize(increment_mask, (Wp, Hp), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 128).astype(np.uint8)

            # Create onehot format
            mask_onehot = all_to_onehot(mask_binary, np.array([1]))
            mask_tensor = torch.from_numpy(mask_onehot).float().cuda()

            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif len(mask_tensor.shape) == 4:
                mask_tensor = mask_tensor.squeeze(1)

            # Resize to padded size
            mask_tensor_resized = transforms.Resize(
                (Hp, Wp), interpolation=InterpolationMode.NEAREST
            )(mask_tensor)

            if len(mask_tensor_resized.shape) == 2:
                mask_tensor_resized = mask_tensor_resized.unsqueeze(0).unsqueeze(1)
            elif len(mask_tensor_resized.shape) == 3:
                mask_tensor_resized = mask_tensor_resized.unsqueeze(1)

            with_bg_msk = torch.cat([1 - torch.sum(mask_tensor_resized, dim=0, keepdim=True), mask_tensor_resized], 0)

            # Encode new key and value
            with torch.amp.autocast('cuda', enabled=True):
                key_k, _, qf16, _, _ = self.stcn_model.encode_key(frame_cuda[:,0])
                key_v = self.stcn_model.encode_value(frame_cuda[:,0], qf16, with_bg_msk[1:])
                key_k = key_k.unsqueeze(2)

                # Add to memory banks
                for i in range(self.num_objects):
                    bank = self.mem_banks[i+1]

                    # Limit memory bank size
                    if bank.mem_k is not None:
                        chunk_size = key_k.flatten(start_dim=2).shape[2]
                        current_frames = bank.mem_k.shape[2] // chunk_size

                        if current_frames >= self.max_memory_frames:
                            old_k = bank.mem_k
                            old_v = bank.mem_v
                            bank.mem_k = bank.mem_k[:, :, chunk_size:].contiguous()
                            bank.mem_v = bank.mem_v[:, :, chunk_size:].contiguous()
                            del old_k, old_v

                    bank.add_memory(key_k, key_v[i:i+1])

            self.get_logger().info("[Memory Increment Update] Frame %d: Added %d pixels to memory" %
                         (self.current_frame_idx, increment_pixels))

        except Exception as e:
            self.get_logger().error("Failed to update memory with increment: %s" % str(e))
            import traceback
            self.get_logger().error(traceback.format_exc())

    def is_tracking_lost(self, mask):
        """
        Improved tracking loss detection with multi-dimensional quality checks
        Returns: (is_lost, quality_score, diagnosis)
        """
        diagnosis = {
            'null_mask': False,
            'area_too_small': False,
            'area_shrinking': False,
            'low_quality': False
        }
        quality_score = 1.0

        if mask is None:
            diagnosis['null_mask'] = True
            self.mask_quality_history.append(0.0)
            self.mask_area_history.append(0)
            return True, 0.0, diagnosis

        # Calculate mask area
        mask_area = np.count_nonzero(mask)

        # 1. Check absolute area threshold
        if mask_area < self.min_mask_area:
            diagnosis['area_too_small'] = True
            quality_score *= 0.3

        # 2. Check if area is rapidly shrinking (sign of tracking drift)
        if len(self.mask_area_history) >= 3:
            recent_areas = list(self.mask_area_history)[-3:]
            avg_recent = np.mean(recent_areas)
            shrink_rate = (self.last_valid_mask_area - mask_area) / (self.last_valid_mask_area + 1e-6)

            # If area drops by >50% in one frame, likely tracking lost
            if shrink_rate > 0.5 and mask_area < self.min_mask_area * 2:
                diagnosis['area_shrinking'] = True
                quality_score *= 0.4

        # 3. Check quality trend
        if len(self.mask_quality_history) >= 3:
            recent_quality = list(self.mask_quality_history)[-3:]
            avg_quality = np.mean(recent_quality)
            # If consistently low quality, mark as lost
            if avg_quality < 0.5:
                diagnosis['low_quality'] = True
                quality_score *= 0.5

        # Record this frame's quality
        self.mask_quality_history.append(quality_score)
        self.mask_area_history.append(mask_area)

        # Consider lost if quality score below threshold or area too small
        is_lost = quality_score < 0.4 or mask_area < self.min_mask_area

        return is_lost, quality_score, diagnosis

    def downsample_by_seam_distance(self, points, v_coords, camera_frame, seam_v_position):
        """
        Downsample points based on distance to stitching seam.
        Points closer to seam are downsampled more aggressively.

        Args:
            points: Nx3 numpy array of 3D points
            v_coords: N array of v (vertical) pixel coordinates in downsampled image space
            camera_frame: 'camera_01_link' or 'camera_02_link'
            seam_v_position: Vertical pixel position of seam in downsampled image

        Returns:
            downsampled_points: Mx3 numpy array after distance-based downsampling
        """
        if points is None or len(points) == 0:
            return points

        if not self.seam_downsample_enabled:
            return points

        # Calculate distance from each point to seam
        distance_to_seam = np.abs(v_coords - seam_v_position)

        # Normalize distance (0 = at seam, 1 = max distance)
        max_distance = np.max(distance_to_seam) + 1e-6
        normalized_distance = distance_to_seam / max_distance

        # Calculate keep probability for each point
        # Points far from seam: high probability (close to max_keep_prob)
        # Points near seam: low probability (close to min_keep_prob)
        min_keep_prob = self.seam_min_keep_prob
        max_keep_prob = self.seam_max_keep_prob

        # Quadratic function: more aggressive near seam
        # Using x^2 gives smooth transition with more aggressive downsampling near seam
        keep_probability = min_keep_prob + (max_keep_prob - min_keep_prob) * (normalized_distance ** 2)

        # Generate random values for each point
        random_values = np.random.random(len(points))

        # Keep points where random value < keep probability
        keep_mask = random_values < keep_probability

        downsampled_points = points[keep_mask]

        kept_ratio = len(downsampled_points) / len(points)
        self.get_logger().info("[%s Seam Downsample] %d -> %d points (%.1f%% kept)" %
                             (camera_frame, len(points), len(downsampled_points), kept_ratio * 100))

        return downsampled_points

    def depth_to_pointcloud(self, depth_image, mask, camera_frame, intrinsics, timestamp, max_points=None):
        """Convert depth image and mask to pointcloud with correct intrinsics and coordinate transform"""
        if depth_image is None or mask is None or intrinsics is None:
            return None

        # Resize mask to match depth image size if needed
        if mask.shape != depth_image.shape:
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Spatial downsampling for Jetson optimization (skip every other pixel)
        # This reduces computation by ~75% while maintaining good coverage
        depth_ds = depth_image[::2, ::2]
        mask_ds = mask[::2, ::2]

        h, w = depth_ds.shape

        # Adjust intrinsics for downsampled image
        fx = intrinsics['fx'] / 2.0
        fy = intrinsics['fy'] / 2.0
        cx = intrinsics['cx'] / 2.0
        cy = intrinsics['cy'] / 2.0

        # Create coordinate grids (faster for downsampled image)
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

        # Convert depth to meters (assuming mm input)
        z = depth_ds.astype(np.float32) / 1000.0

        # Apply mask filter - only keep masked pixels
        valid_mask = (mask_ds > 128) & (z > 0.3) & (z < 5.0)

        # Calculate 3D coordinates only for valid points
        x_cam = (u_grid - cx) * z / fx
        y_cam = (v_grid - cy) * z / fy
        z_cam = z

        # Extract valid points and their v coordinates
        x_cam_valid = x_cam[valid_mask]
        y_cam_valid = y_cam[valid_mask]
        z_cam_valid = z_cam[valid_mask]
        v_coords_valid = v_grid[valid_mask]  # Keep track of vertical pixel positions

        # Transform from camera coordinates (Y-up, Z-forward) to ROS coordinates (Z-up, X-forward)
        # Camera: X-right, Y-down, Z-forward
        # ROS: X-forward, Y-left, Z-up
        # Transformation: X_ros = Z_cam, Y_ros = -X_cam, Z_ros = -Y_cam
        x_ros = z_cam_valid
        y_ros = -x_cam_valid
        z_ros = -y_cam_valid

        # Combine into point array
        points = np.stack((x_ros, y_ros, z_ros), axis=-1)

        if len(points) == 0:
            return None, 0, None

        # Downsample if max_points is specified
        if max_points is not None and len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            v_coords_valid = v_coords_valid[indices]

        # Return points, original count, and v coordinates for seam-based downsampling
        return points, len(x_cam_valid), v_coords_valid

    def grow_mask_with_depth_clustering(self, rgb_mask, depth_image):
        """
        Use depth clustering to grow the RGB mask and capture the full human region.
        The RGB mask is sometimes incomplete - this method uses depth information to
        recover missing parts by region growing with depth similarity.

        Args:
            rgb_mask: Binary mask from RGB tracking (uint8, 0 or 255)
            depth_image: Depth image in mm (uint16 or float)

        Returns:
            expanded_mask: Grown mask that includes full human region (uint8, 0 or 255)
        """
        if rgb_mask is None or depth_image is None:
            return rgb_mask

        # Resize masks to match if needed
        if rgb_mask.shape != depth_image.shape:
            rgb_mask = cv2.resize(rgb_mask, (depth_image.shape[1], depth_image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # Convert to binary
        seed_mask = (rgb_mask > 128).astype(np.uint8)

        # Check if seed mask is valid
        seed_pixels = np.count_nonzero(seed_mask)
        if seed_pixels < self.min_cluster_size:
            self.get_logger().warning("Seed mask too small (%d pixels), skipping depth clustering" % seed_pixels)
            return rgb_mask

        # Extract depth statistics from seed region
        depth_values = depth_image[seed_mask > 0]
        valid_depths = depth_values[(depth_values > 300) & (depth_values < 5000)]  # 0.3m to 5m

        if len(valid_depths) < 100:
            self.get_logger().warning("Not enough valid depth values in seed region, skipping clustering")
            return rgb_mask

        # Use median and MAD (Median Absolute Deviation) for robust statistics
        median_depth = np.median(valid_depths)
        mad = np.median(np.abs(valid_depths - median_depth))

        # Define depth range for clustering
        depth_min = median_depth - self.depth_tolerance_mm
        depth_max = median_depth + self.depth_tolerance_mm

        # Create depth-based mask: pixels with depth in valid range
        depth_cluster_mask = ((depth_image >= depth_min) &
                             (depth_image <= depth_max) &
                             (depth_image > 0)).astype(np.uint8)

        # Morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.morph_kernel_size, self.morph_kernel_size))
        depth_cluster_mask = cv2.morphologyEx(depth_cluster_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components in the depth cluster mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            depth_cluster_mask, connectivity=8
        )

        # Find which component(s) overlap with the seed mask
        seed_labels = np.unique(labels[seed_mask > 0])
        seed_labels = seed_labels[seed_labels != 0]  # Remove background label

        if len(seed_labels) == 0:
            self.get_logger().warning("No depth cluster overlaps with seed mask")
            return rgb_mask

        # Create expanded mask from all components that overlap with seed
        expanded_mask = np.zeros_like(depth_cluster_mask)
        for label in seed_labels:
            component_mask = (labels == label).astype(np.uint8)
            component_size = stats[label, cv2.CC_STAT_AREA]

            # Only include components that are large enough
            if component_size >= self.min_cluster_size:
                expanded_mask = np.maximum(expanded_mask, component_mask)

        # Morphological opening to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, kernel_open)

        # Convert back to 0-255 range
        expanded_mask = (expanded_mask * 255).astype(np.uint8)

        return expanded_mask

    def transform_and_combine_pointclouds(self, points01, points02, timestamp):
        """
        Transform both pointclouds to base_link frame, concatenate, and downsample.

        Args:
            points01: Nx3 numpy array of points in camera_01_link frame (or None)
            points02: Nx3 numpy array of points in camera_02_link frame (or None)
            timestamp: rospy.Time timestamp for TF lookup

        Returns:
            Nx3 numpy array of combined points in base_link frame, or None if both inputs are None
        """
        combined_points = []
        points01_base = None
        points02_base = None

        # Transform camera 01 points to base_link
        if points01 is not None and len(points01) > 0:
            try:
                # Lookup transform from camera to base_link
                # Use Time() to get the latest available transform (like rospy.Time(0) in ROS1)
                from rclpy.time import Time
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_01_link', Time(), rclpy.duration.Duration(seconds=1.0)
                )

                # Extract rotation and translation
                trans = transform.transform.translation
                rot = transform.transform.rotation

                # Convert quaternion to rotation matrix
                # Quaternion: [x, y, z, w]
                qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])

                # Transform points: R @ p + t
                t = np.array([trans.x, trans.y, trans.z])
                points01_base = (R @ points01.T).T + t
                combined_points.append(points01_base)

            except (LookupException, ConnectivityException,
                    ExtrapolationException) as e:
                self.get_logger().warning("Failed to lookup transform for camera 01: %s" % str(e))

        # Transform camera 02 points to base_link
        if points02 is not None and len(points02) > 0:
            try:
                # Lookup transform from camera to base_link
                # Use Time() to get the latest available transform (like rospy.Time(0) in ROS1)
                from rclpy.time import Time
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_02_link', Time(), rclpy.duration.Duration(seconds=1.0)
                )

                # Extract rotation and translation
                trans = transform.transform.translation
                rot = transform.transform.rotation

                # Convert quaternion to rotation matrix
                qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])

                # Transform points: R @ p + t
                t = np.array([trans.x, trans.y, trans.z])
                points02_base = (R @ points02.T).T + t
                combined_points.append(points02_base)

            except (LookupException, ConnectivityException,
                    ExtrapolationException) as e:
                self.get_logger().warning("Failed to lookup transform for camera 02: %s" % str(e))

        # Concatenate all transformed points
        if len(combined_points) == 0:
            return None, None, None

        combined = np.vstack(combined_points)

        # Return combined and individual camera pointclouds for later processing
        return combined, points01_base, points02_base

    def remove_ground_plane(self, points):
        """
        Remove ground points by simple height filtering.
        Removes all points below the height threshold in base_link frame.

        Args:
            points: Nx3 numpy array of points in base_link frame (X-forward, Y-left, Z-up)

        Returns:
            filtered_points: Mx3 numpy array with ground points removed
        """
        if points is None or len(points) < self.min_points_above_ground:
            return points

        if not self.ground_removal_enabled:
            return points

        try:
            n_points = len(points)

            # Filter by Z-axis height (Z-up in base_link frame)
            # Keep only points above the height threshold
            above_ground = points[:, 2] > self.ground_height_threshold

            filtered_points = points[above_ground]

            # Safety check: ensure we didn't remove all points
            if len(filtered_points) < self.min_points_above_ground:
                self.get_logger().warning("[Ground Removal] Too few points remaining (%d), returning original" % len(filtered_points))
                return points

            return filtered_points

        except Exception as e:
            self.get_logger().error("[Ground Removal] Failed: %s" % str(e))
            import traceback
            self.get_logger().error(traceback.format_exc())
            return points

    def process_frame(self, event=None):
        """Main processing loop"""
        with self.lock:
            if self.camera_01_img is None or self.camera_02_img is None:
                return

            # Make copies and downsample for faster processing on Jetson
            # 1280x720 -> 640x360 (0.5x scale)
            img01 = cv2.resize(self.camera_01_img, None, fx=self.input_scale, fy=self.input_scale,
                              interpolation=cv2.INTER_AREA)
            img02 = cv2.resize(self.camera_02_img, None, fx=self.input_scale, fy=self.input_scale,
                              interpolation=cv2.INTER_AREA)

        # Stitch downsampled images
        stitched_bgr = self.stitch_images(img01, img02)
        orig_size = stitched_bgr.shape[:2]

        # Convert to tensor
        frame_tensor, frame_rgb = self.frame_to_tensor(stitched_bgr)

        # Store frame in buffer (only needed before tracking initialization)
        if not self.tracking_initialized:
            self.frame_buffer.append({
                'tensor': frame_tensor,
                'rgb': frame_rgb,
                'bgr': stitched_bgr,
                'size': orig_size
            })
        # After tracking starts, clear buffer to save memory
        elif len(self.frame_buffer) > 0:
            self.frame_buffer.clear()
            self.get_logger().info("Frame buffer cleared after tracking initialization")

        mask = None

        if not self.tracking_initialized:
            # Detection phase: look for first mask
            self.get_logger().info("[YOLO] Running initial detection...")
            mask, num_det, detect_time = self.extract_mask_keep_middle(stitched_bgr)

            if mask is not None:
                self.get_logger().info("[YOLO Detection] Frame %d: %.3f ms (%d detections)" % (self.current_frame_idx, detect_time * 1000, num_det))
                self.first_mask = mask
                # mask_frame_idx should be the last index in the frame buffer (most recent frame)
                self.mask_frame_idx = len(self.frame_buffer) - 1

                # Initialize tracking
                try:
                    self.initialize_tracking(mask, orig_size)
                except Exception as e:
                    self.get_logger().error("Failed to initialize tracking: %s" % str(e))
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    self.tracking_initialized = False

        else:
            # Tracking phase - first track, then check if target is lost
            # Always try to track first
            try:
                mask, track_time = self.track_frame(frame_tensor, orig_size)
                self.get_logger().info("[STCN Tracking] Frame %d: %.3f ms" % (self.current_frame_idx, track_time * 1000))
            except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.get_logger().error("OOM! Clearing memory and reducing memory bank...")
                        # Emergency cleanup
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()

                        # Reduce memory bank size by half
                        for bank in self.mem_banks.values():
                            if bank.mem_k is not None and bank.mem_k.shape[2] > 1000:
                                chunk = bank.mem_k.shape[2] // 2
                                old_k, old_v = bank.mem_k, bank.mem_v
                                bank.mem_k = bank.mem_k[:, :, chunk:].contiguous()
                                bank.mem_v = bank.mem_v[:, :, chunk:].contiguous()
                                del old_k, old_v

                        torch.cuda.empty_cache()
                        mask = np.zeros(orig_size, dtype=np.uint8)
                    else:
                        self.get_logger().error("Tracking failed: %s" % str(e))
                        mask = np.zeros(orig_size, dtype=np.uint8)
            except Exception as e:
                import traceback
                self.get_logger().error("Tracking failed: %s" % str(e))
                self.get_logger().error("Traceback: %s" % traceback.format_exc())
                mask = None

            # Improved: Check tracking quality with detailed diagnosis
            is_lost, quality_score, diagnosis = self.is_tracking_lost(mask)

            if is_lost:
                self.poor_tracking_count += 1
                diag_str = ", ".join([k for k, v in diagnosis.items() if v])
                self.get_logger().warning("Poor tracking detected [%s] - Quality: %.2f - Frame %d (%d/%d)" %
                            (diag_str, quality_score, self.current_frame_idx,
                             self.poor_tracking_count, self.tracking_lost_threshold))

                # Improved: Use fallback mask strategy first
                if self.last_valid_mask is not None and quality_score > 0.2:
                    # If tracking is slightly degraded but not completely lost,
                    # use the last valid mask instead of empty mask
                    self.get_logger().info("Using fallback mask from frame %d (area: %d pixels)" %
                                (self.current_frame_idx - 1, self.last_valid_mask_area))
                    mask = self.last_valid_mask.copy()
                    # Don't increment poor_tracking_count if we recovered with fallback
                    self.poor_tracking_count = max(0, self.poor_tracking_count - 1)

                # Only run YOLO if consistently lost for multiple frames
                if self.poor_tracking_count >= self.tracking_lost_threshold:
                    self.get_logger().info("[YOLO] Running re-detection due to tracking loss...")
                    new_mask, num_det, detect_time = self.extract_mask_keep_middle(stitched_bgr)

                    if new_mask is not None:
                        self.get_logger().info("[YOLO Re-detect] Frame %d: %.3f ms (%d detections)" % (self.current_frame_idx, detect_time * 1000, num_det))

                        # Always use incremental update strategy
                        # Get current tracking mask for comparison
                        current_tracking_mask = mask if mask is not None else None

                        # Compute increment mask (new pixels from YOLO)
                        if current_tracking_mask is not None:
                            increment_mask = self.compute_mask_increment(current_tracking_mask, new_mask)
                        else:
                            # If no current mask, use full YOLO mask as increment
                            increment_mask = new_mask

                        # Update memory bank with increment
                        self.update_memory_bank_with_increment(frame_tensor, increment_mask, orig_size)

                        # Update mask to be the YOLO mask (provides new information)
                        mask = new_mask
                        self.poor_tracking_count = 0  # Reset counter
                        self.get_logger().info("[YOLO Update Strategy] Incremental update completed")
                    else:
                        self.get_logger().warning("YOLO re-detection failed, using fallback mask")
                        if self.last_valid_mask is not None:
                            mask = self.last_valid_mask.copy()
                        else:
                            mask = np.zeros(orig_size, dtype=np.uint8)
                elif self.last_valid_mask is None:
                    # First poor frame and no fallback available, use empty
                    mask = np.zeros(orig_size, dtype=np.uint8)
            else:
                # Tracking is good, reset counter and save as valid mask
                self.poor_tracking_count = 0
                mask_area = np.count_nonzero(mask) if mask is not None else 0
                if mask_area >= self.min_mask_area:
                    self.last_valid_mask = mask.copy()
                    self.last_valid_mask_area = mask_area

                # Periodic YOLO execution for mask refinement
                frames_since_last_yolo = self.current_frame_idx - self.last_yolo_frame
                if frames_since_last_yolo >= self.yolo_period_frames:
                    self.get_logger().info("[YOLO] Running periodic detection (every %d frames)..." % self.yolo_period_frames)
                    yolo_mask, num_det, detect_time = self.extract_mask_keep_middle(stitched_bgr)
                    self.last_yolo_frame = self.current_frame_idx

                    if yolo_mask is not None and num_det > 0:
                        self.get_logger().info("[YOLO Periodic] Frame %d: %.3f ms (%d detections)" %
                                     (self.current_frame_idx, detect_time * 1000, num_det))

                        # Merge YOLO mask with tracking mask (union operation)
                        current_tracking_mask = mask if mask is not None else np.zeros(orig_size, dtype=np.uint8)
                        merged_mask = cv2.bitwise_or(current_tracking_mask, yolo_mask)

                        # Compute increment (what YOLO added)
                        increment_mask = self.compute_mask_increment(current_tracking_mask, merged_mask)
                        increment_pixels = np.count_nonzero(increment_mask > 128)

                        if increment_pixels > 100:
                            # Update memory bank with the increment
                            self.update_memory_bank_with_increment(frame_tensor, increment_mask, orig_size)
                            self.get_logger().info("[YOLO Periodic] Merged masks: +%d pixels from YOLO" % increment_pixels)
                        else:
                            self.get_logger().info("[YOLO Periodic] YOLO mask adds no new pixels")

                        # Use merged mask for this frame
                        mask = merged_mask

        # Publish pointclouds for each camera
        if mask is not None:
            # Split mask back to individual cameras
            # The stitched image was created by stacking top and bottom
            # We need to reverse this process
            with self.lock:
                depth01 = self.camera_01_depth.copy() if self.camera_01_depth is not None else None
                depth02 = self.camera_02_depth.copy() if self.camera_02_depth is not None else None

            if depth01 is not None and depth02 is not None:
                # Reverse the stitching and warping transformations to get masks in original camera space
                h1, w1 = img01.shape[:2]
                h2, w2 = img02.shape[:2]

                # Calculate the split point based on how images were stitched
                # The stitched mask width is the common width after warping and resizing
                w_common = mask.shape[1]
                scale_top = w_common / img01.shape[1]
                scale_bottom = w_common / img02.shape[1]

                top_h = int(round(img01.shape[0] * scale_top))
                bot_h = int(round(img02.shape[0] * scale_bottom))

                # Split the stitched mask
                mask_top_resized = mask[:top_h, :]
                mask_bottom_resized = mask[top_h:, :]

                # Reverse the resize for top camera (to downsampled size)
                mask_top_warped = cv2.resize(mask_top_resized, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Reverse the resize for bottom camera (to downsampled size)
                mask_bottom_warped = cv2.resize(mask_bottom_resized, (img02.shape[1], img02.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Get original camera image sizes (1280x720)
                with self.lock:
                    orig_h1, orig_w1 = self.camera_01_img.shape[:2] if self.camera_01_img is not None else (720, 1280)
                    orig_h2, orig_w2 = self.camera_02_img.shape[:2] if self.camera_02_img is not None else (720, 1280)

                # Reverse the perspective warp for camera 01 (to downsampled size first)
                offset0 = int(img01.shape[1] * self.angle_degrees / 90)
                src0 = np.float32([[0, 0], [img01.shape[1], 0], [img01.shape[1] - offset0, img01.shape[0]], [offset0, img01.shape[0]]])
                dst0 = np.float32([[0, 0], [img01.shape[1], 0], [img01.shape[1], img01.shape[0]], [0, img01.shape[0]]])
                M0_inv = cv2.getPerspectiveTransform(src0, dst0)
                mask_cam01_ds = cv2.warpPerspective(mask_top_warped, M0_inv, (img01.shape[1], img01.shape[0]), flags=cv2.INTER_NEAREST)

                # Upscale mask to original resolution for depth (1280x720)
                mask_cam01 = cv2.resize(mask_cam01_ds, (orig_w1, orig_h1), interpolation=cv2.INTER_NEAREST)

                # Reverse the perspective warp for camera 02 (to downsampled size first)
                offset1 = int(img02.shape[1] * self.angle_degrees / 90)
                src1 = np.float32([[offset1, 0], [img02.shape[1] - offset1, 0], [img02.shape[1], img02.shape[0]], [0, img02.shape[0]]])
                dst1 = np.float32([[0, 0], [img02.shape[1], 0], [img02.shape[1], img02.shape[0]], [0, img02.shape[0]]])
                M1_inv = cv2.getPerspectiveTransform(src1, dst1)
                mask_cam02_ds = cv2.warpPerspective(mask_bottom_warped, M1_inv, (img02.shape[1], img02.shape[0]), flags=cv2.INTER_NEAREST)

                # Upscale mask to original resolution for depth (1280x720)
                mask_cam02 = cv2.resize(mask_cam02_ds, (orig_w2, orig_h2), interpolation=cv2.INTER_NEAREST)

                # Apply depth-based region growing to expand incomplete masks
                # This uses depth clustering to recover missing parts of the human
                # Temporarily disabled to debug pointcloud generation
                mask_cam01_expanded = self.grow_mask_with_depth_clustering(mask_cam01, depth01)
                mask_cam02_expanded = self.grow_mask_with_depth_clustering(mask_cam02, depth02)

                # Use expanded masks for pointcloud generation
                mask_cam01 = mask_cam01_expanded
                mask_cam02 = mask_cam02_expanded

                # Publish masks for debugging - use original camera timestamps
                try:
                    mask_msg01 = self.bridge.cv2_to_imgmsg(mask_cam01, encoding="mono8")
                    # Use original camera timestamp for proper synchronization in RViz
                    mask_msg01.header.stamp = self.camera_01_timestamp if self.camera_01_timestamp else self.get_clock().now().to_msg()
                    mask_msg01.header.frame_id = 'camera_01_link'
                    self.pub_mask01.publish(mask_msg01)

                    mask_msg02 = self.bridge.cv2_to_imgmsg(mask_cam02, encoding="mono8")
                    # Use original camera timestamp for proper synchronization in RViz
                    mask_msg02.header.stamp = self.camera_02_timestamp if self.camera_02_timestamp else self.get_clock().now().to_msg()
                    mask_msg02.header.frame_id = 'camera_02_link'
                    self.pub_mask02.publish(mask_msg02)
                except Exception as e:
                    self.get_logger().error("Failed to publish masks: %s" % str(e))

                # Generate pointclouds from both cameras with seam-based downsampling
                timestamp = self.get_clock().now().to_msg()
                points01, points02 = None, None

                # Calculate seam position in downsampled image space (depth images are at 640x360)
                # Camera 01 is top, camera 02 is bottom
                # Seam is at the bottom of camera 01's view
                cam01_height_ds = depth01.shape[0] // 2  # Downsampled height
                seam_v_cam01 = cam01_height_ds - 1  # Bottom of camera 01

                # For camera 02, seam is at the top
                seam_v_cam02 = 0  # Top of camera 02

                # Extract points from camera 01 with seam-based downsampling
                if self.cam01_intrinsics is not None:
                    result = self.depth_to_pointcloud(
                        depth01, mask_cam01, 'camera_01_link',
                        self.cam01_intrinsics, timestamp, max_points=None
                    )
                    if result[0] is not None:
                        points01_raw = result[0]
                        v_coords01 = result[2]

                        # Apply seam-based downsampling
                        if v_coords01 is not None:
                            points01 = self.downsample_by_seam_distance(
                                points01_raw, v_coords01, 'camera_01_link', seam_v_cam01
                            )
                        else:
                            points01 = points01_raw

                # Extract points from camera 02 with seam-based downsampling
                if self.cam02_intrinsics is not None:
                    result = self.depth_to_pointcloud(
                        depth02, mask_cam02, 'camera_02_link',
                        self.cam02_intrinsics, timestamp, max_points=None
                    )
                    if result[0] is not None:
                        points02_raw = result[0]
                        v_coords02 = result[2]

                        # Apply seam-based downsampling
                        if v_coords02 is not None:
                            points02 = self.downsample_by_seam_distance(
                                points02_raw, v_coords02, 'camera_02_link', seam_v_cam02
                            )
                        else:
                            points02 = points02_raw

                # Transform both pointclouds to base_link and concatenate (no downsampling yet)
                combined_points, points01_base, points02_base = self.transform_and_combine_pointclouds(
                    points01, points02, timestamp
                )

                # Remove ground plane FIRST, then downsample
                if combined_points is not None:
                    points_before_ground = len(combined_points)
                    combined_points = self.remove_ground_plane(combined_points)

                    if combined_points is not None:
                        points_after_ground = len(combined_points)
                        self.get_logger().info("[Ground Removal] %d -> %d points (removed %d)" %
                                             (points_before_ground, points_after_ground,
                                              points_before_ground - points_after_ground))

                        # Now downsample AFTER ground removal
                        if len(combined_points) > 5000:
                            indices = np.random.choice(len(combined_points), 5000, replace=False)
                            combined_points = combined_points[indices]
                            self.get_logger().info("[Downsample] %d -> 5000 points" % points_after_ground)

                # Publish combined pointcloud in base_link frame
                if combined_points is not None:
                    from std_msgs.msg import Header
                    header = Header()
                    header.stamp = timestamp
                    header.frame_id = 'base_link'
                    fields = [
                        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    ]
                    pc_msg = pc2.create_cloud(header, fields, combined_points)
                    self.pub_human_pc.publish(pc_msg)

        self.current_frame_idx += 1

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'yolo_process') and self.yolo_process:
            self.yolo_process.kill()
            self.yolo_process = None


if __name__ == "__main__":
    rclpy.init()
    try:
        tracker = RealtimeSTCNTracker()
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Fatal error: %s" % str(e))
        import traceback
        traceback.print_exc()
    finally:
        if 'tracker' in locals():
            tracker.destroy_node()
        rclpy.shutdown()
