#!/usr/bin/env python
"""
Real-time STCN Tracking ROS1 Node
订阅两个摄像头话题，拼接后实时跟踪人体mask并可视化

Subscribes to:
  - /ob_camera_01/color/image_raw
  - /ob_camera_02/color/image_raw

Workflow:
1) 收集图像帧，从第一帧开始用YOLO检测人体mask (keep_middle策略)
2) 找到第一个有效mask后，初始化STCN跟踪器
3) 后续帧使用STCN实时跟踪并可视化结果
"""

# Set environment variables BEFORE importing any libraries
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from collections import deque
import threading
import subprocess
import pickle
import time
import struct
import tf2_ros

# Import PyTorch components first
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model.eval_network import STCN
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot
from model.aggregate import aggregate


class RealtimeSTCNTracker(object):
    def __init__(self):
        rospy.init_node('realtime_stcn_tracker', anonymous=True)

        # Configuration - Optimized for Jetson
        self.conf_threshold = 0.3
        self.resolution = 480  # Reduced from 720 for faster inference on Jetson
        self.angle_degrees = 25
        self.input_scale = 0.5  # Downsample 1280x720 -> 640x360 before stitching
        self.max_memory_frames = 15  # Reduced for Jetson memory constraints
        self.mem_every = 60  # Update memory every 60 frames (2 sec at 30fps) for smoother performance
        self.top_k = 5  # Reduced for faster processing on Jetson

        # Tracking quality thresholds - Only run YOLO when tracking is lost
        self.min_mask_area = 500  # Minimum pixels for valid tracking
        self.tracking_lost_threshold = 3  # Consecutive frames with poor tracking
        self.poor_tracking_count = 0  # Counter for poor tracking frames

        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Image buffers
        self.camera_01_img = None
        self.camera_02_img = None
        self.camera_01_depth = None
        self.camera_02_depth = None
        self.last_timestamp = None

        # Camera intrinsics - will be populated from camera_info topics
        self.cam01_intrinsics = None
        self.cam02_intrinsics = None

        # TF buffer for transforming pointclouds to base_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
            rospy.loginfo("GPU memory limited to 8GB")

        # Use subprocess for YOLO to isolate CUDA context
        rospy.loginfo("YOLO will run in isolated subprocess...")
        self.yolo_process = None
        self.yolo_ready = False

        rospy.loginfo("Loading STCN model...")
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
            rospy.loginfo("STCN model converted to FP16 for Jetson acceleration")

        rospy.loginfo("Models loaded successfully!")

        # Subscribers - Camera Info
        self.sub_info01 = rospy.Subscriber(
            '/ob_camera_01/depth/camera_info', CameraInfo, self.callback_info01, queue_size=1
        )
        self.sub_info02 = rospy.Subscriber(
            '/ob_camera_02/depth/camera_info', CameraInfo, self.callback_info02, queue_size=1
        )

        # Subscribers - Color images
        self.sub_cam01 = rospy.Subscriber(
            '/ob_camera_01/color/image_raw', Image, self.callback_cam01_color, queue_size=1
        )
        self.sub_cam02 = rospy.Subscriber(
            '/ob_camera_02/color/image_raw', Image, self.callback_cam02_color, queue_size=1
        )

        # Subscribers - Depth images
        self.sub_cam01_depth = rospy.Subscriber(
            '/ob_camera_01/depth/image_raw', Image, self.callback_cam01_depth, queue_size=1
        )
        self.sub_cam02_depth = rospy.Subscriber(
            '/ob_camera_02/depth/image_raw', Image, self.callback_cam02_depth, queue_size=1
        )

        # Publisher - Combined human pointcloud in base_link frame
        self.pub_human_pc = rospy.Publisher(
            '/human_pointcloud', PointCloud2, queue_size=1
        )

        # Publishers - Masks (for debugging)
        self.pub_mask01 = rospy.Publisher(
            '/ob_camera_01/human_mask', Image, queue_size=1
        )
        self.pub_mask02 = rospy.Publisher(
            '/ob_camera_02/human_mask', Image, queue_size=1
        )

        # Timer for processing
        self.timer = rospy.Timer(rospy.Duration(0.033), self.process_frame)  # ~30 Hz

        rospy.loginfo("ROS1 node initialized. Waiting for image messages...")

    def callback_cam01_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_01_img = img
        except Exception as e:
            rospy.logerr("Failed to convert camera_01 color image: %s" % str(e))

    def callback_cam02_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_02_img = img
        except Exception as e:
            rospy.logerr("Failed to convert camera_02 color image: %s" % str(e))

    def callback_cam01_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.camera_01_depth = depth
        except Exception as e:
            rospy.logerr("Failed to convert camera_01 depth image: %s" % str(e))

    def callback_cam02_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.camera_02_depth = depth
        except Exception as e:
            rospy.logerr("Failed to convert camera_02 depth image: %s" % str(e))

    def callback_info01(self, msg):
        """Callback for camera 01 camera_info"""
        if self.cam01_intrinsics is None:
            # Extract intrinsics from camera info K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.cam01_intrinsics = {
                'fx': msg.K[0],
                'fy': msg.K[4],
                'cx': msg.K[2],
                'cy': msg.K[5]
            }
            rospy.loginfo("Camera 01 intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
                         (self.cam01_intrinsics['fx'], self.cam01_intrinsics['fy'],
                          self.cam01_intrinsics['cx'], self.cam01_intrinsics['cy']))

    def callback_info02(self, msg):
        """Callback for camera 02 camera_info"""
        if self.cam02_intrinsics is None:
            # Extract intrinsics from camera info K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.cam02_intrinsics = {
                'fx': msg.K[0],
                'fy': msg.K[4],
                'cx': msg.K[2],
                'cy': msg.K[5]
            }
            rospy.loginfo("Camera 02 intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
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
            rospy.loginfo("Starting YOLO subprocess...")
            try:
                self.yolo_process = subprocess.Popen(
                    ['python3', 'yolo_detector.py'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # Ignore stderr
                    bufsize=0,
                    cwd='/home/woosh/Documents/human-tracking'
                )
                # Wait for READY signal from stdout
                import select
                ready = select.select([self.yolo_process.stdout], [], [], 15.0)
                if ready[0]:
                    line = self.yolo_process.stdout.readline().decode().strip()
                    if "READY" in line:
                        self.yolo_ready = True
                        rospy.loginfo("YOLO subprocess ready!")
                    else:
                        rospy.logerr("YOLO subprocess unexpected output: %s" % line)
                else:
                    rospy.logerr("YOLO subprocess timeout (15s)")
                    if self.yolo_process:
                        self.yolo_process.kill()
                        self.yolo_process = None
            except Exception as e:
                rospy.logerr("Failed to start YOLO subprocess: %s" % str(e))
                import traceback
                rospy.logerr(traceback.format_exc())
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
                rospy.logerr("Failed to read result size from YOLO subprocess")
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
                rospy.logerr("Incomplete result: got %d, expected %d" % (len(result_data), result_size))
                raise RuntimeError("Incomplete data")

            result = pickle.loads(result_data)

            detection_time = time.time() - start_time

            return result['mask'], result['num_det'], detection_time

        except Exception as e:
            rospy.logerr("YOLO subprocess communication error: %s" % str(e))
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

    def is_tracking_lost(self, mask):
        """Check if tracking quality is too poor (target lost)"""
        if mask is None:
            return True

        # Calculate mask area (number of non-zero pixels)
        mask_area = np.count_nonzero(mask)

        # Check if mask area is below threshold
        if mask_area < self.min_mask_area:
            return True

        return False

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

        # Extract valid points
        x_cam_valid = x_cam[valid_mask]
        y_cam_valid = y_cam[valid_mask]
        z_cam_valid = z_cam[valid_mask]

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
            return None, 0

        # Downsample if max_points is specified
        if max_points is not None and len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        # Return points and original count
        return points, len(x_cam_valid)

    def transform_and_combine_pointclouds(self, points01, points02, timestamp):
        """
        Transform both pointclouds to base_link frame, concatenate, and downsample.

        Args:
            points01: Nx3 numpy array of points in ob_camera_01_link frame (or None)
            points02: Nx3 numpy array of points in ob_camera_02_link frame (or None)
            timestamp: rospy.Time timestamp for TF lookup

        Returns:
            Nx3 numpy array of combined points in base_link frame, or None if both inputs are None
        """
        combined_points = []

        # Transform camera 01 points to base_link
        if points01 is not None and len(points01) > 0:
            try:
                # Lookup transform from camera to base_link
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'ob_camera_01_link', timestamp, rospy.Duration(0.1)
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

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Failed to lookup transform for camera 01: %s" % str(e))

        # Transform camera 02 points to base_link
        if points02 is not None and len(points02) > 0:
            try:
                # Lookup transform from camera to base_link
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'ob_camera_02_link', timestamp, rospy.Duration(0.1)
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

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Failed to lookup transform for camera 02: %s" % str(e))

        # Concatenate all transformed points
        if len(combined_points) == 0:
            return None

        combined = np.vstack(combined_points)

        # Downsample to 5000 points if needed
        if len(combined) > 5000:
            indices = np.random.choice(len(combined), 5000, replace=False)
            combined = combined[indices]

        return combined

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
            rospy.loginfo("Frame buffer cleared after tracking initialization")

        mask = None

        if not self.tracking_initialized:
            # Detection phase: look for first mask
            mask, num_det, detect_time = self.extract_mask_keep_middle(stitched_bgr)

            if mask is not None:
                rospy.loginfo("[YOLO Detection] Frame %d: %.3f ms (%d detections)" % (self.current_frame_idx, detect_time * 1000, num_det))
                self.first_mask = mask
                self.mask_frame_idx = self.current_frame_idx

                # Initialize tracking
                try:
                    self.initialize_tracking(mask, orig_size)
                except Exception as e:
                    rospy.logerr("Failed to initialize tracking: %s" % str(e))
                    import traceback
                    rospy.logerr(traceback.format_exc())
                    self.tracking_initialized = False

        else:
            # Tracking phase - first track, then check if target is lost
            # Always try to track first
            try:
                mask, track_time = self.track_frame(frame_tensor, orig_size)
                rospy.loginfo("[STCN Tracking] Frame %d: %.3f ms" % (self.current_frame_idx, track_time * 1000))
            except RuntimeError as e:
                    if "out of memory" in str(e):
                        rospy.logerr("OOM! Clearing memory and reducing memory bank...")
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
                        rospy.logerr("Tracking failed: %s" % str(e))
                        mask = np.zeros(orig_size, dtype=np.uint8)
            except Exception as e:
                import traceback
                rospy.logerr("Tracking failed: %s" % str(e))
                rospy.logerr("Traceback: %s" % traceback.format_exc())
                mask = None

            # Check tracking quality - Only run YOLO when target is lost
            if self.is_tracking_lost(mask):
                self.poor_tracking_count += 1
                rospy.logwarn("Poor tracking quality detected (%d/%d)" % (self.poor_tracking_count, self.tracking_lost_threshold))

                # Only run YOLO if consistently lost for multiple frames
                if self.poor_tracking_count >= self.tracking_lost_threshold:
                    rospy.logwarn("Tracking lost! Running YOLO to re-detect...")
                    new_mask, num_det, detect_time = self.extract_mask_keep_middle(stitched_bgr)

                    if new_mask is not None:
                        rospy.loginfo("[YOLO Re-detect] Frame %d: %.3f ms (%d detections)" % (self.current_frame_idx, detect_time * 1000, num_det))
                        # Reset tracking with new mask
                        try:
                            # Clear old memory banks
                            for bank in self.mem_banks.values():
                                if bank.mem_k is not None:
                                    del bank.mem_k, bank.mem_v
                                    bank.mem_k = None
                                    bank.mem_v = None
                            torch.cuda.empty_cache()

                            # Recreate memory banks
                            from inference_memory_bank import MemoryBank
                            self.mem_banks = {}
                            for oi in range(1, self.num_objects + 1):
                                bank = MemoryBank(k=1, top_k=self.top_k)
                                bank.temp_k = None
                                bank.temp_v = None
                                self.mem_banks[oi] = bank

                            # Re-initialize tracking
                            self.frame_buffer.clear()
                            self.frame_buffer.append({
                                'tensor': frame_tensor,
                                'rgb': frame_rgb,
                                'bgr': stitched_bgr,
                                'size': orig_size
                            })
                            self.mask_frame_idx = 0
                            self.initialize_tracking(new_mask, orig_size)
                            mask = new_mask
                            self.frame_buffer.clear()
                            self.mask_frame_idx = self.current_frame_idx
                            self.poor_tracking_count = 0  # Reset counter
                            rospy.loginfo("Tracking re-initialized successfully")
                        except Exception as e:
                            rospy.logerr("Failed to re-initialize tracking: %s" % str(e))
                            mask = np.zeros(orig_size, dtype=np.uint8)
                    else:
                        rospy.logwarn("YOLO re-detection failed, using empty mask")
                        mask = np.zeros(orig_size, dtype=np.uint8)
                else:
                    # Still tracking poorly but not threshold yet, use empty mask
                    mask = np.zeros(orig_size, dtype=np.uint8)
            else:
                # Tracking is good, reset counter
                self.poor_tracking_count = 0

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
                w_common = min(warped_top.shape[1], warped_bottom.shape[1]) if hasattr(self, 'warped_top') else mask.shape[1]
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

                # Publish masks for debugging
                try:
                    mask_msg01 = self.bridge.cv2_to_imgmsg(mask_cam01, encoding="mono8")
                    mask_msg01.header.stamp = rospy.Time.now()
                    mask_msg01.header.frame_id = 'ob_camera_01_link'
                    self.pub_mask01.publish(mask_msg01)

                    mask_msg02 = self.bridge.cv2_to_imgmsg(mask_cam02, encoding="mono8")
                    mask_msg02.header.stamp = rospy.Time.now()
                    mask_msg02.header.frame_id = 'ob_camera_02_link'
                    self.pub_mask02.publish(mask_msg02)
                except Exception as e:
                    rospy.logerr("Failed to publish masks: %s" % str(e))

                # Generate pointclouds from both cameras (without downsampling yet)
                timestamp = rospy.Time.now()
                points01, points02 = None, None

                # Extract points from camera 01
                if self.cam01_intrinsics is not None:
                    result = self.depth_to_pointcloud(
                        depth01, mask_cam01, 'ob_camera_01_link',
                        self.cam01_intrinsics, timestamp, max_points=None
                    )
                    if result[0] is not None:
                        points01 = result[0]

                # Extract points from camera 02
                if self.cam02_intrinsics is not None:
                    result = self.depth_to_pointcloud(
                        depth02, mask_cam02, 'ob_camera_02_link',
                        self.cam02_intrinsics, timestamp, max_points=None
                    )
                    if result[0] is not None:
                        points02 = result[0]

                # Transform both pointclouds to base_link, concatenate, and downsample to 5000 pts
                combined_points = self.transform_and_combine_pointclouds(points01, points02, timestamp)

                # Publish combined pointcloud in base_link frame
                if combined_points is not None:
                    header = rospy.Header()
                    header.stamp = timestamp
                    header.frame_id = 'base_link'
                    fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
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
    try:
        tracker = RealtimeSTCNTracker()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Fatal error: %s" % str(e))
        import traceback
        traceback.print_exc()
