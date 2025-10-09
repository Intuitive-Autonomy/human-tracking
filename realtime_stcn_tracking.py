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
from sensor_msgs.msg import Image
from collections import deque
import threading
import subprocess
import pickle

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

        # Configuration
        self.conf_threshold = 0.3
        self.resolution = 480
        self.angle_degrees = 25
        self.max_memory_frames = 50  # 减少缓存帧数以节省显存
        self.mem_every = 10  # 增加间隔，减少memory bank更新频率
        self.top_k = 10  # 减少top_k以节省显存
        self.reset_every = 120  # 每120帧重新运行YOLO重置mask

        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Image buffers
        self.camera_01_img = None
        self.camera_02_img = None
        self.last_timestamp = None

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
            # Limit to ~2GB (25% of 8GB GPU) - need more for stable tracking
            torch.cuda.set_per_process_memory_fraction(0.25, device=0)
            torch.cuda.empty_cache()
            self.get_logger().info(f"GPU memory limited to ~2GB")

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
        self.get_logger().info("Models loaded successfully!")

        # QoS profile for image topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_cam01 = self.create_subscription(
            Image, '/camera_01/color/image_raw', self.callback_cam01, qos_profile
        )
        self.sub_cam02 = self.create_subscription(
            Image, '/camera_02/color/image_raw', self.callback_cam02, qos_profile
        )

        # Timer for processing
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 Hz

        self.get_logger().info("ROS2 node initialized. Waiting for image messages...")

    def callback_cam01(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_01_img = img
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera_01 image: {e}")

    def callback_cam02(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_02_img = img
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera_02 image: {e}")

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
                self.yolo_process = subprocess.Popen(
                    ['python3', 'yolo_detector.py'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # Ignore stderr
                    bufsize=0,
                    cwd='/home/oliver/Documents/STCN'
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
                        self.get_logger().error(f"YOLO subprocess unexpected output: {line}")
                else:
                    self.get_logger().error("YOLO subprocess timeout (15s)")
                    if self.yolo_process:
                        self.yolo_process.kill()
                        self.yolo_process = None
            except Exception as e:
                self.get_logger().error(f"Failed to start YOLO subprocess: {e}")
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
            return None, 0

        try:
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
                self.get_logger().error(f"Incomplete result: got {len(result_data)}, expected {result_size}")
                raise RuntimeError("Incomplete data")

            result = pickle.loads(result_data)

            return result['mask'], result['num_det']

        except Exception as e:
            self.get_logger().error(f"YOLO subprocess communication error: {e}")
            self.yolo_ready = False
            if self.yolo_process:
                self.yolo_process.kill()
                self.yolo_process = None
            return None, 0

    def frame_to_tensor(self, frame_bgr):
        """Convert BGR frame to STCN input tensor"""
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.im_transform(frame_pil)
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
        self.get_logger().info(f"✓ STCN tracking initialized at frame {self.mask_frame_idx}")

    def track_frame(self, frame_tensor, orig_size):
        """Track single frame using STCN"""
        # Clear CUDA cache more aggressively
        if self.current_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            if self.current_frame_idx % 30 == 0:
                # Force garbage collection
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

        return mask

    def visualize(self, rgb_frame, mask):
        """Visualize tracking result"""
        # Create overlay
        overlay = rgb_frame.copy()
        overlay[mask > 128] = [255, 0, 0]  # Red for mask
        blend = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)

        # Create display
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Side by side: original, mask, overlay
        h, w = rgb_frame.shape[:2]
        display = np.hstack([
            cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
            mask_3ch,
            cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
        ])

        # Resize display to fit screen (max width 1920 pixels)
        display_h, display_w = display.shape[:2]
        max_width = 1920
        if display_w > max_width:
            scale = max_width / display_w
            new_w = int(display_w * scale)
            new_h = int(display_h * scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Add text
        status = "TRACKING" if self.tracking_initialized else "DETECTING"
        color = (0, 255, 0) if self.tracking_initialized else (0, 165, 255)
        cv2.putText(display, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Frame: {self.current_frame_idx}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if self.tracking_initialized:
            cv2.putText(display, f"Init Frame: {self.mask_frame_idx}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('STCN Real-time Tracking', display)
        cv2.waitKey(1)

    def process_frame(self):
        """Main processing loop"""
        with self.lock:
            if self.camera_01_img is None or self.camera_02_img is None:
                return

            # Make copies
            img01 = self.camera_01_img.copy()
            img02 = self.camera_02_img.copy()

        # Stitch images
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
            if self.current_frame_idx % 30 == 0:  # Throttle logging
                self.get_logger().info(f"Detecting... Frame {self.current_frame_idx}")
            mask, num_det = self.extract_mask_keep_middle(stitched_bgr)

            if mask is not None:
                self.get_logger().info(f"✓ Found first mask at frame {self.current_frame_idx} ({num_det} detections)")
                self.first_mask = mask
                self.mask_frame_idx = self.current_frame_idx

                # Initialize tracking
                try:
                    self.initialize_tracking(mask, orig_size)
                except Exception as e:
                    self.get_logger().error(f"Failed to initialize tracking: {e}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    self.tracking_initialized = False
            else:
                # Show frame without mask during detection
                self.visualize(frame_rgb, np.zeros(orig_size, dtype=np.uint8))

        else:
            # Tracking phase - check if need to reset
            frames_since_init = self.current_frame_idx - self.mask_frame_idx

            if frames_since_init > 0 and frames_since_init % self.reset_every == 0:
                # Time to reset tracking with new YOLO detection
                self.get_logger().info(f"Resetting tracking at frame {self.current_frame_idx} (every {self.reset_every} frames)")

                # Run YOLO detection on current frame
                new_mask, num_det = self.extract_mask_keep_middle(stitched_bgr)

                if new_mask is not None:
                    self.get_logger().info(f"✓ Reset mask found ({num_det} detections)")
                    # Clear old memory banks properly
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
                        # Initialize temp attributes that aren't set in __init__
                        bank.temp_k = None
                        bank.temp_v = None
                        self.mem_banks[oi] = bank

                    # Re-initialize tracking with new mask
                    # Clear buffer and add current frame at index 0
                    self.frame_buffer.clear()
                    self.frame_buffer.append({
                        'tensor': frame_tensor,
                        'rgb': frame_rgb,
                        'bgr': stitched_bgr,
                        'size': orig_size
                    })
                    self.mask_frame_idx = 0  # Frame is at index 0 in buffer

                    try:
                        self.initialize_tracking(new_mask, orig_size)
                        mask = new_mask
                        self.frame_buffer.clear()  # Clear after re-init
                        self.mask_frame_idx = self.current_frame_idx  # Reset to actual frame number
                        self.get_logger().info(f"✓ Tracking re-initialized successfully")
                    except Exception as e:
                        self.get_logger().error(f"Failed to re-initialize tracking: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        mask = np.zeros(orig_size, dtype=np.uint8)
                else:
                    self.get_logger().warn("Reset failed: no mask found, continuing with old tracking")
                    # Continue with normal tracking
                    try:
                        mask = self.track_frame(frame_tensor, orig_size)
                    except Exception as e:
                        self.get_logger().error(f"Tracking failed: {e}")
                        mask = np.zeros(orig_size, dtype=np.uint8)
            else:
                # Normal tracking
                try:
                    mask = self.track_frame(frame_tensor, orig_size)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.get_logger().error(f"OOM! Clearing memory and reducing memory bank...")
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
                        self.get_logger().error(f"Tracking failed: {e}")
                        mask = np.zeros(orig_size, dtype=np.uint8)
                except Exception as e:
                    import traceback
                    self.get_logger().error(f"Tracking failed: {e}")
                    self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                    mask = np.zeros(orig_size, dtype=np.uint8)

        # Visualize
        if mask is not None:
            self.visualize(frame_rgb, mask)

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
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        tracker.destroy_node()
        rclpy.shutdown()
