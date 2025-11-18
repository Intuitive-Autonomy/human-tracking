#!/usr/bin/env python3
"""
YOLO-based Human Pointcloud Reconstruction with Kalman Filtered Cylinder Tracking

Subscribes to:
  - /camera_01/color/image_raw
  - /camera_02/color/image_raw
  - /camera_01/depth/image_raw
  - /camera_02/depth/image_raw
  - /camera_01/depth/camera_info
  - /camera_02/depth/camera_info

Publishes:
  - /human_pointcloud (PointCloud2 in base_footprint frame)
  - /human_mask (Image, stitched mask from YOLO)
  - /detected_human (Image, cropped human bounding box)
  - /human_cylinder (Marker, cylindrical bounding box with Kalman filtering)

Workflow:
1) Stitch two camera images using perspective warping
2) Run YOLO segmentation on stitched image to detect person
3) Split mask back to individual camera spaces
4) Generate pointclouds from depth + mask
5) Filter outliers based on cylindrical distance from central axis
6) Transform to base_footprint frame
7) Compute cylindrical bounding box (radius: 0.4-1.0m, height from ground)
8) Apply Kalman filter for smooth tracking with:
   - Innovation gating (reject measurements > 1.5m position jump)
   - Automatic reinitialization after 5 consecutive outliers
   - Prediction mode when no detection (yellow cylinder)
9) Publish filtered pointcloud and cylinder marker

Features:
  - IQR-based cylindrical outlier filtering
  - Kalman filter with constant velocity model
  - Robust to YOLO detection failures and target re-entry
  - Visual feedback: green (tracking) vs yellow (prediction)

Usage:
  ros2 run <package_name> realtime_yolo_tracking.py
"""

import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


class CylinderKalmanFilter:
    """
    Kalman Filter for tracking cylinder parameters (x, y, radius, height).
    State vector: [x, y, radius, height, vx, vy, vr, vh]
    """
    def __init__(self):
        # State: [x, y, radius, height, vx, vy, vr, vh]
        self.state = np.zeros(8)  # position + velocity

        # State covariance matrix
        self.P = np.eye(8) * 1000  # High initial uncertainty

        # Process noise (how much we expect state to change)
        self.Q = np.eye(8)
        self.Q[0:2, 0:2] *= 0.01   # XY position noise (small, people move smoothly)
        self.Q[2:4, 2:4] *= 0.001  # Size noise (very small, size doesn't change much)
        self.Q[4:6, 4:6] *= 0.1    # XY velocity noise (larger, velocity can change)
        self.Q[6:8, 6:8] *= 0.001  # Size velocity noise (very small, size velocity near zero)

        # Measurement noise (uncertainty in observations)
        self.R = np.eye(4)
        self.R[0, 0] = 0.05  # x measurement noise (5cm)
        self.R[1, 1] = 0.05  # y measurement noise (5cm)
        self.R[2, 2] = 0.1   # radius measurement noise (10cm)
        self.R[3, 3] = 0.1   # height measurement noise (10cm)

        # Measurement matrix (we only observe position, not velocity)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1  # Observe x
        self.H[1, 1] = 1  # Observe y
        self.H[2, 2] = 1  # Observe radius
        self.H[3, 3] = 1  # Observe height

        # Track number of consecutive prediction-only frames
        self.prediction_only_count = 0
        self.max_prediction_frames = 10  # Reset after this many frames without measurement

    def predict(self, dt):
        """Predict next state based on motion model"""
        # State transition matrix (constant velocity model)
        F = np.eye(8)
        F[0, 4] = dt  # x = x + vx * dt
        F[1, 5] = dt  # y = y + vy * dt
        F[2, 6] = dt  # radius = radius + vr * dt
        F[3, 7] = dt  # height = height + vh * dt

        # Predict state
        self.state = F @ self.state

        # Constrain velocities to reasonable ranges (prevent explosion)
        # Position velocity: max 2 m/s (human walking/running speed)
        self.state[4] = np.clip(self.state[4], -2.0, 2.0)  # vx
        self.state[5] = np.clip(self.state[5], -2.0, 2.0)  # vy

        # Size velocity: max 0.1 m/s (size shouldn't change rapidly)
        self.state[6] = np.clip(self.state[6], -0.1, 0.1)  # vr
        self.state[7] = np.clip(self.state[7], -0.1, 0.1)  # vh

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update state with new measurement [x, y, radius, height]"""
        # Innovation (measurement residual)
        z = np.array(measurement)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """Get current state [x, y, radius, height]"""
        return self.state[0:4]


class YoloPointcloudReconstruction(Node):
    def __init__(self):
        super().__init__('yolo_pointcloud_reconstruction')

        # Configuration
        self.angle_degrees = 15

        # YOLO detection parameters
        self.conf_threshold = 0.2    # YOLO confidence threshold (lower = detect more, may include false positives)
        self.iou_threshold = 0.5     # IoU threshold for NMS (lower = keep more overlapping detections)
        self.mask_threshold = 0.3    # Mask binarization threshold (lower = include more pixels)
        self.yolo_imgsz = 640        # YOLO input size (higher = better accuracy, slower)

        self.ground_removal_enabled = True
        self.ground_height_threshold = -0.1  # meters
        self.ceiling_height_threshold = 2.0   # meters

        # Kalman filter for cylinder tracking
        self.cylinder_kf = None
        self.last_cylinder_time = None
        self.cylinder_initialized = False
        self.consecutive_outliers = 0  # Track consecutive outlier rejections
        self.max_outliers_before_reset = 5  # Reset KF after this many consecutive outliers

        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Image buffers
        self.camera_01_img = None
        self.camera_02_img = None
        self.camera_01_depth = None
        self.camera_02_depth = None
        self.camera_01_timestamp = None
        self.camera_02_timestamp = None

        # Camera intrinsics
        self.cam01_intrinsics = None
        self.cam02_intrinsics = None

        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO model
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO("yolov8n-seg.pt")
        self.get_logger().info("YOLO model loaded!")

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

        # Publishers
        self.pub_human_pc = self.create_publisher(PointCloud2, '/human_pointcloud', 1)
        self.pub_human_mask = self.create_publisher(Image, '/human_mask', 1)
        self.pub_detected_human = self.create_publisher(Image, '/detected_human', 1)
        self.pub_cylinder_marker = self.create_publisher(Marker, '/human_cylinder', 1)

        # Processing flag
        self.processing = False
        self.frame_count = 0

        self.get_logger().info("Node initialized. Waiting for images...")

    def callback_info01(self, msg):
        if self.cam01_intrinsics is None:
            k = msg.k
            self.cam01_intrinsics = {
                'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]
            }
            self.get_logger().info("Camera 01 intrinsics received")

    def callback_info02(self, msg):
        if self.cam02_intrinsics is None:
            k = msg.k
            self.cam02_intrinsics = {
                'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]
            }
            self.get_logger().info("Camera 02 intrinsics received")

    def callback_cam01_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_01_img = img
                self.camera_01_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error("Failed to convert camera_01 color: %s" % str(e))

    def callback_cam02_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_02_img = img
                self.camera_02_timestamp = msg.header.stamp

            # Trigger processing
            if not self.processing and self.camera_01_img is not None:
                self.processing = True
                try:
                    self.process_frame()
                finally:
                    self.processing = False
        except Exception as e:
            self.get_logger().error("Failed to convert camera_02 color: %s" % str(e))

    def callback_cam01_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.camera_01_depth = depth
        except Exception as e:
            self.get_logger().error("Failed to convert camera_01 depth: %s" % str(e))

    def callback_cam02_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.camera_02_depth = depth
        except Exception as e:
            self.get_logger().error("Failed to convert camera_02 depth: %s" % str(e))

    def stitch_images(self, img_top, img_bottom):
        """Stitch top and bottom cameras with perspective transform"""
        # # Warp top camera
        # h0, w0 = img_top.shape[:2]
        # src0 = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])
        # offset0 = int(w0 * self.angle_degrees / 90)
        # dst0 = np.float32([[0, 0], [w0, 0], [w0 - offset0, h0], [offset0, h0]])
        # M0 = cv2.getPerspectiveTransform(src0, dst0)
        # warped_top = cv2.warpPerspective(img_top, M0, (w0, h0))

        # # Warp bottom camera
        # h1, w1 = img_bottom.shape[:2]
        # src1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        # offset1 = int(w1 * self.angle_degrees / 90)
        # dst1 = np.float32([[offset1, 0], [w1 - offset1, 0], [w1, h1], [0, h1]])
        # M1 = cv2.getPerspectiveTransform(src1, dst1)
        # warped_bottom = cv2.warpPerspective(img_bottom, M1, (w1, h1))

        # # Resize to common width and stack
        # w_common = min(warped_top.shape[1], warped_bottom.shape[1])
        # scale_top = w_common / warped_top.shape[1]
        # scale_bottom = w_common / warped_bottom.shape[1]

        # top_h = int(round(warped_top.shape[0] * scale_top))
        # bot_h = int(round(warped_bottom.shape[0] * scale_bottom))

        # top_resized = cv2.resize(warped_top, (w_common, top_h), interpolation=cv2.INTER_LINEAR)
        # bot_resized = cv2.resize(warped_bottom, (w_common, bot_h), interpolation=cv2.INTER_LINEAR)

        # stitched = np.vstack([top_resized, bot_resized])
        # return stitched

        # Simplified version: direct stacking without perspective transform
        h0, w0 = img_top.shape[:2]
        h1, w1 = img_bottom.shape[:2]

        # Resize to common width
        w_common = min(w0, w1)
        scale_top = w_common / w0
        scale_bottom = w_common / w1

        top_h = int(round(h0 * scale_top))
        bot_h = int(round(h1 * scale_bottom))

        top_resized = cv2.resize(img_top, (w_common, top_h), interpolation=cv2.INTER_LINEAR)
        bot_resized = cv2.resize(img_bottom, (w_common, bot_h), interpolation=cv2.INTER_LINEAR)

        stitched = np.vstack([top_resized, bot_resized])
        return stitched

    def yolo_detect_human(self, frame_bgr):
        """Run YOLO to detect human and return mask of the one closest to center"""
        h, w = frame_bgr.shape[:2]
        center_x = w * 0.5
        center_y = h * 0.5

        # Downsample to 640 for YOLO
        target_w = 640
        scale = target_w / w
        ds_h = int(h * scale)
        ds_w = target_w

        frame_ds = cv2.resize(frame_bgr, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

        # Run YOLO
        results = self.yolo_model.predict(
            frame_ds[..., ::-1],
            classes=[0],                    # Only detect person class
            conf=self.conf_threshold,       # Confidence threshold
            iou=self.iou_threshold,         # IoU threshold for NMS
            verbose=False,
            device="cuda:0",
            half=False,
            imgsz=self.yolo_imgsz,         # Input image size
            max_det=10,                    # Maximum detections per image
            agnostic_nms=False,            # Class-agnostic NMS
            retina_masks=True              # Use high-resolution segmentation masks
        )

        # Find the mask closest to center - O(n) algorithm
        closest_mask = None
        min_distance = float('inf')

        for r in results:
            if r.masks is None:
                continue

            for m in r.masks.data:
                arr = m.detach().cpu().numpy()

                # Resize to downsampled size
                if arr.shape[-2:] != (ds_h, ds_w):
                    arr = cv2.resize(arr, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)

                mask_ds = (arr > self.mask_threshold).astype(np.uint8) * 255

                # Upscale to original size
                mask = cv2.resize(mask_ds, (w, h), interpolation=cv2.INTER_LINEAR)

                # Calculate centroid of mask - O(n) where n is pixels
                # Use numpy vectorized operations for efficiency
                y_coords, x_coords = np.nonzero(mask)

                if len(x_coords) == 0:
                    continue

                # Compute centroid
                mask_center_x = np.mean(x_coords)
                mask_center_y = np.mean(y_coords)

                # Calculate distance to image center
                distance = (mask_center_x - center_x) ** 2 + (mask_center_y - center_y) ** 2

                # Update if this is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_mask = mask

        # Return the closest mask or empty mask
        if closest_mask is not None:
            return closest_mask
        else:
            return np.zeros((h, w), dtype=np.uint8)

    def extract_human_bbox(self, image, mask, expansion_ratio=0.1):
        """Extract human bounding box from mask with expansion"""
        if mask is None or np.sum(mask) == 0:
            return None

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Get bounding rect of all contours combined
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Expand by 10% in all directions
        expand_w = int(w * expansion_ratio)
        expand_h = int(h * expansion_ratio)

        x_new = max(0, x - expand_w)
        y_new = max(0, y - expand_h)
        x_max = min(image.shape[1], x + w + expand_w)
        y_max = min(image.shape[0], y + h + expand_h)

        # Crop image
        cropped = image[y_new:y_max, x_new:x_max]

        return cropped if cropped.size > 0 else None

    def split_mask(self, stitched_mask, orig_size_01, orig_size_02):
        """Split stitched mask back to individual camera masks"""
        # Get dimensions
        h1, w1 = orig_size_01
        h2, w2 = orig_size_02

        # Calculate split point (same as stitching)
        w_common = stitched_mask.shape[1]
        scale_top = w_common / w1
        scale_bottom = w_common / w2

        top_h = int(round(h1 * scale_top))
        bot_h = int(round(h2 * scale_bottom))

        # Split stitched mask
        mask_top_resized = stitched_mask[:top_h, :]
        mask_bottom_resized = stitched_mask[top_h:, :]

        # Resize to original size (no perspective warp)
        mask_cam01 = cv2.resize(mask_top_resized, (w1, h1), interpolation=cv2.INTER_NEAREST)
        mask_cam02 = cv2.resize(mask_bottom_resized, (w2, h2), interpolation=cv2.INTER_NEAREST)

        # # Reverse perspective warp for camera 01
        # offset0 = int(w1 * self.angle_degrees / 90)
        # src0 = np.float32([[0, 0], [w1, 0], [w1 - offset0, h1], [offset0, h1]])
        # dst0 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        # M0_inv = cv2.getPerspectiveTransform(src0, dst0)
        # mask_cam01 = cv2.warpPerspective(mask_top_warped, M0_inv, (w1, h1), flags=cv2.INTER_NEAREST)

        # # Reverse perspective warp for camera 02
        # offset1 = int(w2 * self.angle_degrees / 90)
        # src1 = np.float32([[offset1, 0], [w2 - offset1, 0], [w2, h2], [0, h2]])
        # dst1 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        # M1_inv = cv2.getPerspectiveTransform(src1, dst1)
        # mask_cam02 = cv2.warpPerspective(mask_bottom_warped, M1_inv, (w2, h2), flags=cv2.INTER_NEAREST)

        return mask_cam01, mask_cam02

    def filter_outliers_by_cylindrical_distance(self, points, iqr_multiplier=1.5):
        """
        Filter outliers based on radial distance from central vertical axis.
        Central axis is defined by XY center of pointcloud, Z from 0 to 1.8m.

        Args:
            points: Nx3 numpy array in base_footprint frame (X-forward, Y-left, Z-up)
            iqr_multiplier: multiplier for IQR (default 1.5)

        Returns:
            filtered_points: Points with outliers removed
        """
        if points is None or len(points) < 100:
            return points

        # Compute center of XY projection
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # Calculate radial distance from central axis (distance in XY plane)
        radial_distances = np.sqrt((points[:, 0] - center_x)**2 +
                                   (points[:, 1] - center_y)**2)

        # Use IQR method to filter radial outliers
        Q1 = np.percentile(radial_distances, 25)
        Q3 = np.percentile(radial_distances, 75)
        IQR = Q3 - Q1

        # Calculate outlier bounds
        radius_max = Q3 + iqr_multiplier * IQR

        # Filter points by radial distance
        valid_radial = radial_distances <= radius_max

        # Also filter by Z height (0 to 1.8m)
        valid_height = (points[:, 2] >= 0.0) & (points[:, 2] <= 1.8)

        # Combine filters
        valid_points = valid_radial & valid_height

        filtered_points = points[valid_points]

        return filtered_points if len(filtered_points) >= 100 else points

    def depth_to_pointcloud(self, depth_image, mask, intrinsics):
        """Convert depth image and mask to pointcloud"""
        if depth_image is None or mask is None or intrinsics is None:
            return None

        # Resize mask to match depth
        if mask.shape != depth_image.shape:
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        # Downsample for efficiency (every other pixel)
        depth_ds = depth_image[::2, ::2]
        mask_ds = mask[::2, ::2]

        h, w = depth_ds.shape

        # Adjust intrinsics
        fx = intrinsics['fx'] / 2.0
        fy = intrinsics['fy'] / 2.0
        cx = intrinsics['cx'] / 2.0
        cy = intrinsics['cy'] / 2.0

        # Create grids
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

        # Convert depth to meters
        z = depth_ds.astype(np.float32) / 1000.0

        # Valid mask
        valid_mask = (mask_ds > 128) & (z > 0.3) & (z < 5.0)

        # Calculate 3D coordinates
        x_cam = (u_grid - cx) * z / fx
        y_cam = (v_grid - cy) * z / fy
        z_cam = z

        # Extract valid points
        x_cam_valid = x_cam[valid_mask]
        y_cam_valid = y_cam[valid_mask]
        z_cam_valid = z_cam[valid_mask]

        # Transform to ROS coordinates (Z-forward, Y-down, X-right -> X-forward, Y-left, Z-up)
        x_ros = z_cam_valid
        y_ros = -x_cam_valid
        z_ros = -y_cam_valid

        points = np.stack((x_ros, y_ros, z_ros), axis=-1)

        return points if len(points) > 0 else None

    def transform_to_base_footprint(self, points, camera_frame):
        """Transform points to base_footprint frame"""
        if points is None or len(points) == 0:
            return None

        try:
            from rclpy.time import Time
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', camera_frame, Time(),
                rclpy.duration.Duration(seconds=1.0)
            )

            # Extract transform
            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Quaternion to rotation matrix
            qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])

            t = np.array([trans.x, trans.y, trans.z])
            points_transformed = (R @ points.T).T + t

            return points_transformed

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning("TF lookup failed for %s: %s" % (camera_frame, str(e)))
            return None

    def remove_ground_plane(self, points):
        """Remove ground and ceiling points by height filtering"""
        if points is None or len(points) < 100:
            return points

        if not self.ground_removal_enabled:
            return points

        # Filter by Z-axis height
        valid_height = (points[:, 2] > self.ground_height_threshold) & \
                      (points[:, 2] < self.ceiling_height_threshold)

        filtered_points = points[valid_height]

        return filtered_points if len(filtered_points) >= 100 else points

    def compute_cylinder_bounding_box(self, points, expansion_ratio=0.05):
        """
        Compute cylindrical bounding box parameters around human pointcloud.
        Cylinder axis is Z (vertical), bottom at ground (Z=0), grows outward by expansion_ratio.

        Args:
            points: Nx3 numpy array in base_footprint frame (X-forward, Y-left, Z-up)
            expansion_ratio: Ratio to expand radius and height (default 5%)

        Returns:
            (center_x, center_y, center_z, radius, height): Cylinder parameters
        """
        if points is None or len(points) < 10:
            return None

        # Get XY projection (horizontal plane)
        xy_points = points[:, :2]  # [X, Y]

        # Compute center of XY projection
        center_x = np.mean(xy_points[:, 0])
        center_y = np.mean(xy_points[:, 1])

        # Compute radius as max distance from center in XY plane
        distances = np.sqrt((xy_points[:, 0] - center_x)**2 + (xy_points[:, 1] - center_y)**2)
        base_radius = np.max(distances)

        # Expand radius by expansion_ratio
        radius = base_radius * (1.0 + expansion_ratio)

        # Ensure radius is within [0.4m, 1.0m]
        radius = max(0.4, min(radius, 1.0))

        # Get Z range (height)
        # Bottom is always at ground (Z=0)
        z_bottom = 0.0
        z_max = np.max(points[:, 2])

        # Expand top by expansion_ratio
        height = z_max * (1.0 + expansion_ratio)

        # Center Z is at half height (since bottom is at 0)
        center_z = height / 2.0

        return (center_x, center_y, center_z, radius, height)

    def update_cylinder_with_kalman(self, measured_params, current_time):
        """
        Update cylinder parameters using Kalman filter for smooth tracking.
        Includes innovation gating to reject outlier measurements (e.g., detection jumps to another person).

        Args:
            measured_params: (center_x, center_y, center_z, radius, height) from point cloud, or None
            current_time: Current timestamp in seconds

        Returns:
            (center_x, center_y, center_z, radius, height): Filtered parameters, or None if no state
        """
        if measured_params is None:
            # No measurement, just predict
            if self.cylinder_kf is not None and self.last_cylinder_time is not None:
                dt = current_time - self.last_cylinder_time
                if dt > 0 and dt < 1.0:  # Sanity check: dt should be reasonable
                    self.cylinder_kf.predict(dt)
                    self.last_cylinder_time = current_time
                    filtered_state = self.cylinder_kf.get_state()

                    # Apply constraints even in prediction mode
                    filtered_x = filtered_state[0]
                    filtered_y = filtered_state[1]
                    filtered_radius = np.clip(filtered_state[2], 0.4, 1.0)
                    filtered_height = np.clip(filtered_state[3], 0.5, 2.5)

                    # Update state with constraints to prevent drift
                    self.cylinder_kf.state[2] = filtered_radius
                    self.cylinder_kf.state[3] = filtered_height

                    center_z = filtered_height / 2.0
                    self.get_logger().info("[Kalman] No detection, using predicted state")
                    return (filtered_x, filtered_y, center_z, filtered_radius, filtered_height)
            return None

        center_x, center_y, center_z, radius, height = measured_params

        # Initialize Kalman filter on first measurement
        if not self.cylinder_initialized:
            self.cylinder_kf = CylinderKalmanFilter()
            self.cylinder_kf.state[0:4] = [center_x, center_y, radius, height]
            self.cylinder_kf.state[4:8] = 0  # Initialize velocities to zero
            self.last_cylinder_time = current_time
            self.cylinder_initialized = True
            self.get_logger().info("[Kalman] Initialized with first measurement")
            return measured_params

        # Calculate time delta
        dt = current_time - self.last_cylinder_time
        self.last_cylinder_time = current_time

        # Sanity check on dt
        if dt <= 0 or dt > 1.0:
            # Time delta too large or invalid, reset
            self.get_logger().warning("Large time gap (%.2f s), resetting Kalman filter" % dt)
            self.cylinder_kf.state[0:4] = [center_x, center_y, radius, height]
            self.cylinder_kf.state[4:8] = 0
            return measured_params

        # Predict step
        self.cylinder_kf.predict(dt)

        # Innovation gating: Check if measurement is too far from prediction
        predicted_state = self.cylinder_kf.get_state()
        measurement = np.array([center_x, center_y, radius, height])
        innovation = measurement - predicted_state

        # Calculate innovation distance (weighted by importance)
        # Position error in meters, radius/height error in meters
        position_error = np.sqrt(innovation[0]**2 + innovation[1]**2)
        size_error = np.sqrt(innovation[2]**2 + innovation[3]**2)

        # Gating thresholds
        max_position_jump = 1.5  # meters (reject if person "jumps" > 1.5m)
        max_size_jump = 0.8      # meters (reject if radius/height changes > 0.8m)

        if position_error > max_position_jump or size_error > max_size_jump:
            # Measurement is likely an outlier (jumped to another person)
            self.consecutive_outliers += 1

            self.get_logger().warning(
                "[Kalman] Rejecting outlier measurement: pos_error=%.2fm, size_error=%.2fm (consecutive: %d)" %
                (position_error, size_error, self.consecutive_outliers)
            )

            # If we've seen too many consecutive outliers, this is likely a new target
            # Reset the Kalman filter to the new measurement
            if self.consecutive_outliers >= self.max_outliers_before_reset:
                self.get_logger().warning(
                    "[Kalman] Too many consecutive outliers (%d), reinitializing with new measurement" %
                    self.consecutive_outliers
                )
                # Reinitialize KF with new measurement
                self.cylinder_kf.state[0:4] = [center_x, center_y, radius, height]
                self.cylinder_kf.state[4:8] = 0  # Reset velocities to zero
                self.cylinder_kf.P = np.eye(8) * 1000  # Reset covariance (high uncertainty)
                self.consecutive_outliers = 0
                return measured_params
            else:
                # Don't update, just return predicted state with constraints
                filtered_x = predicted_state[0]
                filtered_y = predicted_state[1]
                filtered_radius = np.clip(predicted_state[2], 0.4, 1.0)
                filtered_height = np.clip(predicted_state[3], 0.5, 2.5)

                # Update state with constraints to prevent drift
                self.cylinder_kf.state[2] = filtered_radius
                self.cylinder_kf.state[3] = filtered_height

                center_z = filtered_height / 2.0
                return (filtered_x, filtered_y, center_z, filtered_radius, filtered_height)

        # Update step with measurement (measurement is valid)
        self.cylinder_kf.update(measurement)

        # Reset outlier counter on successful update
        self.consecutive_outliers = 0

        # Get filtered state
        filtered_state = self.cylinder_kf.get_state()

        # Apply constraints to filtered state
        filtered_x = filtered_state[0]
        filtered_y = filtered_state[1]
        filtered_radius = np.clip(filtered_state[2], 0.4, 1.0)  # Constrain radius [0.4, 1.0]m
        filtered_height = np.clip(filtered_state[3], 0.5, 2.5)  # Constrain height [0.5, 2.5]m

        # Update state with constraints (prevent drift)
        self.cylinder_kf.state[2] = filtered_radius
        self.cylinder_kf.state[3] = filtered_height

        # Calculate center_z (half of height, since bottom is at 0)
        filtered_center_z = filtered_height / 2.0

        return (filtered_x, filtered_y, filtered_center_z, filtered_radius, filtered_height)

    def process_frame(self):
        """Main processing loop"""
        frame_start = time.time()

        with self.lock:
            if self.camera_01_img is None or self.camera_02_img is None:
                return
            if self.camera_01_depth is None or self.camera_02_depth is None:
                return
            if self.cam01_intrinsics is None or self.cam02_intrinsics is None:
                return

            # Copy data
            img01 = self.camera_01_img.copy()
            img02 = self.camera_02_img.copy()
            depth01 = self.camera_01_depth.copy()
            depth02 = self.camera_02_depth.copy()
            timestamp = self.camera_02_timestamp

        # 1. Stitch images
        stitch_start = time.time()
        stitched = self.stitch_images(img01, img02)
        stitch_time = (time.time() - stitch_start) * 1000

        # 2. YOLO detection
        yolo_start = time.time()
        stitched_mask = self.yolo_detect_human(stitched)
        yolo_time = (time.time() - yolo_start) * 1000

        # 3. Split mask
        split_start = time.time()
        mask_cam01, mask_cam02 = self.split_mask(stitched_mask, img01.shape[:2], img02.shape[:2])
        split_time = (time.time() - split_start) * 1000

        # 4. Generate pointclouds
        pc_start = time.time()
        points01 = self.depth_to_pointcloud(depth01, mask_cam01, self.cam01_intrinsics)
        points02 = self.depth_to_pointcloud(depth02, mask_cam02, self.cam02_intrinsics)

        # 5. Transform to base_footprint
        points01_base = self.transform_to_base_footprint(points01, 'camera_1')
        points02_base = self.transform_to_base_footprint(points02, 'camera_0')

        # 6. Combine pointclouds
        combined_points = []
        if points01_base is not None:
            combined_points.append(points01_base)
        if points02_base is not None:
            combined_points.append(points02_base)

        if len(combined_points) > 0:
            combined = np.vstack(combined_points)

            # 7. Filter outliers by cylindrical distance from central axis
            combined = self.filter_outliers_by_cylindrical_distance(combined, iqr_multiplier=1.5)

            # 8. Remove ground plane (additional safety filter)
            combined = self.remove_ground_plane(combined)

            # 9. Downsample if needed
            if combined is not None and len(combined) > 6000:
                indices = np.random.choice(len(combined), 6000, replace=False)
                combined = combined[indices]
        else:
            combined = None

        pc_time = (time.time() - pc_start) * 1000

        # 9. Publish pointcloud and cylinder
        pub_start = time.time()

        # Publish pointcloud if available
        if combined is not None and len(combined) > 0:
            header = Header()
            header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
            header.frame_id = 'base_footprint'
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg = pc2.create_cloud(header, fields, combined)
            self.pub_human_pc.publish(pc_msg)

        # Compute and publish cylindrical bounding box as Marker with Kalman filtering
        # This runs even if combined is None (using prediction)
        measured_cylinder_params = None
        if combined is not None and len(combined) > 0:
            measured_cylinder_params = self.compute_cylinder_bounding_box(combined, expansion_ratio=0.05)

        # Get current time in seconds
        current_time = time.time()

        # Apply Kalman filter for smooth tracking
        # Will use prediction if measured_cylinder_params is None
        filtered_params = self.update_cylinder_with_kalman(measured_cylinder_params, current_time)

        if filtered_params is not None:
            center_x, center_y, center_z, radius, height = filtered_params

            # Create cylinder marker
            marker = Marker()
            marker.header.frame_id = 'base_footprint'
            marker.header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
            marker.ns = 'human_cylinder'
            marker.id = 0
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Set position (center of cylinder)
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z

            # Set orientation (identity quaternion - cylinder aligned with Z-axis)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Set scale (diameter and height)
            marker.scale.x = radius * 2.0  # Diameter in X
            marker.scale.y = radius * 2.0  # Diameter in Y
            marker.scale.z = height        # Height in Z

            # Set color based on whether we have measurement or prediction
            if measured_cylinder_params is not None:
                # Have measurement: green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                # Prediction only: yellow
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 0.3  # 30% opacity

            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

            self.pub_cylinder_marker.publish(marker)

        # 10. Publish mask
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(stitched_mask, encoding="mono8")
            mask_msg.header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
            mask_msg.header.frame_id = 'stitched'
            self.pub_human_mask.publish(mask_msg)
        except Exception as e:
            self.get_logger().error("Failed to publish mask: %s" % str(e))

        # 11. Extract and publish human bounding box image
        try:
            human_bbox_img = self.extract_human_bbox(stitched, stitched_mask, expansion_ratio=0.1)
            if human_bbox_img is not None:
                bbox_msg = self.bridge.cv2_to_imgmsg(human_bbox_img, encoding="bgr8")
                bbox_msg.header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
                bbox_msg.header.frame_id = 'stitched'
                self.pub_detected_human.publish(bbox_msg)
        except Exception as e:
            self.get_logger().error("Failed to publish detected human: %s" % str(e))

        pub_time = (time.time() - pub_start) * 1000

        # Total time
        total_time = (time.time() - frame_start) * 1000

        # Log latency
        self.get_logger().info(
            "[Frame %d] Total: %.1f ms (Stitch: %.1f | YOLO: %.1f | Split: %.1f | PC: %.1f | Pub: %.1f) | FPS: %.1f" %
            (self.frame_count, total_time, stitch_time, yolo_time, split_time, pc_time, pub_time, 1000.0/total_time)
        )

        self.frame_count += 1


if __name__ == "__main__":
    rclpy.init()
    try:
        node = YoloPointcloudReconstruction()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Fatal error: %s" % str(e))
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
