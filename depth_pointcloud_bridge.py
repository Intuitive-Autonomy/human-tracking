#!/usr/bin/env python3
"""
Depth Point Cloud Bridge Node
Combines human tracking mask with depth camera data to extract human point cloud

Subscribes to:
  - /camera_01/aligned_depth_to_color/image_raw (depth image)
  - /camera_01/color/camera_info (camera intrinsics)
  - Human tracking mask (from STCN tracker - shared memory or topic)

Publishes to:
  - /human_pointcloud (PointCloud2)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DepthPointCloudBridge(Node):
    def __init__(self):
        super().__init__('depth_pointcloud_bridge')

        # Configuration
        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_depth = None
        self.latest_mask = None

        # Camera intrinsics (will be updated from camera_info)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # QoS profile for image topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_01/depth/image_raw',  # Changed from aligned_depth_to_color
            self.depth_callback,
            qos_profile
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_01/color/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        self.mask_sub = self.create_subscription(
            Image,
            '/human_mask',
            self.mask_callback,
            qos_profile
        )

        # Publisher
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/human_pointcloud',
            10
        )

        # Timer for processing (30 Hz)
        self.timer = self.create_timer(0.033, self.process_pointcloud)

        self.get_logger().info('Depth Point Cloud Bridge initialized')
        self.get_logger().info('Waiting for depth images, camera info, and human mask...')
        self.get_logger().info('Subscribing to /human_mask for filtering')

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if self.camera_info is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info = msg
            self.get_logger().info(f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

    def depth_callback(self, msg):
        """Store latest depth image"""
        try:
            # Convert depth image (16-bit, depth in mm)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def mask_callback(self, msg):
        """Store latest human mask"""
        try:
            # Convert mask image (8-bit grayscale, 255=human, 0=background)
            mask_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.latest_mask = mask_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert mask image: {e}')

    def depth_to_pointcloud(self, depth_image, mask=None):
        """
        Convert depth image to 3D point cloud

        Args:
            depth_image: Depth image in mm (16-bit)
            mask: Optional binary mask (255=human, 0=background)

        Returns:
            points: Nx3 numpy array of 3D points in meters
        """
        if self.fx is None:
            return None

        h, w = depth_image.shape

        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Apply mask if provided
        if mask is not None:
            # Ensure mask is same size as depth
            if mask.shape != depth_image.shape:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Filter valid pixels (mask > 128 and depth > 0)
            valid = (mask > 128) & (depth_image > 0)
        else:
            # No mask, use all valid depth pixels
            valid = depth_image > 0

        # Extract valid pixels
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth_image[valid].astype(np.float32) / 1000.0  # Convert mm to meters

        # Back-project to 3D using pinhole camera model
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        z = z_valid

        # Stack into Nx3 array
        points = np.stack([x, y, z], axis=-1)

        return points

    def create_pointcloud2_msg(self, points, timestamp):
        """Create PointCloud2 message from numpy array"""
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_01_color_optical_frame'

        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False

        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        msg.point_step = 12  # 4 bytes * 3 fields
        msg.row_step = msg.point_step * msg.width

        # Pack point data
        msg.data = struct.pack(f'{len(points) * 3}f', *points.flatten())

        return msg

    def process_pointcloud(self):
        """Process depth image and publish point cloud"""
        if self.latest_depth is None or self.fx is None:
            return

        try:
            # Convert depth to point cloud with mask filtering
            if self.latest_mask is not None:
                points = self.depth_to_pointcloud(self.latest_depth, mask=self.latest_mask)
            else:
                # No mask available yet, use full depth image
                points = self.depth_to_pointcloud(self.latest_depth, mask=None)

            if points is None or len(points) == 0:
                self.get_logger().warning('No valid points in depth image')
                return

            # Downsample if too many points (optional, for performance)
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]

            # Create and publish PointCloud2 message
            timestamp = self.get_clock().now().to_msg()
            pc_msg = self.create_pointcloud2_msg(points, timestamp)
            self.pointcloud_pub.publish(pc_msg)

            mask_status = "with mask" if self.latest_mask is not None else "without mask"
            self.get_logger().info(f'Published point cloud with {len(points)} points ({mask_status})')

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')


def main(args=None):
    rclpy.init(args=args)

    try:
        node = DepthPointCloudBridge()

        print("Depth Point Cloud Bridge started")
        print("Topics:")
        print("  - Subscribing to: /camera_01/aligned_depth_to_color/image_raw")
        print("  - Subscribing to: /camera_01/color/camera_info")
        print("  - Subscribing to: /human_mask")
        print("  - Publishing to: /human_pointcloud")
        print("Note: Will use mask filtering when /human_mask is available")
        print("Press Ctrl+C to stop...")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
