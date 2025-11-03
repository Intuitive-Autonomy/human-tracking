#!/usr/bin/env python3
"""
Depth to PointCloud Publisher for ROS2
Subscribes to depth images and publishes full pointclouds for each camera
No tracking - just raw depth to pointcloud conversion
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
import struct

class DepthToPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_publisher')

        self.bridge = CvBridge()

        # Camera intrinsics - will be populated from camera_info topics
        self.cam01_intrinsics = None
        self.cam02_intrinsics = None

        # Depth processing parameters
        self.declare_parameter('min_depth', 0.3)
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('max_points', 5000)

        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.max_points = self.get_parameter('max_points').value

        # Subscribers - Camera Info
        self.sub_info01 = self.create_subscription(
            CameraInfo, '/camera_01/depth/camera_info', self.callback_info01, 1
        )
        self.sub_info02 = self.create_subscription(
            CameraInfo, '/camera_02/depth/camera_info', self.callback_info02, 1
        )

        # Subscribers - Depth images
        self.sub_depth01 = self.create_subscription(
            Image, '/camera_01/depth/image_raw', self.callback_depth01, 1
        )
        self.sub_depth02 = self.create_subscription(
            Image, '/camera_02/depth/image_raw', self.callback_depth02, 1
        )

        # Publishers - Pointclouds
        self.pub_pc01 = self.create_publisher(
            PointCloud2, '/camera_01/pointcloud', 1
        )
        self.pub_pc02 = self.create_publisher(
            PointCloud2, '/camera_02/pointcloud', 1
        )

        self.get_logger().info("Depth to PointCloud Publisher initialized")
        self.get_logger().info("Waiting for camera_info topics to get intrinsics...")
        self.get_logger().info("Depth range: %.2fm to %.2fm" % (self.min_depth, self.max_depth))
        self.get_logger().info("Max points per cloud: %d" % self.max_points)

    def callback_info01(self, msg):
        """Callback for camera 01 camera_info"""
        if self.cam01_intrinsics is None:
            # Extract intrinsics from camera info k matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            k = msg.k
            self.cam01_intrinsics = {
                'fx': k[0],
                'fy': k[4],
                'cx': k[2],
                'cy': k[5]
            }
            self.get_logger().info("Camera 01 intrinsics received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
                         (self.cam01_intrinsics['fx'], self.cam01_intrinsics['fy'],
                          self.cam01_intrinsics['cx'], self.cam01_intrinsics['cy']))

    def callback_info02(self, msg):
        """Callback for camera 02 camera_info"""
        if self.cam02_intrinsics is None:
            # Extract intrinsics from camera info k matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            k = msg.k
            self.cam02_intrinsics = {
                'fx': k[0],
                'fy': k[4],
                'cx': k[2],
                'cy': k[5]
            }
            self.get_logger().info("Camera 02 intrinsics received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" %
                         (self.cam02_intrinsics['fx'], self.cam02_intrinsics['fy'],
                          self.cam02_intrinsics['cx'], self.cam02_intrinsics['cy']))

    def depth_to_pointcloud(self, depth_image, camera_frame, timestamp, intrinsics):
        """Convert depth image to pointcloud"""
        if depth_image is None or intrinsics is None:
            return None

        h, w = depth_image.shape

        # Use intrinsics from camera_info
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # Downsample spatially before converting to 3D (much faster)
        # This reduces the number of points to process
        step = 1
        if h * w > self.max_points * 2:
            # Calculate step size to get approximately max_points
            step = int(np.sqrt((h * w) / self.max_points))

        # Subsample depth image
        depth_sub = depth_image[::step, ::step]
        h_sub, w_sub = depth_sub.shape

        # Create coordinate grids for subsampled image
        u_grid, v_grid = np.meshgrid(
            np.arange(0, w, step),
            np.arange(0, h, step)
        )

        # Convert depth to meters (assuming mm input)
        z = depth_sub.astype(np.float32) / 1000.0

        # Apply depth filter
        valid_mask = (z > self.min_depth) & (z < self.max_depth)

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

        # Combine into point array (already in contiguous memory)
        points = np.stack((x_ros, y_ros, z_ros), axis=-1)

        if len(points) == 0:
            return None

        # Final random downsample if still too many points
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        # Create PointCloud2 message using faster method
        from std_msgs.msg import Header
        header = Header()
        header.stamp = timestamp  # Use original depth image timestamp
        header.frame_id = camera_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Use create_cloud which is faster than manual point packing
        pointcloud_msg = pc2.create_cloud(header, fields, points)
        return pointcloud_msg

    def callback_depth01(self, msg):
        """Callback for camera 01 depth images"""
        try:
            # Skip if intrinsics not yet received
            if self.cam01_intrinsics is None:
                return

            # Convert ROS Image message to OpenCV image
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Rotate camera_01 depth by 180 degrees
            depth = cv2.rotate(depth, cv2.ROTATE_180)

            # Convert to pointcloud with original timestamp and intrinsics
            pc_msg = self.depth_to_pointcloud(depth, 'camera_01_link', msg.header.stamp, self.cam01_intrinsics)

            # Publish pointcloud
            if pc_msg is not None:
                self.pub_pc01.publish(pc_msg)
        except Exception as e:
            self.get_logger().error("Failed to process camera_01 depth: %s" % str(e))

    def callback_depth02(self, msg):
        """Callback for camera 02 depth images"""
        try:
            # Skip if intrinsics not yet received
            if self.cam02_intrinsics is None:
                return

            # Convert ROS Image message to OpenCV image
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert to pointcloud with original timestamp and intrinsics
            pc_msg = self.depth_to_pointcloud(depth, 'camera_02_link', msg.header.stamp, self.cam02_intrinsics)

            # Publish pointcloud
            if pc_msg is not None:
                self.pub_pc02.publish(pc_msg)
        except Exception as e:
            self.get_logger().error("Failed to process camera_02 depth: %s" % str(e))

def main():
    rclpy.init()
    try:
        node = DepthToPointCloudPublisher()
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

if __name__ == '__main__':
    main()
