#!/usr/bin/env python3
"""
Camera TF Publisher for ROS2
Publishes static transforms from cameras to base_link

Global pitch offset: -15 degrees
Camera 1: -45 degrees pitch (-30 - 15), -1.5cm z offset
Camera 2: +15 degrees pitch (+30 - 15), +1.5cm z offset
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import math

class CameraTFPublisher(Node):
    def __init__(self):
        super().__init__('camera_tf_publisher')

        self.br = TransformBroadcaster(self)

        # Overall height offset for both cameras
        base_height = 0.70  # 70cm in meters

        # Global pitch offset applied to both cameras
        global_pitch_offset = math.radians(-15)  # -15 degrees global offset

        # Camera 1: -30 deg pitch rotation + global offset, overall height + relative -1.5cm offset
        pitch_cam1 = math.radians(-30) + global_pitch_offset  # -30 - 15 = -45 degrees total
        r_cam1 = Rotation.from_euler('xyz', [0, pitch_cam1, 0])
        quat_cam1 = r_cam1.as_quat()  # [x, y, z, w]
        trans_cam1 = (0.0, 0.0, base_height - 0.015)

        # Camera 2: +30 deg pitch rotation + global offset, overall height + relative +1.5cm offset
        pitch_cam2 = math.radians(30) + global_pitch_offset  # +30 - 15 = +15 degrees total
        r_cam2 = Rotation.from_euler('xyz', [0, pitch_cam2, 0])
        quat_cam2 = r_cam2.as_quat()  # [x, y, z, w]
        trans_cam2 = (0.0, 0.0, base_height + 0.015)

        self.get_logger().info("Publishing camera TF transforms...")
        self.get_logger().info("Global pitch offset: -15deg")
        self.get_logger().info(f"Base height: {base_height:.2f}m (70cm)")
        self.get_logger().info(f"Camera 1: pitch=-45deg, z={trans_cam1[2]:.3f}m")
        self.get_logger().info(f"Camera 2: pitch=+15deg, z={trans_cam2[2]:.3f}m")

        # Store transforms
        self.trans_cam1 = trans_cam1
        self.quat_cam1 = quat_cam1
        self.trans_cam2 = trans_cam2
        self.quat_cam2 = quat_cam2

        # Create timer to publish transforms at 200Hz (increased from 50Hz)
        self.timer = self.create_timer(0.005, self.publish_transforms)  # 200 Hz = 0.005s

    def publish_transforms(self):
        current_time = self.get_clock().now().to_msg()

        # Publish transform from base_link to camera_01_link
        t1 = TransformStamped()
        t1.header.stamp = current_time
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'camera_01_link'
        t1.transform.translation.x = self.trans_cam1[0]
        t1.transform.translation.y = self.trans_cam1[1]
        t1.transform.translation.z = self.trans_cam1[2]
        t1.transform.rotation.x = self.quat_cam1[0]
        t1.transform.rotation.y = self.quat_cam1[1]
        t1.transform.rotation.z = self.quat_cam1[2]
        t1.transform.rotation.w = self.quat_cam1[3]
        self.br.sendTransform(t1)

        # Publish transform from base_link to camera_02_link
        t2 = TransformStamped()
        t2.header.stamp = current_time
        t2.header.frame_id = 'base_link'
        t2.child_frame_id = 'camera_02_link'
        t2.transform.translation.x = self.trans_cam2[0]
        t2.transform.translation.y = self.trans_cam2[1]
        t2.transform.translation.z = self.trans_cam2[2]
        t2.transform.rotation.x = self.quat_cam2[0]
        t2.transform.rotation.y = self.quat_cam2[1]
        t2.transform.rotation.z = self.quat_cam2[2]
        t2.transform.rotation.w = self.quat_cam2[3]
        self.br.sendTransform(t2)

def main(args=None):
    rclpy.init(args=args)
    node = CameraTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
