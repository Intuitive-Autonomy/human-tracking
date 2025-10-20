#!/usr/bin/env python
"""
Camera TF Publisher for ROS1
Publishes static transforms from cameras to base_link

Global pitch offset: -15 degrees
Camera 1: -45 degrees pitch (-30 - 15), -1.5cm z offset
Camera 2: +15 degrees pitch (+30 - 15), +1.5cm z offset
"""

import rospy
import tf
import tf.transformations as tf_trans
import math

def publish_camera_tfs():
    rospy.init_node('camera_tf_publisher', anonymous=False)

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(50)  # 50 Hz

    # Overall height offset for both cameras
    base_height = 0.70  # 70cm in meters

    # Global pitch offset applied to both cameras
    global_pitch_offset = math.radians(-15)  # -15 degrees global offset

    # Camera 1: -30 deg pitch rotation + global offset, overall height + relative -1.5cm offset
    pitch_cam1 = math.radians(-30) + global_pitch_offset  # -30 - 15 = -45 degrees total
    quat_cam1 = tf_trans.quaternion_from_euler(0, pitch_cam1, 0)  # roll, pitch, yaw
    trans_cam1 = (0.0, 0.0, base_height - 0.015)

    # Camera 2: +30 deg pitch rotation + global offset, overall height + relative +1.5cm offset
    pitch_cam2 = math.radians(30) + global_pitch_offset  # +30 - 15 = +15 degrees total
    quat_cam2 = tf_trans.quaternion_from_euler(0, pitch_cam2, 0)  # roll, pitch, yaw
    trans_cam2 = (0.0, 0.0, base_height + 0.015)

    rospy.loginfo("Publishing camera TF transforms...")
    rospy.loginfo("Global pitch offset: -15deg")
    rospy.loginfo("Base height: %.2fm (70cm)" % base_height)
    rospy.loginfo("Camera 1: pitch=-45deg, z=%.3fm" % trans_cam1[2])
    rospy.loginfo("Camera 2: pitch=+15deg, z=%.3fm" % trans_cam2[2])

    while not rospy.is_shutdown():
        current_time = rospy.Time.now()

        # Publish transform from base_link to ob_camera_01_link
        br.sendTransform(
            trans_cam1,
            quat_cam1,
            current_time,
            "ob_camera_01_link",
            "base_link"
        )

        # Publish transform from base_link to ob_camera_02_link
        br.sendTransform(
            trans_cam2,
            quat_cam2,
            current_time,
            "ob_camera_02_link",
            "base_link"
        )

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_tfs()
    except rospy.ROSInterruptException:
        pass
