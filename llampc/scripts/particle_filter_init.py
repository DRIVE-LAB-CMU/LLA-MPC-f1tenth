#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import time

def main():
    rclpy.init()
    node = rclpy.create_node('initialpose_publisher')  # minimal node
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    # Construct the message
    pose_msg = PoseWithCovarianceStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = "map"

    # Position
    pose_msg.pose.pose.position.x = 0.0
    pose_msg.pose.pose.position.y = 0.0
    pose_msg.pose.pose.position.z = 0.0

    # Orientation (yaw only)
    yaw = 0.0
    pose_msg.pose.pose.orientation.x = 0.0
    pose_msg.pose.pose.orientation.y = 0.0
    pose_msg.pose.pose.orientation.z = math.sin(yaw / 2)
    pose_msg.pose.pose.orientation.w = math.cos(yaw / 2)

    # Covariance
    pose_msg.pose.covariance = [0.0]*36
    pose_msg.pose.covariance[0] = 0.5
    pose_msg.pose.covariance[7] = 0.5
    pose_msg.pose.covariance[35] = 0.2

    # Publish once
    pub.publish(pose_msg)
    node.get_logger().info("Published initial pose with covariance")

    # Sleep briefly to ensure subscribers receive the message
    time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
