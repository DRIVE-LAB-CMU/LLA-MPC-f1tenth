#!/usr/bin/env python3
import rclpy
import math
import time
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class StepResponseNode(Node):
    def __init__(self):
        super().__init__('step_response_node')
        self.declare_params()
        self.sensor_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('sensor_topic').get_parameter_value().string_value,
            self.sensor_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('cmd_topic').get_parameter_value().string_value,
            10
        )
        
        self.sensor_recieved = False
        self.actuated = False
        self.complete = False
        self.sensor_val = 0
        self.bounds = 0.02
        self.target_cmd = 0
        self.callback_timer = self.create_timer(.001, self.callback_func) # run 1000 hz

        self.time_sent = 0
        self.actuation_begin = 0

    def odom_callback(self, msg):       
        self.sensor_recieved = True
        self.sensor_val = msg.angle

        if not self.actuated and abs(self.sensor_val) > self.bounds:
            self.actuated = True
            self.actuation_begin = self.get_clock().now()
            time_diff = self.time_sent - self.actuation_begin
            self.get_logger().info(f"Actuation started in {time_diff}")
        if not self.complete and not self.actuated and abs(self.sensor_val-self.target_cmd) > self.sensor_recieved:
            self.complete = True
            time_diff = self.get_clock().now() - self.actuation_begin 
            self.get_logger().info(f"Actuation completed in {time_diff}")

    def declare_params(self):
        self.declare_parameter('sensor_topic', '/sensors')
        self.declare_parameter('cmd_topic', 'out')

    def callback_func(self):
        if not self.sensor_recieved:
            return
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.speed = 0
        drive_msg.drive.steering_angle = self.target_cmd

        self.time_sent = self.get_clock().now()
        self.cmd_pub.publish()


def main():
    rclpy.init()
    node = StepResponseNode()

    rclpy.spin(node)

  

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
