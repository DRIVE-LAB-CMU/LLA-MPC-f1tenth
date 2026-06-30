#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String # Simple text payload to simulate tracking packet sizes

class CarReflector(Node):
    def __init__(self):
        super().__init__('car_reflector')
        self.pub = self.create_publisher(String, '/network/pong', 10)
        self.sub = self.create_subscription(String, '/network/ping', self.ping_callback, 10)

    def ping_callback(self, msg):
        # Instantly echo the exact message back to the desktop
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(CarReflector())
    rclpy.shutdown()

if __name__ == '__main__':
    main()