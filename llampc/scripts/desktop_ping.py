#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DesktopTimer(Node):
    def __init__(self):
        super().__init__('desktop_timer')
        self.pub = self.create_publisher(String, '/network/ping', 10)
        self.sub = self.create_subscription(String, '/network/pong', self.pong_callback, 10)
        
        # Create a timer to ping the car at 40Hz (matching your vehicle's tracking rate)
        self.create_timer(1.0 / 40.0, self.send_ping)

    def send_ping(self):
        msg = String()
        # Record current Unix epoch nanoseconds using only the Desktop clock
        msg.data = str(time.time_ns())
        self.pub.publish(msg)

    def pong_callback(self, msg):
        t1 = time.time_ns()
        t0 = int(msg.data)
        
        # Calculate full loop transit time
        round_trip_ms = (t1 - t0) / 1e6
        one_way_delay_ms = round_trip_ms / 2.0
        
        self.get_logger().info(f"True One-Way Network Transit Delay: {one_way_delay_ms:.2f} ms (Clock-Independent)")

def main():
    rclpy.init()
    rclpy.spin(DesktopTimer())
    rclpy.shutdown()

if __name__ == '__main__':
    main()