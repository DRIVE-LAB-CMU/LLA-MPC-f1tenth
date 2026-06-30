#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ForwardLegAnalyzer(Node):
    def __init__(self):
        super().__init__('forward_leg_analyzer')
        self.pub = self.create_publisher(String, '/network/pong', 10)
        self.sub = self.create_subscription(String, '/network/ping', self.ping_callback, 10)
        self.last_arrival_steady = None

    def ping_callback(self, msg):
        # Capture local steady (boot) time immediately on packet arrival
        current_steady = self.get_clock().now().nanoseconds
        
        # Calculate consistency of the incoming stream
        if self.last_arrival_steady is not None:
            arrival_gap_ms = (current_steady - self.last_arrival_steady) / 1e6
            # Ideal gap for 40Hz is 25.0ms
            if arrival_gap_ms > 35.0 or arrival_gap_ms < 15.0:
                self.get_logger().warn(f"Forward Leg Jitter Detected! Packet arrival gap: {arrival_gap_ms:.2f} ms")
        
        self.last_arrival_steady = current_steady
        
        # Echo back instantly
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(ForwardLegAnalyzer())
    rclpy.shutdown()

if __name__ == '__main__':
    main()