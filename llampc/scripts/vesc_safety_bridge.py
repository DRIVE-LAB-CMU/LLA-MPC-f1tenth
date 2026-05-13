#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64

class VescSafetyBridge(Node):
    def __init__(self):
        super().__init__('vesc_safety_bridge')
        
        # Subscribe to the SAFE output of the ackermann_mux
        self.sub = self.create_subscription(
            AckermannDriveStamped, '/drive', self.drive_cb, 10)
        
        # Publishers direct to VESC (Updated for PWM / Duty Cycle)
        self.duty_cycle_pub = self.create_publisher(Float64, '/commands/motor/duty_cycle', 10)
        self.speed_pub = self.create_publisher(Float64, '/commands/motor/speed', 10)
        self.servo_pub = self.create_publisher(Float64, '/commands/servo/position', 10)
        
        # YAML Parameters
        self.steer_gain = -1.2135
        self.steer_offset = 0.5304
        self.speed_to_erpm_gain = 4614.0

    def drive_cb(self, msg):
        drive = msg.drive
        
        # 1. ALWAYS handle steering safely using your YAML math
        servo_cmd = (drive.steering_angle * self.steer_gain) + self.steer_offset
        servo_cmd = max(0.15, min(0.85, servo_cmd)) # Clamp to your YAML limits
        print(f"Servo: {servo_cmd}")
        self.servo_pub.publish(Float64(data=servo_cmd))
        
        
        
        # 2. Safety Intercept Logic (Using the Magic Flag)
        if drive.jerk == 1.0:
            # The 'jerk' flag is present! This is the MPC.
            # Safe to publish the smuggled PWM, even if it is 0.0.
            self.duty_cycle_pub.publish(Float64(data=drive.acceleration))
            print(f"Accel: {drive.acceleration}")
        else:
            # No flag. This is Joystick, AEB, or standard Nav fallback.
            # Convert m/s to ERPM using your YAML gain
            erpm_cmd = drive.speed * self.speed_to_erpm_gain
            self.speed_pub.publish(Float64(data=erpm_cmd))

def main(args=None):
    rclpy.init(args=args)
    node = VescSafetyBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()