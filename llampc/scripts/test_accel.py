#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
import time

def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)
    
    # Create a simple procedural node
    node = rclpy.create_node('accel_test_node')
    
    # Create the publisher for the ackermann_cmd topic
    pub = node.create_publisher(AckermannDriveStamped, '/drive', 10)
    
    node.get_logger().warn("=== STARTING ACCELERATION TEST ===")
    node.get_logger().warn("Make sure the car is on a block! Wheels will spin up fast.")
    
    # Give the ROS 2 DDS network a moment to discover the publisher
    time.sleep(3.0) 
    
    def send_acceleration_command(target_accel, duration, step_name):
        """
        Publishes an acceleration command for a specific duration at 50Hz.
        """
        node.get_logger().info(f"Executing: {step_name} | Accel: {target_accel} m/s^2 for {duration}s")
        end_time = time.time() + duration
        
        while rclpy.ok() and time.time() < end_time:
            msg = AckermannDriveStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            # ---------------------------------------------------------
            # THE MAGIC KEY: Speed MUST be exactly 0.0 
            # This forces the low-level node into current/brake mode.
            # ---------------------------------------------------------
            msg.drive.speed = 0.0 
            msg.drive.acceleration = float(target_accel)
            msg.drive.steering_angle = 0.0 # Keep wheels straight
            
            pub.publish(msg)
            
            # Sleep for roughly 20ms to hit ~50Hz
            time.sleep(0.02) 

    try:
        # TEST 1: Gentle acceleration
        # In a frictionless environment (on a block), the wheel RPM should steadily 
        # climb. It will NOT snap to a constant speed like velocity control does.
        send_acceleration_command(1.0, 2.0, "Gentle Spool Up")
        
        # TEST 2: Coasting
        # Zero acceleration means zero current. The wheels will slowly spin down
        # due to internal motor and drivetrain friction.
        send_acceleration_command(0.0, 1.5, "Coasting (0 Current)")
        
        # TEST 3: Active Braking
        # Negative acceleration commands regenerative braking. The wheels should 
        # snap to a halt much faster than coasting.
        send_acceleration_command(-3.0, 1.0, "Active Braking")
        
        # TEST 4: Hard Acceleration
        send_acceleration_command(3.0, 1.0, "Hard Spool Up")
        
        # Final safety stop
        send_acceleration_command(-5.0, 0.5, "Final Stop")
        send_acceleration_command(0.0, 0.5, "Zeroing Output")
        
        node.get_logger().info("Test complete. Car should be stationary.")
        
    except KeyboardInterrupt:
        node.get_logger().error("Test interrupted by user! Sending zero command.")
    
    finally:
        # Failsafe: Always send a single zero command to halt before exiting
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.acceleration = 0.0
        pub.publish(msg)
        
        # Clean up ROS 2 node
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()