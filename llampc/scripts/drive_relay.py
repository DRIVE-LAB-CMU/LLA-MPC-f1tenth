#!/usr/bin/env python3
"""
On-car drive relay + watchdog.

Subscribes to the off-board MPC's command on /mpc/drive and republishes it to
/drive (which the VESC / f1tenth_system actually listens to). If commands stop
arriving (WiFi drop, MPC hang, laptop crash) the watchdog brakes instead of
letting the last throttle/steer latch.

Freshness is judged by LOCAL receive time, never the remote header stamp, so the
two machines do NOT need synced clocks for the safety behaviour to be correct.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped


def control_qos():
    # newest-wins, no retransmission of stale samples -> right choice over WiFi
    q = QoSProfile(depth=1)
    q.reliability = QoSReliabilityPolicy.BEST_EFFORT
    q.history = QoSHistoryPolicy.KEEP_LAST
    q.durability = QoSDurabilityPolicy.VOLATILE
    return q


class DriveRelay(Node):
    def __init__(self):
        super().__init__('drive_relay')

        self.declare_parameter('mpc_drive_topic', '/mpc/drive')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('timeout_s', 0.12)     # ~3 MPC periods at 25 Hz
        self.declare_parameter('watchdog_hz', 100.0)
        # Republish last good cmd at watchdog rate so /drive is a steady local
        # stream decoupled from network arrival jitter. Recommended on a congested
        # link: the VESC sees a smooth 100 Hz feed even if /mpc/drive stutters.
        self.declare_parameter('republish_last', True)

        self.timeout_ns = int(self.get_parameter('timeout_s').value * 1e9)
        self.republish_last = self.get_parameter('republish_last').value

        self.last_rx_ns = 0
        self.last_msg = None
        self.braking = True

        # /mpc/drive comes over WiFi -> best-effort depth-1.
        # /drive is local to the car; match whatever the VESC expects. f1tenth_system
        # uses default (reliable) here, and a best-effort PUBLISHER will NOT connect
        # to a reliable subscriber, so publish /drive reliably to be safe.
        drive_qos = QoSProfile(depth=1)  # reliable by default

        self.sub = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter('mpc_drive_topic').value,
            self.on_cmd, control_qos())
        self.pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('drive_topic').value,
            drive_qos)

        period = 1.0 / self.get_parameter('watchdog_hz').value
        self.timer = self.create_timer(period, self.watchdog)
        self.get_logger().info('Drive relay + watchdog up')

    def on_cmd(self, msg):
        self.last_rx_ns = self.get_clock().now().nanoseconds
        self.last_msg = msg
        out = msg
        out.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(out)
        if self.braking:
            self.braking = False
            self.get_logger().info('Control link live')

    def watchdog(self):
        now = self.get_clock().now().nanoseconds
        stale = (self.last_rx_ns == 0) or (now - self.last_rx_ns > self.timeout_ns)

        if stale:
            brake = AckermannDriveStamped()
            brake.header.stamp = self.get_clock().now().to_msg()
            brake.header.frame_id = 'base_link'
            # Match the MPC's PWM actuation channel: jerk=1.0 is the active-mode
            # flag, acceleration is the PWM command. Zero PWM = no drive.
            # If zero PWM only coasts and you need active braking, set a negative
            # brake PWM here instead of 0.0.
            brake.drive.speed = 0.0
            brake.drive.jerk = 1.0
            brake.drive.acceleration = 0.0
            brake.drive.steering_angle = (
                self.last_msg.drive.steering_angle if self.last_msg else 0.0)
            self.pub.publish(brake)
            if not self.braking:
                self.braking = True
                self.get_logger().warn('Control link stale -> braking')
        elif self.republish_last and self.last_msg is not None:
            out = self.last_msg
            out.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DriveRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()