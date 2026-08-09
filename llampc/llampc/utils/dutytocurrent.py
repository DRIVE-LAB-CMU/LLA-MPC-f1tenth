#!/usr/bin/env python3
"""
Test node for the low-speed duty -> current handoff strategy discussed in
conversation:

    - Below SWITCH_SPEED_ERPM: control via commands/motor/duty_cycle. Duty
      directly sets applied voltage, so it can reliably overcome static
      friction and produce motion even at near-zero speed/current, where
      the current-based torque equation is the most ill-conditioned
      (small denominator, current-mode control has the least useful
      feedback since iq is near zero anyway).
    - Once measured speed crosses SWITCH_SPEED_ERPM: hand off to
      commands/motor/current, since that's what the MPC's dynamics model
      (and VESC's own closed-loop current controller) actually wants to
      command once the car is rolling and back-EMF/voltage-saturation
      effects are the dominant physics, not stiction.

Both stages are simple PID loops on speed error (measured VESC 'speed',
i.e. electrical RPM) -- NOT the real dynamics-model controller, just a
test harness to verify the handoff behaves sensibly (no discontinuity/
kick at the switch, actually gets the car moving, etc.) before wiring
this logic into the real MPC control path.

Prints every command sent alongside live telemetry (speed, current_motor,
duty_cycle) so you can see the transition happen and verify smoothness.

Usage:
    python3 duty_to_current_handoff_test.py --target-erpm 8000
    python3 duty_to_current_handoff_test.py --target-erpm 8000 --switch-erpm 2000
"""

import argparse
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64

from vesc_msgs.msg import VescStateStamped


class PID:
    """Minimal PID with output clamping and anti-windup (clamp-and-hold)."""

    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        # tentative integral update, then clamp-and-check (basic anti-windup)
        tentative_integral = self.integral + error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        out_unclamped = self.kp * error + self.ki * tentative_integral + self.kd * derivative
        out = max(self.out_min, min(self.out_max, out_unclamped))

        # only commit the integral step if we're not saturated in the direction
        # that would make windup worse
        if out == out_unclamped or (out == self.out_max and error < 0) or (out == self.out_min and error > 0):
            self.integral = tentative_integral

        return out


class HandoffTestNode(Node):
    def __init__(self, target_erpm, switch_erpm, duty_pid_gains, current_pid_gains,
                 duty_limits, current_limits, rate_hz):
        super().__init__('duty_to_current_handoff_test')

        self.target_erpm = target_erpm
        self.switch_erpm = switch_erpm
        self.mode = 'DUTY'   # 'DUTY' or 'CURRENT'

        self.duty_pid = PID(*duty_pid_gains, *duty_limits)
        self.current_pid = PID(*current_pid_gains, *current_limits)

        self.latest_speed = 0.0
        self.latest_current_motor = 0.0
        self.latest_duty = 0.0
        self.have_state = False
        self._lock = threading.Lock()

        qos_sensor = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 history=QoSHistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(VescStateStamped, '/sensors/core', self.cb, qos_sensor)

        self.pub_duty = self.create_publisher(Float64, '/commands/motor/duty_cycle', 10)
        self.pub_current = self.create_publisher(Float64, '/commands/motor/current', 10)

        self.dt = 1.0 / rate_hz
        self._last_t = None
        self.timer = self.create_timer(self.dt, self.control_step)

        self.get_logger().info(
            f'target_erpm={target_erpm}  switch_erpm={switch_erpm}  '
            f'starting in DUTY mode. Ctrl+C to stop (sends 0 on exit).')

    def cb(self, msg):
        with self._lock:
            self.latest_speed = msg.state.speed
            self.latest_current_motor = msg.state.current_motor
            self.latest_duty = msg.state.duty_cycle
            self.have_state = True

    def control_step(self):
        if not self.have_state:
            return

        with self._lock:
            speed = self.latest_speed
            current_motor = self.latest_current_motor
            duty_meas = self.latest_duty

        now = time.monotonic()
        dt = self.dt if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now

        error = self.target_erpm - speed

        if self.mode == 'DUTY':
            if abs(speed) >= self.switch_erpm:
                # HANDOFF: switch to current mode. Reset the current PID fresh
                # (don't try to carry over duty's integral state -- different
                # units/scale) and explicitly zero the duty command so nothing
                # is left latched on that topic.
                self.get_logger().info(
                    f'>>> HANDOFF at speed={speed:.1f}erpm (>= switch_erpm={self.switch_erpm}) '
                    f'-- switching DUTY -> CURRENT')
                self.pub_duty.publish(Float64(data=0.0))
                self.current_pid.reset()
                self.mode = 'CURRENT'
            else:
                duty_cmd = self.duty_pid.update(error, dt)
                self.pub_duty.publish(Float64(data=duty_cmd))
                print(f'[DUTY]    speed={speed:8.1f}erpm  err={error:8.1f}  '
                      f'cmd_duty={duty_cmd:+6.3f}  meas: duty={duty_meas:+6.3f} '
                      f'current_motor={current_motor:6.2f}A')
                return

        # CURRENT mode
        current_cmd = self.current_pid.update(error, dt)
        self.pub_current.publish(Float64(data=current_cmd))
        print(f'[CURRENT] speed={speed:8.1f}erpm  err={error:8.1f}  '
              f'cmd_current={current_cmd:+7.2f}A  meas: duty={duty_meas:+6.3f} '
              f'current_motor={current_motor:6.2f}A')

    def stop(self):
        self.pub_duty.publish(Float64(data=0.0))
        self.pub_current.publish(Float64(data=0.0))


def main():
    parser = argparse.ArgumentParser(description='Duty->current handoff test at low speed.')
    parser.add_argument('--target-erpm', type=float, default=8000.0,
                         help='Target speed (electrical RPM) to hold via PID (default %(default)s).')
    parser.add_argument('--switch-erpm', type=float, default=2000.0,
                         help='Speed threshold to hand off from duty control to current control '
                              '(default %(default)s). Should be comfortably above stiction/near-'
                              'zero-speed regime but below where duty control would saturate.')
    parser.add_argument('--duty-kp', type=float, default=3e-5)
    parser.add_argument('--duty-ki', type=float, default=1e-5)
    parser.add_argument('--duty-kd', type=float, default=0.0)
    parser.add_argument('--duty-min', type=float, default=-0.20)
    parser.add_argument('--duty-max', type=float, default=0.20)
    parser.add_argument('--current-kp', type=float, default=0.01)
    parser.add_argument('--current-ki', type=float, default=0.003)
    parser.add_argument('--current-kd', type=float, default=0.0)
    parser.add_argument('--current-min', type=float, default=-40.0)
    parser.add_argument('--current-max', type=float, default=40.0)
    parser.add_argument('--rate', type=float, default=50.0, help='Control loop rate in Hz.')
    args = parser.parse_args()

    rclpy.init()
    node = HandoffTestNode(
        target_erpm=args.target_erpm,
        switch_erpm=args.switch_erpm,
        duty_pid_gains=(args.duty_kp, args.duty_ki, args.duty_kd),
        current_pid_gains=(args.current_kp, args.current_ki, args.current_kd),
        duty_limits=(args.duty_min, args.duty_max),
        current_limits=(args.current_min, args.current_max),
        rate_hz=args.rate,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()