#include "vesc_ackermann/ackermann_to_vesc.hpp"
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <cmath>
#include <algorithm>

namespace vesc_ackermann
{

using ackermann_msgs::msg::AckermannDriveStamped;
using std::placeholders::_1;
using std_msgs::msg::Float64;

AckermannToVesc::AckermannToVesc(const rclcpp::NodeOptions & options)
: Node("ackermann_to_vesc_node", options)
{
  // Get conversion parameters
  speed_to_erpm_gain_ = declare_parameter("speed_to_erpm_gain").get<double>();
  speed_to_erpm_offset_ = declare_parameter("speed_to_erpm_offset").get<double>();
  steering_to_servo_gain_ = declare_parameter("steering_angle_to_servo_gain").get<double>();
  steering_to_servo_offset_ = declare_parameter("steering_angle_to_servo_offset").get<double>();

  // Publishers: Speed (ERPM), Servo (Position), and Duty Cycle (PWM)
  erpm_pub_ = create_publisher<Float64>("commands/motor/speed", 10);
  servo_pub_ = create_publisher<Float64>("commands/servo/position", 10);
  duty_cycle_pub_ = create_publisher<Float64>("commands/motor/duty_cycle", 10);

  // Subscribe to the MUX output (changed from ackermann_cmd to /drive)
  ackermann_sub_ = create_subscription<AckermannDriveStamped>(
    "ackermann_cmd", 10, std::bind(&AckermannToVesc::ackermannCmdCallback, this, _1));
}

void AckermannToVesc::ackermannCmdCallback(const AckermannDriveStamped::SharedPtr cmd)
{
  if (!rclcpp::ok()) return;

  // 1. Handle Steering (Servo) - Always calculated and published
  Float64 servo_msg;
  double servo_cmd = (cmd->drive.steering_angle * steering_to_servo_gain_) + steering_to_servo_offset_;
  
  // Optional: Safety clamp for servo (0.0 to 1.0)
  servo_msg.data = std::max(0.1, std::min(0.9, servo_cmd));
  servo_pub_->publish(servo_msg);

  // 2. Motor Control Logic (The Jerk Flag)
  if (cmd->drive.jerk == 1.0) {
    // --- MPC MODE ---
    // Extract acceleration field as Duty Cycle
    Float64 duty_msg;
    duty_msg.data = cmd->drive.acceleration; 
    duty_cycle_pub_->publish(duty_msg);
  } 
  else if (std::abs(cmd->drive.speed) > 0.01) {
    // --- MANUAL MODE ---
    // Convert speed to ERPM
    Float64 erpm_msg;
    erpm_msg.data = (speed_to_erpm_gain_ * cmd->drive.speed) + speed_to_erpm_offset_;
    erpm_pub_->publish(erpm_msg);
  }
  else {
    // --- STOP STATE ---
    // Explicitly send 0 to stop the motor (Prevents coasting)
    Float64 stop_msg;
    stop_msg.data = 0.0;
    erpm_pub_->publish(stop_msg);
  }
}

}  // namespace vesc_ackermann

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vesc_ackermann::AckermannToVesc)
