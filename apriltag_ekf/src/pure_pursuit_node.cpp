#include <sstream>
#include <string>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
/// CHECK: include needed ROS msg type headers and libraries

// ros2 run pure_pursuit pure_pursuit_node --ros-args --params-file /home/henry/sim_ws/pure_pursuit/config/waypoints.yaml

using namespace std;

class PurePursuit : public rclcpp::Node
{
    // Implement PurePursuit
    // This is just a template, you are free to implement your own node!

private:
    std::vector<double>waypoints_x_;
    std::vector<double>waypoints_y_;
    unsigned int waypoint_index_;
    double horizon_;
    double max_steer_;
    double max_vel_;
    double min_vel_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr waypoint_publisher_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher_; 
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;

    double get_dist(double goal_x, double goal_y, double x, double y){
        return std::sqrt((goal_x-x)*(goal_x-x) + (goal_y-y)*(goal_y-y));
    }

    void pose_callback_(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg)
    {
        // 
        // RCLCPP_INFO(this->get_logger(), "HELLO");

        geometry_msgs::msg::PoseArray untracked_waypoints;
        untracked_waypoints.header.stamp = this->get_clock() -> now();
        untracked_waypoints.header.frame_id = "map";

        geometry_msgs::msg::PoseStamped goal_pose;
        double goal_dist = -1;
        goal_pose.header.frame_id = "map";
        // TODO: find the current waypoint to track using methods mentioned in lecture
        for(int i = 0; i < waypoints_x_.size(); i++){
            geometry_msgs::msg::Pose waypoint;
            int index = (waypoint_index_ + i) % waypoints_x_.size();
            double dist = get_dist(
                waypoints_x_[index], 
                waypoints_y_[index],
                pose_msg->pose.pose.position.x,
                pose_msg->pose.pose.position.y
            );


            if(goal_dist < 0 && dist >= horizon_){
                goal_dist = dist;
                goal_pose.header.stamp = this -> get_clock() -> now();
                waypoint_index_ = index;
                goal_pose.pose.position.x = waypoints_x_[index];
                goal_pose.pose.position.y = waypoints_y_[index];
            } else{
                waypoint.position.x = waypoints_x_[index];
                waypoint.position.y = waypoints_y_[index];
                untracked_waypoints.poses.push_back(waypoint);
            }
        }

        // TODO: transform goal point to vehicle frame of reference
        geometry_msgs::msg::Pose odom_pose = pose_msg->pose.pose;

        tf2::Quaternion robot_angle(
            odom_pose.orientation.x,
            odom_pose.orientation.y,
            odom_pose.orientation.z,
            odom_pose.orientation.w);

        tf2::Vector3 diff(
            goal_pose.pose.position.x - odom_pose.position.x,
            goal_pose.pose.position.y - odom_pose.position.y,
            0.0);

        tf2::Matrix3x3 inverse_rotation(robot_angle.inverse());
        tf2::Vector3 goal_local = inverse_rotation * diff;
    

        // TODO: calculate curvature/steering angle

        // double curvature = 2 * std::abs(p_odom.y())/(goal_dist * goal_dist);
    
        const double wheelbase = 0.33;
        double numerator = 2 * goal_local.y() * wheelbase;
        double denominator = goal_dist * goal_dist;

        double steer;

        if(goal_dist < 1e-3){
            steer = 0;
        } else{
            double steering_clamp = 0.2;
            steer = std::min(std::max(std::atan2(numerator, denominator), -steering_clamp), steering_clamp);
        }
        
        const double scale = max_vel_-min_vel_;
        
        double vel = min_vel_ + scale*std::sqrt(1-std::abs(steer/max_steer_));

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.drive.speed = vel;
        drive_msg.drive.steering_angle =steer;

        // TODO: publish drive message, don't forget to limit the steering angle.
        drive_publisher_->publish(drive_msg);
        goal_publisher_->publish(goal_pose);
        waypoint_publisher_-> publish(untracked_waypoints);

    }

public:
    PurePursuit() : Node("pure_pursuit_node")
    {
        this->declare_parameter<std::vector<double>>("x", {});
        this->declare_parameter<std::vector<double>>("y", {});
        this->declare_parameter<double>("horizon", 1.0);
        this->declare_parameter<double>("max_steer", 0.36);
        this->declare_parameter<double>("max_vel", 3.0);
        this->declare_parameter<double>("min_vel", 0.5);

        this->get_parameter("x", waypoints_x_);
        this->get_parameter("y", waypoints_y_);
        this->get_parameter("horizon", horizon_);
        this->get_parameter("max_steer", max_steer_);
        this->get_parameter("max_vel", max_vel_);
        this->get_parameter("min_vel", min_vel_);

        RCLCPP_INFO(this->get_logger(),std::to_string(waypoints_x_.size()));
        RCLCPP_INFO(this->get_logger(),std::to_string(waypoints_y_.size()));

        if (waypoints_x_.size() != waypoints_y_.size()) {
            RCLCPP_ERROR(this->get_logger(), "'x' and 'y' parameters have different lengths");
            return;
        }
        waypoint_index_ = 0;

        goal_publisher_ =  this->create_publisher<geometry_msgs::msg::PoseStamped>("tracked_waypoint", 10);
        waypoint_publisher_ =  this->create_publisher<geometry_msgs::msg::PoseArray>("untracked_waypoints", 10);
        drive_publisher_ =  this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10, 
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg){
                this -> pose_callback_(msg);
            }
        );
    }

    


    ~PurePursuit() {}
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}