from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # Locate the EKF configuration file. 
    # **NOTE:** Change 'f1tenth_stack' to the actual name of your package if it's different!
    ekf_config_path = PathJoinSubstitution([
        FindPackageShare('f1tenth_stack'),
        'config',
        'ekf.yaml'
    ])

    # Define the EKF Node
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        # If your EKF publishes to a specific topic name, uncomment and adjust this line:
        # remappings=[('odometry/filtered', '/odom')]
    )

    return LaunchDescription([
        ekf_node
    ])