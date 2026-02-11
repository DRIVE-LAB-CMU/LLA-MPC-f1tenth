from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Launch the AprilTag EKF node."""
    
    # AprilTag positions in world frame (6-DOF)
    # Format: tag_id: {x, y, z, roll, pitch, yaw}
    tag_positions = {
        '0': {
            'x': 0.0, 'y': 0.0, 'z': 1.5,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
        '1': {
            'x': 2.0, 'y': 0.0, 'z': 1.5,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
        '2': {
            'x': 2.0, 'y': 2.0, 'z': 1.5,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 1.57
        },
        '3': {
            'x': 0.0, 'y': 2.0, 'z': 1.5,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 1.57
        },
    }
    
    # Optional: Different sizes for different tags (in meters)
    # If not specified, tag_size is used for all tags
    tag_sizes = {
        # '0': 0.16,  # 16cm
        # '1': 0.20,  # 20cm
    }

    config = os.path.join(
        get_package_share_directory('apriltag_ekf'),
        'config',
        'apriltag_ekf_config.yaml'
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_baselink_to_camera',
        arguments=['0.27', '0.0', '0.15', '0.0', '0.0', '0.0', 'base_link', 'camera_link'],
        output='screen'
    )
    
    apriltag_ekf_node = Node(
        package='apriltag_ekf',
        executable='apriltag_ekf_node',
        name='apriltag_ekf',
        output='screen',
        parameters=[config],
        remappings=[
            ('/odom', '/odom'),
            ('/apriltag/detections', '/detections'),
            ('/camera_info', '/camera/camera_info'),
            ('/odom/filtered', '/odom/filtered'),
        ]
    )


    
    
    return LaunchDescription([
        apriltag_ekf_node, static_tf_node
    ])