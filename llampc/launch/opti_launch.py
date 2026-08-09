import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    f1tenth_launch_dir = os.path.join(get_package_share_directory('f1tenth_stack'), 'launch')
    f1tenth_launch_file = os.path.join(f1tenth_launch_dir, 'bringup_launch.py')

    natnet_launch_file = os.path.join(
        get_package_share_directory('natnet_ros2'), 'launch', 'natnet_ros2.launch.py'
    )

    llampc_config_dir = os.path.join(get_package_share_directory('llampc'), 'config')
    ekf_config_path = os.path.join(llampc_config_dir, 'mocap.yaml')

    natnet_mocap_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(natnet_launch_file),
        launch_arguments={
            'serverIP': '172.26.119.139',
            'clientIP': '172.26.112.71',
            'serverType': 'unicast',
            'pub_rigid_body': 'true',
        }.items()
    )

    mocap_topic_la = DeclareLaunchArgument(
        'mocap_topic',
        default_value='/f1tenth/pose',
        description='NatNet rigid-body pose topic name'
    )

    f1tenth_stack_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f1tenth_launch_file)
    )

    zupt_node = Node(
        package='llampc',
        executable='imu_zupt_prep.py',
        name='zupt_node',
        output='screen'
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        remappings=[
            ('/set_pose', '/initialpose')
        ]
    )

    optitrack_node = Node(
        package='llampc',
        executable='optitrack_node.py',
        name='optitrack_subscriber',
        output='screen',
        parameters=[{'topic': LaunchConfiguration('mocap_topic')}]
    )

    cg_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_baselink_to_cg',
            arguments=['0.15', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'cg']
        )
    
    imu_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_baselink_to_imu',
        arguments=['0.065', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'imu_link']
    )

    return LaunchDescription([
        natnet_mocap_action,
        mocap_topic_la,
        f1tenth_stack_action,
        zupt_node,
        ekf_node,
        optitrack_node
    ])
