import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
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

    # =================================================================
    # 1. Mocap (NatNet) action — wrapped so we can respawn it on crash
    # =================================================================
    def make_natnet_action():
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(natnet_launch_file),
            launch_arguments={
                'serverIP': '172.26.119.139',
                'clientIP': '172.26.112.71',
                'serverType': 'unicast',
                'pub_rigid_body': 'true',
            }.items()
        )

    natnet_mocap_action = make_natnet_action()

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

    # =================================================================
    # Respawn handler: if natnet_ros2's process dies, relaunch it
    # =================================================================
    natnet_respawn_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=natnet_mocap_action,
            on_exit=[make_natnet_action()]
        )
    )

    return LaunchDescription([
        natnet_mocap_action,
        natnet_respawn_handler,
        mocap_topic_la,
        f1tenth_stack_action,
        zupt_node,
        ekf_node,
        optitrack_node
    ])