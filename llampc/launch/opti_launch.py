import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Path to f1tenth_stack launch file
    f1tenth_launch_dir = os.path.join(get_package_share_directory('f1tenth_stack'), 'launch')
    f1tenth_launch_file = os.path.join(f1tenth_launch_dir, 'bringup_launch.py')

    # Path to natnet_ros2 launch file
    natnet_launch_file = os.path.join(
        get_package_share_directory('natnet_ros2'), 'launch', 'natnet_ros2.launch.py'
    )

    # Path to your ekf.yaml config file in the llampc package
    llampc_config_dir = os.path.join(get_package_share_directory('llampc'), 'config')
    ekf_config_path = os.path.join(llampc_config_dir, 'mocap.yaml')

    # =================================================================
    # 1. Mocap (NatNet) action
    # =================================================================

    natnet_mocap_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(natnet_launch_file),
        launch_arguments={
            'serverIP': '172.26.119.139',   # Host PC IP (Local Interface in Motive)
            'clientIP': '172.26.112.71',   # <-- IP of THIS PC (where you launch)
            'serverType': 'unicast',        # 'multicast' or 'unicast'
            'pub_rigid_body': 'true', 
        }.items()
    )

    # Argument for the mocap pose topic.
    # natnet_ros2 publishes per-rigid-body topics: /natnet_ros2/<body_name>/pose
    mocap_topic_la = DeclareLaunchArgument(
        'mocap_topic',
        default_value='/f1tenth/pose',
        description='NatNet rigid-body pose topic name'
    )

    # =================================================================
    # 2. Create Launch Actions
    # =================================================================

    # Action to launch the f1tenth stack
    f1tenth_stack_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f1tenth_launch_file)
    )

    # Action to run your ZUPT script
    zupt_node = Node(
        package='llampc',
        executable='imu_zupt_prep.py',
        name='zupt_node',
        output='screen'
    )

    # Action to run robot_localization EKF with your yaml file
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        remappings=[
            ('/set_pose', '/initialpose')  # Routes RViz directly into the EKF's reset switch
        ]
    )

    # Action to run your Optitrack subscriber script
    optitrack_node = Node(
        package='llampc',
        executable='optitrack_node.py',
        name='optitrack_subscriber',
        output='screen',
        parameters=[{'topic': LaunchConfiguration('mocap_topic')}]
    )

    # =================================================================
    # 3. Return the LaunchDescription
    # =================================================================

    return LaunchDescription([
        natnet_mocap_action,
        mocap_topic_la,
        f1tenth_stack_action,
        zupt_node,
        ekf_node,
        optitrack_node
    ])