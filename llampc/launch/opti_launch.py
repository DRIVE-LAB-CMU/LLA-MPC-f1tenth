import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Path to f1tenth_stack launch file
    f1tenth_launch_dir = os.path.join(get_package_share_directory('f1tenth_stack'), 'launch')
    f1tenth_launch_file = os.path.join(f1tenth_launch_dir, 'bringup_launch.py')

    # # Path to particle_filter launch file 
    # pf_launch_dir = os.path.join(get_package_share_directory('particle_filter'), 'launch')
    # pf_launch_file = os.path.join(pf_launch_dir, 'localize_launch.py')
    vrpn_launch_file = os.path.join(get_package_share_directory('vrpn_mocap'), 'launch', 'client.launch.yaml')
    
    # Path to your ekf.yaml config file in the llampc package
    llampc_config_dir = os.path.join(get_package_share_directory('llampc'), 'config')
    ekf_config_path = os.path.join(llampc_config_dir, 'mocap.yaml')

    # =================================================================
    # =================================================================
    
    vrpn_mocap_action = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(vrpn_launch_file),
        launch_arguments={
            'server': '172.26.119.139',
            'port': '3883',
            'frame_id': 'world'
        }.items()
    )
    
    # Argument for Optitrack topic, defaulting to what was in your python script
    mocap_topic_la = DeclareLaunchArgument(
        'mocap_topic',
        default_value='/vrpn_mocap/f1tenth/pose',
        description='Optitrack pose topic name'
    )
    

    # =================================================================
    # 3. Create Launch Actions
    # =================================================================

    # Action to launch the f1tenth stack
    f1tenth_stack_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f1tenth_launch_file)
    )

    # # Action to launch the particle filter
    # particle_filter_action = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(pf_launch_file)
    # )

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
            ('/set_pose', '/initialpose') # <--- Routes RViz directly into the EKF's reset switch
        ]
    )

    # Action to run your new Optitrack Subscriber script
    optitrack_node = Node(
        package='llampc',             # <-- UPDATE THIS if the script is in a different package
        executable='optitrack_node.py',  # <-- UPDATE THIS to your exact Python script name (e.g., 'optitrack_node.py')
        name='optitrack_subscriber',
        output='screen',
        parameters=[{'topic': LaunchConfiguration('mocap_topic')}]
    )

    # =================================================================
    # 4. Return the LaunchDescription
    # =================================================================
    
    return LaunchDescription([
        vrpn_mocap_action, 
        mocap_topic_la,
        f1tenth_stack_action,
        # particle_filter_action,
        zupt_node,
        ekf_node,
        optitrack_node
    ])