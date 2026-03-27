import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # =================================================================
    # 1. Define Paths to Packages and Files
    # =================================================================
    
    # Path to f1tenth_stack launch file
    # (Note: Update 'bringup_launch.py' if the f1tenth package uses a different name)
    f1tenth_launch_dir = os.path.join(get_package_share_directory('f1tenth_stack'), 'launch')
    f1tenth_launch_file = os.path.join(f1tenth_launch_dir, 'bringup_launch.py')

    # Path to particle_filter launch file 
    # (Note: Update 'localize_launch.py' if your PF package uses a different name)
    pf_launch_dir = os.path.join(get_package_share_directory('particle_filter'), 'launch')
    pf_launch_file = os.path.join(pf_launch_dir, 'localize_launch.py')

    # Path to your ekf.yaml config file in the llampc package
    llampc_config_dir = os.path.join(get_package_share_directory('llampc'), 'config')
    ekf_config_path = os.path.join(llampc_config_dir, 'ekf.yaml')


    # =================================================================
    # 2. Create Launch Actions
    # =================================================================

    # Action to launch the f1tenth stack
    f1tenth_stack_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f1tenth_launch_file)
    )

    # Action to launch the particle filter
    particle_filter_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(pf_launch_file)
    )

    # Action to run your ZUPT script 
    # (Assuming the script is installed as an executable named 'zupt.py' or 'zupt_node.py')
    zupt_node = Node(
        package='llampc',
        executable='imu_zupt_prep.py',  # <-- UPDATE THIS to your exact Python script name
        name='zupt_node',
        output='screen'
    )

    # Action to run robot_localization EKF with your yaml file
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path]
    )


    # =================================================================
    # 3. Return the LaunchDescription
    # =================================================================
    
    return LaunchDescription([
        f1tenth_stack_action,
        particle_filter_action,
        zupt_node,
        ekf_node
    ])