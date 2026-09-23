"""Simulation + the autonomy stack (milestone 1: odometry only).

Args (passed through to autopark_sim/sim.launch.py): seed, gui, mode
  heading_source  imu | steering   odometry heading source   default imu
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('autopark_sim'), 'launch', 'sim.launch.py')),
        launch_arguments={'seed': LaunchConfiguration('seed'),
                          'gui': LaunchConfiguration('gui'),
                          'mode': LaunchConfiguration('mode')}.items(),
    )
    odometry = Node(package='autopark', executable='odometry', output='screen',
                    parameters=[{'use_sim_time': True,
                                 'heading_source': LaunchConfiguration('heading_source')}])
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('mode', default_value='realtime'),
        DeclareLaunchArgument('heading_source', default_value='imu'),
        sim,
        odometry,
    ])
