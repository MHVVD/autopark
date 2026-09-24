"""Simulation + the autonomy stack (odometry, bird's-eye view, slot detector).

Args (passed through to autopark_sim/sim.launch.py): seed, gui, mode
  heading_source  imu | steering   odometry heading source   default imu
  bev             true | false     run the BEV node          default true
  detector        true | false     run the slot detector     default true
  model           path             detector weights          default ~/autopark_models/slotnet.pt
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
    bev = Node(package='autopark', executable='bev', output='screen',
               parameters=[{'use_sim_time': True}], condition=IfCondition(LaunchConfiguration('bev')))
    detector = Node(package='autopark', executable='slot_detector', output='screen',
                    parameters=[{'use_sim_time': True, 'model': LaunchConfiguration('model')}],
                    condition=IfCondition(LaunchConfiguration('detector')))
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('mode', default_value='realtime'),
        DeclareLaunchArgument('heading_source', default_value='imu'),
        DeclareLaunchArgument('bev', default_value='true'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('model', default_value=os.path.expanduser('~/autopark_models/slotnet.pt')),
        sim,
        odometry,
        bev,
        detector,
    ])
