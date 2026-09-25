"""Simulation + the autonomy stack (odometry, bird's-eye view, slot detector, slot tracker,
planner, controller, parking manager).

Args (passed through to autopark_sim/sim.launch.py): seed, gui, mode
  heading_source  imu | steering   odometry heading source   default imu
  bev             true | false     run the BEV node          default true
  detector        true | false     run the slot detector     default true
  model           path             detector weights          default ~/autopark_models/slotnet_v2.pt
  view_yaw_tol    deg              tracker uses detections of slots seen within this   default 90
                                   of perpendicular (20 for the aisle-only slotnet.pt)
  tracker         true | false     run the slot tracker      default true
  noise_pos       m                injected detection noise (entrance position std)   default 0
  noise_yaw_deg   deg              injected detection noise (heading std)             default 0
  noise_dropout   0..1             probability of dropping a detected slot            default 0
  noise_false     per frame        mean number of false slots per frame               default 0
  planner         true | false     run the planner (/parking/plan service)            default true
  park            true | false     run the controller + parking manager: the car      default true
                                   searches for a vacant slot and parks by itself
                                   (use park:=false for the evaluation tools that drive)
  replan          closed | open    closed: replan at cusps and correct the final       default closed
                                   approach with the slot estimate; open: plan once
The tracker always reads /slots/detections_noisy (a pass-through when all noise is 0) and is
told the injected noise level (extra_pos_std / extra_yaw_std_deg).

OPENBLAS_NUM_THREADS=1: numpy's OpenBLAS otherwise keeps extra threads spinning after every
small matrix call (measured: the tracker used 3x the CPU for the same speed, about two cores
in total), which starved the rest of the stack.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
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
    noise = Node(package='autopark', executable='detection_noise', output='screen',
                 parameters=[{'use_sim_time': True,
                              'pos_std': LaunchConfiguration('noise_pos'),
                              'yaw_std_deg': LaunchConfiguration('noise_yaw_deg'),
                              'dropout': LaunchConfiguration('noise_dropout'),
                              'false_per_frame': LaunchConfiguration('noise_false')}],
                 condition=IfCondition(LaunchConfiguration('tracker')))
    tracker = Node(package='autopark', executable='slot_tracker', output='screen',
                   parameters=[{'use_sim_time': True,
                                'detections_topic': '/slots/detections_noisy',
                                'extra_pos_std': LaunchConfiguration('noise_pos'),
                                'extra_yaw_std_deg': LaunchConfiguration('noise_yaw_deg'),
                                'view_yaw_tol_deg': LaunchConfiguration('view_yaw_tol')}],
                   condition=IfCondition(LaunchConfiguration('tracker')))
    planner = Node(package='autopark', executable='planner', output='screen',
                   parameters=[{'use_sim_time': True}],
                   condition=IfCondition(LaunchConfiguration('planner')))
    controller = Node(package='autopark', executable='controller', output='screen',
                      parameters=[{'use_sim_time': True}],
                      condition=IfCondition(LaunchConfiguration('park')))
    manager = Node(package='autopark', executable='parking_manager', output='screen',
                   parameters=[{'use_sim_time': True, 'replan': LaunchConfiguration('replan')}],
                   condition=IfCondition(LaunchConfiguration('park')))
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('mode', default_value='realtime'),
        DeclareLaunchArgument('heading_source', default_value='imu'),
        DeclareLaunchArgument('bev', default_value='true'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('model', default_value=os.path.expanduser('~/autopark_models/slotnet_v2.pt')),
        DeclareLaunchArgument('tracker', default_value='true'),
        DeclareLaunchArgument('view_yaw_tol', default_value='90.0',
                              description='deg from perpendicular within which the tracker uses detections'),
        DeclareLaunchArgument('noise_pos', default_value='0.0'),
        DeclareLaunchArgument('noise_yaw_deg', default_value='0.0'),
        DeclareLaunchArgument('noise_dropout', default_value='0.0'),
        DeclareLaunchArgument('noise_false', default_value='0.0'),
        DeclareLaunchArgument('planner', default_value='true'),
        DeclareLaunchArgument('park', default_value='true'),
        DeclareLaunchArgument('replan', default_value='closed'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        sim,
        odometry,
        bev,
        detector,
        noise,
        tracker,
        planner,
        controller,
        manager,
    ])
