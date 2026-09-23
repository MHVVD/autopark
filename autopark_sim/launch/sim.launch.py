"""Webots parking-row simulation: ego car + ground-truth supervisor.

Args:
  seed   int                 scenario seed applied at start-up          default 0
  gui    true | false        false = minimised, 3D view not rendered    default true
  mode   realtime | fast                                                default realtime
"""
import os
import tempfile

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher

from autopark_sim.descriptions import (CAMERAS, image_topic, supervisor_description,
                                       vehicle_description, webots_image_topic)


def _write(name, text):
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, 'w') as f:
        f.write(text)
    return path


def launch_setup(context):
    seed = int(LaunchConfiguration('seed').perform(context))
    world = os.path.join(get_package_share_directory('autopark_sim'), 'worlds', 'parking_row.wbt')

    # ros2_supervisor=True: the Ros2Supervisor node publishes /clock.
    webots = WebotsLauncher(world=world, gui=LaunchConfiguration('gui'),
                            mode=LaunchConfiguration('mode'), ros2_supervisor=True)
    ego = WebotsController(
        robot_name='ego_car',
        parameters=[{'robot_description': _write('autopark_ego_car.urdf', vehicle_description()),
                     'use_sim_time': True}],
        remappings=[(webots_image_topic(c), image_topic(c)) for c in CAMERAS],
    )
    gt = WebotsController(
        robot_name='gt_supervisor',
        parameters=[{'robot_description': _write('autopark_gt_supervisor.urdf',
                                                  supervisor_description(seed)),
                     'use_sim_time': True}],
    )
    return [
        webots, webots._supervisor, ego, gt,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('mode', default_value='realtime'),
        OpaqueFunction(function=launch_setup),
    ])
