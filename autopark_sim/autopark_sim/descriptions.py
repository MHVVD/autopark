"""URDF strings read by webots_ros2_driver (which Python plugin runs for each robot). No ROS.

The cameras are read and published by our own VehiclePlugin (not by webots_ros2's camera
device), so every image is stamped with the exact simulation time of its render. webots_ros2
stamps images with its ROS clock, which follows /clock from another process and lagged the
render by 0-2 steps from frame to frame (measured: 88 % one step, the rest 0 or 2 steps).
"""
from autopark_sim.rig import CAMERAS as _RIG

CAMERAS = [c.name for c in _RIG]
CAMERA_PERIOD_MS = 100  # 10 Hz


def image_topic(cam):
    return f'/cam_{cam}/image'


def vehicle_description(noise=None):
    """noise: optional dict of VehiclePlugin properties (gyroNoise, gyroBias, speedNoise,
    steerNoise, noiseSeed)."""
    props = ''.join(f'\n      <{k}>{v}</{k}>' for k, v in (noise or {}).items())
    return f"""<?xml version="1.0" ?>
<robot name="ego_car">
  <webots>
    <plugin type="autopark_sim.vehicle_plugin.VehiclePlugin">{props}
    </plugin>
  </webots>
</robot>
"""


def supervisor_description(seed=0):
    return f"""<?xml version="1.0" ?>
<robot name="gt_supervisor">
  <webots>
    <plugin type="autopark_sim.supervisor_plugin.GroundTruthSupervisor">
      <seed>{seed}</seed>
    </plugin>
  </webots>
</robot>
"""
