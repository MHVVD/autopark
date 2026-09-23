"""URDF strings read by webots_ros2_driver (which devices to publish + which Python plugin). No ROS.

Camera note: webots_ros2 publishes when `time - last < 1/updateRate` is false. With exactly
10 Hz that float comparison sometimes fails (0.7 - 0.6 = 0.0999...), dropping to ~8.9 Hz;
requesting 10.5 Hz gives exactly one frame every 5th 20 ms step, i.e. 10 Hz (measured).
"""
CAMERAS = ['front', 'rear', 'left', 'right']
CAMERA_UPDATE_RATE = 10.5  # -> 10 Hz actual, see above


def image_topic(cam):
    return f'/cam_{cam}/image'


def webots_image_topic(cam):
    """webots_ros2 appends /image_color; the launch file remaps it to image_topic(cam)."""
    return f'/cam_{cam}/image_color'


def vehicle_description(noise=None):
    """noise: optional dict of VehiclePlugin properties (gyroNoise, gyroBias, speedNoise,
    steerNoise, noiseSeed)."""
    devices = ''.join(f"""
    <device reference="cam_{c}" type="Camera">
      <ros>
        <topicName>/cam_{c}</topicName>
        <frameName>cam_{c}</frameName>
        <updateRate>{CAMERA_UPDATE_RATE}</updateRate>
        <alwaysOn>true</alwaysOn>
      </ros>
    </device>""" for c in CAMERAS)
    props = ''.join(f'\n      <{k}>{v}</{k}>' for k, v in (noise or {}).items())
    return f"""<?xml version="1.0" ?>
<robot name="ego_car">
  <webots>{devices}
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
