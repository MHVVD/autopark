"""Camera rig of the ego car: the single source of truth for the Webots camera nodes (written by
generate_world.py) and for the camera calibration used by the bird's-eye view. No ROS/Webots.

Frames
  base_link  rear-axle centre, x forward, y left, z up (the Car PROTO origin)
  ground     directly below base_link on the ground plane, z up, same yaw as base_link
  camera     Webots convention: x = optical axis, y left, z up (image u right, v down)

Each camera is mounted in the ToyotaPrius sensorsSlotCenter (whose origin is base_link) as
    Pose { translation x y z  rotation 0 0 1 yaw  children [ Camera { rotation 0 1 0 pitch } ] }
so R_base_cam = Rz(yaw) @ Ry(pitch); positive pitch looks down.
"""
import math
from dataclasses import dataclass

import numpy as np

# Rest attitude of base_link relative to the ground, measured with the ground-truth supervisor
# (identical across scenarios): origin height, roll, pitch. Negative pitch = nose up.
REST_HEIGHT = 0.0247   # m
REST_ROLL = 0.0        # rad
REST_PITCH = -0.0201   # rad


@dataclass(frozen=True)
class CameraSpec:
    """projection 'planar': pinhole, hfov = horizontal field of view (< pi).
    projection 'spherical': Webots fisheye, equidistant: the angle from the optical axis is
    |(u - cx) / W, (v - cy) / H| * hfov (merge_spherical.frag), so use a square image."""
    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    width: int = 640
    height: int = 480
    hfov: float = 2.0      # rad (Webots Camera.fieldOfView)
    projection: str = 'planar'

    @property
    def cx(self):
        return (self.width - 1) / 2   # pixel-index coordinates: pixel i covers [i - 0.5, i + 0.5]

    @property
    def cy(self):
        return (self.height - 1) / 2

    def K(self):
        """Pinhole intrinsics (planar cameras: square pixels, no distortion)."""
        f = (self.width / 2) / math.tan(self.hfov / 2)
        return np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1.0]])

    def T_base_cam(self):
        return _T(_Rz(self.yaw) @ _Ry(self.pitch), (self.x, self.y, self.z))


# Surround-view rig: four 189 deg equidistant fisheyes, 640x640. Front at the bumper, rear at
# the tailgate, left/right under the mirrors. Compared with 115 deg pinholes (blind wedges at the
# car corners, 62 % coverage within 1 m of the body) this sees 100 % of the ground around the car
# at a median 4.0 cm/px at the slot entrances (analysis in the milestone 2 notes).
_FISHEYE = dict(width=640, height=640, hfov=3.3, projection='spherical')
CAMERAS = [
    CameraSpec('front', 3.66, 0.0, 0.65, 0.0, 0.5, **_FISHEYE),
    CameraSpec('rear', -0.90, 0.0, 0.80, math.pi, 0.5, **_FISHEYE),
    CameraSpec('left', 1.90, 0.95, 0.75, math.pi / 2, 0.9, **_FISHEYE),
    CameraSpec('right', 1.90, -0.95, 0.75, -math.pi / 2, 0.9, **_FISHEYE),
]

# Ego footprint in base_link (ToyotaPrius: bumpers at x = -0.85 .. 3.635, width 1.76 m)
FOOTPRINT = (-0.95, 3.73, -0.93, 0.93)  # x_min, x_max, y_min, y_max with a small margin

# Car body as a box in the ground frame, used for camera occlusion: a ground point is only
# visible to a camera if the line of sight does not pass through it. The cameras sit just
# outside it (bumpers, tailgate, below the mirrors).
BODY_BOX = (-0.85, 3.635, -0.88, 0.88, 0.12, 1.49)  # x_min, x_max, y_min, y_max, z_min, z_max


def T_ground_base(height=REST_HEIGHT, roll=REST_ROLL, pitch=REST_PITCH):
    return _T(_Ry(pitch) @ _Rx(roll), (0.0, 0.0, height))


def T_ground_cam(cam):
    return T_ground_base() @ cam.T_base_cam()


def camera_by_name(name):
    return next(c for c in CAMERAS if c.name == name)


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
