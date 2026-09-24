# autopark: vision-based autonomous parking (Webots + ROS 2 Jazzy)

A simulated car drives past a row of parking slots, detects an empty slot with its cameras
and reverses into it. Work in progress, built milestone by milestone.

| Milestone | Status |
|---|---|
| 1. Simulation foundation: world, vehicle interface, ground truth, odometry | done |
| 2. Bird's-eye view (camera calibration, IPM, stitching) | done |
| 3. Slot detector (dataset from ground truth, PyTorch) | next |
| 4. Slot tracker (Kalman filter) | |
| 5. Planner (Hybrid A* + Reeds-Shepp) | |
| 6. Controller (Stanley fwd/rev) + parking manager | |
| 7. Experiments, visualisation, write-up | |

## Packages

- `autopark_msgs`: `ParkingSlot`, `ParkingSlotArray`, `ResetScenario`.
- `autopark_sim`: Webots world (generated from `lot.py`), vehicle plugin, ground-truth
  supervisor. Pure geometry/scenario/vehicle maths lives in modules without ROS imports.
- `autopark`: the autonomy stack (currently odometry) and evaluation tools.

## Interfaces

| Topic / service | Type | Notes |
|---|---|---|
| `/cmd_ackermann` | ackermann_msgs/AckermannDriveStamped | speed m/s (+fwd), steering rad (+left); rate limited, 0.5 s timeout |
| `/vehicle/state` | ackermann_msgs/AckermannDriveStamped | speed from rear-wheel encoders, steering from steering joints, 50 Hz |
| `/imu` | sensor_msgs/Imu | gyro + accelerometer, 50 Hz |
| `/cam_{front,rear,left,right}/image` | sensor_msgs/Image | 640x640 equidistant fisheye, 189 deg, 10 Hz |
| `/bev/image` | sensor_msgs/Image | stitched bird's-eye view, 18 x 18 m at 4 cm/px, frame base_footprint |
| `/odom` + TF odom->base_link | nav_msgs/Odometry | Ackermann dead reckoning, heading from IMU (or steering) |
| `/ground_truth/pose`, `/ground_truth/slots` | Odometry, ParkingSlotArray | evaluation/labels only, never used by the stack |
| `/ground_truth/reset` | autopark_msgs/srv/ResetScenario | deterministic scenario from a seed |

Conventions: `map` frame x along the aisle, y across; `base_link` at the rear-axle centre,
x forward, y left. All sensor messages are stamped with simulation time.

## Bird's-eye view

- Camera rig (poses, intrinsics, projection) is defined once in `autopark_sim/rig.py`; the
  world generator writes the Webots cameras from it and `autopark/bev.py` uses it for the
  inverse perspective mapping, so calibration is exact by construction.
- The fisheye model matches Webots' spherical projection (equidistant: angle from the
  optical axis = normalised image radius x field of view).
- A ground point is taken from a camera only if the line of sight clears the own car
  (`BODY_BOX`) and misses the camera's body mask (`autopark_sim/config/masks`, computed with
  `ros2 launch autopark_sim sim.launch.py world:=calibration` + `calibrate_masks` +
  `drive_test`). Overlapping cameras are blended by viewing elevation.
- BEV pixel (row r, col c): x = x_max - (r + 0.5) res, y = y_max - (c + 0.5) res
  (top = forward, left = vehicle left), see `BevGrid`.
- `ros2 run autopark bev_eval` measures how far the painted lines in the BEV are from
  their ground-truth positions (use `seed:=-1`, an empty lot).

## Run

```bash
cd ~/p_WS && colcon build --symlink-install && source install/setup.bash
ros2 launch autopark bringup.launch.py            # add gui:=false for no 3D view, seed:=N
ros2 run autopark odom_eval --ros-args -p duration:=28.0   # terminal 2
ros2 run autopark drive_test                               # terminal 3
ros2 service call /ground_truth/reset autopark_msgs/srv/ResetScenario "{seed: 5}"
ros2 run rqt_image_view rqt_image_view /bev/image      # view the bird's-eye view
```

Tests: `python3 -m pytest -q autopark_sim/test autopark/test` (with ROS sourced).

After changing `lot.py`, `scenario.py` or the cameras, regenerate the world:
`python3 -m autopark_sim.generate_world worlds` (run in `autopark_sim/`).
