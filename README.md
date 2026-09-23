# autopark: vision-based autonomous parking (Webots + ROS 2 Jazzy)

A simulated car drives past a row of parking slots, detects an empty slot with its cameras
and reverses into it. Work in progress, built milestone by milestone.

| Milestone | Status |
|---|---|
| 1. Simulation foundation: world, vehicle interface, ground truth, odometry | done |
| 2. Bird's-eye view (camera calibration, IPM, stitching) | next |
| 3. Slot detector (dataset from ground truth, PyTorch) | |
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
| `/cam_{front,rear,left,right}/image` | sensor_msgs/Image | 320x240, 10 Hz |
| `/odom` + TF odom->base_link | nav_msgs/Odometry | Ackermann dead reckoning, heading from IMU (or steering) |
| `/ground_truth/pose`, `/ground_truth/slots` | Odometry, ParkingSlotArray | evaluation/labels only, never used by the stack |
| `/ground_truth/reset` | autopark_msgs/srv/ResetScenario | deterministic scenario from a seed |

Conventions: `map` frame x along the aisle, y across; `base_link` at the rear-axle centre,
x forward, y left. All sensor messages are stamped with simulation time.

## Run

```bash
cd ~/p_WS && colcon build --symlink-install && source install/setup.bash
ros2 launch autopark bringup.launch.py            # add gui:=false for no 3D view, seed:=N
ros2 run autopark odom_eval --ros-args -p duration:=28.0   # terminal 2
ros2 run autopark drive_test                               # terminal 3
ros2 service call /ground_truth/reset autopark_msgs/srv/ResetScenario "{seed: 5}"
```

Tests: `python3 -m pytest -q autopark_sim/test autopark/test` (with ROS sourced).

After changing `lot.py`, `scenario.py` or the cameras, regenerate the world:
`python3 -m autopark_sim.generate_world autopark_sim/worlds/parking_row.wbt` (run in `autopark_sim/`).
