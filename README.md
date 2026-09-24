# autopark: vision-based autonomous parking (Webots + ROS 2 Jazzy)

A simulated car drives past a row of parking slots, detects an empty slot with its cameras
and reverses into it. Work in progress, built milestone by milestone.

| Milestone | Status |
|---|---|
| 1. Simulation foundation: world, vehicle interface, ground truth, odometry | done |
| 2. Bird's-eye view (camera calibration, IPM, stitching) | done |
| 3. Slot detector (dataset from ground truth, PyTorch) | done |
| 4. Slot tracker (Kalman filter) | done |
| 5. Planner (Hybrid A* + Reeds-Shepp) | next |
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
| `/cam_{front,rear,left,right}/image` | sensor_msgs/Image | 640x640 equidistant fisheye, 189 deg, 10 Hz, stamped with the exact simulation time of the render |
| `/bev/image` | sensor_msgs/Image | stitched bird's-eye view, 18 x 18 m at 4 cm/px, frame base_footprint |
| `/slots/detections` | autopark_msgs/ParkingSlotArray | detected slots in base_footprint (entrance, heading, width, occupied, confidence) |
| `/slots/debug_image` | sensor_msgs/Image | BEV with detections drawn |
| `/slots/detections_noisy` | autopark_msgs/ParkingSlotArray | detections after optional injected noise (pass-through by default) |
| `/slots/tracked` | autopark_msgs/ParkingSlotArray | confirmed slot tracks in odom: id, entrance, covariance, fused vacancy |
| `/slots/tracked_image` | sensor_msgs/Image | BEV with the tracks drawn |
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

## Slot detector

Marking-point approach: the network (`slot_net.py`, 1.6 M parameters, input 448x448 BEV,
output stride 4) predicts where each painted side line ends at the aisle (heatmap + sub-cell
offset), the direction into the slot, and a vacancy map. `slot_codec.py` pairs neighbouring
points one slot width apart into slots and averages the vacancy over the slot entrance.

```bash
# 1. dataset (simulation running: ros2 launch autopark bringup.launch.py gui:=false mode:=fast detector:=false)
ros2 run autopark collect_slots --ros-args -p seed_start:=10000 -p seed_count:=150 -p out_dir:=$HOME/autopark_data/slots_v1/train
ros2 run autopark collect_slots --ros-args -p seed_start:=10150 -p seed_count:=25 -p out_dir:=$HOME/autopark_data/slots_v1/val
ros2 run autopark collect_slots --ros-args -p seed_start:=0 -p seed_count:=30 -p out_dir:=$HOME/autopark_data/slots_v1/test
# 2. train (CPU: ~11 min/epoch on the i5-8265U) and 3. evaluate on the test scenarios
ros2 run autopark train_slots --data ~/autopark_data/slots_v1 --epochs 12
ros2 run autopark eval_slots --split test
```

Seeds >= 10000 are dataset scenarios with 3-10 empty slots (balanced vacancy labels); the
test split uses the normal 1-3-empty scenarios. The splits never share a scenario.

**Training on Colab (GPU):** upload `autopark/autopark` and `autopark_sim/autopark_sim`
(pure Python, no ROS needed) and the dataset folder, then
`!pip install opencv-python-headless` and
`!PYTHONPATH=. python -m autopark.train_slots --data slots_v1 --out slotnet.pt --workers 2`.
Copy `slotnet.pt` back to `~/autopark_models/`.

## Slot tracker

One Kalman filter per slot on the entrance pose (x, y, heading) in the odom frame
(`slot_tracker.py`). The measurement noise is the detector's error measured against range
(0.5 cm + 0.25 cm/m, 0.3 deg + 0.04 deg/m) plus any injected noise; prediction adds odometry
drift proportional to the distance driven. Association: Mahalanobis gate (chi2, 3 dof, 99.9 %)
+ linear assignment. Tracks are confirmed after 3 hits and only accumulate misses while the
slot is predicted to be inside the detector's view, so slots are remembered when they leave
the view during a manoeuvre. Vacancy is a log-odds filter whose weight falls with range
(far-range vacancy is unreliable, see milestone 3).

`ros2 run autopark track_eval` drives past the rows and back (using ground truth to steer)
and scores detections and tracks against ground truth in the car frame. Detection noise for
experiments: `ros2 launch autopark bringup.launch.py noise_pos:=0.2 noise_yaw_deg:=4
noise_dropout:=0.2 noise_false:=0.3`.

Camera timing: the vehicle plugin publishes the camera images itself, stamped with the
simulation time of the render. webots_ros2's own camera publisher stamps with a ROS clock fed
by /clock from another process; measured per frame, its stamps lagged the render by one step
88 % of the time and by 0 or 2 steps otherwise (2-3 cm of error at 1.2 m/s).

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
