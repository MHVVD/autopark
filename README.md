# autopark: vision-based autonomous parking (Webots + ROS 2 Jazzy)

A simulated car drives past a row of parking slots, detects an empty slot with its cameras
and reverses into it. **Technical write-up with the results: [docs/writeup.md](docs/writeup.md).**

| Milestone | Status |
|---|---|
| 1. Simulation foundation: world, vehicle interface, ground truth, odometry | done |
| 2. Bird's-eye view (camera calibration, IPM, stitching) | done |
| 3. Slot detector (dataset from ground truth, PyTorch) | done |
| 4. Slot tracker (Kalman filter) | done |
| 5. Planner (Hybrid A* + Reeds-Shepp) | done |
| 6. Controller (Stanley fwd/rev) + parking manager | done |
| 7. Experiments, visualisation, demo video, write-up | done |

## Packages

- `autopark_msgs`: `ParkingSlot`, `ParkingSlotArray`, `ParkingPath`, `ControlStatus`, `ParkingStatus`,
  `ResetScenario`, `PlanParking`.
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
| `/parking/plan` | autopark_msgs/srv/PlanParking | plan into a tracked slot (`slot_id`, -1 = nearest selectable vacant) |
| `/parking/path` | autopark_msgs/ParkingPath | latest plan (odom): rear-axle poses every 0.1 m, gear and curvature per sample |
| `/parking/path_viz`, `/parking/path_image` | nav_msgs/Path, sensor_msgs/Image | the plan for RViz, and drawn on the BEV |
| `/parking/path_exec` | autopark_msgs/ParkingPath | what the controller follows (sent by the parking manager) |
| `/parking/control_status`, `/parking/status` | ControlStatus, ParkingStatus | controller and parking manager state |
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

**Views while parking (slots_v2, `slotnet_v2.pt`, the default).** `collect_slots` only drives
along the aisle, so the first model (`slotnet.pt`) found no slot once the car turned more than
45 deg (0 % recall). `collect_poses` teleports the car (`/ground_truth/set_pose`, keeping the
settled suspension so every image is at rest) to poses along ground-truth parking paths,
anywhere in the aisle at any heading, and inside empty slots, and saves the BEV:

```bash
# simulation running: ros2 launch autopark bringup.launch.py gui:=false mode:=fast detector:=false tracker:=false planner:=false park:=false
ros2 run autopark collect_poses --ros-args -p seed_start:=10200 -p seed_count:=200 -p out_dir:=$HOME/autopark_data/slots_v2/train
ros2 run autopark collect_poses --ros-args -p seed_start:=10400 -p seed_count:=30 -p out_dir:=$HOME/autopark_data/slots_v2/val
ros2 run autopark collect_poses --ros-args -p seed_start:=30 -p seed_count:=40 -p poses_per_seed:=15 -p out_dir:=$HOME/autopark_data/slots_v2/test
# fine-tune from slotnet.pt on every 2nd slots_v1 image + slots_v2 (8 epochs, ~13.5 min each on CPU)
ros2 run autopark train_slots --data ~/autopark_data/slots_v1:2,~/autopark_data/slots_v2 --init ~/autopark_models/slotnet.pt --out ~/autopark_models/slotnet_v2.pt --epochs 8 --lr 1e-3
ros2 run autopark eval_slots --data ~/autopark_data/slots_v2 --split test
```

Slot recall / heading error p90 on the slots_v2 test set (600 images, scenarios 30-69), by
the car's heading relative to the aisle:

| heading vs aisle | images | slotnet.pt | slotnet_v2.pt |
|---|---|---|---|
| 0-20 deg | 109 | 99.9 % / 0.76 deg | 100.0 % / 0.61 deg |
| 20-45 deg | 102 | 70.2 % / 5.74 deg | 99.8 % / 0.90 deg |
| 45-90 deg (turned, in a slot) | 389 | 0.0 % | 99.7 % / 1.27 deg |

On the aisle test set (slots_v1) slotnet_v2 is as good as before (100 % recall and precision,
entrance error 1.0 cm median, heading 0.23 deg median). The slot heading blends the painted
lines' direction with the normal of the 2.6 m entrance segment, weighted towards the normal
near the car (`slot_codec.blend_heading`): near the car the side lines are short stubs,
partly under the car body. This halved the heading error within 5 m and was what made the
closed loop work (see below). Per-frame "vacant" calls of turned views are less precise (79 %
at the 0.5 threshold), but the wrong ones are far away: 1 of 459 within 4 m, and 4 of 763
above 0.95 (the tracker needs a fused vacancy > 0.95 and discounts far-range vacancy).

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

## Planner

Hybrid A* (`hybrid_astar.py`) with Reeds-Shepp analytic expansion (`reeds_shepp.py`: the
CSC, CCC, CCCC, CCSC and CCSCC families with their symmetries, 44 candidate formulas as in
OMPL; verified: every candidate ends exactly on the goal, distances are symmetric and satisfy
the triangle inequality on random poses).

- Planning problem (`parking_goal.py`), built only from perception: the goal is the rear-axle
  pose that centres the car in the tracked slot, facing out (reverse-in). Every tracked slot
  that is not selectable-vacant (vacancy > 0.95 and >= 3 close observations) is a keep-out
  box, grown 0.25 m towards the aisle and 0.05 m sideways to bound a car parked in it (worst
  case in 3000 simulated scenarios: 0.20 m and 0.03 m); the band behind each row is keep-out.
  Only explored space is drivable: the planner node records the odometry pose every metre while
  the car heads along the aisle, and the explored area is the union of a car-frame box
  (-3.5..5.5 m along, +-9 m across) at those poses. The box is chosen so that every slot
  reaching into it had its entrance in the tracker's view (checked exhaustively in the tests),
  so an unseen occupied slot can never look free. Obstacles that are not parked cars in slots
  (pedestrians, pillars) are not modelled: a real system would add a free-space map.
- Search: 7 steering angles (|steer| <= 0.55 rad of the 0.6 rad actuator limit, min turning
  radius 4.57 m), forward and reverse arcs of 0.6 m, 0.25 m / 5 deg pruning grid. Cost:
  length (reverse x1.5) + 3 m per gear change + steering penalties. Heuristic: max(obstacle
  aware 2D distance, obstacle-free Reeds-Shepp distance), weight 2. Pieces between gear
  changes are at least 0.5 m.
- Collision check: points every 5 cm on the car outline need clearance >= 10 cm margin in a
  2.5 cm distance field that is made conservative for its resolution; tests check it against
  exact polygon geometry.

`ros2 run autopark plan_bench` benchmarks the planner offline on ground-truth scenarios and
checks every path with exact geometry against the true parked-car rectangles.
`ros2 run autopark plan_eval` does the same on the live stack (perception + planner).

Results (milestone 5):
- Offline, 50 scenarios, 214 problems (start 3 m before to 8 m past the slot): 214 solved,
  all collision-free against the true cars (min clearance 0.33 m), kinematically feasible
  and ending inside the slot; planning time median 0.14 s, p90 0.97 s, max 2.5 s (laptop CPU);
  0-3 gear changes.
- Slot pose errors given to the planner (independent per slot): 5 cm / 0.5 deg: 104 / 107
  solved, all safe; 15 cm / 2 deg: 56 solved; 30 cm / 6 deg (a single raw detection at the
  highest noise level): 35 solved, one touching a car, only 12 ending inside the slot.
- Live (8 scenarios, full perception stack, planning at two stops): every request for a slot
  the car had passed was solved (36 / 36), all collision-free against the true cars (min
  0.39 m) and ending inside the true slot (car within 4.6 cm lateral, 2.8 cm depth, 0.5 deg
  of the slot centre). Slots still ahead of the car are refused (outside the explored area);
  planning time median 2.0 s, max 7.4 s with the simulation and perception on the same CPU.

## Controller and parking manager

`stanley.py`: path tracking per segment of constant direction. Forward: Stanley at the front
axle (lateral error measured across the front axle's direction of travel, yaw + steering).
Reverse: the same heading + cross-track structure at the rear axle with gains per metre
(error dynamics e'' + 2 e' + e = 0 per metre; Stanley's speed-scaled gains at the rear axle
settle over ~5 m when reversing, too slow for a parking slot). Curvature feedforward. The
speed profile brakes to creep speed before every cusp and every steering jump in the path
(planned paths switch between full-lock arcs and the steering actuator needs ~1.4 s for
that), and slows to a stop while the steering lags its command. Each segment starts with the
wheels turned at standstill. In a kinematic model with the actuator limits and 60 ms delay
(`kinematic_sim.py`), on 31 planned manoeuvres: max tracking error 4.7 cm, final error < 1 cm.

`parking_manager_node.py`: search (follows the aisle centre estimated from the tracked slots)
-> stop once a selectable-vacant slot has been passed by 3 m -> plan -> execute -> done / failed.
`replan:=closed` sends the path up to the first cusp, replans at each cusp (standing) with the
latest slot estimate, and on the final reverse moves the remaining path rigidly with the goal
when the tracked slot moves (1 cm .. 25 cm). `replan:=open` plans once and executes the whole
path. `ros2 run autopark park_eval` scores autonomous runs against ground truth.

Results (milestone 6, 8 scenarios per mode, full stack in Webots, scored against ground truth):

| | plan once (`open`) | closed loop (`closed`) |
|---|---|---|
| parked, no contact | 8 / 8 | 8 / 8 |
| min clearance to parked cars | 0.50 m | 0.51 m |
| final lateral error (median / max) | 1.1 / 2.7 cm | 1.1 / 3.2 cm |
| final depth error (median / max) | 2.4 / 3.0 cm | 1.4 / 1.7 cm |
| final heading error (median / max) | 0.35 / 0.75 deg | 0.58 / 1.37 deg |
| time from start of search (median) | 45 s | 47 s |

The first detector only worked on views from the aisle, so in the runs above the tracker used
detections only within 20 deg of the trained view angle, and during the final reverse (car at
~90 deg) there was no new information: closed loop ~ plan once. With `slotnet_v2.pt` (trained
on turned, mid-manoeuvre and in-slot views, see *Slot detector*) the tracker uses every view
(`view_yaw_tol:=90`, the default). Same 8 scenarios:

| slotnet_v2, blended heading | plan once (`open`) | closed loop (`closed`) |
|---|---|---|
| parked, no contact | 8 / 8 | 8 / 8 |
| min clearance to parked cars | 0.49 m | 0.50 m |
| final lateral error (median / max) | 1.4 / 4.4 cm | 1.4 / 2.5 cm |
| final depth error (median / max) | 1.5 / 2.9 cm | 0.2 / 0.5 cm |
| final heading error (median / max) | 0.72 / 1.04 deg | 0.31 / 0.97 deg |
| time from start of search (median) | 43 s | 47 s |

Closed loop now corrects the final reverse with what the cameras see from inside the slot
(34-46 corrections per run). Before the heading blend (line direction only) it was worse than
plan once: lateral median 4.3 / max 12.2 cm. A trace of the goal the controller steered to
showed its lateral error swinging from +9 to -10 cm, in step with the heading estimate (4 m
from the entrance to the goal: 1 deg ~ 7 cm); per-frame positions, stamps and odometry were
all accurate. Old configuration: `model:=$HOME/autopark_models/slotnet.pt view_yaw_tol:=20`.

## Visualizer, demo video, experiment

`ros2 launch autopark bringup.launch.py` also starts the visualizer (`viz:=false` to skip):
`/viz/image` shows the bird's-eye view with this frame's detections, the tracked slots, the
path and the goal, next to a map of the manoeuvre and the manager / controller state.
`record:=<dir>` saves its frames plus a 1280x720 view of the car from the supervisor's demo
camera (renders off-screen, works with `gui:=false`); `ros2 run autopark make_demo_video`
turns recordings, title cards and plots into an mp4.

Detection-noise experiment (plan once vs closed loop; `noise_mode:=white|field`, see
`detection_noise.py`): run `park_eval` per condition into one directory, then
`ros2 run autopark noise_report --dir <dir>` writes the table, paired statistics and plots.
Results and discussion: [docs/writeup.md](docs/writeup.md), section 4.

![Parking accuracy vs detection noise](docs/figures/corner_error.png)

## Run

```bash
cd ~/p_WS && colcon build --symlink-install && source install/setup.bash
ros2 launch autopark bringup.launch.py seed:=5     # the car finds a vacant slot and parks (gui:=false: no 3D view)
ros2 topic echo /parking/status                    # what the parking manager is doing
# the evaluation tools that drive the car need park:=false:
ros2 launch autopark bringup.launch.py park:=false
ros2 run autopark odom_eval --ros-args -p duration:=28.0   # terminal 2
ros2 run autopark drive_test                               # terminal 3
ros2 service call /ground_truth/reset autopark_msgs/srv/ResetScenario "{seed: 5}"
ros2 run rqt_image_view rqt_image_view /bev/image      # view the bird's-eye view
ros2 service call /parking/plan autopark_msgs/srv/PlanParking "{slot_id: -1}"   # plan into the nearest vacant slot
ros2 run rqt_image_view rqt_image_view /parking/path_image                      # view the plan
ros2 run rqt_image_view rqt_image_view /viz/image                               # everything in one view
```

Tests: `python3 -m pytest -q autopark_sim/test autopark/test` (with ROS sourced).

After changing `lot.py`, `scenario.py` or the cameras, regenerate the world:
`python3 -m autopark_sim.generate_world worlds` (run in `autopark_sim/`).
