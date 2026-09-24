"""webots_ros2_driver Python plugin for the ego car (ToyotaPrius PROTO, vehicle.Driver).

Subscribes
  /cmd_ackermann   ackermann_msgs/AckermannDriveStamped  speed m/s (+fwd), steering rad (+left)
  /scenario/reset  std_msgs/Int64                         zero the actuator state after a reset
Publishes (every simulation step, stamped with Webots time)
  /vehicle/state   ackermann_msgs/AckermannDriveStamped  measured speed (rear-wheel encoders)
                                                         and steering angle (steering joints)
  /imu             sensor_msgs/Imu                       gyro + accelerometer, frame base_link
  /cam_<name>/image sensor_msgs/Image (bgra8)            every CAMERA_PERIOD_MS, stamped with the
                                                         exact simulation time of the render

The actuators are rate limited (acceleration, steering rate) and the car stops if no command
arrives for CMD_TIMEOUT seconds. Optional sensor noise is set with <plugin> properties
(gyroNoise, gyroBias, speedNoise, steerNoise; defaults 0) and seeded by noiseSeed.
"""
import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Int64

from autopark_sim.descriptions import CAMERA_PERIOD_MS, CAMERAS, image_topic

from autopark_sim.vehicle_model import (CMD_TIMEOUT, MAX_ACCEL, MAX_SPEED, MAX_STEER,
                                        MAX_STEER_RATE, ackermann_center_angle, clamp,
                                        encoder_speed, rate_limit, to_webots)

# Sign of the Webots joint sensors relative to the ROS conventions, verified against ground
# truth with `autopark drive_test` + `odom_eval`: the rear-wheel encoders increase when driving
# forward; the steering-joint sensors are positive for a RIGHT turn.
ENCODER_SIGN = 1.0
STEER_SENSOR_SIGN = -1.0


def _stamp(t):
    sec = int(t)
    return sec, int(round((t - sec) * 1e9))


class VehiclePlugin:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        step_ms = int(self.__robot.getBasicTimeStep())
        self.__dt = step_ms / 1000.0

        def dev(name):
            d = self.__robot.getDevice(name)
            d.enable(step_ms)
            return d

        self.__enc_l, self.__enc_r = dev('left_rear_sensor'), dev('right_rear_sensor')
        self.__steer_l, self.__steer_r = dev('left_steer_sensor'), dev('right_steer_sensor')
        self.__gyro, self.__accel = dev('gyro'), dev('accelerometer')
        self.__cams = []
        for name in CAMERAS:
            cam = self.__robot.getDevice(f'cam_{name}')
            cam.enable(CAMERA_PERIOD_MS)
            self.__cams.append((name, cam))
        # sensors are sampled every period counted from the moment they were enabled
        self.__cam_t0_ms = int(round(self.__robot.getTime() * 1000))

        self.__gyro_noise = float(properties.get('gyroNoise', 0.0))
        self.__gyro_bias = float(properties.get('gyroBias', 0.0))
        self.__speed_noise = float(properties.get('speedNoise', 0.0))
        self.__steer_noise = float(properties.get('steerNoise', 0.0))
        self.__rng = np.random.default_rng(int(properties.get('noiseSeed', 0)))

        self.__cmd_speed = 0.0
        self.__cmd_steer = 0.0
        self.__max_accel = MAX_ACCEL
        self.__max_steer_rate = MAX_STEER_RATE
        self.__last_cmd = -1e9
        self.__speed = 0.0     # rate-limited actuator state
        self.__steer = 0.0
        self.__prev_enc = None

        if not rclpy.ok():
            rclpy.init(args=None)
        self.__node = rclpy.create_node('vehicle')
        self.__node.create_subscription(AckermannDriveStamped, '/cmd_ackermann', self.__on_cmd, 1)
        self.__node.create_subscription(Int64, '/scenario/reset', self.__on_reset, 10)
        self.__state_pub = self.__node.create_publisher(AckermannDriveStamped, '/vehicle/state', 10)
        self.__imu_pub = self.__node.create_publisher(Imu, '/imu', 10)
        self.__cam_pubs = {name: self.__node.create_publisher(Image, image_topic(name), 2)
                           for name, _ in self.__cams}

    def __on_cmd(self, msg):
        d = msg.drive
        self.__cmd_speed = clamp(d.speed, -MAX_SPEED, MAX_SPEED)
        self.__cmd_steer = clamp(d.steering_angle, -MAX_STEER, MAX_STEER)
        self.__max_accel = d.acceleration if d.acceleration > 0 else MAX_ACCEL
        self.__max_steer_rate = (d.steering_angle_velocity if d.steering_angle_velocity > 0
                                 else MAX_STEER_RATE)
        self.__last_cmd = self.__robot.getTime()

    def __on_reset(self, _msg):
        self.__cmd_speed = self.__speed = 0.0
        self.__cmd_steer = self.__steer = 0.0
        self.__prev_enc = None

    def __publish_cameras(self, t):
        sec, nsec = _stamp(t)
        for name, cam in self.__cams:
            data = cam.getImage()
            if not data:
                continue
            msg = Image()
            msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nsec
            msg.header.frame_id = f'cam_{name}'
            msg.width, msg.height = cam.getWidth(), cam.getHeight()
            msg.encoding = 'bgra8'
            msg.step = 4 * msg.width
            msg.data = data
            self.__cam_pubs[name].publish(msg)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        t = self.__robot.getTime()
        dt_ms = int(round(t * 1000)) - self.__cam_t0_ms
        if dt_ms > 0 and dt_ms % CAMERA_PERIOD_MS == 0:
            self.__publish_cameras(t)

        # Actuators
        target = self.__cmd_speed if t - self.__last_cmd <= CMD_TIMEOUT else 0.0
        self.__speed = rate_limit(self.__speed, target, self.__max_accel, self.__dt)
        self.__steer = rate_limit(self.__steer, self.__cmd_steer, self.__max_steer_rate, self.__dt)
        kmh, wb_steer = to_webots(self.__speed, self.__steer)
        self.__robot.setCruisingSpeed(kmh)
        self.__robot.setSteeringAngle(wb_steer)

        # Sensors (NaN until the first sample is available)
        enc = (self.__enc_l.getValue(), self.__enc_r.getValue())
        if any(math.isnan(v) for v in enc):
            return
        if self.__prev_enc is None:
            self.__prev_enc = enc
            return
        speed = ENCODER_SIGN * encoder_speed(enc[0] - self.__prev_enc[0],
                                             enc[1] - self.__prev_enc[1], self.__dt)
        self.__prev_enc = enc
        steer = ackermann_center_angle(STEER_SENSOR_SIGN * self.__steer_l.getValue(),
                                       STEER_SENSOR_SIGN * self.__steer_r.getValue())
        if self.__speed_noise > 0:
            speed += self.__rng.normal(0.0, self.__speed_noise)
        if self.__steer_noise > 0:
            steer += self.__rng.normal(0.0, self.__steer_noise)

        sec, nsec = _stamp(t)
        state = AckermannDriveStamped()
        state.header.stamp.sec, state.header.stamp.nanosec = sec, nsec
        state.header.frame_id = 'base_link'
        state.drive.speed = float(speed)
        state.drive.steering_angle = float(steer)
        self.__state_pub.publish(state)

        gx, gy, gz = self.__gyro.getValues()
        if self.__gyro_noise > 0 or self.__gyro_bias != 0:
            gz += self.__gyro_bias + self.__rng.normal(0.0, self.__gyro_noise)
        ax, ay, az = self.__accel.getValues()
        imu = Imu()
        imu.header = state.header
        imu.orientation_covariance[0] = -1.0  # no orientation estimate
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gx, gy, gz
        imu.linear_acceleration.x, imu.linear_acceleration.y = ax, ay
        imu.linear_acceleration.z = az
        self.__imu_pub.publish(imu)
