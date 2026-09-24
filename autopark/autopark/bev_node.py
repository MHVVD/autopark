"""BEV node: 4 fisheye images -> stitched bird's-eye view on /bev/image.

The BEV is in the `base_footprint` frame (ground plane below base_link, same yaw); see
autopark.bev for the pixel <-> metre convention. Parameters x_min, x_max, y_min, y_max, res
define the grid (defaults: BevGrid()); mask_dir the camera body masks.
"""
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from autopark.bev import BevGrid, BevStitcher, load_body_masks
from autopark_sim.descriptions import image_topic
from autopark_sim.rig import CAMERAS


class BevNode(Node):
    def __init__(self):
        super().__init__('bev')
        d = BevGrid()
        for k in ('x_min', 'x_max', 'y_min', 'y_max', 'res'):
            self.declare_parameter(k, getattr(d, k))
        self.declare_parameter('mask_dir', os.path.join(
            get_package_share_directory('autopark_sim'), 'config', 'masks'))
        grid = BevGrid(**{k: self.get_parameter(k).value
                          for k in ('x_min', 'x_max', 'y_min', 'y_max', 'res')})

        t0 = time.monotonic()
        self.stitcher = BevStitcher(grid, masks=load_body_masks(self.get_parameter('mask_dir').value))
        self.get_logger().info(f'BEV {grid.shape[1]}x{grid.shape[0]} px at {grid.res} m/px, '
                               f'tables built in {time.monotonic() - t0:.1f} s')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/bev/image', 2)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        subs = [Subscriber(self, Image, image_topic(c.name), qos_profile=qos) for c in CAMERAS]
        # All cameras are rendered in the same simulation step; stamps can differ by ~1 step.
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=4, slop=0.03)
        self.sync.registerCallback(self.on_images)
        self.n, self.t_proc = 0, 0.0

    def on_images(self, *msgs):
        t0 = time.monotonic()
        images = {c.name: self.bridge.imgmsg_to_cv2(m, 'bgr8') for c, m in zip(CAMERAS, msgs)}
        bev = self.stitcher.stitch(images)
        out = self.bridge.cv2_to_imgmsg(bev, 'bgr8')
        out.header.stamp = msgs[0].header.stamp
        out.header.frame_id = 'base_footprint'
        self.pub.publish(out)
        self.n += 1
        self.t_proc += time.monotonic() - t0
        if self.n % 100 == 0:
            self.get_logger().info(f'{self.n} BEV frames, mean processing {1000 * self.t_proc / self.n:.1f} ms')


def main():
    rclpy.init()
    node = BevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
