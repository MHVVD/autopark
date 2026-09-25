from glob import glob

from setuptools import setup

package_name = 'autopark'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhvd',
    maintainer_email='mahmudlawal75@gmail.com',
    description='Vision-based autonomous parking stack',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odometry = autopark.odometry_node:main',
            'drive_test = autopark.drive_test:main',
            'odom_eval = autopark.odom_eval:main',
            'calibrate_masks = autopark.calibrate_masks:main',
            'bev = autopark.bev_node:main',
            'bev_eval = autopark.bev_eval:main',
            'collect_slots = autopark.collect_slots:main',
            'collect_poses = autopark.collect_poses:main',
            'train_slots = autopark.train_slots:main',
            'eval_slots = autopark.eval_slots:main',
            'slot_detector = autopark.slot_detector_node:main',
            'slot_tracker = autopark.slot_tracker_node:main',
            'detection_noise = autopark.detection_noise:main',
            'track_eval = autopark.track_eval:main',
            'planner = autopark.planner_node:main',
            'plan_bench = autopark.plan_bench:main',
            'plan_eval = autopark.plan_eval:main',
            'controller = autopark.controller_node:main',
            'parking_manager = autopark.parking_manager_node:main',
            'park_eval = autopark.park_eval:main',
        ],
    },
)
