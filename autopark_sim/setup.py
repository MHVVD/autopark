from glob import glob

from setuptools import setup

package_name = 'autopark_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.wbt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhvd',
    maintainer_email='mahmudlawal75@gmail.com',
    description='Webots parking-row world, vehicle interface and ground-truth supervisor',
    license='TODO',
    entry_points={'console_scripts': []},
)
