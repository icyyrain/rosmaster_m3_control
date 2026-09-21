import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'm3pro_arm_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='M3 Pro extension maintainers',
    maintainer_email='devnull@example.com',
    description='Launch and configuration files for the M3 Pro arm extension.',
    license='Apache-2.0',
)
