from setuptools import find_packages, setup


package_name = 'm3pro_arm_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='M3 Pro extension maintainers',
    maintainer_email='devnull@example.com',
    description='Convert raw M3 Pro arm servo feedback to sensor_msgs/JointState.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_state_bridge = m3pro_arm_bridge.arm_state_bridge:main',
        ],
    },
)
