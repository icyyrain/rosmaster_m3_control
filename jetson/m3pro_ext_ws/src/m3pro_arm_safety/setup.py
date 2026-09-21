from setuptools import find_packages, setup


package_name = 'm3pro_arm_safety'

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
    description='Fail-closed arm command arbitration and validation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_command_mux = m3pro_arm_safety.arm_command_mux:main',
        ],
    },
)
