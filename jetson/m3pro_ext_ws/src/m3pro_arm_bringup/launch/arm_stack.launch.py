from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    share_directory = get_package_share_directory('m3pro_arm_bringup')
    bridge_config = os.path.join(share_directory, 'config', 'arm_bridge.yaml')
    safety_config = os.path.join(share_directory, 'config', 'arm_safety.yaml')

    return LaunchDescription([
        Node(
            package='m3pro_arm_bridge',
            executable='arm_state_bridge',
            name='arm_state_bridge',
            output='screen',
            parameters=[bridge_config],
        ),
        Node(
            package='m3pro_arm_safety',
            executable='arm_command_mux',
            name='arm_command_mux',
            output='screen',
            parameters=[safety_config],
        ),
    ])
