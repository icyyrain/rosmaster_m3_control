#!/usr/bin/env python3

import json
import math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from m3pro_arm_msgs.msg import ArmServoState


class ArmStateBridge(Node):
    """Convert compact MCU feedback into a standard JointState stream."""

    JOINT_COUNT = 6

    def __init__(self) -> None:
        super().__init__('arm_state_bridge')

        self.declare_parameter('raw_feedback_topic', '/arm/servo_states_raw')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('status_topic', '/arm/feedback_status')
        self.declare_parameter(
            'joint_names',
            [f'arm_joint_{index}' for index in range(1, self.JOINT_COUNT + 1)],
        )
        self.declare_parameter('calibration_valid', False)
        self.declare_parameter('raw_center', [2000.0, 2000.0, 2000.0, 2000.0, 1486.0, 3100.0])
        self.declare_parameter('position_at_center_deg', [90.0] * self.JOINT_COUNT)
        self.declare_parameter('degrees_per_count', [270.0 / (4000.0 - 96.0)] * self.JOINT_COUNT)
        self.declare_parameter('direction', [1.0] * self.JOINT_COUNT)
        self.declare_parameter('velocity_filter_alpha', 0.25)

        self.joint_names = self._get_vector('joint_names', str)
        self.raw_center = self._get_vector('raw_center', float)
        self.center_position = self._get_vector('position_at_center_deg', float)
        self.degrees_per_count = self._get_vector('degrees_per_count', float)
        self.direction = self._get_vector('direction', float)
        self.calibration_valid = bool(self.get_parameter('calibration_valid').value)
        self.velocity_alpha = float(self.get_parameter('velocity_filter_alpha').value)
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError('velocity_filter_alpha must be in (0, 1].')

        self.previous_position: List[float] | None = None
        self.previous_velocity = [math.nan] * self.JOINT_COUNT
        self.previous_time_ns: int | None = None

        feedback_topic = str(self.get_parameter('raw_feedback_topic').value)
        joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)

        self.joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.feedback_sub = self.create_subscription(
            ArmServoState,
            feedback_topic,
            self._feedback_callback,
            10,
        )

        if self.calibration_valid:
            self.get_logger().info(f'Publishing calibrated feedback on {joint_state_topic}.')
        else:
            self.get_logger().warning(
                'Calibration is marked invalid. Raw feedback status will be reported, '
                'but /joint_states will not be published.'
            )

    def _get_vector(self, name: str, value_type: type) -> list:
        values = list(self.get_parameter(name).value)
        if len(values) != self.JOINT_COUNT:
            raise ValueError(f'{name} must contain exactly {self.JOINT_COUNT} values.')
        return [value_type(value) for value in values]

    def _feedback_callback(self, message: ArmServoState) -> None:
        now = self.get_clock().now()
        valid = [bool(message.valid_mask & (1 << index)) for index in range(self.JOINT_COUNT)]
        self._publish_status(message, valid, now.nanoseconds)

        if not self.calibration_valid:
            return

        positions = []
        for index, raw_value in enumerate(message.position_raw):
            if not valid[index]:
                positions.append(math.nan)
                continue
            position_deg = (
                self.center_position[index]
                + self.direction[index]
                * (float(raw_value) - self.raw_center[index])
                * self.degrees_per_count[index]
            )
            positions.append(math.radians(position_deg))

        velocities = self._estimate_velocity(positions, now.nanoseconds)
        joint_state = JointState()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = self.joint_names
        joint_state.position = positions
        joint_state.velocity = velocities
        self.joint_state_pub.publish(joint_state)

    def _estimate_velocity(self, positions: List[float], now_ns: int) -> List[float]:
        if self.previous_position is None or self.previous_time_ns is None:
            filtered = [math.nan] * self.JOINT_COUNT
        else:
            delta_seconds = (now_ns - self.previous_time_ns) / 1_000_000_000.0
            filtered = []
            for index, position in enumerate(positions):
                previous = self.previous_position[index]
                if delta_seconds <= 0.0 or not math.isfinite(position) or not math.isfinite(previous):
                    filtered.append(math.nan)
                    continue
                instantaneous = (position - previous) / delta_seconds
                old_velocity = self.previous_velocity[index]
                if math.isfinite(old_velocity):
                    instantaneous = (
                        self.velocity_alpha * instantaneous
                        + (1.0 - self.velocity_alpha) * old_velocity
                    )
                filtered.append(instantaneous)

        self.previous_position = list(positions)
        self.previous_velocity = list(filtered)
        self.previous_time_ns = now_ns
        return filtered

    def _publish_status(self, message: ArmServoState, valid: List[bool], received_ns: int) -> None:
        payload = {
            'seq': int(message.seq),
            'sample_time_ms': int(message.sample_time_ms),
            'received_time_ns': int(received_ns),
            'valid_mask': int(message.valid_mask),
            'valid': valid,
            'raw_position': [int(value) for value in message.position_raw],
            'calibration_valid': self.calibration_valid,
        }
        status = String()
        status.data = json.dumps(payload, separators=(',', ':'))
        self.status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmStateBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
