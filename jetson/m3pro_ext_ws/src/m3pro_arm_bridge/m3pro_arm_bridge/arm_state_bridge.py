#!/usr/bin/env python3

import json
import math
from typing import List

import rclpy
from arm_msgs.msg import ArmJoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from m3pro_arm_msgs.msg import ArmServoState


class ArmStateBridge(Node):
    """Convert MCU servo feedback into a standard JointState stream."""

    JOINT_COUNT = 6

    def __init__(self) -> None:
        super().__init__('arm_state_bridge')

        self.declare_parameter('input_mode', 'degrees')
        self.declare_parameter('degree_feedback_topic', '/arm6_joints_feedback')
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
        self.declare_parameter('position_offset_deg', [0.0] * self.JOINT_COUNT)
        self.declare_parameter('degree_min', [0.0] * self.JOINT_COUNT)
        self.declare_parameter('degree_max', [180.0, 180.0, 180.0, 180.0, 270.0, 180.0])
        self.declare_parameter('velocity_filter_alpha', 0.25)

        self.input_mode = str(self.get_parameter('input_mode').value)
        if self.input_mode not in ('degrees', 'raw'):
            raise ValueError("input_mode must be either 'degrees' or 'raw'.")
        self.joint_names = self._get_vector('joint_names', str)
        self.raw_center = self._get_vector('raw_center', float)
        self.center_position = self._get_vector('position_at_center_deg', float)
        self.degrees_per_count = self._get_vector('degrees_per_count', float)
        self.direction = self._get_vector('direction', float)
        self.position_offset_deg = self._get_vector('position_offset_deg', float)
        self.degree_min = self._get_vector('degree_min', float)
        self.degree_max = self._get_vector('degree_max', float)
        self.calibration_valid = bool(self.get_parameter('calibration_valid').value)
        self.velocity_alpha = float(self.get_parameter('velocity_filter_alpha').value)
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError('velocity_filter_alpha must be in (0, 1].')

        self.previous_position: List[float] | None = None
        self.previous_velocity = [math.nan] * self.JOINT_COUNT
        self.previous_time_ns: int | None = None

        joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)

        self.joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        if self.input_mode == 'degrees':
            feedback_topic = str(self.get_parameter('degree_feedback_topic').value)
            self.feedback_sub = self.create_subscription(
                ArmJoints,
                feedback_topic,
                self._degree_feedback_callback,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f'Publishing vendor-converted servo feedback from {feedback_topic} '
                f'on {joint_state_topic}.'
            )
        elif self.calibration_valid:
            feedback_topic = str(self.get_parameter('raw_feedback_topic').value)
            self.feedback_sub = self.create_subscription(
                ArmServoState,
                feedback_topic,
                self._raw_feedback_callback,
                10,
            )
            self.get_logger().info(f'Publishing calibrated feedback on {joint_state_topic}.')
        else:
            feedback_topic = str(self.get_parameter('raw_feedback_topic').value)
            self.feedback_sub = self.create_subscription(
                ArmServoState,
                feedback_topic,
                self._raw_feedback_callback,
                10,
            )
            self.get_logger().warning(
                'Calibration is marked invalid. Raw feedback status will be reported, '
                'but /joint_states will not be published.'
            )

    def _get_vector(self, name: str, value_type: type) -> list:
        values = list(self.get_parameter(name).value)
        if len(values) != self.JOINT_COUNT:
            raise ValueError(f'{name} must contain exactly {self.JOINT_COUNT} values.')
        return [value_type(value) for value in values]

    def _degree_feedback_callback(self, message: ArmJoints) -> None:
        now = self.get_clock().now()
        angles = [
            int(message.joint1),
            int(message.joint2),
            int(message.joint3),
            int(message.joint4),
            int(message.joint5),
            int(message.joint6),
        ]
        valid = [
            self.degree_min[index] <= angle <= self.degree_max[index]
            for index, angle in enumerate(angles)
        ]
        self._publish_status(
            source='degrees',
            sequence=int(message.time),
            values=angles,
            valid=valid,
            received_ns=now.nanoseconds,
        )

        positions = [
            math.radians(
                self.position_offset_deg[index]
                + self.direction[index] * float(angle)
            )
            if valid[index]
            else math.nan
            for index, angle in enumerate(angles)
        ]
        self._publish_joint_state(positions, now.nanoseconds, now.to_msg())

    def _raw_feedback_callback(self, message: ArmServoState) -> None:
        now = self.get_clock().now()
        valid = [bool(message.valid_mask & (1 << index)) for index in range(self.JOINT_COUNT)]
        self._publish_status(
            source='raw',
            sequence=int(message.seq),
            values=[int(value) for value in message.position_raw],
            valid=valid,
            received_ns=now.nanoseconds,
            sample_time_ms=int(message.sample_time_ms),
        )

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

        self._publish_joint_state(positions, now.nanoseconds, now.to_msg())

    def _publish_joint_state(self, positions: List[float], now_ns: int, stamp) -> None:
        velocities = self._estimate_velocity(positions, now_ns)
        joint_state = JointState()
        joint_state.header.stamp = stamp
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

    def _publish_status(
        self,
        source: str,
        sequence: int,
        values: List[int],
        valid: List[bool],
        received_ns: int,
        sample_time_ms: int | None = None,
    ) -> None:
        payload = {
            'source': source,
            'seq': sequence,
            'received_time_ns': int(received_ns),
            'valid': valid,
            'position': values,
        }
        if sample_time_ms is not None:
            payload['sample_time_ms'] = sample_time_ms
        if source == 'raw':
            payload['calibration_valid'] = self.calibration_valid
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
