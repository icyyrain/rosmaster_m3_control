#!/usr/bin/env python3

import json
import math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from arm_msgs.msg import ArmJoints


class ArmCommandMux(Node):
    """Fail-closed command arbiter for teleoperation, tests, and RL."""

    JOINT_COUNT = 6
    SOURCES = ('teleop', 'test', 'rl')

    def __init__(self) -> None:
        super().__init__('arm_command_mux')

        self.declare_parameter('output_topic', '/arm6_joints')
        self.declare_parameter('feedback_topic', '/joint_states')
        self.declare_parameter('mode_topic', '/arm/control_mode')
        self.declare_parameter('status_topic', '/arm/safety_status')
        self.declare_parameter('teleop_topic', '/arm/command/teleop')
        self.declare_parameter('test_topic', '/arm/command/test')
        self.declare_parameter('rl_topic', '/arm/command/rl')
        self.declare_parameter('require_feedback', True)
        self.declare_parameter('feedback_timeout_sec', 0.25)
        self.declare_parameter('command_timeout_sec', 0.5)
        self.declare_parameter('min_command_period_sec', 0.05)
        self.declare_parameter('min_motion_time_ms', 100)
        self.declare_parameter('max_motion_time_ms', 2000)
        self.declare_parameter('joint_min_deg', [0.0] * self.JOINT_COUNT)
        self.declare_parameter('joint_max_deg', [180.0] * self.JOINT_COUNT)
        self.declare_parameter('max_step_deg', [2.0] * self.JOINT_COUNT)
        self.declare_parameter('max_tracking_error_deg', [15.0] * self.JOINT_COUNT)

        self.require_feedback = bool(self.get_parameter('require_feedback').value)
        self.feedback_timeout_ns = self._seconds_to_ns('feedback_timeout_sec')
        self.command_timeout_ns = self._seconds_to_ns('command_timeout_sec')
        self.min_command_period_ns = self._seconds_to_ns('min_command_period_sec')
        self.min_motion_time_ms = int(self.get_parameter('min_motion_time_ms').value)
        self.max_motion_time_ms = int(self.get_parameter('max_motion_time_ms').value)
        self.joint_min = self._get_float_vector('joint_min_deg')
        self.joint_max = self._get_float_vector('joint_max_deg')
        self.max_step = self._get_float_vector('max_step_deg')
        self.max_tracking_error = self._get_float_vector('max_tracking_error_deg')

        self.active_mode = 'disabled'
        self.feedback_deg: List[float] | None = None
        self.feedback_time_ns: int | None = None
        self.last_command: List[float] | None = None
        self.last_command_time_ns: int | None = None

        self.output_pub = self.create_publisher(
            ArmJoints,
            str(self.get_parameter('output_topic').value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            10,
        )
        self.mode_sub = self.create_subscription(
            String,
            str(self.get_parameter('mode_topic').value),
            self._mode_callback,
            10,
        )
        self.feedback_sub = self.create_subscription(
            JointState,
            str(self.get_parameter('feedback_topic').value),
            self._feedback_callback,
            10,
        )

        self.command_subs = []
        for source in self.SOURCES:
            topic = str(self.get_parameter(f'{source}_topic').value)
            self.command_subs.append(
                self.create_subscription(
                    ArmJoints,
                    topic,
                    lambda message, command_source=source: self._command_callback(
                        message,
                        command_source,
                    ),
                    10,
                ),
            )

        self.watchdog = self.create_timer(0.05, self._watchdog_callback)
        self._publish_status('initialized', 'Control mode is disabled.')
        self.get_logger().warning(
            'Arm command mux started fail-closed. Publish disabled/test/teleop/rl '
            'to /arm/control_mode to change mode.'
        )

    def _seconds_to_ns(self, parameter_name: str) -> int:
        return int(float(self.get_parameter(parameter_name).value) * 1_000_000_000)

    def _get_float_vector(self, parameter_name: str) -> List[float]:
        values = [float(value) for value in self.get_parameter(parameter_name).value]
        if len(values) != self.JOINT_COUNT:
            raise ValueError(f'{parameter_name} must contain exactly {self.JOINT_COUNT} values.')
        return values

    def _feedback_callback(self, message: JointState) -> None:
        if len(message.position) != self.JOINT_COUNT:
            self._publish_status('feedback_rejected', 'Expected six joint positions.')
            return
        positions = [math.degrees(value) for value in message.position]
        if not all(math.isfinite(value) for value in positions):
            self._publish_status('feedback_rejected', 'Feedback contains a non-finite value.')
            return
        self.feedback_deg = positions
        self.feedback_time_ns = self.get_clock().now().nanoseconds

    def _mode_callback(self, message: String) -> None:
        requested = message.data.strip().lower()
        if requested not in ('disabled',) + self.SOURCES:
            self._publish_status('mode_rejected', f'Unknown mode: {requested}')
            return
        if requested != 'disabled' and self.require_feedback and not self._feedback_is_fresh():
            self.active_mode = 'disabled'
            self._publish_status('mode_rejected', 'Fresh calibrated feedback is required.')
            return

        self.active_mode = requested
        self.last_command = None
        self.last_command_time_ns = None
        self._publish_status('mode_changed', f'Active mode: {requested}')

    def _command_callback(self, message: ArmJoints, source: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.active_mode != source:
            return
        if self.require_feedback and not self._feedback_is_fresh(now_ns):
            self._disable('feedback_timeout', 'Feedback is missing or stale.')
            return
        if (
            self.last_command_time_ns is not None
            and now_ns - self.last_command_time_ns < self.min_command_period_ns
        ):
            self._publish_status('command_rejected', 'Command rate is too high.')
            return

        positions = [
            float(message.joint1),
            float(message.joint2),
            float(message.joint3),
            float(message.joint4),
            float(message.joint5),
            float(message.joint6),
        ]
        error = self._validate_command(positions, int(message.time))
        if error:
            self._publish_status('command_rejected', error)
            return

        self.output_pub.publish(message)
        self.last_command = positions
        self.last_command_time_ns = now_ns
        self._publish_status('command_forwarded', f'Accepted command from {source}.')

    def _validate_command(self, positions: List[float], motion_time_ms: int) -> str | None:
        if not self.min_motion_time_ms <= motion_time_ms <= self.max_motion_time_ms:
            return (
                f'Motion time {motion_time_ms} ms is outside '
                f'[{self.min_motion_time_ms}, {self.max_motion_time_ms}].'
            )
        for index, position in enumerate(positions):
            if not self.joint_min[index] <= position <= self.joint_max[index]:
                return (
                    f'Joint {index + 1} target {position:.2f} deg is outside '
                    f'[{self.joint_min[index]:.2f}, {self.joint_max[index]:.2f}].'
                )

        reference = self.last_command if self.last_command is not None else self.feedback_deg
        if reference is None:
            return 'No safe reference position is available.'
        for index, (position, previous) in enumerate(zip(positions, reference)):
            if abs(position - previous) > self.max_step[index]:
                return (
                    f'Joint {index + 1} step {abs(position - previous):.2f} deg '
                    f'exceeds {self.max_step[index]:.2f} deg.'
                )

        if self.last_command is not None and self.feedback_deg is not None:
            for index, (target, actual) in enumerate(zip(self.last_command, self.feedback_deg)):
                if abs(target - actual) > self.max_tracking_error[index]:
                    return (
                        f'Joint {index + 1} tracking error '
                        f'{abs(target - actual):.2f} deg is too large.'
                    )
        return None

    def _feedback_is_fresh(self, now_ns: int | None = None) -> bool:
        if self.feedback_time_ns is None or self.feedback_deg is None:
            return False
        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds
        return now_ns - self.feedback_time_ns <= self.feedback_timeout_ns

    def _watchdog_callback(self) -> None:
        if self.active_mode == 'disabled':
            return
        now_ns = self.get_clock().now().nanoseconds
        if self.require_feedback and not self._feedback_is_fresh(now_ns):
            self._disable('feedback_timeout', 'Feedback watchdog expired.')
            return
        if (
            self.last_command_time_ns is not None
            and now_ns - self.last_command_time_ns > self.command_timeout_ns
        ):
            self._disable('command_timeout', 'Command watchdog expired.')

    def _disable(self, event: str, detail: str) -> None:
        self.active_mode = 'disabled'
        self.last_command = None
        self.last_command_time_ns = None
        self._publish_status(event, detail)

    def _publish_status(self, event: str, detail: str) -> None:
        message = String()
        message.data = json.dumps(
            {
                'event': event,
                'detail': detail,
                'active_mode': self.active_mode,
                'feedback_fresh': self._feedback_is_fresh(),
            },
            separators=(',', ':'),
        )
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmCommandMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
