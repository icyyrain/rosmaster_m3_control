#!/usr/bin/env python3

"""Shared open-loop arm control for the M3 Pro PC clients.

The control board reports no joint positions, so nothing here can know where
the arm actually is. Callers track only the poses they have themselves
commanded, and every session must start from a pose the operator established.

This module is the single place where the motion envelope is defined. Keep it
that way: two copies of a joint limit is one copy too many.
"""

import time

import roslibpy

# Factory rest pose used by the Yahboom M3Pro_demo nodes.
HOME_POSE = [90, 120, 0, 0, 90, 90]

# Conservative envelope. Joints 1-5 are the arm; joint 6 is the gripper, whose
# factory demos only ever use 30 (open) through 180 (closed).
JOINT_MIN = [0, 0, 0, 0, 0, 30]
JOINT_MAX = [180, 180, 180, 180, 180, 180]

GRIPPER_OPEN = 30
GRIPPER_CLOSED = 180

STEP_CHOICES = [1, 2, 5, 10]
HOMING_TIME_MS = 3000
MIN_COMMAND_PERIOD_SEC = 0.1

COMMAND_TOPIC = '/arm6_joints'
COMMAND_TYPE = 'arm_msgs/msg/ArmJoints'

EXECUTE_WARNING = """This will MOVE the real arm.
Clear the workspace, support the robot, and be ready to cut power.
The arm has no position feedback, so this tool is flying blind: it knows
only what it has commanded, never where the arm really is.
Note: the factory /joy_ctrl node also publishes {topic}.
      Put the gamepad down so the two do not fight.""".format(topic=COMMAND_TOPIC)


def clamp_pose(pose):
    return [
        max(JOINT_MIN[index], min(JOINT_MAX[index], int(value)))
        for index, value in enumerate(pose)
    ]


def motion_time_ms(step):
    """Give every jog a deliberately slow constant speed."""
    return max(150, abs(step) * 60)


def confirm_execute():
    """Ask on the terminal before anything is allowed to move. True to proceed."""
    print(EXECUTE_WARNING)
    return input('Type MOVE to continue: ') == 'MOVE'


def connect(host, port, timeout=10):
    client = roslibpy.Ros(host=host, port=port)
    client.run(timeout=timeout)
    if not client.is_connected:
        raise RuntimeError('Unable to connect to ws://{}:{}'.format(host, port))
    return client


class ArmLink(object):
    """Publishes absolute joint targets, rate-limited.

    A client may be supplied even when execute is False: the camera clients
    share one connection, and roslibpy cannot open a second one in the same
    process. Nothing is advertised or published unless execute is True.
    """

    def __init__(self, client, execute):
        self.execute = execute
        self.client = client
        self.topic = None
        self.last_sent = 0.0
        if execute:
            if client is None:
                raise ValueError('execute=True needs a connected client')
            self.topic = roslibpy.Topic(client, COMMAND_TOPIC, COMMAND_TYPE)
            self.topic.advertise()

    def send(self, pose, time_ms):
        payload = {'joint{}'.format(index + 1): int(value) for index, value in enumerate(pose)}
        payload['time'] = int(time_ms)
        if not self.execute:
            return False
        wait = MIN_COMMAND_PERIOD_SEC - (time.time() - self.last_sent)
        if wait > 0:
            time.sleep(wait)
        self.topic.publish(roslibpy.Message(payload))
        self.last_sent = time.time()
        return True

    def close(self):
        if self.topic is not None:
            self.topic.unadvertise()


def apply_key(key, pose, selected, step, link):
    """Handle one jog keystroke.

    Returns (pose, selected, step, note). The pose is always clamped, and a
    command is sent only when the pose actually changes or the key is an
    explicit re-send.
    """
    note = ''
    if key in '123456':
        selected = int(key) - 1
    elif key in ('j', 'k'):
        delta = step if key == 'k' else -step
        target = clamp_pose([
            value + delta if index == selected else value
            for index, value in enumerate(pose)
        ])
        if target == pose:
            note = 'limit reached on joint {}'.format(selected + 1)
        else:
            pose = target
            link.send(pose, motion_time_ms(step))
    elif key in ('[', ']'):
        index = STEP_CHOICES.index(step)
        index = max(0, index - 1) if key == '[' else min(len(STEP_CHOICES) - 1, index + 1)
        step = STEP_CHOICES[index]
    elif key in ('o', 'c'):
        pose = list(pose)
        pose[5] = GRIPPER_OPEN if key == 'o' else GRIPPER_CLOSED
        link.send(pose, 1500)
        note = 'gripper {}'.format('open' if key == 'o' else 'closed')
    elif key == 'h':
        pose = list(HOME_POSE)
        link.send(pose, HOMING_TIME_MS)
        note = 'homing'
    elif key == ' ':
        link.send(pose, 300)
        note = 'hold'
    return pose, selected, step, note
