#!/usr/bin/env python3

"""Keyboard jog demo for the M3 Pro arm over rosbridge, without a camera.

For the combined camera + teleop window see teleop_view.py. This one only
needs a console, which is handy when the camera driver is not running.

Open loop: see arm_control for why the operator must declare the starting
pose.
"""

import argparse
import sys

import arm_control

try:
    import msvcrt
except ImportError:
    msvcrt = None

HELP = """
  1..6      select joint (6 = gripper)
  j / k     jog selected joint down / up
  [ / ]     step size {steps}
  o / c     gripper open / close
  h         return to home pose {home}
  space     hold (re-send current pose)
  q         quit
""".format(steps=arm_control.STEP_CHOICES, home=arm_control.HOME_POSE)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Open-loop keyboard jog demo for the M3 Pro arm. Dry-run by default.'
    )
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--home', action='store_true',
                        help='Start by driving the arm to the factory rest pose.')
    parser.add_argument('--assume-pose', type=int, nargs=6,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Declare the pose the arm is already in, without moving it.')
    parser.add_argument('--step', type=int, default=2, choices=arm_control.STEP_CHOICES)
    parser.add_argument('--execute', action='store_true',
                        help='Actually publish. Without this flag nothing is sent to the robot.')
    return parser.parse_args()


def format_pose(pose, selected):
    parts = []
    for index, value in enumerate(pose):
        label = 'grip' if index == 5 else 'j{}'.format(index + 1)
        text = '{}={}'.format(label, value)
        parts.append('[{}]'.format(text) if index == selected else ' {} '.format(text))
    return ' '.join(parts)


def read_key():
    key = msvcrt.getch()
    if key in (b'\x00', b'\xe0'):
        msvcrt.getch()
        return ''
    if key == b'\x03':
        return 'q'
    try:
        return key.decode('ascii').lower()
    except UnicodeDecodeError:
        return ''


def main():
    args = parse_args()
    if msvcrt is None:
        raise SystemExit('This demo needs the Windows console (msvcrt).')
    if bool(args.home) == bool(args.assume_pose):
        raise SystemExit('Choose exactly one of --home or --assume-pose.')
    if args.execute and not arm_control.confirm_execute():
        raise SystemExit('Cancelled.')

    client = arm_control.connect(args.host, args.port) if args.execute else None
    link = arm_control.ArmLink(client, args.execute)
    step = args.step
    selected = 0

    try:
        if args.home:
            pose = list(arm_control.HOME_POSE)
            print('Homing to {} over {}ms.'.format(pose, arm_control.HOMING_TIME_MS))
            link.send(pose, arm_control.HOMING_TIME_MS)
        else:
            pose = arm_control.clamp_pose(args.assume_pose)
            print('Assuming the arm is at {} without moving it.'.format(pose))

        print(HELP)
        print('mode: {}'.format('EXECUTE' if args.execute else 'dry-run'))
        print(format_pose(pose, selected), ' step={}'.format(step))

        while True:
            key = read_key()
            if key == 'q':
                break
            if not key:
                continue
            pose, selected, step, note = arm_control.apply_key(key, pose, selected, step, link)
            print(format_pose(pose, selected), ' step={}  {}'.format(step, note))
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        if client is not None:
            client.terminate()
        print('\nLast commanded pose: {}'.format(pose))
        print('Resume with --assume-pose {}'.format(' '.join(str(v) for v in pose)))

    return 0


if __name__ == '__main__':
    sys.exit(main())
