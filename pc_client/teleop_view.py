#!/usr/bin/env python3

"""Live camera view with arm jog control in one window, over one rosbridge link.

The camera and the arm share a single roslibpy connection because the twisted
reactor underneath cannot be restarted inside one process.

This is open loop. The arm publishes no joint feedback, so the pose shown on
screen is what this program has commanded, not what the hardware did. Start
every session from a known pose with --home or --assume-pose.
"""

import argparse
import collections
import pathlib
import time

import cv2
import roslibpy

import arm_control
import camera_view

LEGEND = '1-6 joint  j/k jog  [/] step  o/c grip  h home  spc hold  s save  q quit'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera view plus arm teleop for the M3 Pro. Dry-run by default.'
    )
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--stream', choices=sorted(camera_view.STREAMS), default='color')
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--save-dir', default='.')
    parser.add_argument('--home', action='store_true',
                        help='Start by driving the arm to the factory rest pose.')
    parser.add_argument('--assume-pose', type=int, nargs=6,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Declare the pose the arm is already in, without moving it.')
    parser.add_argument('--step', type=int, default=2, choices=arm_control.STEP_CHOICES)
    parser.add_argument('--execute', action='store_true',
                        help='Actually publish arm commands. Without this nothing moves.')
    return parser.parse_args()


def pose_line(pose, selected):
    parts = []
    for index, value in enumerate(pose):
        label = 'grip' if index == 5 else 'j{}'.format(index + 1)
        text = '{}:{}'.format(label, value)
        parts.append('<{}>'.format(text) if index == selected else ' {} '.format(text))
    return ''.join(parts)


def main():
    args = parse_args()
    if bool(args.home) == bool(args.assume_pose):
        raise SystemExit('Choose exactly one of --home or --assume-pose.')

    if args.execute and not arm_control.confirm_execute():
        raise SystemExit('Cancelled.')

    spec = camera_view.STREAMS[args.stream]
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    client = arm_control.connect(args.host, args.port)
    print('Connected to ws://{}:{}'.format(args.host, args.port))
    link = arm_control.ArmLink(client, args.execute)

    image_topic = roslibpy.Topic(
        client,
        spec['topic'],
        spec['type'],
        throttle_rate=int(1000.0 / args.fps),
        queue_length=1,
        queue_size=1,
    )
    latest = {}
    arrivals = collections.deque(maxlen=30)

    def on_frame(message):
        latest['message'] = message
        arrivals.append(time.time())

    image_topic.subscribe(on_frame)

    step = args.step
    selected = 0
    note = ''
    dropped = 0
    last_seen = None
    window = 'M3 Pro teleop ({})'.format(args.stream)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        if args.home:
            pose = list(arm_control.HOME_POSE)
            print('Homing to {} over {}ms.'.format(pose, arm_control.HOMING_TIME_MS))
            link.send(pose, arm_control.HOMING_TIME_MS)
            if args.execute:
                time.sleep(arm_control.HOMING_TIME_MS / 1000.0 + 0.5)
            note = 'homed'
        else:
            pose = arm_control.clamp_pose(args.assume_pose)
            print('Assuming the arm is at {} without moving it.'.format(pose))
            note = 'assumed pose'

        while True:
            message = latest.get('message')
            if message is not None and message is not last_seen:
                last_seen = message
                image, byte_count, depth_mm = camera_view.decode(message, spec['kind'])
                if image is None:
                    dropped += 1
                else:
                    rate = 0.0
                    if len(arrivals) > 1:
                        span = arrivals[-1] - arrivals[0]
                        if span > 0:
                            rate = (len(arrivals) - 1) / span
                    if args.scale != 1.0:
                        image = cv2.resize(image, None, fx=args.scale, fy=args.scale)

                    head = '{:.1f} fps  {} KB'.format(rate, byte_count // 1024)
                    if depth_mm is not None:
                        centre = int(depth_mm[depth_mm.shape[0] // 2, depth_mm.shape[1] // 2])
                        head += '  centre {} mm'.format(centre if centre else '--')
                    mode = 'EXECUTE' if args.execute else 'DRY-RUN'
                    lines = [
                        '{}   [{}]'.format(head, mode),
                        pose_line(pose, selected),
                        'step {}   {}'.format(step, note),
                        LEGEND,
                    ]
                    camera_view.annotate(image, lines)
                    cv2.imshow(window, image)

            raw_key = cv2.waitKey(10) & 0xFF
            if raw_key != 255:
                key = chr(raw_key).lower() if 32 <= raw_key < 127 else ''
                if key == 'q':
                    break
                if key == 's' and last_seen is not None:
                    snap, _, _ = camera_view.decode(last_seen, spec['kind'])
                    if snap is not None:
                        path = save_dir / 'm3pro_{}_{}.png'.format(args.stream, int(time.time()))
                        cv2.imwrite(str(path), snap)
                        note = 'saved {}'.format(path.name)
                elif key:
                    pose, selected, step, new_note = arm_control.apply_key(
                        key, pose, selected, step, link
                    )
                    if new_note:
                        note = new_note
                    elif key in 'jk123456[]':
                        note = '' if args.execute else 'dry-run, nothing sent'

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        image_topic.unsubscribe()
        link.close()
        client.terminate()
        cv2.destroyAllWindows()
        print('Last commanded pose: {}'.format(pose))
        print('Resume with --assume-pose {}'.format(' '.join(str(v) for v in pose)))
        if dropped:
            print('Dropped {} undecodable frames.'.format(dropped))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
