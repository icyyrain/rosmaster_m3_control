#!/usr/bin/env python3

"""Read the gripper from the camera, and tell whether it closed on something.

The gripper's angle cannot be recovered the way joints 1-4 can, because it does
not move the camera: it hangs off arm5, downstream of the arm4 mount. But it is
plainly visible. The two jaws appear as dark shapes along the bottom of the
frame either side of the fixed gripper pixel (707, 680), and because joint 5
turns about an axis nearly collinear with the offset to the gripper, that
position does not shift when joints 1-4 move.

So the jaws are simply measured. The first attempt tracked the gap between the
outermost dark regions and failed, correlating only +0.46 with the command,
because the strip also holds a keyboard edge, cables and shadows and the
extreme regions jump between them. Dark *area* works instead:

  * +0.962 correlation with the command over the 30-120 band, +97 px/degree
  * saturates above 120, where the jaws are fully in view
  * 8-14% hysteresis between closing and opening, matching the backlash the
    arm joints show, so always approach a reading from the same direction

Absolute area depends on whatever else is dark in the strip, so it shifts when
the arm moves and the background changes. The robust quantity is the
difference between closed and open at one pose, where the background cancels:

    grasp_signal = area(closed) - area(open)

Measured on air over six cycles that is 7786 px with a standard deviation of
9 px, which is 0.09 degrees of jaw travel. A three-sigma deviation is 0.28
degrees, and a sweet is tens of degrees thick, so grasp detection has an
enormous margin.
"""

import argparse
import sys
import time

import cv2
import numpy

import arm_control
import reach_candy

# Bottom strip holding both jaws, kept below the keyboard edge in the test
# scene. Both jaws sit near x 490 and x 930, either side of the gripper pixel.
STRIP = dict(y0=665, y1=720, x0=400, x1=1020)
DARK_LEVEL = 80

# Usable command band. Above CLOSED the reading saturates.
OPEN_CMD = 30
CLOSED_CMD = 120
PX_PER_DEGREE = 97.0

# Grasp signal on air, measured over six cycles at one pose. Treat as a
# starting point only: it depends on the strip's background, so re-measure it
# at the working pose with --mode probe and no object present.
EMPTY_SIGNAL_PX = 7786.0
EMPTY_SIGNAL_SD_PX = 9.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera readout of the gripper, and grasp detection.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--mode', required=True, choices=('read', 'probe', 'sweep'),
                        help='read: report the area now, no motion. '
                             'probe: open then close and report the grasp signal. '
                             'sweep: characterise area against command.')
    parser.add_argument('--assume-pose', type=int, nargs=6,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Required for probe and sweep: the arm pose to hold '
                             'while only joint 6 moves.')
    parser.add_argument('--cycles', type=int, default=3,
                        help='probe: how many open/close cycles to average.')
    parser.add_argument('--execute', action='store_true',
                        help='Required for probe and sweep, which move the gripper.')
    return parser.parse_args()


def dark_area(gray):
    """Dark pixels in the jaw strip, after removing speckle."""
    strip = gray[STRIP['y0']:STRIP['y1'], STRIP['x0']:STRIP['x1']]
    mask = (strip < DARK_LEVEL).astype(numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    return int(mask.sum())


def read_area(streams, samples=6):
    """Median dark area over several frames, with the frame-to-frame spread."""
    values = []
    for _ in range(samples):
        bgr = streams.colour()
        if bgr is None:
            continue
        values.append(dark_area(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        time.sleep(0.05)
    if not values:
        return None, None
    return float(numpy.median(values)), float(numpy.std(values))


def set_gripper(link, pose, value, settle=1.9):
    target = list(pose)
    target[5] = value
    link.send(arm_control.clamp_pose(target), 1000)
    time.sleep(settle)


def grasp_signal(link, streams, pose, cycles=3):
    """(signal_px, spread_px, open_area, closed_area).

    Always closes from the open end so hysteresis does not enter the reading.
    """
    signals = []
    opens = []
    closeds = []
    for _ in range(cycles):
        set_gripper(link, pose, OPEN_CMD)
        open_area, _ = read_area(streams)
        set_gripper(link, pose, CLOSED_CMD)
        closed_area, _ = read_area(streams)
        if open_area is None or closed_area is None:
            continue
        opens.append(open_area)
        closeds.append(closed_area)
        signals.append(closed_area - open_area)
    if not signals:
        return None
    return (float(numpy.median(signals)), float(numpy.std(signals)),
            float(numpy.median(opens)), float(numpy.median(closeds)))


def verdict(signal):
    """Interpret a grasp signal against the empty-air reference."""
    shortfall = EMPTY_SIGNAL_PX - signal
    degrees = shortfall / PX_PER_DEGREE
    sigmas = shortfall / EMPTY_SIGNAL_SD_PX
    if sigmas > 3.0:
        return ('SOMETHING IN THE JAWS', shortfall, degrees, sigmas)
    if sigmas < -3.0:
        return ('closed further than empty air; re-measure the reference',
                shortfall, degrees, sigmas)
    return ('closed on air', shortfall, degrees, sigmas)


def main():
    args = parse_args()
    if args.mode in ('probe', 'sweep'):
        if not args.assume_pose:
            raise SystemExit('--mode {} needs --assume-pose.'.format(args.mode))
        if not args.execute:
            raise SystemExit('--mode {} moves the gripper; pass --execute.'.format(args.mode))

    client = arm_control.connect(args.host, args.port)
    link = arm_control.ArmLink(client, args.execute)
    streams = reach_candy.Streams(client)
    pose = arm_control.clamp_pose(args.assume_pose) if args.assume_pose else None

    try:
        if not streams.wait():
            raise RuntimeError('No camera frames. Is the camera driver running?')
        print('strip y {y0}-{y1}, x {x0}-{x1}, dark < {level}'.format(
            level=DARK_LEVEL, **STRIP))

        if args.mode == 'read':
            area, spread = read_area(streams, samples=10)
            print('\ndark area now: {:.0f} px (frame-to-frame sd {:.1f})'.format(area, spread))
            print('That is an absolute reading, so it includes whatever else is dark')
            print('in the strip. Use --mode probe for a background-free measurement.')
            return 0

        print('holding arm at {}, moving only joint 6'.format(pose))

        if args.mode == 'sweep':
            print()
            print('{:>8} {:>6} {:>12} {:>10}'.format('joint6', 'phase', 'area px', 'frame sd'))
            print('-' * 40)
            commands = [30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30]
            previous = None
            rows = []
            for value in commands:
                phase = '' if previous is None else ('close' if value > previous else 'open')
                previous = value
                set_gripper(link, pose, value)
                area, spread = read_area(streams)
                rows.append((value, phase, area))
                print('{:>8} {:>6} {:>12.0f} {:>10.1f}'.format(value, phase, area, spread))
            print('-' * 40)
            band = [(a, v) for a, p, v in rows if a <= CLOSED_CMD]
            if len(band) >= 4:
                angles = numpy.array([a for a, _ in band], float)
                areas = numpy.array([v for _, v in band], float)
                slope = float(numpy.polyfit(angles, areas, 1)[0])
                corr = float(numpy.corrcoef(angles, areas)[0, 1])
                print()
                print('over the {}-{} band: {:+.1f} px/deg, correlation {:+.3f}'.format(
                    OPEN_CMD, CLOSED_CMD, slope, corr))
            return 0

        result = grasp_signal(link, streams, pose, args.cycles)
        if result is None:
            print('\nno usable readings')
            return 1
        signal, spread, open_area, closed_area = result
        print()
        print('open   @{:>3}: {:8.0f} px'.format(OPEN_CMD, open_area))
        print('closed @{:>3}: {:8.0f} px'.format(CLOSED_CMD, closed_area))
        print('grasp signal: {:8.0f} px  (spread over {} cycles: {:.0f} px)'.format(
            signal, args.cycles, spread))
        label, shortfall, degrees, sigmas = verdict(signal)
        print()
        print('against the empty-air reference of {:.0f} px:'.format(EMPTY_SIGNAL_PX))
        print('  shortfall {:+.0f} px = {:+.2f} deg of jaw travel = {:+.1f} sigma'.format(
            shortfall, degrees, sigmas))
        print('  -> {}'.format(label))
        print()
        print('The reference depends on the strip background, so at a new pose run')
        print('this once with nothing in the jaws and use that number instead.')
        return 0
    finally:
        if pose is not None and args.execute:
            set_gripper(link, pose, 90, settle=1.2)
        streams.close()
        link.close()
        client.terminate()


if __name__ == '__main__':
    sys.exit(main())
