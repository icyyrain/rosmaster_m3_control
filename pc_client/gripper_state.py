#!/usr/bin/env python3

"""Read the gripper from the camera, and limit how hard it grips.

The gripper's angle cannot be recovered the way joints 1-4 can, because it does
not move the camera: it hangs off arm5, downstream of the arm4 mount. But it is
plainly visible. The two jaws appear as dark shapes along the bottom of the
frame either side of the fixed gripper pixel (707, 680), and because joint 5
turns about an axis nearly collinear with the offset to the gripper, that
position does not shift when joints 1-4 move.

So the jaws are simply measured. Tracking the gap between the outermost dark
regions fails, correlating only +0.46 with the command, because the strip also
holds a keyboard edge, cables and shadows and the extreme regions jump between
them. Dark *area* works.

**But every number describing that curve is pose and lighting dependent.** At
pose [90, 120, 0, 23] the jaws registered from 30 degrees at about 97 px per
degree, saturating near 120. At the home pose the reading was flat zero below
55, ran at about 161 px per degree, and saturated at 110. So the band, the rate
and the saturation angle all have to be measured where the work happens, with
`characterise`; an earlier version that trusted one global constant read the
flat zero as a blockage and reported a firm grip on empty jaws.

Two things follow for grip force. There is no force or current feedback on this
arm at all: ArmJoints carries angles only, so a servo told to reach 180 on an
object that stops it at 120 keeps pushing with its stall torque, which is how a
chocolate gets crushed. Force is therefore set by how far past contact the
command goes, and contact is found optically: free jaws move the dark area at a
measurable rate, blocked jaws stop moving while the command keeps rising.

For grasp detection the robust quantity is the difference between closed and
open at one pose, where the strip's background cancels:

    grasp_signal = area(closed) - area(open)

On air over six cycles that was 7786 px with a standard deviation of 9 px,
which is 0.09 degrees of jaw travel. That reference does not survive a large
posture change either: at a grasp pose it read 11008 px, larger rather than
smaller. To confirm a grasp across poses, prefer the kinematic test in
docs/MEASUREMENTS.md, which watches whether the object moves with the gripper.
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

OPEN_CMD = 30
CLOSED_CMD = 120

# Measured at pose [90, 120, 0, 23] only. Use characterise() at the working
# pose rather than trusting this.
PX_PER_DEGREE = 97.0

# Below this the jaws are not in the strip at all, and an increment of zero
# means "cannot see", not "blocked".
VISIBLE_MIN_PX = 400

# Grasp signal on air at one pose. A starting point, nothing more.
EMPTY_SIGNAL_PX = 7786.0
EMPTY_SIGNAL_SD_PX = 9.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera readout of the gripper, grasp detection and grip limiting.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--mode', required=True,
                        choices=('read', 'probe', 'sweep', 'calibrate', 'grip'),
                        help='read: report the area now, no motion. '
                             'probe: open then close and report the grasp signal. '
                             'sweep: print area against command. '
                             'calibrate: sweep EMPTY jaws to find the band, rate '
                             'and saturation angle. '
                             'grip: close in steps and stop at contact plus a margin.')
    parser.add_argument('--assume-pose', type=int, nargs=6,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Required for every mode that moves: the arm pose to '
                             'hold while only joint 6 moves.')
    parser.add_argument('--cycles', type=int, default=3,
                        help='probe: how many open/close cycles to average.')
    parser.add_argument('--grip-step', type=int, default=5,
                        help='degrees per increment for calibrate and grip.')
    parser.add_argument('--grip-margin', type=int, default=8,
                        help='grip: degrees commanded past contact. This is the '
                             'force knob; larger squeezes harder.')
    parser.add_argument('--contact-frac', type=float, default=0.35,
                        help='grip: an increment below this fraction of the free '
                             'rate counts as contact.')
    parser.add_argument('--ceiling', type=int, default=0,
                        help='grip: angle at which empty jaws saturate, from '
                             'calibrate. A collapse there is not contact.')
    parser.add_argument('--visible-from', type=int, default=OPEN_CMD,
                        help='grip: first angle at which the jaws show in the strip.')
    parser.add_argument('--free-rate', type=float, default=0.0,
                        help='grip: free-jaw rate in px per degree, from calibrate.')
    parser.add_argument('--execute', action='store_true',
                        help='Required for every mode that moves the gripper.')
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
    """(signal_px, spread_px, open_area, closed_area), closing from open only."""
    signals, opens, closeds = [], [], []
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


def characterise(link, streams, pose, step=5, low=OPEN_CMD, high=140, verbose=True):
    """Sweep EMPTY jaws and report what the readout can actually see here.

    Returns the first angle at which the jaws are visible, the free rate in px
    per degree, and the angle where the reading stops rising. That last one is
    the ceiling a contact test must stay below, because saturation and a real
    blockage look identical in this signal.
    """
    rows = []
    for angle in range(low, high + 1, step):
        set_gripper(link, pose, angle, settle=1.5)
        area, _ = read_area(streams)
        if area is None:
            continue
        rows.append((angle, area))
        if verbose:
            print('  {:>5} deg  {:>8.0f} px'.format(angle, area))
    if len(rows) < 5:
        return None

    visible = next((a for a, v in rows if v >= VISIBLE_MIN_PX), None)
    if visible is None:
        return None
    rising = [(a, v) for a, v in rows if a >= visible]
    increments = [(rising[i][0], rising[i][1] - rising[i - 1][1])
                  for i in range(1, len(rising))]
    strong = sorted(d for _, d in increments if d > 0)
    rate = float(numpy.median(strong[-4:])) / step if strong else 0.0

    ceiling = rising[-1][0]
    for angle, delta in increments:
        if rate > 0 and delta < 0.25 * rate * step:
            ceiling = angle
            break
    return dict(visible_from=visible, rate_px_per_deg=rate, ceiling=ceiling, rows=rows)


def close_until_contact(link, streams, pose, step=5, margin=8,
                        contact_fraction=0.35, baseline=None, verbose=True):
    """Close in increments and stop shortly after the jaws meet resistance.

    The free rate is measured as it goes rather than assumed, and nothing is
    judged until the jaws are actually visible in the strip: at the home pose
    the reading is flat zero below 55 degrees, and reading that as a collapse is
    exactly how an earlier version claimed a grip on empty jaws.

    `baseline` comes from characterise() with nothing in the jaws at this same
    pose. It supplies the saturation angle, so a collapse there is reported as
    "closed on nothing" rather than as a false grip.
    """
    ceiling = baseline['ceiling'] if baseline else CLOSED_CMD
    visible_from = baseline['visible_from'] if baseline else OPEN_CMD
    known_rate = baseline.get('rate_px_per_deg') if baseline else None
    if not known_rate:
        known_rate = None

    set_gripper(link, pose, OPEN_CMD, settle=2.2)
    previous, _ = read_area(streams)
    if previous is None:
        return None

    rows, increments = [], []
    weak = 0
    contact = None
    if verbose:
        print('{:>8} {:>11} {:>11} {:>9} {:>12}'.format(
            'joint6', 'area px', 'increment', 'of free', 'state'))
        print('-' * 56)

    angle = OPEN_CMD
    while angle + step <= ceiling:
        angle += step
        set_gripper(link, pose, angle, settle=1.5)
        area, _ = read_area(streams)
        if area is None:
            continue
        increment = area - previous
        previous = area

        fraction = float('nan')
        if area < VISIBLE_MIN_PX or angle < visible_from:
            state = 'not visible'
        else:
            increments.append(increment)
            rate = known_rate
            if rate is None and len(increments) >= 3:
                rate = float(numpy.median(sorted(increments)[-3:])) / step
            if not rate or rate <= 0:
                state = 'learning'
            else:
                fraction = increment / (rate * step)
                weak = weak + 1 if fraction < contact_fraction else 0
                state = 'weak x{}'.format(weak) if weak else 'free'
        rows.append((angle, area, increment, fraction))
        if verbose:
            print('{:>8} {:>11.0f} {:>11.0f} {:>9} {:>12}'.format(
                angle, area, increment,
                '--' if fraction != fraction else '{:.0%}'.format(fraction), state))
        if weak >= 2:
            contact = angle - step
            break

    if verbose:
        print('-' * 56)

    if contact is None:
        if verbose:
            print('no contact below the {} deg ceiling: the jaws closed on nothing,'
                  .format(ceiling))
            print('or whatever is between them is too thin to block them there.')
        set_gripper(link, pose, OPEN_CMD, settle=1.5)
        return dict(contact=None, commanded=OPEN_CMD, rows=rows, ceiling=ceiling)

    if contact >= ceiling - step:
        if verbose:
            print('the reading stopped rising at {} deg, which is where empty jaws'
                  .format(contact))
            print('saturate too, so this is not evidence of contact.')
        set_gripper(link, pose, OPEN_CMD, settle=1.5)
        return dict(contact=None, commanded=OPEN_CMD, rows=rows, ceiling=ceiling)

    commanded = min(ceiling, contact + margin)
    set_gripper(link, pose, commanded, settle=2.0)
    if verbose:
        print('contact at about {} deg; commanding {} deg ({} deg of squeeze)'.format(
            contact, commanded, commanded - contact))
    return dict(contact=contact, commanded=commanded, rows=rows, ceiling=ceiling)


def verdict(signal):
    """Interpret a grasp signal against the empty-air reference at one pose."""
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
    moving = args.mode in ('probe', 'sweep', 'calibrate', 'grip')
    if moving:
        if not args.assume_pose:
            raise SystemExit('--mode {} needs --assume-pose.'.format(args.mode))
        if not args.execute:
            raise SystemExit(
                '--mode {} moves the gripper; pass --execute.'.format(args.mode))

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
            print('\ndark area now: {:.0f} px (frame-to-frame sd {:.1f})'.format(
                area, spread))
            print('Absolute, so it includes whatever else is dark in the strip.')
            return 0

        print('holding arm at {}, moving only joint 6'.format(pose))

        if args.mode in ('sweep', 'calibrate'):
            print()
            if args.mode == 'calibrate':
                print('sweeping EMPTY jaws; make sure nothing is between them')
            base = characterise(link, streams, pose, args.grip_step)
            set_gripper(link, pose, 90, settle=1.2)
            if base is None:
                print('\nthe jaws never became visible in the strip at this pose.')
                return 1
            print()
            print('visible from {} deg, free rate {:.0f} px/deg, saturates at {} deg'.format(
                base['visible_from'], base['rate_px_per_deg'], base['ceiling']))
            print()
            print('pass these to --mode grip:')
            print('  --visible-from {} --free-rate {:.0f} --ceiling {}'.format(
                base['visible_from'], base['rate_px_per_deg'], base['ceiling']))
            print('and re-run this whenever the pose or the lighting changes.')
            return 0

        if args.mode == 'grip':
            print()
            baseline = None
            if args.ceiling:
                baseline = dict(visible_from=args.visible_from,
                                rate_px_per_deg=args.free_rate,
                                ceiling=args.ceiling)
            else:
                print('no --ceiling given, so saturation cannot be told from contact.')
                print('Run --mode calibrate on empty jaws first.')
                print()
            outcome = close_until_contact(
                link, streams, pose, args.grip_step, args.grip_margin,
                args.contact_frac, baseline)
            if outcome is None:
                print('no usable readings')
                return 1
            print()
            if outcome['contact'] is None:
                print('left open; nothing gripped.')
            else:
                print('holding at joint6 = {}. Raise --grip-margin to squeeze'.format(
                    outcome['commanded']))
                print('harder, lower it for a gentler hold. Commanding 180 is what')
                print('crushes things.')
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
        print('  shortfall {:+.0f} px = {:+.2f} deg = {:+.1f} sigma'.format(
            shortfall, degrees, sigmas))
        print('  -> {}'.format(label))
        return 0
    finally:
        if pose is not None and args.execute:
            set_gripper(link, pose, 90, settle=1.2)
        streams.close()
        link.close()
        client.terminate()


if __name__ == '__main__':
    sys.exit(main())
