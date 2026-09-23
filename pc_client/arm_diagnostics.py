#!/usr/bin/env python3

"""Characterise the M3 Pro arm using the eye-in-hand camera as the sensor.

The control board publishes no joint feedback, so none of these figures can be
read from ROS. They are measured optically: the camera is fixed to arm link
arm4, so arm motion moves the whole image, and the global image transform gives
motion in pixels at a 0.03-0.05 px noise floor. See vision_metrology.

THESE TESTS MOVE THE ARM. Clear the workspace and be ready to cut power. Every
test starts by homing, returns home between trials, and homes again on the way
out, including on failure.

Tests:

  coupling  Which joints move the camera. Confirms the URDF claim that the
            camera hangs off arm4, and that joint 5 and the gripper are
            therefore invisible. Gives the invisible joints the largest
            commands, so the experiment is biased against its own hypothesis.

  sweep     Joint 1 in one-degree steps, up then down then up, to find the
            deadband, the linearity and the backlash. One degree is the finest
            ArmJoints command, so "does one degree do anything" decides whether
            a one-degree action space means anything.

  step      Step response against several ArmJoints.time values, to check that
            the field really sets trajectory duration and to measure latency.

  hold      Whether the arm hunts after a move. Reads a reference at the new
            pose, which is what makes this trustworthy: measuring against a
            pre-move reference exaggerates the wander roughly twentyfold
            because the inlier count collapses.

Results measured on 2026-09-23 are recorded in docs/MEASUREMENTS.md.
"""

import argparse
import math
import sys
import time

import numpy
import roslibpy

import arm_control
import vision_metrology as vm

JOINT1 = 0
MOVE_MS = 1500
SETTLE_S = 2.5

# Deliberately large commands for the joints predicted to be invisible.
COUPLING_TRIALS = [
    ('joint1 base yaw   ', 0, +10, False),
    ('joint2 shoulder   ', 1, +10, False),
    ('joint3 elbow      ', 2, +10, False),
    ('joint4 wrist pitch', 3, +10, False),
    ('joint5 wrist roll ', 4, +60, True),
    ('joint5 wrist roll-', 4, -60, True),
    ('joint6 gripper    ', 5, +60, True),
    ('joint6 gripper -  ', 5, -60, True),
]

STEP_DURATIONS = [200, 500, 1000, 1500]
STEP_DEG = 6

# Local sensitivity of joint 1 near home, px of image translation per degree,
# from the linear fit in the sweep test. Used only to print degrees alongside
# pixels; the pixel figures are the measurement.
PX_PER_DEG = 18.81


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optical characterisation of the M3 Pro arm. MOVES THE ARM.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--test', required=True,
                        choices=('coupling', 'sweep', 'step', 'hold'))
    parser.add_argument('--yes', action='store_true',
                        help='Skip the interactive confirmation.')
    return parser.parse_args()


def home(link, wait_extra=1.5):
    link.send(arm_control.HOME_POSE, arm_control.HOMING_TIME_MS)
    time.sleep(arm_control.HOMING_TIME_MS / 1000.0 + wait_extra)


def pose_with(index, value):
    return arm_control.clamp_pose(
        [value if i == index else v for i, v in enumerate(arm_control.HOME_POSE)])


def test_coupling(link, camera):
    print('{:<20} {:>6} {:>9} {:>10} {:>10} {:>10}'.format(
        'joint', 'delta', 'noise px', 'moved px', 'roll deg', 'return px'))
    print('-' * 70)
    rows = []
    for label, index, delta, invisible in COUPLING_TRIALS:
        reference = vm.reference_of(camera.grab())
        if reference is None:
            print('{:<20} no usable reference frame'.format(label))
            continue

        quiet = vm.median_shift(reference, camera)
        noise = quiet[0] if quiet else float('nan')

        link.send(pose_with(index, arm_control.HOME_POSE[index] + delta), MOVE_MS)
        time.sleep(SETTLE_S)
        moved = vm.median_shift(reference, camera)

        link.send(arm_control.HOME_POSE, MOVE_MS)
        time.sleep(SETTLE_S)
        back = vm.median_shift(reference, camera)

        moved_px = moved[0] if moved else float('nan')
        roll = moved[1] if moved else float('nan')
        back_px = back[0] if back else float('nan')
        rows.append((label, moved_px, noise, back_px, invisible))
        print('{:<20} {:>+5}d {:>9.2f} {:>10.2f} {:>10.3f} {:>10.2f}'.format(
            label, delta, noise, moved_px, roll, back_px))

    print('-' * 70)
    print()
    print('A joint moves the camera only if it clearly beats both the static')
    print('noise and the return-to-home residual.')
    print()
    for label, moved_px, noise, back_px, invisible in rows:
        floor = max(v for v in (noise, back_px) if not math.isnan(v)) \
            if not (math.isnan(noise) and math.isnan(back_px)) else float('nan')
        if math.isnan(moved_px):
            verdict = 'NO DATA'
        elif moved_px > max(5.0 * floor, 5.0):
            verdict = 'MOVES CAMERA'
        else:
            verdict = 'no camera motion'
        expected = 'no camera motion' if invisible else 'MOVES CAMERA'
        mark = 'OK' if verdict == expected else 'UNEXPECTED'
        print('  {} {:8.2f} px vs floor {:5.2f} px -> {:<18} [{}]'.format(
            label, moved_px, floor, verdict, mark))


def test_sweep(link, camera):
    angles = ([(a, 'up') for a in range(91, 101)] +
              [(a, 'down') for a in range(99, 79, -1)] +
              [(a, 'up2') for a in range(81, 91)])

    anchor_frame = camera.grab()
    anchor = vm.reference_of(anchor_frame)
    still = vm.shift_of(anchor, camera.grab())
    print('anchor at joint1=90; still-frame residual {:.2f} px'.format(
        still.magnitude if still else float('nan')))
    print()
    print('{:>5} {:>6} {:>13} {:>14} {:>9}'.format(
        'cmd', 'phase', 'cum shift px', 'step shift px', 'inliers'))
    print('-' * 52)

    rows = []
    previous = anchor_frame
    for angle, phase in angles:
        link.send(pose_with(JOINT1, angle), 400)
        time.sleep(1.2)
        frame = camera.grab()
        cumulative = vm.shift_of(anchor, frame)
        incremental = vm.shift_of(vm.reference_of(previous), frame)
        previous = frame
        if cumulative is None:
            print('{:>5} {:>6}   lost tracking'.format(angle, phase))
            continue
        step_px = incremental.magnitude if incremental else float('nan')
        rows.append((angle, phase, cumulative.signed, step_px))
        print('{:>5} {:>6} {:>13.2f} {:>14.2f} {:>9}'.format(
            angle, phase, cumulative.signed, step_px, cumulative.inliers))
    print('-' * 52)

    steps = [r[3] for r in rows if not math.isnan(r[3])]
    if not steps:
        print('no usable steps')
        return
    print()
    print('per-degree step: median {:.2f} px, min {:.2f}, max {:.2f}'.format(
        float(numpy.median(steps)), min(steps), max(steps)))
    dead = [r for r in rows if not math.isnan(r[3]) and r[3] < 2.0]
    print('steps under 2 px (candidate dead commands): {} of {}'.format(len(dead), len(rows)))

    rising = {r[0]: r[2] for r in rows if r[1] == 'up'}
    falling = {r[0]: r[2] for r in rows if r[1] == 'down'}
    shared = sorted(set(rising) & set(falling))
    if shared:
        scale = float(numpy.median(steps))
        diffs = [abs(rising[a] - falling[a]) for a in shared]
        print()
        print('backlash: same command, rising vs falling')
        for angle in shared:
            print('  {:>5}  rising {:>8.2f}  falling {:>8.2f}  diff {:>7.2f} px'.format(
                angle, rising[angle], falling[angle], rising[angle] - falling[angle]))
        print('  mean |diff| {:.2f} px = {:.3f} deg at {:.2f} px/deg'.format(
            float(numpy.mean(diffs)), float(numpy.mean(diffs)) / scale, scale))

    if len(rising) >= 5:
        xs = numpy.array(sorted(rising), dtype=float) - 90.0
        ys = numpy.array([rising[int(90 + x)] for x in xs])
        slope, intercept = numpy.polyfit(xs, ys, 1)
        worst = float(numpy.abs(ys - (slope * xs + intercept)).max())
        print()
        print('linearity on the rising sweep: {:.2f} px/deg, worst residual '
              '{:.2f} px ({:.3f} deg)'.format(slope, worst, worst / abs(slope)))


def test_step(link, camera):
    print('Latency below includes the camera pipeline and is not subtracted.')
    print('Frames arrive near 30 fps, so timing resolution is about 33 ms.')
    print()
    print('{:>9} {:>12} {:>10} {:>12} {:>9}'.format(
        'cmd time', 'travel px', 'deg', 'onset ms', '95% ms'))
    print('-' * 56)
    for duration in STEP_DURATIONS:
        # Approach the start angle from a fixed direction so that backlash does
        # not contaminate the timing.
        link.send(pose_with(JOINT1, 90 - STEP_DEG), 1200)
        time.sleep(2.0)
        link.send(arm_control.HOME_POSE, 1200)
        time.sleep(2.0)

        reference = vm.reference_of(camera.grab())
        camera.record()
        sent_at = time.time()
        link.send(pose_with(JOINT1, 90 + STEP_DEG), duration)
        time.sleep(3.0)
        frames = camera.stop()

        trace = []
        for stamp, payload in frames:
            result = vm.shift_of(reference, vm.decode_gray(payload))
            if result is not None:
                trace.append(((stamp - sent_at) * 1000.0, result.magnitude))
        if len(trace) < 5:
            print('{:>9} not enough usable frames'.format(duration))
            continue

        final = float(numpy.median([v for _, v in trace[-5:]]))

        def crossing(fraction):
            for moment, value in trace:
                if value >= fraction * final:
                    return moment
            return float('nan')

        print('{:>9} {:>12.1f} {:>10.2f} {:>12.0f} {:>9.0f}'.format(
            duration, final, final / PX_PER_DEG, crossing(0.05), crossing(0.95)))
        link.send(arm_control.HOME_POSE, 1200)
        time.sleep(2.0)
    print('-' * 56)
    print('If 95% tracks the commanded time, ArmJoints.time sets the duration.')


def test_hold(link, camera):
    def observe(reference, seconds, label):
        values = []
        inliers = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            result = vm.shift_of(reference, camera.grab())
            if result is not None:
                values.append(result.magnitude)
                inliers.append(result.inliers)
        if len(values) < 10:
            print('{:<44} too few frames'.format(label))
            return
        array = numpy.array(values)
        print('{:<44} n={:3d} | sd {:6.3f} px | range {:6.3f} px | inliers {:4.0f}'.format(
            label, len(values), array.std(), array.max() - array.min(),
            float(numpy.mean(inliers))))

    observe(vm.reference_of(camera.grab()), 3.0, 'A idle, long after last command')

    link.send(pose_with(JOINT1, 96), 500)
    time.sleep(1.5)
    observe(vm.reference_of(camera.grab()), 3.0, 'B 1.5-4.5 s after a 6 deg step')

    time.sleep(6.0)
    observe(vm.reference_of(camera.grab()), 3.0, 'C 10-13 s after the same step')

    for _ in range(6):
        link.send(pose_with(JOINT1, 96), 300)
        time.sleep(0.3)
    observe(vm.reference_of(camera.grab()), 3.0, 'D while re-sent every 300 ms')
    print()
    print('B is expected to be mildly elevated because its reference predates')
    print('less overlap. If all four are under a few tenths of a px, the arm is')
    print('holding position and not hunting.')


TESTS = {
    'coupling': test_coupling,
    'sweep': test_sweep,
    'step': test_step,
    'hold': test_hold,
}


def main():
    args = parse_args()

    print('This test MOVES THE ARM through its own scripted sequence.')
    print('Clear the workspace, support the robot, be ready to cut power.')
    print('The factory /joy_ctrl node also publishes /arm6_joints; put the')
    print('gamepad down so the two do not fight.')
    if not args.yes:
        if input('Type MOVE to continue: ') != 'MOVE':
            return 1

    client = arm_control.connect(args.host, args.port)
    link = arm_control.ArmLink(client, execute=True)
    camera = vm.Camera(client, roslibpy)
    try:
        if not camera.wait_for_stream():
            raise RuntimeError(
                'No frames on {}. Is the camera driver running?'.format(vm.COLOR_TOPIC))
        print('\nhoming to {}'.format(arm_control.HOME_POSE))
        home(link)
        print('running: {}\n'.format(args.test))
        TESTS[args.test](link, camera)
    finally:
        print('\nreturning home')
        home(link, wait_extra=1.0)
        camera.close()
        link.close()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
