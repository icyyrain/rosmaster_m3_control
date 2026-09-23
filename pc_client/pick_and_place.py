#!/usr/bin/env python3

"""Pick a red object off the table and put it back, using vision only.

No learning, no joint feedback, no inverse kinematics at run time. Five phases,
each of which exists because a simpler version of it failed on hardware. The
reasoning and the measurements behind every constant are in
docs/MEASUREMENTS.md.

  align    Turn the arm until the target sits under the gripper's fixed image
           pixel, driving joints 1 and 4 through the measured image Jacobian.
           Pointing only; it never lowers the arm.

  descend  Interpolate joints 2 and 3 along a forward-kinematics path towards
           the grasp pose, while joints 1 and 4 keep the target under the
           gripper pixel. Image-space servoing alone cannot do this: joint 2
           changes the target's range by about 1.4 mm per degree, so closing a
           187 mm gap would need 134 degrees and the travel does not exist.
           Height above the fitted table plane is the stopping signal, because
           the FK goal is computed from a target estimate that sits about 20 mm
           too high.

  creep    Half-sized steps for the last few millimetres. Depth stops being
           usable below roughly 120 mm, so this phase ends up open loop on
           range and relies on the image alignment.

  grip     Close to the aperture that suits the object's width. The servos have
           no force or current feedback, so a command of 180 pushes with stall
           torque and flattens soft things; grip force is set purely by where
           the command stops. See gripper_state.grip_command.

  verify   Lift, and check the target refused to move. The Jacobian predicts
           how far a stationary object's image would shift, so a target that
           stays put is being held. The dark-area grasp signal is not used: it
           does not survive a change of pose.

Dry-run by default: it prints the phases it would run and the first alignment
step, and touches nothing.
"""

import argparse
import math
import sys
import time

import numpy

import arm_control
import gripper_state
import reach_candy
import table_plane

# Grasp pose that forward kinematics found for a target on the desk at roughly
# 265 mm, from the aligned pose. Joints 2 and 3 carry the descent; joint 4 only
# needs about 8 degrees over the whole path, so it is left to the aligner.
GOAL_J2, GOAL_J3 = 18, 62

DESCENT_STEPS = 20
CREEP_STEPS = 5
HEIGHT_FLOOR_MM = 30.0
ALIGN_TOLERANCE_PX = 25.0
GATE_PX = 170.0

# Retracting and lowering both follow this joint 2 / joint 3 ladder.
LIFT_LADDER = [(33, 53), (54, 40), (90, 20)]

J1_BAND = (70, 110)
J4_BAND = (0, 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pick a red object off the table and put it back. Dry-run by default.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--a-star', type=int, default=140,
                        help='Lab a* threshold for "red". 128 is neutral.')
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--width-mm', type=float, default=28.0,
                        help='Object width, which sets the grip aperture.')
    parser.add_argument('--squeeze-mm', type=float, default=5.0,
                        help='How much narrower than the object to close. The '
                             'force knob; 5 mm held a sweet without marking it.')
    parser.add_argument('--keep-holding', action='store_true',
                        help='Stop after verifying the grasp instead of putting '
                             'the object back down.')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move. Without this nothing is sent.')
    return parser.parse_args()


class Session(object):
    """Shared connection, camera streams and detector state for one run."""

    def __init__(self, args):
        self.args = args
        self.client = arm_control.connect(args.host, args.port)
        self.link = arm_control.ArmLink(self.client, args.execute)
        self.streams = reach_candy.Streams(self.client)
        self.rng = numpy.random.default_rng(31)
        self.pose = list(arm_control.HOME_POSE)
        self.last_px = None

    def close(self):
        self.streams.close()
        self.link.close()
        self.client.terminate()

    def send(self, pose, duration, settle):
        self.pose = arm_control.clamp_pose(pose)
        self.link.send(self.pose, duration)
        time.sleep(settle)

    def look(self, predicted=None):
        """Detect the target and measure what depth still supports.

        Returns a dict, or None when the target is not where it should be.
        """
        bgr = self.streams.colour()
        if bgr is None:
            return None
        found = reach_candy.find_target(
            bgr, self.args.a_star, self.args.min_area, predicted, max_jump=GATE_PX)
        if found is None:
            return None
        cx, cy, area, rivals = found
        self.last_px = (cx, cy)
        height, support = table_plane.gripper_height_retry(self.streams, self.rng)
        return dict(cx=cx, cy=cy, area=area, rivals=rivals,
                    height=height, support=support,
                    range_mm=reach_candy.target_range_mm(self.streams.depth(), (cx, cy)))

    def error_px(self, state):
        return (state['cx'] - reach_candy.GRIPPER_PX[0],
                state['cy'] - reach_candy.GRIPPER_PX[1])

    def aligned_step(self, state, gain, max_step):
        """Joint 1 and 4 corrections that reduce the current pixel error."""
        delta = reach_candy.solve_align(self.error_px(state), gain, max_step)
        j1 = max(J1_BAND[0], min(J1_BAND[1], int(round(self.pose[0] + delta[0]))))
        j4 = max(J4_BAND[0], min(J4_BAND[1], int(round(self.pose[3] + delta[1]))))
        return j1, j4


def row(step, pose, state, note=''):
    return '  {:>2} {:<18} height {:>6} range {:>5} target {:>4.0f},{:<4.0f} {}'.format(
        step, str(pose[:4]),
        '{:.1f}'.format(state['height']) if state['height'] else '--',
        '{:.0f}'.format(state['range_mm']) if state['range_mm'] else '--',
        state['cx'], state['cy'], note)


def measure_object(session, state):
    """Report the object size, so --width-mm can be set from evidence.

    Height comes from the fitted table plane and is trustworthy: it read 26.4 mm
    for a sweet measured at 28 mm.

    Apparent width is the colour blob equated to a circle, and it is a LOWER
    BOUND, not a measurement. The a* mask only catches the part of the object
    that is red enough, so the same sweet came out at 18.2 mm. Feeding that to
    grip_command would ask for a 13 mm aperture and crush it. It is printed to
    sanity-check the detection, not to set the grip.
    """
    height = table_plane.target_height(
        session.streams.depth(), session.rng, (state['cx'], state['cy']))
    width_mm = None
    if state['range_mm']:
        diameter_px = math.sqrt(4.0 * state['area'] / math.pi)
        width_mm = diameter_px * state['range_mm'] / table_plane.FX_COLOUR
    print('  object: height above table {}, red blob at least {} wide'.format(
        '{:.1f} mm'.format(height) if height else 'unknown',
        '{:.1f} mm'.format(width_mm) if width_mm else 'unknown'))
    print('  gripping as {:.0f} mm wide (from --width-mm). The blob width is a'.format(
        session.args.width_mm))
    print('  lower bound only, so do not copy it into --width-mm.')


def align(session):
    print('=== align ===')
    state = session.look()
    if state is None:
        print('  target not visible; nothing to do')
        return None
    measure_object(session, state)
    for step in range(1, 16):
        error = session.error_px(state)
        magnitude = (error[0] ** 2 + error[1] ** 2) ** 0.5
        print('  {:>2} target {:>4.0f},{:<4.0f} error {:>5.0f},{:<5.0f} |e| {:5.0f} '
              'range {}'.format(step, state['cx'], state['cy'], error[0], error[1],
                                magnitude,
                                '{:.0f}'.format(state['range_mm'])
                                if state['range_mm'] else '--'))
        if magnitude <= ALIGN_TOLERANCE_PX:
            print('  aligned')
            return state
        if not session.args.execute:
            print('  dry run: one step shown, add --execute to continue')
            return None
        j1, j4 = session.aligned_step(state, 0.35, 3.0)
        predicted = reach_candy.predict_target(
            session.last_px, (j1 - session.pose[0], 0, 0, j4 - session.pose[3]))
        pose = list(session.pose)
        pose[0], pose[3] = j1, j4
        session.send(pose, 600, 1.2)
        state = session.look(predicted)
        if state is None:
            print('  lost the target while aligning')
            return None
    print('  did not converge')
    return None


def descend(session, state):
    print('=== descend ===')
    start = list(session.pose)
    lost = 0
    for step in range(1, DESCENT_STEPS + 1):
        j1, j4 = session.aligned_step(state, 0.6, 4.0)
        fraction = step / float(DESCENT_STEPS)
        j2 = int(round(start[1] + fraction * (GOAL_J2 - start[1])))
        j3 = int(round(start[2] + fraction * (GOAL_J3 - start[2])))
        delta = (j1 - session.pose[0], j2 - session.pose[1],
                 j3 - session.pose[2], j4 - session.pose[3])
        predicted = reach_candy.predict_target(session.last_px, delta)
        session.send([j1, j2, j3, j4, session.pose[4], session.pose[5]], 900, 1.6)

        fresh = session.look(predicted)
        if fresh is None:
            lost += 1
            print('  {:>2} {:<18} target lost x{}'.format(step, str(session.pose[:4]), lost))
            if lost >= 3:
                print('  giving up: lost three times in a row')
                return None
            continue
        lost = 0
        state = fresh
        print(row(step, session.pose, state))
        if state['height'] is not None and state['height'] <= HEIGHT_FLOOR_MM:
            print('  stopping at the {:.0f} mm height floor'.format(HEIGHT_FLOOR_MM))
            break
    return state


def creep(session, state):
    print('=== creep ===')
    for step in range(1, CREEP_STEPS + 1):
        j1, j4 = session.aligned_step(state, 0.5, 2.0)
        j2 = max(0, session.pose[1] - 3)
        j3 = min(180, session.pose[2] + 2)
        delta = (j1 - session.pose[0], j2 - session.pose[1],
                 j3 - session.pose[2], j4 - session.pose[3])
        predicted = reach_candy.predict_target(session.last_px, delta)
        session.send([j1, j2, j3, j4, session.pose[4], session.pose[5]], 800, 1.6)

        fresh = session.look(predicted)
        if fresh is None:
            print('  {:>2} {:<18} target lost; stopping here'.format(
                step, str(session.pose[:4])))
            return state
        state = fresh
        print(row(step, session.pose, state))
        if state['range_mm'] and state['range_mm'] <= table_plane.CAM_TO_GRIPPER_MM:
            print('  target reached the gripper distance')
            break
    return state


def grip(session, state):
    command = gripper_state.grip_command(session.args.width_mm, session.args.squeeze_mm)
    print('=== grip ===')
    print('  object {:.0f} mm, squeeze {:.0f} mm -> aperture {:.1f} mm, joint6 {}'.format(
        session.args.width_mm, session.args.squeeze_mm,
        gripper_state.gap_mm(command), command))
    print('  (120 would leave {:.0f} mm and hold nothing; 180 crushes)'.format(
        gripper_state.gap_mm(120)))
    pose = list(session.pose)
    pose[5] = command
    session.send(pose, 1500, 2.5)
    fresh = session.look(session.last_px)
    if fresh:
        state = fresh
        print('  target {:>4.0f},{:<4.0f} area {}'.format(
            state['cx'], state['cy'], state['area']))
    return state


def travel(session, state, ladder, label):
    """Walk joints 2 and 3 along the ladder, reporting how far the target moved.

    A target that moves far less than the Jacobian predicts for a stationary
    object is attached to the gripper. Returns (state, held).
    """
    print('  {:<18} {:>12} {:>14} {:>14}'.format(
        'pose j1..j4', 'target px', 'if stationary', 'moved vs that'))
    held = True
    # Skip a leading rung the arm is already standing on: it commands nothing
    # and prints a meaningless "0 vs 0" row.
    ladder = [rung for rung in ladder
              if rung != (session.pose[1], session.pose[2])] or ladder
    for j2, j3 in ladder:
        previous = session.last_px
        delta = (0, j2 - session.pose[1], j3 - session.pose[2], 0)
        predicted = reach_candy.predict_target(previous, delta)
        pose = list(session.pose)
        pose[1], pose[2] = j2, j3
        session.send(pose, 1100, 1.9)

        fresh = session.look(previous)
        if fresh is None:
            print('  {:<18} {:>12} {:>7.0f},{:<6.0f} {:>14}'.format(
                str(session.pose[:4]), 'not found', predicted[0], predicted[1], '--'))
            held = False
            session.last_px = predicted
            continue
        moved = ((fresh['cx'] - previous[0]) ** 2 + (fresh['cy'] - previous[1]) ** 2) ** 0.5
        expected = ((predicted[0] - previous[0]) ** 2
                    + (predicted[1] - previous[1]) ** 2) ** 0.5
        print('  {:<18} {:>5.0f},{:<6.0f} {:>7.0f},{:<6.0f} {:>6.0f} vs {:<6.0f}'.format(
            str(session.pose[:4]), fresh['cx'], fresh['cy'],
            predicted[0], predicted[1], moved, expected))
        if expected > 60 and moved > 0.4 * expected:
            held = False
        state = fresh
    print('  -> {} during {}'.format('HELD' if held else 'not held', label))
    return state, held


def place(session, state):
    print('=== lower and release ===')
    state, still_held = travel(session, state, list(reversed(LIFT_LADDER)), 'the descent')
    pose = list(session.pose)
    pose[5] = gripper_state.OPEN_CMD
    session.send(pose, 1200, 2.5)
    fresh = session.look(session.last_px)
    if fresh:
        state = fresh
        print('  jaws open: target {:>4.0f},{:<4.0f} area {}'.format(
            state['cx'], state['cy'], state['area']))
    print('  retracting; a released object must now drift as predicted')
    state, still_attached = travel(session, state, LIFT_LADDER, 'the retraction')
    print('  -> release {}'.format(
        'confirmed' if not still_attached else 'NOT confirmed: it came back up with us'))
    return state


def main():
    args = parse_args()
    if args.execute:
        print('This runs a full pick sequence on the real arm, down to the table.')
        print('Clear the workspace and be ready to cut power.')
        if not arm_control.confirm_execute():
            raise SystemExit('Cancelled.')

    session = Session(args)
    try:
        if not session.streams.wait():
            raise RuntimeError('No camera frames. Is the camera driver running?')
        print('mode: {}\n'.format('EXECUTE' if args.execute else 'DRY-RUN'))
        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.5 if args.execute else 0.1)

        state = align(session)
        if state is None or not args.execute:
            return 0

        state = descend(session, state)
        if state is None:
            return 1
        state = creep(session, state)
        state = grip(session, state)

        print('=== lift and verify ===')
        state, held = travel(session, state, LIFT_LADDER, 'the lift')
        final = session.look()
        offset = None
        if final:
            offset = ((final['cx'] - reach_candy.GRIPPER_PX[0]) ** 2
                      + (final['cy'] - reach_candy.GRIPPER_PX[1]) ** 2) ** 0.5
            print('  fresh look: target {:.0f},{:.0f}, {:.0f} px from the gripper '
                  'pixel, area {}'.format(final['cx'], final['cy'], offset, final['area']))
        else:
            print('  fresh look: target not visible')
        verdict = held and offset is not None and offset < 150
        print('  VERDICT: {}'.format('HELD' if verdict else 'not held'))

        if not verdict:
            print('\nnot holding anything, so not attempting to place it')
        elif args.keep_holding:
            print('\n--keep-holding given; leaving the object in the jaws')
            print('The servo holds with stall torque, so do not leave it long.')
        else:
            place(session, state or final)

        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
        print('\nhome. final pose {}'.format(session.pose))
        return 0
    finally:
        session.close()


if __name__ == '__main__':
    sys.exit(main())
