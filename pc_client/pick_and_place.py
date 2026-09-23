#!/usr/bin/env python3

"""Pick a coloured object off the table and put it back, using vision only.

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

Before any of that, `preflight` checks what it can check while standing
still, and refuses to move if something fails. It is cheaper to find out that
the object is out of range now than with the arm half way down.

Detection is split, and geometry forces the split. `red` thresholds the Lab a*
channel; it is precise and works right up to contact, but only on red things,
finding nothing at all on a blue biscuit wrapper. `table` fits the table plane
and takes whatever stands proud of it, which works on any colour and, unlike
the colour blob, measures real size: 28 mm for a sweet the colour blob called
18 mm.

But `table` cannot track. Depth stops being usable below about 120 mm, and the
gripper sits a fixed 111.9 mm from the camera, so a held object is permanently
inside the depth dead zone. A run that tried to track with depth lost the
target at the second creep step and could not verify the grasp at all. So
`table` acquires and measures, then hands off to colour, and preflight refuses
the run if colour cannot see the object, because then nothing can track it
through the grasp.

Dry-run by default: it runs preflight, prints the first alignment step, and
touches nothing.
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

CREEP_STEPS = 5

# Where the gripper origin should end up, relative to the top of the object.
# Anchored on the one grasp that worked: a 28 mm sweet was held when the creep
# finished with the gripper origin 35-38 mm above the table, so the origin sits
# roughly 8 mm above the object's top at a good grasp. One data point, so treat
# it as a starting value rather than a calibration.
#
# It has to follow the object. Left fixed at the sweet's value, the descent
# stopped 38.4 mm up while trying to grasp a 15.8 mm biscuit bar and closed the
# jaws about 20 mm above it.
GRASP_OFFSET_MM = 8.0
HEIGHT_FLOOR_MM = 18.0          # absolute floor, whatever the object

# How short an object this path can actually pick up. Empirical, and only two
# points wide.
#
# An earlier version of this claimed a "reachable floor" of 38 mm, computed
# from the lowest gripper height the descent ever reported. That was wrong: the
# depth stream stops returning anything below about 120 mm of range, so the
# height reading disappears while the arm is still descending. 38 mm was where
# measurement stopped, not where motion stopped, and the check built on it
# refused a sweet that had already been picked up twice.
#
# What is actually known: a 28 mm sweet was grasped successfully twice, and a
# 15.9 mm biscuit bar failed twice, the creep running joint 2 out of travel
# with the jaws closing about 20 mm above it. The true limit is somewhere
# between, and finding it properly needs the FK goal re-solved per target,
# which means carrying forward kinematics at run time.
PROVEN_GRASP_HEIGHT_MM = 28.0
FAILED_GRASP_HEIGHT_MM = 15.9
ALIGN_TOLERANCE_PX = 25.0
GATE_PX = 170.0

# Retracting and lowering both follow this joint 2 / joint 3 ladder.
LIFT_LADDER = [(33, 53), (54, 40), (90, 20)]

J1_BAND = (70, 110)
J4_BAND = (0, 70)

# Range band the FK descent path is valid over. It was solved at 265 mm, and
# this is not a workspace check: a proper one needs forward kinematics at run
# time, which this tool does not carry.
REACH_BAND_MM = (200.0, 340.0)

# What the jaws can actually take, from the aperture calibration.
GRIPPABLE_MM = (8.0, 60.0)

# No graspable object fills this much of the frame. A blob this size is a
# monitor, a shadow or a sleeve, and accepting one as the target is how a run
# ends up reporting a 198017 px "target" at the edge of the image.
MAX_PLAUSIBLE_AREA_PX = 60000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pick a coloured object off the table and put it back. Dry-run by default.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--detector', choices=('red', 'table'), default='red',
                        help='red: Lab a* threshold throughout. table: acquire '
                             'and measure with the depth plane, which works on '
                             'any colour, then hand off to colour tracking, '
                             'which is the only thing that still sees the '
                             'object inside the depth dead zone.')
    parser.add_argument('--colour', default='red',
                        help='Which colour to chase: a name (red, blue, green, '
                             'yellow, orange, pink, cyan, purple, magenta, '
                             'red-sat), a literal Lab direction "dx,dy", or '
                             '"@x,y" to sample the colour off that pixel of the '
                             'object. Sampling is the robust choice for '
                             'anything that is not an idealised colour.')
    parser.add_argument('--a-star', type=int, default=140,
                        help='Chroma threshold on the Lab a* scale, where 128 '
                             'is neutral, so 140 means a chroma of 12. The same '
                             'number applies whichever colour is chosen.')
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--width-mm', type=float, default=28.0,
                        help='Object width, which sets the grip aperture.')
    parser.add_argument('--auto-width', action='store_true',
                        help='Take the grip width from the depth measurement '
                             'instead of --width-mm. Only the depth measurement '
                             'is trustworthy for this; the colour blob is a '
                             'lower bound and would ask for a crushing grip.')
    parser.add_argument('--descent-steps', type=int, default=20,
                        help='Waypoints in the descent. Fewer is quicker but '
                             'each one moves the image further, so the tracker '
                             'has more to predict and the plane fit sees more '
                             'motion blur.')
    parser.add_argument('--settle', type=float, default=1.0,
                        help='Multiplier on every settling wait. Below about '
                             '0.5 the arm is still moving when it is measured.')
    parser.add_argument('--fast', action='store_true',
                        help='Shorthand for --descent-steps 10 --settle 0.6. '
                             'Roughly halves the run at some cost in '
                             'monitoring quality.')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Run anyway when a preflight check fails.')
    parser.add_argument('--grasp-offset-mm', type=float, default=GRASP_OFFSET_MM,
                        help='Where to stop the gripper origin above the top of '
                             'the object. Anchored on one successful grasp, so '
                             'raise it if the jaws hit the table.')
    parser.add_argument('--squeeze-mm', type=float, default=5.0,
                        help='How much narrower than the object to close. The '
                             'force knob; 5 mm held a sweet without marking it.')
    parser.add_argument('--keep-holding', action='store_true',
                        help='Stop after verifying the grasp instead of putting '
                             'the object back down.')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move. Without this nothing is sent.')
    args = parser.parse_args()
    if args.fast:
        args.descent_steps = 10
        args.settle = 0.6
    return args


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
        self.width_mm = args.width_mm
        self.acquired = False
        self.colour = None          # resolved once a frame is available
        self.colour_chroma = None
        self.object_top_mm = None   # filled in by preflight
        self.stop_height_mm = HEIGHT_FLOOR_MM

    def set_object_top(self, top_mm):
        """Derive the descent stop height from the object's own height."""
        self.object_top_mm = top_mm
        self.stop_height_mm = max(HEIGHT_FLOOR_MM,
                                  top_mm + self.args.grasp_offset_mm)

    def close(self):
        self.streams.close()
        self.link.close()
        self.client.terminate()

    def send(self, pose, duration, settle):
        self.pose = arm_control.clamp_pose(pose)
        self.link.send(self.pose, duration)
        time.sleep(max(0.15, settle * self.args.settle))

    def resolve_colour(self):
        """Turn --colour into a direction, sampling from a frame if asked."""
        bgr = self.streams.colour()
        self.colour, self.colour_chroma = reach_candy.parse_colour(
            self.args.colour, bgr)
        return self.colour

    def wait_for_depth(self, timeout=8.0):
        deadline = time.time() + timeout
        while self.streams.depth() is None and time.time() < deadline:
            time.sleep(0.1)
        return self.streams.depth() is not None

    def look(self, predicted=None):
        """Detect the target and measure what depth still supports.

        Returns a dict, or None when the target is not where it should be. Both
        detectors are gated the same way: a candidate more than GATE_PX from
        where the Jacobian says the target should have moved is rejected rather
        than followed, because the alternative is steering from a bad fix.
        """
        # The table detector acquires and measures; colour tracks. That split
        # is forced by geometry, not taste. Depth stops being usable below
        # about 120 mm, and the gripper sits a fixed 111.9 mm from the camera,
        # so a held object is permanently inside the depth dead zone. A run
        # that tried to track with depth lost the target at the second creep
        # step and could not verify the grasp at all.
        if self.args.detector == 'table' and predicted is None and not self.acquired:
            found = self._look_table(predicted)
            if found is not None:
                self.acquired = True
        else:
            found = self._look_red(predicted)
        if found is None:
            return None
        found['height'], found['support'] = table_plane.gripper_height_retry(
            self.streams, self.rng)
        self.last_px = (found['cx'], found['cy'])
        return found

    def _look_red(self, predicted):
        bgr = self.streams.colour()
        if bgr is None:
            return None
        found = reach_candy.find_target(
            bgr, self.args.a_star, self.args.min_area, predicted,
            max_jump=GATE_PX, colour=self.colour)
        if found is None:
            return None
        cx, cy, area, rivals = found
        if area > MAX_PLAUSIBLE_AREA_PX:
            return None
        return dict(cx=cx, cy=cy, area=area, rivals=rivals, width_mm=None,
                    range_mm=reach_candy.target_range_mm(self.streams.depth(), (cx, cy)))

    def _look_table(self, predicted):
        """Depth for geometry, colour for identity.

        Depth alone chooses by size and centrality, which ignores what the
        object is: with a sweet and a biscuit bar both on the table it picked
        the sweet while --colour said blue, and preflight rightly refused the
        run. So candidates are first required to match the requested colour,
        and only then judged on size and position.
        """
        chosen, total = self.pick_object(predicted)
        if chosen is None:
            return None
        return dict(cx=chosen['colour_px'][0], cy=chosen['colour_px'][1],
                    area=chosen['area_px'], rivals=total - 1,
                    width_mm=table_plane.grip_width_mm(chosen),
                    range_mm=chosen['dist_mm'])

    def pick_object(self, predicted=None):
        """The depth object this run is about: colour first, then geometry.

        Preflight and the tracker must agree on which object they mean. They
        did not at first: preflight used geometry alone, so with a sweet and a
        biscuit on the table it measured the biscuit, 16 mm tall, while
        --colour red-sat tracked the 28 mm sweet. The stop height and
        --auto-width were then set from the wrong object.
        """
        objects = table_plane.find_objects(self.streams.depth(), self.rng)
        if not objects:
            return None, 0
        total = len(objects)
        if predicted is not None:
            objects = [o for o in objects
                       if ((o['colour_px'][0] - predicted[0]) ** 2
                           + (o['colour_px'][1] - predicted[1]) ** 2) ** 0.5 <= GATE_PX]
        matching = self._colour_filter(objects)
        return table_plane.pick_graspable(matching or objects), total

    def _colour_filter(self, objects, window=12):
        """Keep the objects whose pixels lean towards the requested colour."""
        bgr = self.streams.colour()
        if bgr is None or self.colour is None or not objects:
            return objects
        _, projection = reach_candy.chroma_mask(bgr, self.colour, 0)
        threshold = self.args.a_star - 128
        keep = []
        for item in objects:
            x, y = int(item['colour_px'][0]), int(item['colour_px'][1])
            patch = projection[max(0, y - window):y + window + 1,
                               max(0, x - window):x + window + 1]
            if patch.size and float(numpy.percentile(patch, 75)) > threshold:
                keep.append(item)
        return keep

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
    blob_mm = None
    if state['range_mm']:
        diameter_px = math.sqrt(4.0 * state['area'] / math.pi)
        blob_mm = diameter_px * state['range_mm'] / table_plane.FX_COLOUR
    width_text = ('depth says {:.1f} mm wide'.format(state['width_mm'])
                  if state.get('width_mm')
                  else 'depth width is in the preflight report above')
    print('  object: {} above the table, {}, colour blob {}'.format(
        '{:.1f} mm'.format(height) if height else 'height unknown',
        width_text,
        '{:.1f} mm'.format(blob_mm) if blob_mm else 'unknown'))
    print('  gripping as {:.0f} mm wide. Trust the depth figure, not the colour'.format(
        session.width_mm))
    print('  blob: the a* mask covers only the red enough part and reads low.')


def preflight(session):
    """Check what can be checked standing still. Returns True to proceed.

    Deliberately modest about what it can prove. A real workspace check needs
    forward kinematics at run time, which this tool does not carry, so
    reachability is approximated by the range band the descent path was solved
    for. That is stated rather than dressed up.
    """
    args = session.args
    results = []

    def check(name, ok, detail, fatal=True):
        results.append((name, ok, detail, fatal))

    check('colour stream', session.streams.colour() is not None, 'frames arriving')
    try:
        direction = session.resolve_colour()
        detail = 'direction ({:+.3f}, {:+.3f})'.format(*direction)
        if session.colour_chroma:
            detail += ', sampled chroma {:.0f}'.format(session.colour_chroma)
        check('colour resolved', True, detail)
    except ValueError as error:
        check('colour resolved', False, str(error))
    check('depth stream', session.wait_for_depth(), 'frames arriving')

    state = session.look()
    check('target detected', state is not None,
          'none found with the {} detector'.format(args.detector)
          if state is None else
          'at ({:.0f}, {:.0f}), area {}'.format(state['cx'], state['cy'], state['area']))

    if state is not None:
        check('no rival detections', state['rivals'] == 0,
              '{} other candidate(s); the largest or nearest wins'.format(state['rivals']),
              fatal=False)

        bgr = session.streams.colour()
        if bgr is not None and session.colour is not None:
            hit, background = reach_candy.colour_margin(
                bgr, session.colour, (state['cx'], state['cy']))
            if hit is not None:
                check('colour separates the target', hit - background > 3.0,
                      'target chroma {:.0f} against the frame 99.5th percentile '
                      'of {:.0f}, margin {:+.0f}'.format(hit, background, hit - background))

        check('table plane found', state['height'] is not None,
              'gripper {:.0f} mm above the table, {} inliers'.format(
                  state['height'], state['support'])
              if state['height'] is not None else 'plane fit failed')

        range_mm = state['range_mm']
        check('target range known', range_mm is not None,
              '{:.0f} mm'.format(range_mm) if range_mm else 'no depth at the target')

        if range_mm is not None:
            in_band = REACH_BAND_MM[0] <= range_mm <= REACH_BAND_MM[1]
            check('range within the solved band', in_band,
                  '{:.0f} mm against {:.0f}-{:.0f} mm; the FK descent path was '
                  'solved at 265 mm'.format(range_mm, *REACH_BAND_MM))

        graspable, _ = session.pick_object()
        objects = table_plane.find_objects(session.streams.depth(), session.rng)
        if graspable is not None:
            measured = table_plane.grip_width_mm(graspable)
            check('object fits the jaws',
                  GRIPPABLE_MM[0] <= measured <= GRIPPABLE_MM[1],
                  '{:.0f} mm across its narrow axis ({:.0f} x {:.0f}), jaws take '
                  '{:.0f}-{:.0f} mm'.format(
                      measured, graspable['width_mm'], graspable['depth_mm'],
                      *GRIPPABLE_MM))
            check('object stands off the table', graspable['top_mm'] >= 8.0,
                  '{:.1f} mm tall'.format(graspable['top_mm']))
            if args.auto_width:
                session.width_mm = measured
            session.set_object_top(graspable['top_mm'])
            top = graspable['top_mm']
            check('tall enough to grasp', top >= PROVEN_GRASP_HEIGHT_MM - 2.0,
                  '{:.1f} mm; {:.0f} mm has been picked up, {:.1f} mm failed twice '
                  'with the jaws closing above it'.format(
                      top, PROVEN_GRASP_HEIGHT_MM, FAILED_GRASP_HEIGHT_MM),
                  fatal=False)

            # Two independent measurements of the same object that ought to
            # agree for anything roughly round. When they do not, the narrow
            # axis is probably an underestimate, and --auto-width would then
            # ask for too tight a grip: the sweet measured 21 mm across but
            # 27 mm tall, and gripping it as 21 mm gives a 15 mm aperture.
            if args.auto_width:
                narrow = min(graspable['width_mm'], graspable['depth_mm'])
                check('width estimate consistent', abs(narrow - top) <= 0.5 * top,
                      'narrow axis {:.0f} mm, height {:.1f} mm, gripping as '
                      '{:.0f} mm; the mask only sees the cap of a rounded object '
                      'so the larger is used'.format(narrow, top, measured),
                      fatal=False)
            # Compare an INDEPENDENT colour detection against depth. Comparing
            # the chosen detector against itself passes by construction and
            # tells us nothing, which is worse than having no check.
            bgr = session.streams.colour()
            independent = (reach_candy.find_target(bgr, args.a_star, args.min_area)
                           if bgr is not None else None)
            if independent is None:
                check('colour can track it', False,
                      'depth can acquire this object but no colour fix exists, and '
                      'colour is the only tracker that works inside the depth dead '
                      'zone below 120 mm. A non-red object needs an appearance '
                      'tracker, which is not implemented.',
                      fatal=args.detector == 'table')
            else:
                (dx, dy), _ = table_plane.preflight_offset(
                    (independent[0], independent[1]), objects)
                check('detectors agree', abs(dx) < 40 and abs(dy) < 40,
                      'independent colour fix differs from depth by '
                      '({:+.0f}, {:+.0f}) px'.format(dx, dy), fatal=False)
        else:
            check('object fits the jaws', False,
                  'nothing graspable on the table plane', fatal=args.detector == 'table')

    command = gripper_state.grip_command(session.width_mm, args.squeeze_mm)
    check('grip aperture sane', gripper_state.OPEN_CMD < command < gripper_state.MAX_GRIP_CMD,
          'joint6 {} for a {:.0f} mm object, aperture {:.1f} mm'.format(
              command, session.width_mm, gripper_state.gap_mm(command)))

    print('=== preflight ===')
    failed = []
    for name, ok, detail, fatal in results:
        mark = 'pass' if ok else ('FAIL' if fatal else 'warn')
        print('  [{}] {:<28} {}'.format(mark, name, detail))
        if not ok and fatal:
            failed.append(name)
    if failed:
        print('  {} check(s) failed: {}'.format(len(failed), ', '.join(failed)))
        if args.skip_preflight:
            print('  --skip-preflight given, going anyway')
            return True
        return False
    print('  all clear')
    return True


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
    for step in range(1, session.args.descent_steps + 1):
        j1, j4 = session.aligned_step(state, 0.6, 4.0)
        fraction = step / float(session.args.descent_steps)
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
        if session.pose[1] <= 0:
            print('  joint 2 is out of travel at {:.1f} mm; the path cannot go '
                  'lower'.format(state['height'] if state['height'] else float('nan')))
            break
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
        if state['height'] is not None and state['height'] <= session.stop_height_mm:
            print('  reached {:.1f} mm, the height this object needs'.format(
                session.stop_height_mm))
            break
        if state['range_mm'] and state['range_mm'] <= table_plane.CAM_TO_GRIPPER_MM:
            print('  target reached the gripper distance')
            break
    return state


def grip(session, state):
    command = gripper_state.grip_command(session.width_mm, session.args.squeeze_mm)
    print('=== grip ===')
    print('  object {:.0f} mm, squeeze {:.0f} mm -> aperture {:.1f} mm, joint6 {}'.format(
        session.width_mm, session.args.squeeze_mm,
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
        print('mode: {}   detector: {}   colour: {}   steps: {}   settle x{:.2f}\n'.format(
            'EXECUTE' if args.execute else 'DRY-RUN', args.detector, args.colour,
            args.descent_steps, args.settle))
        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.5 if args.execute else 0.1)

        if not preflight(session):
            print('\nnot moving. Fix the failures, or pass --skip-preflight.')
            return 1
        print()

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
        # Gate this one too. Ungated, it once returned a 198017 px region at
        # (189, 233) after the object had left the frame, which is the dark
        # monitor rather than a 34 mm biscuit, and reported it as the target.
        final = session.look(session.last_px)
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
