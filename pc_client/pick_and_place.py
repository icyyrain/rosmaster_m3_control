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
import base64
import math
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy

import arm_control
import gripper_state
import kinematics
import reach_candy
import table_plane

# Grasp pose that forward kinematics found for a target on the desk at roughly
# 265 mm, from the aligned pose. Joints 2 and 3 carry the descent; joint 4 only
# needs about 8 degrees over the whole path, so it is left to the aligner.
GOAL_J2, GOAL_J3 = 18, 62

CREEP_STEPS = 5
ADAPTIVE_CREEP_STEPS = 12
ADAPTIVE_MAX_JOINT_STEP = 1
ALIGN_RETRIES = 2
ACQUIRE_ATTEMPTS = 10
MODEL_BIAS_LIMIT_MM = 25.0
ELONGATED_ASPECT = 1.8
CLIPPED_OFFSET_SMOOTHING = 0.5

# Where the gripper origin should end up: the object's mid-height plus this.
# The finger pads sit roughly this far below the origin, so this aims them at
# the object's waist. Two regimes used to live here, "top + 2 mm" below a 26 mm
# switch and "top + 8 mm" above it, and the red sweet measures 24.7-28.8 mm, so
# each run picked one at random on a millimetre of depth noise. Across 35
# labelled sweet grasps (calibrated FK at closing, minus measured top) the jaws
# held 9 of 11 times at 1.2 mm or more *below* the top, 4 of 17 within -1 to
# +3.3 mm and 1 of 7 at 4.6 mm or more above. One rule covers both objects: a
# 16 mm biscuit gives 18 mm, which held it, and a 27 mm sweet 23.5 mm.
GRIP_DEPTH_MM = 10.0
HEIGHT_FLOOR_MM = 18.0          # absolute floor, whatever the object

# Empirical lower bound: a 16 mm biscuit is the flattest object held so far.
# Lower objects remain untested and should still warn.
PROVEN_LOWEST_OBJECT_MM = 16.0
ALIGN_TOLERANCE_PX = 25.0
GATE_PX = 170.0
CLIPPED_OFFSET_SAFETY_PX = GATE_PX * 0.85

# Joint 5 rolls the jaws in the image without moving the camera or the gripper
# origin. Existing 0/90 degree snapshots put the fingers' long axis horizontal
# at 0 and vertical at 90, so image-line degrees and joint degrees share this
# zero/sign. Keep a bounded range until more oblique grasps are exercised.
WRIST_AXIS_AT_J5_90_DEG = 90.0
WRIST_IMAGE_DEG_PER_JOINT_DEG = 1.0
WRIST_BAND = (20, 160)
WRIST_TOLERANCE_DEG = 4.0
WRIST_MAX_MOVE_DEG = 60

# Retracting and lowering both follow this joint 2 / joint 3 ladder.
LIFT_LADDER = [(33, 53), (54, 40), (90, 20)]
PLACE_RELEASE_CLEARANCE_MM = 10.0
HELD_PIXEL_DRIFT_PX = 60.0

J1_BAND = (70, 110)
J4_BAND = (0, 70)

# Range band the FK descent path is valid over. It was solved at 265 mm, and
# this is not a workspace check: a proper one needs forward kinematics at run
# time, which this tool does not carry.
REACH_BAND_MM = (200.0, 340.0)
COLOUR_GUIDED_WORKING_AREA = (0.15, 0.85, 0.15, 0.90)

# What the jaws can actually take, from the aperture calibration.
GRIPPABLE_MM = (8.0, 60.0)

# No graspable object fills this much of the frame. A blob this size is a
# monitor, a shadow or a sleeve, and accepting one as the target is how a run
# ends up reporting a 198017 px "target" at the edge of the image.
MAX_PLAUSIBLE_AREA_PX = 60000
MAX_SCALED_TARGET_AREA_PX = 100000
RECORD_FPS = 10.0


def recording_path(spec):
    """Resolve --record to an absolute MP4 path."""
    if spec == 'auto':
        result = Path('recordings') / time.strftime('pick-%Y%m%d-%H%M%S.mp4')
    else:
        result = Path(spec)
        if not result.suffix:
            result = result.with_suffix('.mp4')
        elif result.suffix.lower() != '.mp4':
            raise ValueError('--record output must use the .mp4 extension')
    return result.resolve()


class VideoRecorder(object):
    """Write compressed colour messages to MP4 without blocking rosbridge."""

    def __init__(self, path, fps=RECORD_FPS):
        self.path = Path(path)
        self.fps = float(fps)
        self.frames = 0
        self.dropped = 0
        self.error = None
        self._queue = queue.Queue(maxsize=30)
        self._sentinel = object()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._write_loop, name='pick-video-writer', daemon=True)
        self._thread.start()
        print('recording colour video to {}'.format(self.path))

    def submit(self, payload_b64):
        if self._closed or self.error is not None:
            return
        try:
            self._queue.put_nowait(payload_b64)
        except queue.Full:
            self.dropped += 1

    def _write_loop(self):
        writer = None
        try:
            while True:
                payload = self._queue.get()
                if payload is self._sentinel:
                    break
                raw = base64.b64decode(payload)
                frame = cv2.imdecode(numpy.frombuffer(raw, numpy.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    self.dropped += 1
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(self.path), cv2.VideoWriter_fourcc(*'mp4v'),
                        self.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('OpenCV could not open the MP4 video writer')
                writer.write(frame)
                self.frames += 1
        except Exception as error:
            self.error = error
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        while self._thread.is_alive():
            try:
                self._queue.put(self._sentinel, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            print('WARNING: video writer did not finish flushing {}'.format(self.path))
        elif self.error is not None:
            print('WARNING: recording failed: {}'.format(self.error))
        else:
            suffix = ', {} frame(s) dropped'.format(self.dropped) if self.dropped else ''
            print('saved recording: {} ({} frames at {:.1f} fps{})'.format(
                self.path, self.frames, self.fps, suffix))


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
    parser.add_argument('--creep-controller', choices=('auto', 'legacy', 'adaptive'),
                        default='auto',
                        help='Final descent controller. adaptive steps to the '
                             'grasp height with online-calibrated forward '
                             'kinematics. legacy is the original fixed ladder; its '
                             'stop test rarely fires because depth fails at that '
                             'range. auto uses adaptive whenever preflight measured '
                             'the object, and legacy otherwise.')
    parser.add_argument('--no-orient-gripper', action='store_true',
                        help='Do not rotate joint 5 to follow the long axis of '
                             'an elongated target. Round targets never rotate it.')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Run anyway when a preflight check fails.')
    parser.add_argument('--grasp-offset-mm', type=float,
                        help='Where to stop the gripper origin relative to the '
                             'top of the object; negative is below the top. By '
                             'default aim at mid-height + {:.0f} mm. Raise it if '
                             'the jaws hit the table.'.format(GRIP_DEPTH_MM))
    parser.add_argument('--squeeze-mm', type=float, default=5.0,
                        help='How much narrower than the object to close. The '
                             'force knob; 5 mm held a sweet without marking it.')
    parser.add_argument('--keep-holding', action='store_true',
                        help='Stop after verifying the grasp instead of putting '
                             'the object back down.')
    parser.add_argument('--record', nargs='?', const='auto', metavar='PATH',
                        help='Record the colour camera to MP4. With no PATH, save '
                             'recordings/pick-YYYYMMDD-HHMMSS.mp4.')
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
        self.recorder = None
        if args.record is not None:
            try:
                self.recorder = VideoRecorder(recording_path(args.record))
            except ValueError as error:
                self.link.close()
                self.client.terminate()
                raise SystemExit(str(error))
        self.streams = reach_candy.Streams(
            self.client,
            colour_sink=self.recorder.submit if self.recorder is not None else None)
        self.rng = numpy.random.default_rng(31)
        self.pose = list(arm_control.HOME_POSE)
        self.last_px = None
        self.width_mm = args.width_mm
        self.acquired = False
        self.colour = None          # resolved once a frame is available
        self.colour_chroma = None
        self.object_top_mm = None   # filled in by preflight
        self.grasp_offset_mm = None
        self.stop_height_mm = HEIGHT_FLOOR_MM
        self.object_aspect = None
        self.lock_clipped_target = False
        self.clipped_target_edge = None
        self.target_geometry_initial = None
        self.target_offset_px = (0.0, 0.0)
        self.acquired_object = None

    def set_object_top(self, top_mm):
        """Derive the descent stop height from the object's own height."""
        self.object_top_mm = top_mm
        self.grasp_offset_mm = self.args.grasp_offset_mm
        if self.grasp_offset_mm is None:
            self.grasp_offset_mm = GRIP_DEPTH_MM - top_mm / 2.0
        self.stop_height_mm = max(HEIGHT_FLOOR_MM,
                                  top_mm + self.grasp_offset_mm)

    def close(self):
        self.streams.close()
        if self.recorder is not None:
            self.recorder.close()
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
            # After depth acquisition, the first colour-only look has no
            # motion-model prediction yet. Anchor it to the acquired target;
            # otherwise an unrelated red blob elsewhere in the frame can win.
            if (self.args.detector == 'table' and predicted is None
                    and self.acquired and self.last_px is not None):
                predicted = self.last_px
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
        # `last_px` is the corrected centre used by the servo. Identity gating
        # must follow the actually visible component, so remove the previous
        # correction before asking find_target which blob is nearby.
        predicted_observed = predicted
        if predicted is not None:
            predicted_observed = (
                predicted[0] - self.target_offset_px[0],
                predicted[1] - self.target_offset_px[1])
        found = reach_candy.find_target(
            bgr, self.args.a_star, self.args.min_area, predicted_observed,
            max_jump=GATE_PX, colour=self.colour)
        if found is None:
            return None
        cx, cy, area, rivals = found
        observed = (cx, cy)
        geometry = reach_candy.target_blob_geometry(
            bgr, self.colour, observed, self.args.a_star - 128)
        plausible_area = MAX_PLAUSIBLE_AREA_PX
        if (self.lock_clipped_target and geometry is not None
                and self.target_geometry_initial is not None
                and self.target_geometry_initial['short_px'] > 1.0):
            scale = geometry['short_px'] / self.target_geometry_initial['short_px']
            expected_area = self.target_geometry_initial['area_px'] * scale * scale
            plausible_area = min(MAX_SCALED_TARGET_AREA_PX,
                                 max(plausible_area, expected_area * 1.5))
        if area > plausible_area:
            return None
        bounds = geometry['bounds'] if geometry is not None else None
        clipped = ()
        if bounds is not None:
            x, y, width, height = bounds
            frame_height, frame_width = bgr.shape[:2]
            clipped = tuple(edge for edge, touches in (
                ('left', x <= 1), ('right', x + width >= frame_width - 1),
                ('top', y <= 1), ('bottom', y + height >= frame_height - 1))
                            if touches)
        if (self.lock_clipped_target and geometry is not None
                and self.target_geometry_initial is None and not clipped):
            self.target_geometry_initial = geometry

        centre_locked = False
        if (self.lock_clipped_target and predicted is not None
                and self.clipped_target_edge is None and clipped):
            for edge in ('bottom', 'top', 'right', 'left'):
                if edge in clipped:
                    self.clipped_target_edge = edge
                    break
        compensation = None
        if (self.lock_clipped_target and predicted is not None
                and self.clipped_target_edge is not None
                and self.target_geometry_initial is not None
                and geometry is not None):
            # Estimate how much of the long side has disappeared. The short
            # side supplies the current image scale; half of the missing long
            # side is exactly the shift from the visible centroid to the full
            # object's centre. This replaces the old fixed 100 px guess.
            compensation = reach_candy.clipped_centre_offset(
                self.target_geometry_initial, geometry,
                self.clipped_target_edge, self.target_offset_px,
                smoothing=CLIPPED_OFFSET_SMOOTHING,
                max_offset_px=CLIPPED_OFFSET_SAFETY_PX)
            if compensation is not None:
                self.target_offset_px = compensation['offset']
                cx += self.target_offset_px[0]
                cy += self.target_offset_px[1]
                centre_locked = compensation['magnitude_px'] > 0.5
        else:
            self.target_offset_px = (0.0, 0.0)
        return dict(cx=cx, cy=cy, area=area, rivals=rivals, width_mm=None,
                    range_mm=reach_candy.target_range_mm(
                        self.streams.depth(), observed),
                    observed_px=observed, bounds=bounds, clipped=clipped,
                    geometry=geometry, compensation=compensation,
                    centre_locked=centre_locked)

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
        self.acquired_object = chosen
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
        if matching:
            # A strong requested-colour match is a better discriminator than
            # the conservative central work-area heuristic. This lets a known
            # target that was set down near the lower edge be reacquired while
            # leaving geometry-only acquisition on the original central 60%.
            return table_plane.pick_graspable(
                matching, working_area=COLOUR_GUIDED_WORKING_AREA), total
        return table_plane.pick_graspable(objects), total

    def _colour_filter(self, objects, window=12):
        """Keep the objects whose pixels lean towards the requested colour."""
        bgr = self.streams.colour()
        if bgr is None or self.colour is None or not objects:
            return objects
        primary = reach_candy.find_target(
            bgr, self.args.a_star, self.args.min_area, colour=self.colour)
        if primary is not None:
            near_primary = [item for item in objects
                            if ((item['colour_px'][0] - primary[0]) ** 2
                                + (item['colour_px'][1] - primary[1]) ** 2) ** 0.5
                            <= 100.0]
            if near_primary:
                return near_primary
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
    compensation = state.get('compensation')
    if compensation is not None and compensation['magnitude_px'] > 0.5:
        note = '{}{}centre {:+.0f},{:+.0f}px ({:.0f}/{:.0f}px visible)'.format(
            note, ', ' if note else '',
            compensation['offset'][0], compensation['offset'][1],
            compensation['visible_long_px'], compensation['expected_long_px'])
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

    state = None
    for attempt in range(ACQUIRE_ATTEMPTS):
        state = session.look()
        if state is not None:
            break
        if attempt + 1 < ACQUIRE_ATTEMPTS:
            time.sleep(0.25)
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

        # Reuse the exact depth component that produced `state`. A second
        # random plane fit on the next frame occasionally returns no objects
        # and used to make an already-successful acquisition fail preflight.
        graspable = session.acquired_object
        if graspable is None:
            graspable, _ = session.pick_object()
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
            narrow = min(graspable['width_mm'], graspable['depth_mm'])
            wide = max(graspable['width_mm'], graspable['depth_mm'])
            session.object_aspect = wide / max(narrow, 1e-6)
            top = graspable['top_mm']
            check('grasp height calibrated',
                  top >= PROVEN_LOWEST_OBJECT_MM - 1.0,
                  '{:.1f} mm tall, grasp at {:.1f} mm; objects of {:.0f} to '
                  '29 mm have been held'.format(
                      top, session.stop_height_mm, PROVEN_LOWEST_OBJECT_MM),
                  fatal=False)

            # Two independent measurements of the same object that ought to
            # agree for anything roughly round. When they do not, the narrow
            # axis is probably an underestimate, and --auto-width would then
            # ask for too tight a grip: the sweet measured 21 mm across but
            # 27 mm tall, and gripping it as 21 mm gives a 15 mm aperture.
            if args.auto_width:
                check('width estimate consistent', abs(narrow - top) <= 0.5 * top,
                      'narrow axis {:.0f} mm, height {:.1f} mm, gripping as '
                      '{:.0f} mm; the mask only sees the cap of a rounded object '
                      'so the larger is used'.format(narrow, top, measured),
                      fatal=False)
            # Compare an INDEPENDENT colour detection against depth. Comparing
            # the chosen detector against itself passes by construction and
            # tells us nothing, which is worse than having no check.
            bgr = session.streams.colour()
            independent = (reach_candy.find_target(
                               bgr, args.a_star, args.min_area,
                               colour=session.colour)
                           if bgr is not None else None)
            if independent is None:
                check('colour can track it', False,
                      'depth can acquire this object but no colour fix exists, and '
                      'colour is the only tracker that works inside the depth dead '
                      'zone below 120 mm. A non-red object needs an appearance '
                      'tracker, which is not implemented.',
                      fatal=args.detector == 'table')
            else:
                dx = independent[0] - graspable['colour_px'][0]
                dy = independent[1] - graspable['colour_px'][1]
                check('detectors agree', abs(dx) < 40 and abs(dy) < 40,
                      'independent colour fix differs from depth by '
                      '({:+.0f}, {:+.0f}) px'.format(dx, dy), fatal=False)
                geometry = reach_candy.target_blob_geometry(
                    bgr, session.colour, independent[:2], args.a_star - 128)
                if geometry is not None:
                    colour_aspect = geometry['long_px'] / max(
                        geometry['short_px'], 1e-6)
                    if colour_aspect >= 1.5:
                        # Save the complete, high-altitude view. Waiting until
                        # align finishes can leave only a square-looking tail
                        # at the frame edge, which is too late to recover the
                        # original length or long-axis direction.
                        session.target_geometry_initial = geometry
                    if session.object_aspect is not None and session.object_aspect >= ELONGATED_ASPECT:
                        check('colour long axis', colour_aspect >= 1.5,
                              '{:.1f}:1 at {:.1f} deg in the complete view'.format(
                                  colour_aspect, geometry['angle_deg']),
                              fatal=False)
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


def creep_legacy(session, state):
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


def _update_model_bias(state, pose, previous):
    """Calibrate model height from a real plane measurement when available."""
    if state.get('height') is None:
        return previous
    observed = state['height'] - kinematics.gripper_height_mm(pose)
    if abs(observed) > MODEL_BIAS_LIMIT_MM:
        print('  WARNING: rejecting implausible FK height correction {:+.1f} mm'.format(
            observed))
        return previous
    return observed


def _look_adaptive(session, predicted, tries=3):
    """Retry only the camera read; never issue another command while blind."""
    for attempt in range(tries):
        fresh = session.look(predicted)
        if fresh is not None:
            return fresh
        if attempt + 1 < tries:
            time.sleep(0.2)
    return None


def orient_gripper(session, state):
    """At safe height, roll the open jaws parallel to an elongated target."""
    if (session.args.no_orient_gripper or not session.lock_clipped_target
            or session.object_aspect is None
            or session.object_aspect < ELONGATED_ASPECT):
        return state
    geometry = state.get('geometry')
    geometry_source = 'current view'
    if (geometry is None or geometry['short_px'] <= 1.0
            or geometry['long_px'] / geometry['short_px'] < 1.5):
        geometry = session.target_geometry_initial
        geometry_source = 'complete preflight view'
    if geometry is None or geometry['short_px'] <= 1.0:
        print('=== orient gripper ===')
        print('  no reliable colour long axis; leaving joint5 at {}'.format(
            session.pose[4]))
        return state
    colour_aspect = geometry['long_px'] / geometry['short_px']
    if colour_aspect < 1.5:
        print('=== orient gripper ===')
        print('  colour component is only {:.1f}:1; leaving joint5 at {}'.format(
            colour_aspect, session.pose[4]))
        return state

    object_angle = geometry['angle_deg']
    wrist_axis = (WRIST_AXIS_AT_J5_90_DEG
                  + WRIST_IMAGE_DEG_PER_JOINT_DEG
                  * (session.pose[4] - 90.0)) % 180.0
    image_delta = reach_candy.line_angle_delta(object_angle, wrist_axis)
    joint_delta = image_delta / WRIST_IMAGE_DEG_PER_JOINT_DEG
    joint_delta = max(-WRIST_MAX_MOVE_DEG,
                      min(WRIST_MAX_MOVE_DEG, joint_delta))
    target = int(round(max(WRIST_BAND[0],
                           min(WRIST_BAND[1], session.pose[4] + joint_delta))))
    remaining = reach_candy.line_angle_delta(
        object_angle,
        (WRIST_AXIS_AT_J5_90_DEG
         + WRIST_IMAGE_DEG_PER_JOINT_DEG * (target - 90.0)) % 180.0)

    print('=== orient gripper ===')
    print('  object long axis {:.1f} deg ({}), jaw axis {:.1f} deg at joint5 {}'.format(
        object_angle, geometry_source, wrist_axis, session.pose[4]))
    if abs(image_delta) <= WRIST_TOLERANCE_DEG:
        print('  already parallel (error {:+.1f} deg)'.format(image_delta))
        return state
    print('  joint5 {} -> {}; expected residual {:+.1f} deg'.format(
        session.pose[4], target, remaining))
    pose = list(session.pose)
    pose[4] = target
    session.send(pose, max(700, abs(target - session.pose[4]) * 35), 1.5)
    fresh = session.look(session.last_px)
    if fresh is None:
        print('  target became occluded after wrist rotation; restoring joint5')
        pose[4] = 90
        session.send(pose, 1200, 1.8)
        return None
    refreshed = fresh.get('geometry')
    if refreshed is not None:
        print('  target retained; measured long axis {:.1f} deg'.format(
            refreshed['angle_deg']))
    return fresh


def _recentre_adaptive(session, state):
    """Bring the target back under the fixed gripper pixel after a height step."""
    for _ in range(ALIGN_RETRIES):
        error = session.error_px(state)
        magnitude = (error[0] ** 2 + error[1] ** 2) ** 0.5
        if magnitude <= ALIGN_TOLERANCE_PX:
            break
        j1, j4 = session.aligned_step(state, 0.6, 4.0)
        if (j1, j4) == (session.pose[0], session.pose[3]):
            break
        pose = list(session.pose)
        pose[0], pose[3] = j1, j4
        predicted = reach_candy.predict_target(
            session.last_px, (j1 - session.pose[0], 0, 0,
                              j4 - session.pose[3]))
        session.send(pose, 500, 1.0)
        fresh = _look_adaptive(session, predicted)
        if fresh is None:
            print('     re-centre lost the target')
            return None
        state = fresh
        error = session.error_px(state)
        print('     re-centre {:<14} target {:>4.0f},{:<4.0f} |e| {:.0f}'.format(
            str(session.pose[:4]), state['cx'], state['cy'],
            (error[0] ** 2 + error[1] ** 2) ** 0.5))
    return state


def creep_adaptive(session, state):
    """Approach the grasp height using FK calibrated by the live table plane.

    The URDF gets the *change* in gripper height right, but its absolute table
    offset moved by 7-9 mm between real runs.  Calibrating that offset at the
    last pose where depth still works keeps the model useful inside the depth
    camera's near blind zone without trusting an old global constant.
    """
    print('=== creep (adaptive) ===')
    target = session.stop_height_mm
    if state.get('height') is None:
        print('  no measured height at the hand-off; refusing an uncalibrated descent')
        return None

    bias = _update_model_bias(state, session.pose, None)
    if bias is None:
        print('  could not calibrate FK height; refusing the low-profile descent')
        return None
    model = kinematics.gripper_height_mm(session.pose)
    print('  aiming for {:.1f} mm; measured {:.1f}, model {:.1f}, correction {:+.1f}'.format(
        target, state['height'], model, bias))

    for step in range(1, ADAPTIVE_CREEP_STEPS + 1):
        j1, j4 = session.aligned_step(state, 0.5, 2.0)
        proposal = list(session.pose)
        proposal[0], proposal[3] = j1, j4
        model_target = target - bias
        moved, modelled = kinematics.approach_height(
            proposal, model_target, max_step=ADAPTIVE_MAX_JOINT_STEP)
        corrected = modelled + bias
        if moved is None:
            print('  model cannot improve on {:.1f} mm towards {:.1f} mm'.format(
                corrected, target))
            return state if corrected <= target + 2.0 else None

        delta = tuple(moved[i] - session.pose[i] for i in range(4))
        predicted = reach_candy.predict_target(session.last_px, delta)
        session.send(moved, 700, 1.4)
        fresh = _look_adaptive(session, predicted)
        if fresh is None:
            print('  {:>2} {:<18} target lost; stopping before the grip'.format(
                step, str(session.pose[:4])))
            return None
        state = fresh
        bias = _update_model_bias(state, session.pose, bias)
        corrected = kinematics.gripper_height_mm(session.pose) + bias
        note = 'FK+cal {:.1f} mm'.format(corrected)
        if state['height'] is not None:
            note += ', measured {:.1f}'.format(state['height'])
        print(row(step, session.pose, state, note))

        state = _recentre_adaptive(session, state)
        if state is None:
            return None
        bias = _update_model_bias(state, session.pose, bias)
        corrected = kinematics.gripper_height_mm(session.pose) + bias
        if state['height'] is not None and state['height'] < target - 2.0:
            print('  measured {:.1f} mm, more than 2 mm below the guarded target; '
                  'stopping without closing'.format(state['height']))
            return None
        measured_arrived = (state['height'] is not None
                            and state['height'] <= target + 1.0)
        model_arrived = state['height'] is None and corrected <= target + 1.0
        if measured_arrived or model_arrived:
            source = 'measurement' if measured_arrived else 'calibrated model'
            print('  arrived at {:.1f} mm by {}'.format(
                state['height'] if measured_arrived else corrected, source))
            return state

    print('  adaptive creep exhausted {} guarded steps before reaching {:.1f} mm'.format(
        ADAPTIVE_CREEP_STEPS, target))
    return None


def creep(session, state):
    controller = session.args.creep_controller
    if controller == 'auto':
        # Without a measured top there is no grasp height to aim for, only the
        # floor, so fall back to the original ladder.
        controller = 'adaptive' if session.object_top_mm is not None else 'legacy'
    print('creep controller: {}, aiming for {:.1f} mm'.format(
        controller, session.stop_height_mm))
    if controller == 'legacy':
        return creep_legacy(session, state)
    return creep_adaptive(session, state)


def grip(session, state):
    # From here on verification needs the observed colour movement, not the
    # approach anchor used to keep an elongated object's physical centre.
    session.lock_clipped_target = False
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


def low_release_pose(grasp_pose, first_lift_rung=LIFT_LADDER[0],
                     clearance_mm=PLACE_RELEASE_CLEARANCE_MM):
    """Choose a point on the proven lift path just above the grasp pose."""
    start_height = kinematics.gripper_height_mm(grasp_pose)
    target_height = start_height + clearance_mm
    span = max(abs(first_lift_rung[0] - grasp_pose[1]),
               abs(first_lift_rung[1] - grasp_pose[2]), 1)
    best = (abs(clearance_mm), list(grasp_pose), start_height)
    seen = set()
    for sample in range(span * 4 + 1):
        fraction = sample / float(span * 4)
        pose = list(grasp_pose)
        pose[1] = int(round(grasp_pose[1]
                            + fraction * (first_lift_rung[0] - grasp_pose[1])))
        pose[2] = int(round(grasp_pose[2]
                            + fraction * (first_lift_rung[1] - grasp_pose[2])))
        rung = (pose[1], pose[2])
        if rung in seen:
            continue
        seen.add(rung)
        height = kinematics.gripper_height_mm(pose)
        score = abs(height - target_height)
        if score < best[0]:
            best = (score, pose, height)
    return best[1], best[2] - start_height


def place(session, state, grasp_pose, ladder=LIFT_LADDER):
    print('=== lower and release ===')
    descent = list(reversed(ladder))
    release_pose, release_clearance = low_release_pose(grasp_pose, ladder[0])
    release_rung = (release_pose[1], release_pose[2])
    if descent[-1] != release_rung:
        descent.append(release_rung)
    state, still_held = travel(session, state, descent, 'the descent')
    print('  low release point {}: about {:.1f} mm above the grasp height'.format(
        release_rung, release_clearance))
    pose = list(session.pose)
    pose[5] = gripper_state.OPEN_CMD
    session.send(pose, 1200, 2.5)
    fresh = session.look(session.last_px)
    if fresh:
        state = fresh
        print('  jaws open: target {:>4.0f},{:<4.0f} area {}'.format(
            state['cx'], state['cy'], state['area']))
    print('  retracting; a released object must now drift as predicted')
    state, still_attached = travel(session, state, ladder, 'the retraction')
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
        session.lock_clipped_target = (
            args.detector == 'table'
            and session.object_aspect is not None
            and session.object_aspect >= ELONGATED_ASPECT)
        if session.lock_clipped_target:
            print('tracking the full centre of the elongated target from its '
                  'measured visible length (aspect {:.1f}:1)'.format(
                      session.object_aspect))
        print()

        state = align(session)
        if state is None or not args.execute:
            return 0

        state = orient_gripper(session, state)
        if state is None:
            print('\nwrist orientation lost the target; returning home')
            session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                         arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
            print('home. final pose {}'.format(session.pose))
            return 1

        state = descend(session, state)
        if state is None:
            return 1
        state = creep(session, state)
        if state is None:
            print('\nadaptive descent stopped safely; not closing the jaws')
            session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                         arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
            print('home. final pose {}'.format(session.pose))
            return 1
        state = grip(session, state)
        grasp_pose = list(session.pose)
        grasp_px = ((state['cx'], state['cy']) if state is not None else None)
        elongated = (session.object_aspect is not None
                     and session.object_aspect >= ELONGATED_ASPECT)
        lift_ladder = LIFT_LADDER

        print('=== lift and verify ===')
        state, held = travel(session, state, lift_ladder, 'the lift')
        # Gate this one too. Ungated, it once returned a 198017 px region at
        # (189, 233) after the object had left the frame, which is the dark
        # monitor rather than a 34 mm biscuit, and reported it as the target.
        final = session.look(session.last_px)
        offset = None
        drift = None
        if final:
            offset = ((final['cx'] - reach_candy.GRIPPER_PX[0]) ** 2
                      + (final['cy'] - reach_candy.GRIPPER_PX[1]) ** 2) ** 0.5
            if grasp_px is not None:
                drift = ((final['cx'] - grasp_px[0]) ** 2
                         + (final['cy'] - grasp_px[1]) ** 2) ** 0.5
            print('  fresh look: target {:.0f},{:.0f}, {:.0f} px from the gripper '
                  'pixel, {:.0f} px from its closed-jaw position, area {}'.format(
                      final['cx'], final['cy'], offset,
                      drift if drift is not None else float('nan'), final['area']))
        else:
            print('  fresh look: target not visible')
        # A clipped long package has a visible centroid far from its physical
        # centre even while correctly held. For it, invariance relative to the
        # closed-jaw frame is the meaningful test. Round/full-view objects keep
        # the old absolute gripper-pixel gate.
        if elongated:
            # The complete presentation-height lift intentionally carries a
            # long package out of the eye-in-hand camera's useful field of
            # view. A tiny visible corner is no more conclusive than no blob:
            # centroid drift while clipping makes both tests unreliable.
            # Returning along the known lift path is safer than opening at
            # HOME.
            verdict = None
        else:
            verdict = held and offset is not None and offset < 150
        if verdict is None:
            print('  VERDICT: inconclusive (elongated target left the frame)')
        else:
            print('  VERDICT: {}'.format('HELD' if verdict else 'not held'))

        released = False
        if verdict is False:
            print('\ngrasp not confirmed, so not attempting to place it')
            print('returning HOME with joint6 unchanged; the jaws will not open')
        elif args.keep_holding:
            print('\n--keep-holding given; leaving the object in the jaws')
            print('The servo holds with stall torque, so do not leave it long.')
        else:
            if verdict is None:
                print('\ntarget visibility is inconclusive; returning along the known '
                      'lift path and releasing only at the table')
            place(session, state or final, grasp_pose, lift_ladder)
            released = True

        home_pose = list(arm_control.HOME_POSE)
        if not released:
            home_pose[5] = session.pose[5]
        session.send(home_pose, arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
        print('\nhome. final pose {}'.format(session.pose))
        return 0
    finally:
        session.close()


if __name__ == '__main__':
    sys.exit(main())
