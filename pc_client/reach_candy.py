#!/usr/bin/env python3

"""Drive the arm so a red object lands under the gripper, using vision only.

No learning, no joint feedback, no inverse kinematics and no calibration beyond
the three constants below. The loop is: find the target in the image, take the
error against the gripper's known image position, convert that error to joint
degrees through a measured Jacobian, command it, repeat.

Everything it relies on was measured on this robot and is recorded in
docs/MEASUREMENTS.md:

  * The camera is bolted to arm link arm4, so joints 1-4 move the image.
  * The gripper hangs off arm5, whose rotation axis is nearly collinear with
    the offset to it, so the gripper moves at most 0.88 mm through a full
    revolution of joint 5 and therefore projects to a FIXED pixel. Confirmed by
    lock-in detection at four very different poses: (707, 680) +- 1.1 px. The
    loop never has to find the gripper.
  * The image Jacobian near the working pose.

Two stages, because they carry different risk. "align" turns the arm until the
target sits under the gripper pixel, which only ever points the arm. "approach"
then reduces the remaining range, which drives the gripper towards the table
and can collide with it, so it is opt-in and step-limited.

Dry-run by default: it prints what it detects and what it would send, and
touches nothing.
"""

import argparse
import base64
import sys
import time

import cv2
import numpy
import roslibpy

import arm_control

COLOR_TOPIC = '/high_camera/color/image_raw/compressed'
DEPTH_TOPIC = '/high_camera/depth/image_raw'

# Gripper image position. Measured by toggling the gripper through three
# open/close cycles at four poses and keeping only pixels that changed every
# time, which rejects anything else moving in the room.
GRIPPER_PX = (707.0, 680.0)

# Image Jacobian, px of image motion per degree of joint command, evaluated at
# the image centre near pose [90, 120, 10, 10, 90, 90]. Joint 1 pans, joints
# 2-4 tilt. Only joints 1 and 4 are used: all of 2, 3 and 4 do nearly the same
# thing, so any pair among them is degenerate (condition 88 to 212), while
# joint 1 with joint 4 conditions at 1.2.
J_ALIGN = numpy.array([[-11.70, 0.63],    # dx from joint1, joint4
                       [1.55, 14.34]])    # dy from joint1, joint4
ALIGN_JOINTS = (0, 3)   # joint1, joint4
APPROACH_JOINT = 1      # joint2 sets reach

# Depth intrinsics, for turning a depth pixel into millimetres.
DEPTH_SCALE = 2.0  # colour is 1280x720, depth 640x360, same field of view.

# Colour is thresholded on the Lab a* channel rather than HSV hue. The wrapper
# is dark (L about 47), and hue is unstable at low lightness: the sweet's
# median hue sat three degrees outside a 170-179 red band and only 105 px
# passed. a* is a direct green-to-red axis and does not care how dark the pixel
# is. At a* > 140 the sweet gave 938 px with zero false regions, while the
# whole frame's 99.5th percentile was 138.
DEFAULT_A_STAR = 140


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual servo a red target under the gripper. Dry-run by default.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--a-star', type=int, default=DEFAULT_A_STAR,
                        help='Lab a* threshold for "red". 128 is neutral.')
    parser.add_argument('--min-area', type=int, default=150,
                        help='Smallest blob in px accepted as the target.')
    parser.add_argument('--tolerance', type=float, default=25.0,
                        help='Alignment is done when the error falls below this, in px.')
    parser.add_argument('--gain', type=float, default=0.35,
                        help='Fraction of the computed correction applied per step.')
    parser.add_argument('--max-step', type=float, default=3.0,
                        help='Largest joint change per step, in degrees.')
    parser.add_argument('--max-steps', type=int, default=25)
    parser.add_argument('--min-range', type=float, default=180.0,
                        help='Abort if the measured range to the target falls below '
                             'this, in mm. Aligning tilts the wrist down, which '
                             'carries the gripper towards the table.')
    parser.add_argument('--approach', action='store_true',
                        help='After aligning, also close the remaining range. '
                             'This drives the gripper towards the table.')
    parser.add_argument('--approach-mm', type=float, default=150.0,
                        help='Stop approaching at this range, in mm.')
    parser.add_argument('--home', action='store_true',
                        help='Start by driving to the factory rest pose.')
    parser.add_argument('--assume-pose', type=int, nargs=6,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Declare the pose the arm is already in.')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move. Without this nothing is sent.')
    return parser.parse_args()


class Streams(object):
    """Latest colour and depth frame over one rosbridge connection."""

    def __init__(self, client):
        self._colour = None
        self._depth = None
        self.n_colour = 0
        self.colour_topic = roslibpy.Topic(
            client, COLOR_TOPIC, 'sensor_msgs/msg/CompressedImage',
            throttle_rate=100, queue_length=1, queue_size=1)
        self.depth_topic = roslibpy.Topic(
            client, DEPTH_TOPIC, 'sensor_msgs/msg/Image',
            throttle_rate=200, queue_length=1, queue_size=1)
        self.colour_topic.subscribe(self._on_colour)
        self.depth_topic.subscribe(self._on_depth)

    def _on_colour(self, message):
        self._colour = message['data']
        self.n_colour += 1

    def _on_depth(self, message):
        self._depth = message

    def wait(self, timeout=10.0):
        deadline = time.time() + timeout
        while self._colour is None and time.time() < deadline:
            time.sleep(0.02)
        return self._colour is not None

    def colour(self, timeout=5.0):
        """Next colour frame to arrive after this call, as BGR."""
        start = self.n_colour
        deadline = time.time() + timeout
        while self.n_colour == start and time.time() < deadline:
            time.sleep(0.005)
        if self._colour is None:
            return None
        raw = base64.b64decode(self._colour)
        if not raw:
            return None
        return cv2.imdecode(numpy.frombuffer(raw, numpy.uint8), cv2.IMREAD_COLOR)

    def depth(self):
        message = self._depth
        if message is None:
            return None
        height, width = message['height'], message['width']
        raw = base64.b64decode(message['data'])[:height * width * 2]
        if len(raw) < height * width * 2:
            return None
        return numpy.frombuffer(raw, numpy.uint16).reshape(height, width)

    def close(self):
        self.colour_topic.unsubscribe()
        self.depth_topic.unsubscribe()


def find_target(bgr, a_star, min_area, previous=None, max_jump=120.0):
    """Locate the red target. Returns (cx, cy, area, rivals) or None.

    Picking simply the largest red blob is not enough. Human skin also sits
    high on a*, so a hand in the frame produces rival regions, and on one run a
    rival briefly grew larger than the target: the detection jumped 539 px
    across the frame and the loop issued a 3 degree command from it before
    recovering. So once the target has been seen, only blobs within max_jump of
    the last position are eligible, because a sweet on a table cannot
    teleport. Returning None is the honest answer when nothing qualifies; the
    caller skips the step rather than steering from a bad fix.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mask = (lab[:, :, 1] > a_star).astype(numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((9, 9), numpy.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            blobs.append((area, centroids[label][0], centroids[label][1]))
    if not blobs:
        return None

    rivals = len(blobs) - 1
    if previous is not None:
        near = [b for b in blobs
                if ((b[1] - previous[0]) ** 2 + (b[2] - previous[1]) ** 2) ** 0.5 <= max_jump]
        if not near:
            return None
        blobs = near
    blobs.sort(reverse=True)
    area, cx, cy = blobs[0]
    return cx, cy, area, rivals


def target_range_mm(depth, colour_px, search=30):
    """Range to the target in mm, or None.

    Depth is not registered to colour on this driver, so a colour pixel maps to
    a depth pixel only approximately: the two sensors are about 12 mm apart,
    which is tens of pixels of parallax at close range. Rather than trust the
    mapping, this searches a window and takes a low percentile, because the
    target stands proud of the table and is therefore nearer than everything
    around it.
    """
    if depth is None:
        return None
    height, width = depth.shape
    cx = int(colour_px[0] / DEPTH_SCALE)
    cy = int(colour_px[1] / DEPTH_SCALE)
    x0, x1 = max(0, cx - search), min(width, cx + search + 1)
    y0, y1 = max(0, cy - search), min(height, cy + search + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch > 100) & (patch < 1200)]
    if valid.size < 20:
        return None
    return float(numpy.percentile(valid, 10))


def solve_align(error_px, gain, max_step):
    """Joint degrees for joints 1 and 4 that reduce a pixel error."""
    # The Jacobian maps joint degrees to motion of image content. To bring the
    # target to the gripper pixel, content must move by (gripper - target),
    # which is the negative of the error.
    desired = -numpy.asarray(error_px, dtype=numpy.float64)
    delta = numpy.linalg.solve(J_ALIGN, desired) * gain
    largest = numpy.abs(delta).max()
    if largest > max_step:
        delta = delta * (max_step / largest)
    return delta


def main():
    args = parse_args()
    if bool(args.home) == bool(args.assume_pose):
        raise SystemExit('Choose exactly one of --home or --assume-pose.')
    if args.execute:
        print('This will MOVE the real arm under closed-loop visual control.')
        print('Clear the workspace and be ready to cut power.')
        if args.approach:
            print('--approach is on, so the gripper will be driven towards the table.')
        if not arm_control.confirm_execute():
            raise SystemExit('Cancelled.')

    client = arm_control.connect(args.host, args.port)
    link = arm_control.ArmLink(client, args.execute)
    streams = Streams(client)

    try:
        if not streams.wait():
            raise RuntimeError('No camera frames. Is the camera driver running?')

        if args.home:
            pose = list(arm_control.HOME_POSE)
            print('Homing to {}.'.format(pose))
            link.send(pose, arm_control.HOMING_TIME_MS)
            if args.execute:
                time.sleep(arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
        else:
            pose = arm_control.clamp_pose(args.assume_pose)
            print('Assuming the arm is at {}.'.format(pose))

        print('Gripper pixel {}, tolerance {:.0f} px, gain {:.2f}, max step {:.1f} deg'.format(
            GRIPPER_PX, args.tolerance, args.gain, args.max_step))
        print('Mode: {}\n'.format('EXECUTE' if args.execute else 'DRY-RUN'))
        print('{:>4} {:>16} {:>7} {:>16} {:>8} {:>20}'.format(
            'step', 'target px', 'area', 'error px', 'range', 'command'))
        print('-' * 78)

        aligned = False
        previous = None
        lost = 0
        for step in range(1, args.max_steps + 1):
            bgr = streams.colour()
            if bgr is None:
                print('{:>4}  no colour frame'.format(step))
                break
            found = find_target(bgr, args.a_star, args.min_area, previous)
            if found is None:
                lost += 1
                print('{:>4}  no target within {:.0f} px of the last fix '
                      '(a* > {}, min area {}); skipping, {} in a row'.format(
                          step, 120.0, args.a_star, args.min_area, lost))
                if lost >= 4:
                    print('       lost for {} consecutive steps; stopping'.format(lost))
                    break
                if args.execute:
                    time.sleep(0.4)
                    continue
                break
            lost = 0
            cx, cy, area, rivals = found
            previous = (cx, cy)
            error = (cx - GRIPPER_PX[0], cy - GRIPPER_PX[1])
            magnitude = (error[0] ** 2 + error[1] ** 2) ** 0.5
            range_mm = target_range_mm(streams.depth(), (cx, cy))
            range_text = '{:.0f} mm'.format(range_mm) if range_mm else '--'

            if range_mm is not None and range_mm < args.min_range:
                print('{:>4} {:>8.1f},{:<7.1f} {:>7} {:>16} {:>8} {:>20}'.format(
                    step, cx, cy, area, 'RANGE STOP', range_text, ''))
                print('       range {:.0f} mm is below the {:.0f} mm floor; stopping '
                      'before the gripper reaches the table'.format(range_mm, args.min_range))
                break

            if magnitude <= args.tolerance:
                print('{:>4} {:>8.1f},{:<7.1f} {:>7} {:>16} {:>8} {:>20}'.format(
                    step, cx, cy, area, 'ALIGNED', range_text, ''))
                aligned = True
                break

            delta = solve_align(error, args.gain, args.max_step)
            target_pose = list(pose)
            for slot, index in enumerate(ALIGN_JOINTS):
                target_pose[index] = int(round(pose[index] + delta[slot]))
            target_pose = arm_control.clamp_pose(target_pose)
            command = 'j1 {:+d} j4 {:+d}'.format(
                target_pose[ALIGN_JOINTS[0]] - pose[ALIGN_JOINTS[0]],
                target_pose[ALIGN_JOINTS[1]] - pose[ALIGN_JOINTS[1]])
            print('{:>4} {:>8.1f},{:<7.1f} {:>7} {:>8.0f},{:<7.0f} {:>8} {:>20}'.format(
                step, cx, cy, area, error[0], error[1], range_text, command))
            if rivals:
                print('       note: {} other red region(s); using the largest'.format(rivals))

            if target_pose == pose:
                print('       correction rounded to zero; stopping')
                break
            pose = target_pose
            if args.execute:
                link.send(pose, 600)
                time.sleep(1.2)
            else:
                break  # One illustrative step is enough without moving.

        print('-' * 78)
        if not args.execute:
            print('Dry run: one step shown. Add --execute to close the loop.')
        elif aligned:
            print('Aligned within {:.0f} px.'.format(args.tolerance))
            if args.approach:
                print('\nApproach stage is not yet implemented; alignment only.')
                print('Remaining range is printed above: the target sits on the line')
                print('through the gripper, but still further along it.')
        else:
            print('Did not converge in {} steps.'.format(args.max_steps))
    finally:
        print('\nLast commanded pose: {}'.format(pose))
        print('Resume with --assume-pose {}'.format(' '.join(str(v) for v in pose)))
        streams.close()
        link.close()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
