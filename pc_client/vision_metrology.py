#!/usr/bin/env python3

"""Measure camera motion in pixels, using the scene itself as the reference.

The arm reports no joint positions, but the depth camera is bolted to arm link
arm4, so any motion of joints 1-4 moves the whole image. Estimating the global
image transform between two frames therefore measures arm motion, with no tag
and no calibration.

This is a differential measurement and it is far more sensitive than absolute
pose from a tag: the static noise floor is 0.03-0.05 px against roughly 18.8 px
per degree on joint 1, which resolves a few thousandths of a degree. It says
nothing about absolute angles. Use visual_state for that.

A hard-won caveat about reference frames. The first version of the coupling
experiment compared every trial against one reference captured at the start.
Joints 5 and 6 then reported 9.67 and 9.44 px, suspiciously equal to each other
and enough to look like real motion, because the arm does not return exactly to
a commanded pose and late trials inherited the accumulated error. A second
trap: measuring against a reference taken before a large move inflates the
apparent noise, because the views only partly overlap and the inlier count
collapses, which made a settled arm look like it was hunting with a 2.04 px
standard deviation when the true figure was 0.12 px.

So: always take the reference close to what is being measured, and read the
inlier count. If it drops by half, the numbers are not comparable.
"""

import base64
import math
import time

import cv2
import numpy

COLOR_TOPIC = '/high_camera/color/image_raw/compressed'

_ORB = cv2.ORB_create(nfeatures=1500)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

MIN_KEYPOINTS = 20
MIN_MATCHES = 20
MIN_INLIERS = 15


def decode_gray(payload_b64):
    """Decode a base64 CompressedImage payload to greyscale, or None."""
    raw = base64.b64decode(payload_b64)
    if not raw:
        return None
    return cv2.imdecode(numpy.frombuffer(raw, numpy.uint8), cv2.IMREAD_GRAYSCALE)


def reference_of(frame):
    """Precompute the features of a reference frame."""
    keypoints, descriptors = _ORB.detectAndCompute(frame, None)
    if descriptors is None or len(keypoints) < MIN_KEYPOINTS:
        return None
    return keypoints, descriptors


DEFAULT_CENTRE = (640.0, 360.0)  # Centre of the 1280x720 colour frame.


class Shift(object):
    """Global image motion of a frame relative to a reference.

    Beware which point the translation refers to. `dx`/`dy` come straight from
    the affine fit, so they describe the displacement at the image ORIGIN, the
    top-left corner. Whenever the transform also rotates, that figure is
    inflated by the rotation acting over the distance to the origin: joint 1
    measured 11.8 px per degree by median inlier displacement but 18.8 px per
    degree by origin translation, purely from this effect.

    For anything that reasons about where content actually moved, including any
    image Jacobian, use `displacement_at` with the point of interest.
    """

    __slots__ = ('dx', 'dy', 'roll_deg', 'inliers', 'matrix')

    def __init__(self, dx, dy, roll_deg, inliers, matrix=None):
        self.dx = dx
        self.dy = dy
        self.roll_deg = roll_deg
        self.inliers = inliers
        self.matrix = matrix

    @property
    def magnitude(self):
        """Translation magnitude in px at the image origin."""
        return math.hypot(self.dx, self.dy)

    @property
    def signed(self):
        """Magnitude carrying the sign of the x component, for sweeps."""
        return math.copysign(self.magnitude, self.dx)

    def displacement_at(self, point=DEFAULT_CENTRE):
        """How far image content at `point` moved, in px, as (dx, dy).

        This is the number to build a Jacobian from.
        """
        if self.matrix is None:
            return self.dx, self.dy
        x, y = point
        moved = self.matrix @ numpy.array([x, y, 1.0])
        return float(moved[0] - x), float(moved[1] - y)

    def __repr__(self):
        return 'Shift(origin {:.2f} px, roll {:.3f} deg, {} inliers)'.format(
            self.magnitude, self.roll_deg, self.inliers)


def shift_of(reference, frame):
    """Estimate the global image transform of frame against reference.

    RANSAC is used so that something moving inside the frame, such as a hand,
    is rejected as an outlier rather than counted as camera motion.
    """
    if reference is None or frame is None:
        return None
    keypoints, descriptors = _ORB.detectAndCompute(frame, None)
    if descriptors is None or len(keypoints) < MIN_KEYPOINTS:
        return None
    matches = _MATCHER.match(reference[1], descriptors)
    if len(matches) < MIN_MATCHES:
        return None
    src = numpy.float32(
        [reference[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = numpy.float32(
        [keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None:
        return None
    count = int(inliers.ravel().sum())
    if count < MIN_INLIERS:
        return None
    return Shift(float(matrix[0, 2]), float(matrix[1, 2]),
                 math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), count, matrix)


class Camera(object):
    """Frame source over rosbridge, with local arrival timestamps.

    Timestamps are local, so every interval includes a fixed transport cost:
    JPEG encode on the Jetson, the WebSocket hop, and decode here. Measured end
    to end that is 240-330 ms for a commanded move to become visible, which is
    the dead time of any vision-in-the-loop controller built on this.
    """

    def __init__(self, client, roslibpy, rate_hz=30.0):
        self._latest = None
        self.count = 0
        self.buffer = []
        self.recording = False
        self.topic = roslibpy.Topic(
            client, COLOR_TOPIC, 'sensor_msgs/msg/CompressedImage',
            throttle_rate=int(1000.0 / rate_hz), queue_length=1, queue_size=1)
        self.topic.subscribe(self._on_frame)

    def _on_frame(self, message):
        self._latest = message['data']
        self.count += 1
        if self.recording:
            self.buffer.append((time.time(), message['data']))

    def wait_for_stream(self, timeout=10.0):
        deadline = time.time() + timeout
        while self.count == 0 and time.time() < deadline:
            time.sleep(0.02)
        return self.count > 0

    def grab(self, timeout=5.0):
        """The next frame to arrive after this call, as greyscale."""
        start = self.count
        deadline = time.time() + timeout
        while self.count == start and time.time() < deadline:
            time.sleep(0.005)
        if self._latest is None:
            return None
        return decode_gray(self._latest)

    def record(self):
        self.buffer = []
        self.recording = True

    def stop(self):
        self.recording = False
        return list(self.buffer)

    def close(self):
        self.topic.unsubscribe()


def median_shift(reference, camera, count=4, gap=0.15):
    """Median of several shift estimates, to damp single-frame outliers.

    Returns (magnitude_px, roll_deg, inliers, last_frame) or None.
    """
    samples = []
    frame = None
    for _ in range(count):
        frame = camera.grab()
        result = shift_of(reference, frame)
        if result is not None:
            samples.append(result)
        time.sleep(gap)
    if not samples:
        return None
    return (float(numpy.median([s.magnitude for s in samples])),
            float(numpy.median([s.roll_deg for s in samples])),
            int(numpy.median([s.inliers for s in samples])),
            frame)
