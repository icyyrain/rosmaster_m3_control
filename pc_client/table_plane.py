#!/usr/bin/env python3

"""How far the gripper is above the table, measured from depth.

This is the safety signal for descending onto something, and it deliberately
never uses the target's own range, because that estimate proved to sit about
36 mm too high.

How it works. The gripper occupies a fixed image pixel, which gives its
direction in the camera's optical frame straight from the intrinsics; the URDF
gives its distance from the camera, 111.9 mm. Those two fix the gripper's 3D
position in the camera frame without needing the camera's orientation relative
to the arm, which is the fiddly part. The table is then fitted as a plane from
the depth image, thousands of pixels strong, and the gripper's signed distance
from that plane is the height.

Cross-checked against forward kinematics at the aligned pose: 215.6 mm measured
against 230.1 mm from FK, a 14.5 mm difference, which also told us that
base_link z = 0 is about table level.

Two limits worth knowing. The plane is fitted in the depth sensor's frame while
the gripper ray is in the colour sensor's frame, and the two sit about 12 mm
apart, almost entirely sideways, which barely affects a height. And depth stops
being usable below roughly 120 mm: at the final grasp pose the whole depth
frame came back 0% valid, most likely the wrist occluding the module. So the
last few millimetres of an approach are necessarily open loop on height.
"""

import math

import numpy

# Colour intrinsics, and the gripper's fixed pixel (see reach_candy).
FX_COLOUR, FY_COLOUR = 720.501, 720.4833
CX_COLOUR, CY_COLOUR = 649.1655, 359.6946
GRIPPER_PX = (707.0, 680.0)
CAM_TO_GRIPPER_MM = 111.9

# Depth intrinsics. Same field of view as colour, at half the resolution.
FX_DEPTH, FY_DEPTH = 360.250, 360.242
CX_DEPTH, CY_DEPTH = 324.583, 179.847

NEAR_MM, FAR_MM = 120.0, 900.0
INLIER_MM = 4.0
MIN_POINTS = 4000
MIN_INLIERS = 1500


def gripper_point_mm():
    """Gripper position in the camera optical frame, in mm."""
    direction = numpy.array([(GRIPPER_PX[0] - CX_COLOUR) / FX_COLOUR,
                             (GRIPPER_PX[1] - CY_COLOUR) / FY_COLOUR,
                             1.0])
    direction = direction / numpy.linalg.norm(direction)
    return direction * CAM_TO_GRIPPER_MM


GRIPPER_POINT = gripper_point_mm()


def to_points(depth):
    """Depth image to a 3D point per pixel, in mm, in the depth camera frame."""
    height, width = depth.shape
    us, vs = numpy.meshgrid(numpy.arange(width), numpy.arange(height))
    z = depth.astype(numpy.float32)
    return numpy.stack([(us - CX_DEPTH) * z / FX_DEPTH,
                        (vs - CY_DEPTH) * z / FY_DEPTH,
                        z], axis=2)


def fit_plane(depth, rng, iterations=250):
    """RANSAC plane through the dominant surface, refined by least squares.

    Returns (normal, offset, inlier_count) with the normal pointing back at the
    camera, so a point between the camera and the table has a positive height.
    Returns None if there is not enough valid depth to trust.
    """
    points = to_points(depth)
    valid = (depth > NEAR_MM) & (depth < FAR_MM)
    pts = points[valid]
    if len(pts) < MIN_POINTS:
        return None

    sample = pts[rng.choice(len(pts), min(15000, len(pts)), replace=False)]
    best = (None, None, -1)
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = numpy.cross(b - a, c - a)
        norm = numpy.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        offset = -float(normal @ a)
        count = int((numpy.abs(sample @ normal + offset) < INLIER_MM).sum())
        if count > best[2]:
            best = (normal, offset, count)

    normal, offset, count = best
    if normal is None or count < MIN_INLIERS:
        return None

    inliers = sample[numpy.abs(sample @ normal + offset) < INLIER_MM]
    centroid = inliers.mean(axis=0)
    _, _, vt = numpy.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[2] / numpy.linalg.norm(vt[2])
    offset = -float(normal @ centroid)
    if normal[2] > 0:
        normal, offset = -normal, -offset
    return normal, offset, len(inliers)


def gripper_height(depth, rng):
    """(height_mm, plane_support) of the gripper above the table, or (None, 0)."""
    if depth is None:
        return None, 0
    plane = fit_plane(depth, rng)
    if plane is None:
        return None, 0
    normal, offset, support = plane
    return float(normal @ GRIPPER_POINT + offset), support


def gripper_height_retry(streams, rng, tries=3, pause=0.25):
    """Same, retrying a few times because single depth frames drop out."""
    import time
    for _ in range(tries):
        height, support = gripper_height(streams.depth(), rng)
        if height is not None:
            return height, support
        time.sleep(pause)
    return None, 0


def target_height(depth, rng, colour_px, search=30):
    """Height above the table of whatever is at a colour pixel, in mm.

    Depth is not registered to colour on this driver, so the mapping is only
    approximate; a window is searched and the nearest surface taken, since an
    object on a table stands proud of everything around it.
    """
    if depth is None:
        return None
    plane = fit_plane(depth, rng)
    if plane is None:
        return None
    normal, offset, _ = plane
    points = to_points(depth)
    height, width = depth.shape
    cx = int(colour_px[0] / 2.0)
    cy = int(colour_px[1] / 2.0)
    x0, x1 = max(0, cx - search), min(width, cx + search + 1)
    y0, y1 = max(0, cy - search), min(height, cy + search + 1)
    patch = points[y0:y1, x0:x1].reshape(-1, 3)
    depths = depth[y0:y1, x0:x1].reshape(-1)
    patch = patch[(depths > NEAR_MM) & (depths < FAR_MM)]
    if len(patch) < 20:
        return None
    heights = patch @ normal + offset
    return float(numpy.percentile(heights, 90))


def summarise(normal):
    """Tilt of the fitted plane from the camera's optical axis, in degrees."""
    return math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
