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


# --- colour-agnostic object detection --------------------------------------
#
# Anything standing proud of the fitted plane is an object on the table. This
# works on a blue biscuit wrapper as readily as a red sweet, which the a*
# colour detector cannot: it found nothing on a blue wrapper.
#
# It also measures real size, unlike the colour blob. On the sweet it returned
# 28 x 26 mm against a true 28 mm, where the a* mask gave 18 mm because it
# only covers the sufficiently red part. That makes it the right source for
# gripper_state.grip_command.
#
# The catch is the depth-to-colour mapping. depth_registration is off on this
# driver, so the two sensors sit about 12 mm apart and a depth pixel maps to a
# colour pixel only approximately. Halving the coordinates put the sweet within
# 8 px at 302 mm, but the parallax grows as the object gets nearer, so
# DEPTH_TO_COLOUR_SHIFT_PX exists to correct it and defaults to none because it
# has not been calibrated. preflight_offset measures it whenever both detectors
# can see the same object.

OBJECT_MIN_HEIGHT_MM = 6.0
OBJECT_MAX_HEIGHT_MM = 80.0
OBJECT_MIN_AREA_PX = 40
DEPTH_TO_COLOUR_SHIFT_PX = (0.0, 0.0)


def find_objects(depth, rng, min_height=OBJECT_MIN_HEIGHT_MM,
                 max_height=OBJECT_MAX_HEIGHT_MM, min_area=OBJECT_MIN_AREA_PX):
    """Objects standing on the table, largest first.

    Each entry carries its real size in mm, its height above the table, its
    distance, and the colour pixel it maps to.
    """
    import cv2

    if depth is None:
        return []
    plane = fit_plane(depth, rng)
    if plane is None:
        return []
    normal, offset, _ = plane
    points = to_points(depth)
    valid = (depth > NEAR_MM) & (depth < FAR_MM)

    heights = numpy.full(depth.shape, numpy.nan, numpy.float32)
    heights[valid] = points[valid] @ normal + offset
    if numpy.nanmedian(heights[valid]) < 0:
        heights = -heights

    mask = ((heights > min_height) & (heights < max_height)).astype(numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((5, 5), numpy.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blob = labels == label
        distance = float(numpy.median(depth[blob]))
        if not NEAR_MM < distance < FAR_MM:
            continue
        width_px = int(stats[label, cv2.CC_STAT_WIDTH])
        depth_px = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        found.append(dict(
            area_px=area,
            width_mm=width_px * distance / FX_DEPTH,
            depth_mm=depth_px * distance / FY_DEPTH,
            top_mm=float(numpy.nanpercentile(heights[blob], 95)),
            dist_mm=distance,
            depth_px=(cx, cy),
            colour_px=(cx * 2.0 + DEPTH_TO_COLOUR_SHIFT_PX[0],
                       cy * 2.0 + DEPTH_TO_COLOUR_SHIFT_PX[1]),
        ))
    found.sort(key=lambda item: -item['area_px'])
    return found


# Working area, as a fraction of the colour frame, inside which a target is
# accepted. This does more of the discrimination than it looks. On one desk the
# size filter alone left four candidates: the sweet 96 px from the frame centre,
# and three fragments at x = 9, 18 and 26, which are the monitor and the frame
# edge rather than objects on the table. The sweet won only on "nearest the
# centre", a convention rather than a measurement. Restricting to the central
# band excludes the edge fragments by construction.
WORKING_AREA = (0.20, 0.80, 0.20, 0.80)   # x_min, x_max, y_min, y_max
FRAME_PX = (1280.0, 720.0)


def in_working_area(colour_px, area=WORKING_AREA, frame=FRAME_PX):
    x_min, x_max, y_min, y_max = area
    return (x_min * frame[0] <= colour_px[0] <= x_max * frame[0]
            and y_min * frame[1] <= colour_px[1] <= y_max * frame[1])


def pick_graspable(objects, max_width_mm=60.0, min_width_mm=8.0,
                   min_top_mm=8.0, centre_px=(640.0, 360.0),
                   working_area=WORKING_AREA, max_long_mm=200.0):
    """Choose the object most likely to be the intended target.

    Geometry alone returns everything standing on the table, the keyboard
    included. The keyboard is thrown out robustly, at 201 x 188 mm against a
    60 mm limit. The size filter is thinner than it looks though: one desk
    object measured 61.3 mm and missed by 1.3 mm. So the working area does the
    rest of the work, and nearness to the centre only breaks what is left.
    """
    # The jaws close across the narrower axis, so that is the dimension that
    # has to fit. Requiring BOTH axes to fit threw out a 25 x 110 mm biscuit
    # bar that the jaws could hold perfectly well across its width. The long
    # axis still gets a generous ceiling, which keeps the keyboard out at
    # 201 x 188 mm.
    candidates = []
    for o in objects:
        narrow = min(o['width_mm'], o['depth_mm'])
        wide = max(o['width_mm'], o['depth_mm'])
        if not min_width_mm <= narrow <= max_width_mm:
            continue
        if wide > max_long_mm:
            continue
        if o['top_mm'] < min_top_mm:
            continue
        if working_area is not None and not in_working_area(o['colour_px'], working_area):
            continue
        candidates.append(o)
    if not candidates:
        return None
    return min(candidates, key=lambda o: (
        (o['colour_px'][0] - centre_px[0]) ** 2 + (o['colour_px'][1] - centre_px[1]) ** 2))


def preflight_offset(colour_px, objects):
    """Offset between a colour detection and the nearest depth object, in px.

    Run whenever both detectors see the same thing; the result is what
    DEPTH_TO_COLOUR_SHIFT_PX should be set to.
    """
    if not objects:
        return None
    nearest = min(objects, key=lambda o: (
        (o['colour_px'][0] - colour_px[0]) ** 2 + (o['colour_px'][1] - colour_px[1]) ** 2))
    return (colour_px[0] - nearest['colour_px'][0],
            colour_px[1] - nearest['colour_px'][1]), nearest


def grip_width_mm(obj):
    """Width the jaws have to span, in mm.

    The jaws close across the narrower horizontal axis, so that is the starting
    point; using the wider one would ask for an aperture wider than the jaws
    open for anything long, such as a 33 x 83 mm biscuit bar.

    But the narrow axis systematically UNDERESTIMATES a rounded object. The
    object mask is everything more than 6 mm above the table, which on a sphere
    captures only the upper cap, narrower than the widest section. The sweet
    measured 21 mm across that way while standing 27 mm tall, and gripping it
    as 21 mm asks for a 15 mm aperture, about what flattened one earlier at 180.
    For an object resting on a table, its height is an independent estimate of
    its diameter, so the larger of the two is taken. That reproduces the
    aperture that was proven to hold the sweet without marking it.
    """
    narrow = min(obj['width_mm'], obj['depth_mm'])
    return max(narrow, obj['top_mm'])
