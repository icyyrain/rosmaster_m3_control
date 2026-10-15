"""Put a held red sweet down at a random spot the next trial can reach.

Used by unattended collection. The spot is drawn from the part of the desk
where the rule teacher has already picked the sweet up, so the next trial's
preflight, alignment and descent cover ground they have covered before. Every
pose comes from forward kinematics. The sweet is carried at the short-lift
height, lowered to the same clearance pick_and_place uses for its gentle
release, and only then let go; every leg of the path is checked so the sweet
never travels lower than intended.
"""

import math

import numpy as np

import arm_control
import gripper_state
import kinematics
import pick_and_place as pick

# Gripper-origin positions, in base_link polar coordinates, of the 17
# human-confirmed grasps up to 2026-10-14: radius 185-242 mm, azimuth -17.2 to
# +14.6 degrees. Shrunk by a few mm and degrees so placements stay inside.
PLACE_RADIUS_MM = (200.0, 236.0)
PLACE_AZIMUTH_DEG = (-13.0, 11.0)
MIN_MOVE_MM = 30.0               # do not put it back where it was picked up

CARRY_CLEARANCE_MM = 35.0        # the short lift's height above the grasp
MIN_TRAVEL_CLEARANCE_MM = 20.0   # lowest the sweet may be while moving sideways
RELEASE_CLEARANCE_MM = pick.PLACE_RELEASE_CLEARANCE_MM
MAX_LOWERING_DRIFT_MM = 6.0      # sideways motion allowed while lowering

RADIUS_TOLERANCE_MM = 2.0
HEIGHT_TOLERANCE_MM = 1.0
POSITION_CHECK_MM = 3.5          # after rounding joint 1 to a whole degree
PITCH_TOLERANCE_DEG = 8          # joints 2+3+4 against the grasp pose
MS_PER_DEGREE = 60               # slow enough not to fling a held sweet
CARRY_DRIFT_PX = 60.0            # a held sweet stays this close to its lift pixel
MIN_MOVE_MS = 900

SEARCH_J2 = np.arange(0, 46)
SEARCH_J3 = np.arange(25, 91)
SEARCH_J4 = np.arange(pick.J4_BAND[0], pick.J4_BAND[1] + 1)
SAMPLE_ATTEMPTS = 300

_ARM1 = kinematics.JOINTS['arm1_Joint']
_ARM1_ROTATION = kinematics._rpy(*_ARM1['rpy'])
_ARM1_ORIGIN_MM = np.asarray(_ARM1['xyz']) * 1000.0
_TABLE = None


def _reach_table():
    """Gripper origin in the arm1 frame, mm, for every (j2, j3, j4) searched.

    Joint 1 only turns this vector about the arm1 axis, so one table serves
    every azimuth. Joint 5 is held at its HOME angle by the red collector.
    """
    global _TABLE
    if _TABLE is None:
        joints = kinematics.JOINTS

        def stack(name, values):
            return np.stack([
                kinematics._transform(joints[name], math.radians(
                    value - kinematics.SERVO_ZERO_DEG)) for value in values])

        upper = np.einsum('aij,bjk->abik', stack('arm2_Joint', SEARCH_J2),
                          stack('arm3_Joint', SEARCH_J3))
        chain = np.einsum('abij,cjk->abcik', upper,
                          stack('arm4_Joiint', SEARCH_J4))
        tail = (kinematics._transform(joints['arm5_Joint'], 0.0)
                @ kinematics._transform(joints['Gripping_Joint']))
        _TABLE = (chain @ tail[:, 3])[..., :3] * 1000.0
    return _TABLE


def solve_pose(xy_mm, height_mm, pitch_sum, gripper_cmd):
    """Integer joint pose that puts the gripper origin at (x, y, height).

    `height_mm` is in kinematics.gripper_height_mm terms. Among the solutions
    within tolerance, prefer the one whose joints 2+3+4 are closest to
    `pitch_sum`, which keeps the jaws at the tilt the sweet was gripped at.
    Returns None when nothing within the search box fits.
    """
    target = np.asarray([xy_mm[0], xy_mm[1],
                         height_mm + kinematics.MODEL_MINUS_MEASURED_MM])
    local = _ARM1_ROTATION.T @ (target - _ARM1_ORIGIN_MM)
    table = _reach_table()
    radius = np.hypot(table[..., 0], table[..., 1])
    radius_error = radius - math.hypot(local[0], local[1])
    height_error = table[..., 2] - local[2]
    pitch = (SEARCH_J2[:, None, None] + SEARCH_J3[None, :, None]
             + SEARCH_J4[None, None, :])
    fits = ((np.abs(radius_error) <= RADIUS_TOLERANCE_MM)
            & (np.abs(height_error) <= HEIGHT_TOLERANCE_MM)
            & (np.abs(pitch - pitch_sum) <= PITCH_TOLERANCE_DEG))
    if not fits.any():
        return None
    cost = (radius_error ** 2 + 4.0 * height_error ** 2
            + 0.5 * (pitch - pitch_sum) ** 2)
    cost = np.where(fits, cost, np.inf)
    a, b, c = np.unravel_index(np.argmin(cost), cost.shape)
    vector = table[a, b, c]
    turn = math.atan2(local[1], local[0]) - math.atan2(vector[1], vector[0])
    j1 = int(round(kinematics.SERVO_ZERO_DEG + math.degrees(turn)))
    if not pick.J1_BAND[0] <= j1 <= pick.J1_BAND[1]:
        return None
    pose = [j1, int(SEARCH_J2[a]), int(SEARCH_J3[b]), int(SEARCH_J4[c]),
            arm_control.HOME_POSE[4], int(gripper_cmd)]
    reached = kinematics.gripper_mm(pose)
    if (math.hypot(reached[0] - xy_mm[0], reached[1] - xy_mm[1]) > POSITION_CHECK_MM
            or abs(kinematics.gripper_height_mm(pose) - height_mm)
            > HEIGHT_TOLERANCE_MM + 0.5):
        return None
    return pose


def _path(start, end, samples=10):
    for index in range(samples + 1):
        fraction = index / float(samples)
        yield [start[j] + fraction * (end[j] - start[j]) for j in range(6)]


def path_clearance(start, end, grasp_height_mm):
    """Lowest gripper height along a joint-space move, above the grasp height."""
    return min(kinematics.gripper_height_mm(pose) for pose in _path(start, end)) \
        - grasp_height_mm


def lowering_drift_mm(start, end):
    first = kinematics.gripper_mm(start)
    return max(math.hypot(*(kinematics.gripper_mm(pose)[:2] - first[:2]))
               for pose in _path(start, end))


def plan(grasp_pose, lift_pose, rng):
    """Choose a random reachable spot and the poses to put the sweet there.

    Returns a dict with the travel waypoints (turn, carry, release, clear) and
    where the sweet goes, or None when no sampled spot passes every check.
    """
    grasp_height = kinematics.gripper_height_mm(grasp_pose)
    picked = kinematics.gripper_mm(grasp_pose)[:2]
    pitch_sum = sum(grasp_pose[1:4])
    for _ in range(SAMPLE_ATTEMPTS):
        radius = rng.uniform(*PLACE_RADIUS_MM)
        azimuth = rng.uniform(*PLACE_AZIMUTH_DEG)
        xy = (radius * math.cos(math.radians(azimuth)),
              radius * math.sin(math.radians(azimuth)))
        if math.hypot(xy[0] - picked[0], xy[1] - picked[1]) < MIN_MOVE_MM:
            continue
        release = solve_pose(xy, grasp_height + RELEASE_CLEARANCE_MM,
                             pitch_sum, grasp_pose[5])
        if release is None:
            continue
        carry = solve_pose(xy, grasp_height + CARRY_CLEARANCE_MM,
                           sum(release[1:4]), grasp_pose[5])
        if carry is None:
            continue
        turn = list(lift_pose)
        turn[0] = carry[0]
        if path_clearance(turn, carry, grasp_height) < MIN_TRAVEL_CLEARANCE_MM:
            continue
        if (path_clearance(carry, release, grasp_height)
                < RELEASE_CLEARANCE_MM - HEIGHT_TOLERANCE_MM - 0.5):
            continue
        if lowering_drift_mm(carry, release) > MAX_LOWERING_DRIFT_MM:
            continue
        clear = list(carry)
        clear[5] = gripper_state.OPEN_CMD
        return dict(radius_mm=round(radius, 1), azimuth_deg=round(azimuth, 1),
                    xy_mm=[round(xy[0], 1), round(xy[1], 1)],
                    moved_mm=round(math.hypot(xy[0] - picked[0],
                                              xy[1] - picked[1]), 1),
                    turn=turn, carry=carry, release=release, clear=clear)
    return None


def move_ms(start, end):
    largest = max(abs(end[j] - start[j]) for j in range(4))
    return max(MIN_MOVE_MS, int(MS_PER_DEGREE * largest))


def carried(session, held_px):
    """Is the sweet still in the jaws? Look where it sat after the short lift.

    A held sweet is rigid with the camera, so it stays near that pixel; one
    dropped on the way is either out of view or somewhere else on the table.
    Returns (still_held, pixel_or_None).
    """
    fresh = session.look(held_px)
    if fresh is None:
        return False, None
    pixel = (fresh['cx'], fresh['cy'])
    return math.hypot(pixel[0] - held_px[0], pixel[1] - held_px[1]) <= CARRY_DRIFT_PX, pixel


def execute(session, route, snapshot=None, held_px=None):
    """Carry, lower, open, lift clear and go HOME. Joint 5 never moves.

    With `held_px`, the pixel of the sweet after the short lift, the sweet is
    checked at the carry pose before it is lowered: an operator saw one grasp
    that passed the lift check fall out on the way. A dropped sweet is not
    lowered onto; the jaws open where they are and the arm goes HOME.

    `snapshot` is called once the jaws have lifted clear of the released
    sweet, for an audit photograph. Returns dict(dropped, carry_px, image).
    """
    print('=== relocate ===')
    print('  to radius {radius_mm} mm, azimuth {azimuth_deg} deg, {moved_mm} mm '
          'from where it was picked up'.format(**route))
    carry_px = None
    for name in ('turn', 'carry', 'release'):
        target = route[name]
        duration = move_ms(session.pose, target)
        print('  {:<8} {}'.format(name, target[:4]))
        session.send(target, duration, duration / 1000.0 + 0.8)
        if name == 'carry' and held_px is not None:
            still_held, carry_px = carried(session, held_px)
            if not still_held:
                print('  the sweet is no longer in the jaws (seen at {}, held at '
                      '{:.0f},{:.0f}); not putting anything down'.format(
                          'nowhere' if carry_px is None else
                          '{:.0f},{:.0f}'.format(*carry_px), *held_px))
                opened = list(session.pose)
                opened[5] = gripper_state.OPEN_CMD
                session.send(opened, 1200, 2.0)
                session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                             arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
                return dict(dropped=True, carry_px=carry_px, image=None)
            print('  still held at {:.0f},{:.0f}'.format(*carry_px))
    opened = list(route['release'])
    opened[5] = gripper_state.OPEN_CMD
    session.send(opened, 1200, 2.0)
    print('  released about {:.0f} mm above the grasp height'.format(
        RELEASE_CLEARANCE_MM))
    duration = move_ms(session.pose, route['clear'])
    session.send(route['clear'], duration, duration / 1000.0 + 0.8)
    image = snapshot() if snapshot is not None else None
    session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                 arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
    return dict(dropped=False, carry_px=carry_px, image=image)
