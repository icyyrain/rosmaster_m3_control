#!/usr/bin/env python3

"""Forward kinematics for the M3 Pro arm, on the PC, from the factory URDF.

This exists because its absence kept biting. Three separate limitations traced
back to not knowing where the gripper is without measuring it:

  * The descent followed a fixed joint ladder solved for one target distance,
    so a target at a very different range needed the ladder recomputed by hand.
  * The creep decided which way to move joints 2 and 3 by watching the measured
    height, and the depth stream dies below about 120 mm of range, exactly
    where that decision matters. On a 15.9 mm biscuit bar the height reading
    vanished at the fourth creep step and the remaining steps carried on
    raising joint 3, which at joint 2 = 0 *lifts* the gripper rather than
    lowering it. The run ended with the jaws well above the object.
  * A check invented a "reachable floor" from the lowest height ever measured,
    which was really the lowest height ever *reported*, and refused a sweet
    that had already been picked up twice.

None of those need measurement. The arm is rigid and the geometry is published.

The parameters are copied from
`yahboomcar_ws/src/yahboom_M3Pro_description/urdf/M3Pro.urdf` on the robot, so
they are the factory numbers rather than anything fitted here. Joint commands
are servo degrees, 0..180, with 90 as the joint's zero, matching `ArmJoints`.

What this does NOT give: where the arm really is. Commands are not feedback,
and the measured backlash is about one degree, so treat every position here as
where the arm was told to go. `table_plane` remains the only thing that
measures rather than models, and the two agreed to 14.5 mm when both were
available.
"""

import math

import numpy

# Straight from the URDF. Lengths in metres, angles in radians.
JOINTS = {
    'arm1_Joint': dict(xyz=(0.033617, -1.2421e-05, 0.185),
                       rpy=(6.3751e-05, 0.0, 0.028395),
                       axis=(0.0, 0.0, 1.0), type='revolute'),
    'arm2_Joint': dict(xyz=(-0.0021522, 6.1185e-05, 0.033),
                       rpy=(-6.3725e-05, 0.0, -0.028422),
                       axis=(0.0, -1.0, -6.3725e-05), type='revolute'),
    'arm3_Joint': dict(xyz=(0.0, 0.0, 0.08285),
                       rpy=(6.3702e-05, 0.0, 0.027102),
                       axis=(-0.027098, -0.99963, 0.0), type='revolute'),
    'arm4_Joiint': dict(xyz=(0.0, 0.0, 0.08285),
                        rpy=(0.0, 0.0, -0.027075),
                        axis=(-2.6392e-05, -1.0, 0.0), type='revolute'),
    'arm5_Joint': dict(xyz=(0.00057305, 0.0, 0.074815),
                       rpy=(0.0, 0.0, 0.0),
                       axis=(0.0, 0.0, 1.0), type='revolute'),
    'Gripping_Joint': dict(xyz=(-0.00062499, 0.00023987, 0.093235),
                           rpy=(-0.041648, -1.5708, 0.041619),
                           axis=(0.0, 0.0, 0.0), type='fixed'),
    'DCW2_Joint': dict(xyz=(-0.0483883324, 0.00245180098, 0.06719),
                       rpy=(1.84287098, -1.57079631, -1.81447539),
                       axis=(0.0, 0.0, 0.0), type='fixed'),
}

CHAIN = ('arm1_Joint', 'arm2_Joint', 'arm3_Joint', 'arm4_Joiint')
SERVO_ZERO_DEG = 90.0

# base_link z = 0 is about table level, from comparing this model against the
# plane fit: 230.1 mm here against 215.6 mm measured, a 14.5 mm difference.
# Subtract this to compare a modelled height with a measured one.
MODEL_MINUS_MEASURED_MM = 14.5


def _rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return numpy.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _axis_rotation(axis, angle):
    vector = numpy.asarray(axis, dtype=float)
    vector = vector / numpy.linalg.norm(vector)
    x, y, z = vector
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return numpy.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _transform(joint, angle=0.0):
    rotation = _rpy(*joint['rpy'])
    if joint['type'] == 'revolute':
        rotation = rotation @ _axis_rotation(joint['axis'], angle)
    matrix = numpy.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = joint['xyz']
    return matrix


def arm4_frame(commands):
    """Pose of link arm4, which carries the camera, for servo commands."""
    matrix = numpy.eye(4)
    for index, name in enumerate(CHAIN):
        angle = math.radians(commands[index] - SERVO_ZERO_DEG)
        matrix = matrix @ _transform(JOINTS[name], angle)
    return matrix


def gripper_mm(commands):
    """Gripper origin in base_link, in mm."""
    matrix = arm4_frame(commands)
    matrix = matrix @ _transform(JOINTS['arm5_Joint'],
                                 math.radians(commands[4] - SERVO_ZERO_DEG))
    matrix = matrix @ _transform(JOINTS['Gripping_Joint'])
    return matrix[:3, 3] * 1000.0


def camera_mm(commands):
    """Camera origin in base_link, in mm."""
    matrix = arm4_frame(commands) @ _transform(JOINTS['DCW2_Joint'])
    return matrix[:3, 3] * 1000.0


def gripper_height_mm(commands):
    """Modelled gripper height above the table, in mm.

    base_link z = 0 sits about table level, and the model reads about 14.5 mm
    high against the plane fit, so that offset is removed here to make this
    directly comparable with `table_plane.gripper_height`.
    """
    return float(gripper_mm(commands)[2]) - MODEL_MINUS_MEASURED_MM


def descent_step(commands, joints=(1, 2), step=3, limits=(0, 180)):
    """The small joint change that lowers the gripper most.

    This is what the creep needed and could not get from depth. Once joint 2
    reaches its limit the useful direction for joint 3 reverses: the model puts
    [85, 0, 74] at 40.5 mm, [85, 0, 82] at 62.4 and [85, 0, 90] at 87.4, so a
    ladder that keeps raising joint 3 climbs back up. Searching the immediate
    neighbourhood finds the right sign without anyone having to know it.

    Returns (new_commands, modelled_height_mm), or (None, current_height) when
    no neighbouring move goes lower.
    """
    current = gripper_height_mm(commands)
    best, best_height = None, current
    for first in (-step, 0, step):
        for second in (-step, 0, step):
            if first == 0 and second == 0:
                continue
            candidate = list(commands)
            candidate[joints[0]] = candidate[joints[0]] + first
            candidate[joints[1]] = candidate[joints[1]] + second
            if not all(limits[0] <= candidate[j] <= limits[1] for j in joints):
                continue
            height = gripper_height_mm(candidate)
            if height < best_height - 0.05:
                best, best_height = candidate, height
    return best, best_height


def lowest_reachable_mm(commands, joints=(1, 2), limits=(0, 180), coarse=3):
    """Lowest modelled gripper height with the other joints held, in mm."""
    best = (1e9, None)
    for first in range(limits[0], limits[1] + 1, coarse):
        for second in range(limits[0], limits[1] + 1, coarse):
            candidate = list(commands)
            candidate[joints[0]] = first
            candidate[joints[1]] = second
            height = gripper_height_mm(candidate)
            if height < best[0]:
                best = (height, (first, second))
    return best


def approach_height(commands, target_mm, joints=(1, 2), max_step=3,
                    limits=(0, 180)):
    """Small joint change that best lands the gripper at `target_mm`.

    `descent_step` only asks for lower, which overshoots near the end: from the
    pose a biscuit run got stuck at, one 3 degree move took the modelled height
    from 32.9 mm to 14.0 mm, straight past the 24 mm that object needed. This
    searches the same neighbourhood but scores on distance to the target and
    refuses to go below it, so it slows down as it arrives.

    Returns (new_commands, modelled_height_mm), or (None, height) when nothing
    nearby is better than standing still.
    """
    current = gripper_height_mm(commands)
    best, best_error, best_height = None, abs(current - target_mm), current
    for first in range(-max_step, max_step + 1):
        for second in range(-max_step, max_step + 1):
            if first == 0 and second == 0:
                continue
            candidate = list(commands)
            candidate[joints[0]] += first
            candidate[joints[1]] += second
            if not all(limits[0] <= candidate[j] <= limits[1] for j in joints):
                continue
            height = gripper_height_mm(candidate)
            if height < target_mm - 1.0:
                continue                      # never dive below the target
            error = abs(height - target_mm)
            if error < best_error - 0.05:
                best, best_error, best_height = candidate, error, height
    return best, best_height
