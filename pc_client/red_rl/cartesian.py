"""Small end-effector steps for the RL policy, realised on integer servos.

The policy acts like HIL-SERL's end-effector delta: a few millimetres along
three axes. The arm only takes whole servo degrees, about 3 mm of gripper
motion each near the table, so every step searches the nearby integer poses
for the one whose forward-kinematics gripper position lands closest to the
request. Joint 5 never moves.

The tilt of the jaws (joints 2+3+4) may wander PITCH_TOLERANCE_DEG from the
episode's start at a small cost. Holding it within 2 degrees, as a first
version did, left only one way down: fold joint 2 back while opening joint 3.
Ten such steps drove joint 2 onto its 0 degree stop, after which the arm
could no longer reach outward at all; one real episode spent its last nine
steps stuck there. Reaching out is joint 4's job, which tilts the jaws, as
the rule teacher does (its grasps span about 20 degrees of tilt). A barrier
keeps joint 2 off its stop. In a kinematic simulation of 80 harder starts
this closed on the sweet 75 times against 61, and ten "out 4 mm, down 3 mm"
requests reached 28 mm outward instead of 10.

Axes are local to the arm's current heading, so an action means the same
thing whichever way the base is turned: `radial` away from the base, `lateral`
to the arm's left, `up` vertical.
"""

import itertools
import math

import numpy as np

import kinematics
import pick_and_place as pick

MAX_STEP_MM = 4.0
JOINT_REACH_DEG = 2
PITCH_TOLERANCE_DEG = 10
PITCH_PENALTY = 0.03             # mm^2 per degree^2 of tilt away from the start
MOTION_PENALTY = 0.05            # mm^2 per degree^2: prefer the smaller move
JOINT2_FLOOR_DEG = 6             # keep joint 2 clear of its 0 degree stop
FLOOR_PENALTY = 25.0             # mm^2 per degree^2 below that

_DELTAS = np.asarray(list(itertools.product(
    range(-JOINT_REACH_DEG, JOINT_REACH_DEG + 1), repeat=4)), dtype=int)


def heading(pose):
    """Unit radial and lateral vectors of the arm in base_link, from FK."""
    position = kinematics.gripper_mm(pose)
    norm = math.hypot(position[0], position[1])
    radial = np.asarray([position[0] / norm, position[1] / norm, 0.0])
    lateral = np.asarray([-radial[1], radial[0], 0.0])
    return radial, lateral


def local_to_base(pose, local_mm):
    radial, lateral = heading(pose)
    return (local_mm[0] * radial + local_mm[1] * lateral
            + np.asarray([0.0, 0.0, local_mm[2]]))


def base_to_local(pose, base_mm):
    radial, lateral = heading(pose)
    return np.asarray([float(np.dot(base_mm, radial)),
                       float(np.dot(base_mm, lateral)), float(base_mm[2])])


def in_bounds(pose):
    return (pick.J1_BAND[0] <= pose[0] <= pick.J1_BAND[1]
            and 0 <= pose[1] <= 180 and 0 <= pose[2] <= 180
            and pick.J4_BAND[0] <= pose[3] <= pick.J4_BAND[1])


def step(pose, local_mm, pitch_sum):
    """Nearest integer pose to a local displacement of the gripper.

    Returns (new_pose, achieved_local_mm). `new_pose` equals `pose` when no
    nearby pose gets closer to the request than standing still.
    """
    request = np.clip(np.asarray(local_mm, dtype=float), -MAX_STEP_MM, MAX_STEP_MM)
    origin = kinematics.gripper_mm(pose)
    target = origin + local_to_base(pose, request)
    best_pose, best_position = list(pose), origin
    best_cost = (float(np.sum(request ** 2))           # standing still
                 + PITCH_PENALTY * (sum(pose[1:4]) - pitch_sum) ** 2
                 + FLOOR_PENALTY * max(0, JOINT2_FLOOR_DEG - pose[1]) ** 2)
    for delta in _DELTAS:
        if not delta.any():
            continue
        candidate = list(pose)
        for joint in range(4):
            candidate[joint] = int(pose[joint] + delta[joint])
        if not in_bounds(candidate):
            continue
        tilt = sum(candidate[1:4]) - pitch_sum
        if abs(tilt) > PITCH_TOLERANCE_DEG:
            continue
        position = kinematics.gripper_mm(candidate)
        cost = (float(np.sum((position - target) ** 2))
                + MOTION_PENALTY * float(np.sum(delta ** 2))
                + PITCH_PENALTY * tilt ** 2
                + FLOOR_PENALTY * max(0, JOINT2_FLOOR_DEG - candidate[1]) ** 2)
        if cost < best_cost:
            best_pose, best_cost, best_position = candidate, cost, position
    return best_pose, base_to_local(pose, best_position - origin)


def offset(pose, local_mm, pitch_sum, max_steps=8):
    """Walk to a larger local offset in MAX_STEP_MM pieces; for perturbing."""
    remaining = np.asarray(local_mm, dtype=float)
    current = list(pose)
    for _ in range(max_steps):
        if np.linalg.norm(remaining) < 1.0:
            break
        chunk = np.clip(remaining, -MAX_STEP_MM, MAX_STEP_MM)
        moved_pose, achieved = step(current, chunk, pitch_sum)
        if moved_pose == current:
            break
        # Re-express what is left in the new heading before the next piece.
        remaining_base = local_to_base(current, remaining - achieved)
        current = moved_pose
        remaining = base_to_local(current, remaining_base)
    return current
