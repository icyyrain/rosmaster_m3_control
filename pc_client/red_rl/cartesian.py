"""Small end-effector steps for the RL policy, realised on integer servos.

The policy acts like HIL-SERL's end-effector delta: a few millimetres along
three axes. The arm only takes whole servo degrees, about 3 mm of gripper
motion each near the table, so every step searches the nearby integer poses
for the one whose forward-kinematics gripper position lands closest to the
request. Joint 5 never moves.

Joints 2+3+4 set the tilt of the jaws, and of the camera on arm4. Each step
first searches within TIGHT_PITCH_DEG of the episode's starting tilt, which
converges quickly. That alone has a trap: the only way down at constant tilt
folds joint 2 back while opening joint 3, and one real episode folded joint 2
onto its 0 degree stop and spent its last nine steps unable to reach outward.
So when the tight search leaves more than FALLBACK_MM of the request undone,
the search widens to WIDE_PITCH_DEG with a small cost on tilt, letting joint 4
reach out as the rule teacher does (its grasps span about 20 degrees of
tilt); and a barrier keeps joint 2 off its stop in both. Allowing the wide
tilt on every step instead removed the trap but made real episodes slower
(11-19 steps against about 12), as joint 2 still drifted down to the barrier
and every descent then pulled the gripper inward.

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
TIGHT_PITCH_DEG = 2
WIDE_PITCH_DEG = 10
PITCH_TOLERANCE_DEG = WIDE_PITCH_DEG
PITCH_PENALTY = 0.03             # mm^2 per degree^2 of tilt, in the wide search
FALLBACK_MM = 1.5                # widen when the tight search misses by more
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


def _search(pose, origin, target, pitch_sum, band, pitch_penalty):
    """Best nearby integer pose for `target`; standing still is a candidate."""
    def cost(candidate, position, delta):
        tilt = sum(candidate[1:4]) - pitch_sum
        return (float(np.sum((position - target) ** 2))
                + MOTION_PENALTY * float(np.sum(np.square(delta)))
                + pitch_penalty * tilt ** 2
                + FLOOR_PENALTY * max(0, JOINT2_FLOOR_DEG - candidate[1]) ** 2)

    best_pose, best_position = list(pose), origin
    best_cost = cost(pose, origin, np.zeros(4))
    for delta in _DELTAS:
        if not delta.any():
            continue
        candidate = list(pose)
        for joint in range(4):
            candidate[joint] = int(pose[joint] + delta[joint])
        if not in_bounds(candidate):
            continue
        if abs(sum(candidate[1:4]) - pitch_sum) > band:
            continue
        position = kinematics.gripper_mm(candidate)
        candidate_cost = cost(candidate, position, delta)
        if candidate_cost < best_cost:
            best_pose, best_cost, best_position = candidate, candidate_cost, position
    return best_pose, best_position


def step(pose, local_mm, pitch_sum):
    """Nearest integer pose to a local displacement of the gripper.

    Returns (new_pose, achieved_local_mm). `new_pose` equals `pose` when no
    nearby pose gets closer to the request than standing still.
    """
    request = np.clip(np.asarray(local_mm, dtype=float), -MAX_STEP_MM, MAX_STEP_MM)
    origin = kinematics.gripper_mm(pose)
    target = origin + local_to_base(pose, request)
    best_pose, best_position = _search(pose, origin, target, pitch_sum,
                                       TIGHT_PITCH_DEG, 0.0)
    miss = float(np.linalg.norm(best_position - target))
    if miss > FALLBACK_MM:
        wide_pose, wide_position = _search(pose, origin, target, pitch_sum,
                                           WIDE_PITCH_DEG, PITCH_PENALTY)
        if float(np.linalg.norm(wide_position - target)) < miss - 0.5:
            best_pose, best_position = wide_pose, wide_position
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
