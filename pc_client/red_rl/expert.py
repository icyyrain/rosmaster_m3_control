"""The rule teacher re-expressed as a policy over the RL environment's spaces.

expert(observation) -> action, using nothing the learner does not also see.
It plays three parts in docs/BC_RL_PLAN.md: the M1 check that the episode is
solvable from the random starts, the demonstrations, and later the rule
corrector that stands in for a person.

What it does each step, like the rule creep it replaces:
  * horizontal: drive the sweet back under the gripper pixel. A sweet low in
    the image means the gripper is beyond it (move in, radial -); a sweet to
    the left means the gripper is to its left (move right, lateral -);
  * vertical: descend toward the grasp height, slowly while badly aligned;
  * close once within HEIGHT_READY_MM of the grasp height and ALIGN_PX of the
    gripper pixel, or on the last step if it got low enough to try.
"""

import math

import numpy as np

from pc_client.red_rl import cartesian
from pc_client.red_rl import env as red_env

MM_PER_PX = 0.17                 # near the table: about range / focal length
HORIZONTAL_GAIN = 0.8            # under-correct a little; the loop closes it
ALIGN_PX = 25.0                  # the rule teacher's re-centre tolerance
SLOW_DESCENT_PX = 60.0           # descend slowly while further off than this
SLOW_DESCENT_MM = 1.0
HEIGHT_READY_MM = 1.5
LOST_CLIMB_MM = 3.0              # climb when the sweet is out of view


def _index(name):
    return red_env.OBSERVATION_NAMES.index(name)


def read(observation):
    """(dx_px, dy_px, visible, height_to_grasp_mm, step) from an observation."""
    o = np.asarray(observation, dtype=float)
    return (o[_index('target_dx')] * 640.0, o[_index('target_dy')] * 360.0,
            o[_index('target_visible')] > 0.5, o[_index('height_to_grasp')] * 50.0,
            int(round(o[_index('step_fraction')] * red_env.MAX_STEPS)))


def expert(observation):
    dx, dy, visible, height, step = read(observation)
    limit = cartesian.MAX_STEP_MM
    if not visible:
        return np.asarray([0.0, 0.0, LOST_CLIMB_MM / limit, -1.0], dtype=np.float32)
    error = math.hypot(dx, dy)
    last_step = step >= red_env.MAX_STEPS - 1
    if height <= HEIGHT_READY_MM and (error <= ALIGN_PX or last_step):
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    radial = -dy * MM_PER_PX * HORIZONTAL_GAIN
    lateral = dx * MM_PER_PX * HORIZONTAL_GAIN
    descent = min(limit, max(0.0, height))
    if error > SLOW_DESCENT_PX:
        descent = min(descent, SLOW_DESCENT_MM)
    action = np.clip(np.asarray([radial, lateral, -descent]) / limit, -1.0, 1.0)
    return np.asarray([*action, -1.0], dtype=np.float32)
