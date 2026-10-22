"""The rule expert as the corrector: S3's stand-in for a person with a SpaceMouse.

HIL-SERL puts the person in an environment wrapper: while they push the
SpaceMouse, the wrapper executes their action instead of the policy's and
reports it as info["intervene_action"]; the learner stores that executed
action and also copies the transition into the demonstration buffer. This
wrapper does the same with the rule expert, so swapping in a SpaceMouse later
changes this file and nothing in the learner.

It only takes over when the policy is clearly going wrong, so the policy is
corrected rather than replaced (docs/BC_RL_PLAN.md section 8):
  * the sweet is out of view, or more than ERROR_PX from the gripper pixel or
    BOTTOM_DY_PX below it (the gripper pixel is only 40 px above the frame's
    bottom edge), while the policy is not bringing it back (moving less than
    TOWARD_MM toward it this step);
  * below the grasp height (within 3 mm of the safety floor) and still going down;
  * STALL_STEPS steps with neither the height nor the alignment improving,
    which includes hovering ready to close without closing;
  * LAST_STEPS or fewer steps left and the policy is not closing;
  * closing while still well above the grasp height or far off the sweet.
    A person watching would hold the gripper open; HIL-SERL's own gripper
    has a separate discrete critic and a grasp penalty, which this
    continuous close does not, and a fresh policy (S4) would otherwise close
    in mid-air on about half its steps.
Each takeover lasts up to TAKEOVER_STEPS steps, then control goes back.

Out of view, the corrector moves toward where the sweet was last seen
rather than only climbing: the episode ends after two steps without it.

Off unless the episode asks: reset(options={"intervene": True}). Evaluation
never asks, so it always measures the policy alone.

With options["intervene_progress"] = f (0 < f <= 1) it is also attentive, as
the person in the HIL-SERL paper is early in training: it takes over whenever
the policy's move advances toward the goal by less than f of what the
expert's move would (the expert's direction, descent included). This needs
no schedule: as the policy comes to move like the expert, it is taken over
less. The rules above alone only catch clear mistakes, and in the kinematic
fake a policy started from scratch needed about 225 episodes before it
finished any episode on its own.

The first S3 run (2026-10-20) took over whenever the sweet was more than
80 px off. Ten of its fifteen takeovers came in the first steps, where the
deliberately offset start alone puts the sweet 90-170 px off and the policy
was already heading back, only descending more slowly than the expert.
"""

import math

import gymnasium as gym
import numpy as np

from pc_client.red_rl import cartesian
from pc_client.red_rl import env as red_env
from pc_client.red_rl import expert as red_expert

TAKEOVER_STEPS = 3
ERROR_PX = 80.0
TOWARD_MM = 1.0                  # off target and moving less than this toward it
# A fresh policy in the kinematic fake (S4 dry run, 2026-10-20) lost the sweet
# out of the bottom of the frame in 3-6 steps, never more than 80 px off.
BOTTOM_DY_PX = 25.0
# Of 43 held evaluation grasps up to 2026-10-20 the highest closed 4.6 mm
# above the grasp height and the furthest 65 px off the gripper pixel.
CLOSE_HEIGHT_MM = 6.0
MIN_WANTED_MM = 1.0              # attentive rule: ignore moves the expert hardly wants
CLOSE_ERROR_PX = 70.0
STALL_STEPS = 3
STALL_HEIGHT_MM = 0.5            # less descent than this over STALL_STEPS steps
STALL_ERROR_PX = 10.0            # and less alignment gain than this: no progress
LAST_STEPS = 3

UP = red_env.ACTION_NAMES.index('up')
CLOSE = red_env.ACTION_NAMES.index('close')


def reason_to_intervene(observation, action, history, steps_left, min_progress=0.0):
    """Why the expert should take over now, or None.

    `history` holds (height_to_grasp_mm, error_px) of the observations since
    the policy last took control, latest last, including this one.
    `min_progress` > 0 adds the attentive rule (see the module docstring).
    """
    dx, dy, visible, height, _ = red_expert.read(observation)
    closing = action[CLOSE] > red_env.CLOSE_THRESHOLD
    if not visible:
        return 'target_not_visible'
    error = math.hypot(dx, dy)
    if closing and (height > CLOSE_HEIGHT_MM or error > CLOSE_ERROR_PX):
        return 'premature_close'
    if ((error > ERROR_PX or dy > BOTTOM_DY_PX)
            and toward_mm(observation, action) < TOWARD_MM):
        return 'off_target'
    if height < 0.0 and action[UP] < 0.0 and not closing:
        return 'below_grasp_and_descending'
    if steps_left <= LAST_STEPS and not closing:
        return 'running_out_of_steps'
    if len(history) > STALL_STEPS and not closing:
        then, now = history[-1 - STALL_STEPS], history[-1]
        if (then[0] - now[0] < STALL_HEIGHT_MM
                and then[1] - now[1] < STALL_ERROR_PX):
            return 'no_progress'
    if min_progress > 0.0 and not closing:
        advance, wanted = progress_mm(observation, action)
        if wanted >= MIN_WANTED_MM and advance < min_progress * wanted:
            return 'not_heading_for_the_goal'
    return None


def progress_mm(observation, action):
    """(how far this move goes along the expert's move, the expert's move length), mm."""
    desired = np.asarray(red_expert.expert(observation)[:CLOSE], dtype=float) \
        * cartesian.MAX_STEP_MM
    wanted = float(np.linalg.norm(desired))
    if wanted < 1e-6:
        return 0.0, 0.0
    move = np.asarray(action[:CLOSE], dtype=float) * cartesian.MAX_STEP_MM
    return float(move @ desired / wanted), wanted


def toward_mm(observation, action):
    """How far this action moves the gripper toward the sweet, horizontally, mm.

    The direction is the expert's: it moves straight at the sweet.
    """
    desired = np.asarray(red_expert.expert(observation)[:UP], dtype=float)
    length = float(np.linalg.norm(desired))
    if length < 1e-6:
        return 0.0
    move = np.asarray(action[:UP], dtype=float) * cartesian.MAX_STEP_MM
    return float(move @ desired / length)


class RuleIntervention(gym.Wrapper):
    def __init__(self, env, policy=red_expert.expert, takeover_steps=TAKEOVER_STEPS):
        super().__init__(env)
        self.corrector = policy
        self.takeover_steps = takeover_steps
        self.active = False
        self.min_progress = 0.0
        self._clear()

    def _clear(self):
        self.observation = None
        self.last_seen = None
        self.history = []
        self.steps = 0
        self.remaining = 0
        self.reason = None
        self.count = 0
        self.intervention_steps = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.active = bool(options.pop('intervene', False))
        self.min_progress = float(options.pop('intervene_progress', 0.0) or 0.0)
        observation, info = self.env.reset(seed=seed, options=options)
        self._clear()
        self.observation = observation
        if red_expert.read(observation)[2]:
            self.last_seen = observation
        return observation, dict(info, intervention_enabled=self.active)

    def correction(self):
        """The corrector's action; out of view, toward where the sweet was last seen."""
        if red_expert.read(self.observation)[2] or self.last_seen is None:
            return self.corrector(self.observation)
        back = np.array(self.corrector(self.last_seen), dtype=np.float32)
        back[UP] = max(float(back[UP]), 0.0)        # never descend blind
        back[CLOSE] = -1.0
        return back

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        executed, intervened = action, False
        if self.active:
            dx, dy, _, height, _ = red_expert.read(self.observation)
            self.history.append((height, math.hypot(dx, dy)))
            if self.remaining == 0:
                steps_left = getattr(self.env.unwrapped, 'max_steps',
                                     red_env.MAX_STEPS) - self.steps
                self.reason = reason_to_intervene(self.observation, action,
                                                  self.history, steps_left,
                                                  self.min_progress)
                if self.reason is not None:
                    self.remaining = self.takeover_steps
                    self.count += 1
                    print('  corrector takes over: {}'.format(self.reason))
            if self.remaining > 0:
                executed = np.asarray(self.correction(), dtype=np.float32)
                intervened = True
                self.remaining -= 1
                self.intervention_steps += 1
                if executed[CLOSE] > red_env.CLOSE_THRESHOLD:
                    self.remaining = 0
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self.observation = observation
        if red_expert.read(observation)[2]:
            self.last_seen = observation
        self.steps += 1
        if intervened:
            # Lists, not arrays: the info dictionary crosses the RPC as JSON.
            info = dict(info, intervene_action=[float(v) for v in executed],
                        intervention_reason=self.reason)
            if self.remaining == 0:
                self.history = []      # judge the policy afresh once it is back
        if terminated or truncated:
            info = dict(info, intervention_count=self.count,
                        intervention_steps=self.intervention_steps)
        return observation, reward, terminated, truncated, info
