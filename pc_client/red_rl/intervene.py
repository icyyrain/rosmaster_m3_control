"""The rule expert as the corrector: S3's stand-in for a person with a SpaceMouse.

HIL-SERL puts the person in an environment wrapper: while they push the
SpaceMouse, the wrapper executes their action instead of the policy's and
reports it as info["intervene_action"]; the learner stores that executed
action and also copies the transition into the demonstration buffer. This
wrapper does the same with the rule expert, so swapping in a SpaceMouse later
changes this file and nothing in the learner.

It only takes over when the policy is clearly going wrong, so the policy is
corrected rather than replaced (docs/BC_RL_PLAN.md section 8):
  * the sweet is out of view, or more than ERROR_PX from the gripper pixel
    while the policy is not bringing it back (moving less than TOWARD_MM
    toward it this step);
  * below the grasp height (within 3 mm of the safety floor) and still going down;
  * STALL_STEPS steps with neither the height nor the alignment improving,
    which includes hovering ready to close without closing;
  * LAST_STEPS or fewer steps left and the policy is not closing.
Each takeover lasts up to TAKEOVER_STEPS steps, then control goes back.

Off unless the episode asks: reset(options={"intervene": True}). Evaluation
never asks, so it always measures the policy alone.

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
STALL_STEPS = 3
STALL_HEIGHT_MM = 0.5            # less descent than this over STALL_STEPS steps
STALL_ERROR_PX = 10.0            # and less alignment gain than this: no progress
LAST_STEPS = 3

UP = red_env.ACTION_NAMES.index('up')
CLOSE = red_env.ACTION_NAMES.index('close')


def reason_to_intervene(observation, action, history, steps_left):
    """Why the expert should take over now, or None.

    `history` holds (height_to_grasp_mm, error_px) of the observations since
    the policy last took control, latest last, including this one.
    """
    dx, dy, visible, height, _ = red_expert.read(observation)
    closing = action[CLOSE] > red_env.CLOSE_THRESHOLD
    if not visible:
        return 'target_not_visible'
    if math.hypot(dx, dy) > ERROR_PX and toward_mm(observation, action) < TOWARD_MM:
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
    return None


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
        self._clear()

    def _clear(self):
        self.observation = None
        self.history = []
        self.steps = 0
        self.remaining = 0
        self.reason = None
        self.count = 0
        self.intervention_steps = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.active = bool(options.pop('intervene', False))
        observation, info = self.env.reset(seed=seed, options=options)
        self._clear()
        self.observation = observation
        return observation, dict(info, intervention_enabled=self.active)

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
                                                  self.history, steps_left)
                if self.reason is not None:
                    self.remaining = self.takeover_steps
                    self.count += 1
                    print('  corrector takes over: {}'.format(self.reason))
            if self.remaining > 0:
                executed = np.asarray(self.corrector(self.observation), dtype=np.float32)
                intervened = True
                self.remaining -= 1
                self.intervention_steps += 1
                if executed[CLOSE] > red_env.CLOSE_THRESHOLD:
                    self.remaining = 0
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self.observation = observation
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
