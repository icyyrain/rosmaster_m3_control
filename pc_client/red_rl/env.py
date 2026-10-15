"""Gym environment for the last few centimetres of the red-sweet grasp.

One episode, as in docs/BC_RL_PLAN.md:

  reset()  HOME -> preflight -> rule alignment -> rule descent to ~50 mm
           -> random end-effector offset. Returns the first observation.
  step(a)  a = (radial, lateral, up, close), each in [-1, 1].
           close > 0: fixed grip, short lift and camera verdict, reward 1 if
           the camera confirms the hold and 0 otherwise; then the sweet is put
           down (at a random reachable spot if held) and the arm goes HOME.
           Otherwise one end-effector step of up to cartesian.MAX_STEP_MM
           along each axis, then a fresh look.

Safety follows HIL-SERL: a move that would leave the box (too low, or too far
from the episode's start) is clipped rather than punished. Losing the target
for LOST_LIMIT looks in a row, or MAX_STEPS without closing, ends the episode
with reward 0. Every ending leaves the arm HOME with the jaws open, so the
learner may stop between any two episodes.

The observation is a flat float32 vector (OBSERVATION_NAMES); the learner wraps
it as {"state": ...} like the simulation project. Joint values are the last
*commanded* angles; this arm reports none back.

Reset failures (preflight, alignment, descent, no depth calibration) raise:
they need a person, not another episode.
"""

import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np

import arm_control
import kinematics
import pick_and_place as pick
import reach_candy
from pc_client.red_rl import cartesian
from pc_client.red_rl import collect

MAX_STEPS = 20
LOST_LIMIT = 2
START_OFFSET_MM = dict(radial=(-12.0, 12.0), lateral=(-12.0, 12.0), up=(0.0, 10.0))
BOX_HALF_WIDTH_MM = 30.0         # horizontal distance allowed from the start
BELOW_GRASP_LIMIT_MM = 3.0       # lowest point: this far under the grasp height
STEP_MS = 600
STEP_SETTLE_S = 1.0
CLOSE_THRESHOLD = 0.0

OBSERVATION_NAMES = (
    'target_dx', 'target_dy', 'target_visible', 'target_area',
    'height_to_grasp', 'joint1', 'joint2', 'joint3', 'joint4', 'step_fraction',
)
ACTION_NAMES = ('radial', 'lateral', 'up', 'close')


def teacher_args(host='192.168.2.4', port=9090, execute=True):
    """The red-sweet collector's settings, which the rule phases reuse."""
    return SimpleNamespace(
        host=host, port=port, detector='table', colour='red-sat',
        a_star=140, min_area=150, width_mm=28.0, auto_width=True,
        descent_steps=10, settle=0.6, fast=True, creep_controller='auto',
        no_orient_gripper=True, skip_preflight=False, grasp_offset_mm=None,
        squeeze_mm=5.0, keep_holding=False, record=None, execute=execute)


def observation_space():
    return gym.spaces.Box(low=-np.inf, high=np.inf,
                          shape=(len(OBSERVATION_NAMES),), dtype=np.float32)


def action_space():
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_NAMES),),
                          dtype=np.float32)


def encode(target, height_to_grasp_mm, pose, step_index):
    """Observation vector. `target` is (cx, cy, area) or None."""
    visible = target is not None
    cx, cy, area = target if visible else (reach_candy.GRIPPER_PX[0],
                                           reach_candy.GRIPPER_PX[1], 0.0)
    return np.asarray([
        (cx - reach_candy.GRIPPER_PX[0]) / 640.0,
        (cy - reach_candy.GRIPPER_PX[1]) / 360.0,
        float(visible),
        area / 50000.0,
        height_to_grasp_mm / 50.0,
        *[pose[j] / 180.0 for j in range(4)],
        step_index / float(MAX_STEPS),
    ], dtype=np.float32)


class AuditLog:
    """Per-episode folder with the photos and a JSON summary, for checking."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = None

    def start(self, meta):
        name = time.strftime('%Y%m%d-%H%M%S') + '-' + '%06x' % random.getrandbits(24)
        self.path = self.root / name
        (self.path / 'images').mkdir(parents=True)
        self.meta = dict(meta, episode=name, steps=[])
        return name

    def _image(self, frame, name):        # the interface collect._audit_image uses
        if frame is None or self.path is None:
            return None
        target = self.path / 'images' / name
        if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise OSError('Could not save {}'.format(target))
        return 'images/' + name

    def step(self, record):
        if self.path is not None:
            self.meta['steps'].append(record)

    def finish(self, **summary):
        if self.path is None:
            return
        self.meta.update(summary)
        (self.path / 'episode.json').write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        self.path = None


def _fresh_episode(session):
    """Clear what one episode leaves in a reused Session.

    One process can open rosbridge only once, so the Session lives across
    episodes. These are the fields pick_and_place.Session.__init__ sets and the
    pick phases change; the resolved colour direction is deliberately kept.
    """
    session.last_px = None
    session.width_mm = session.args.width_mm
    session.acquired = False
    session.object_top_mm = None
    session.grasp_offset_mm = None
    session.stop_height_mm = pick.HEIGHT_FLOOR_MM
    session.object_aspect = None
    session.lock_clipped_target = False
    session.clipped_target_edge = None
    session.target_geometry_initial = None
    session.target_offset_px = (0.0, 0.0)
    session.acquired_object = None


class RedSweetApproachEnv(gym.Env):
    metadata = {'name': 'red_sweet_last_cm', 'version': 1,
                'observation_names': list(OBSERVATION_NAMES),
                'action_names': list(ACTION_NAMES), 'max_steps': MAX_STEPS}

    def __init__(self, session=None, args=None, audit_root='datasets/red_sweet_rl',
                 seed=None):
        super().__init__()
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.args = args or teacher_args()
        self.session = session if session is not None else collect.TeacherSession(self.args)
        self.audit = AuditLog(audit_root)
        self.rng = random.Random(seed)
        self.bias_mm = 0.0
        self.start_xy = None
        self.pitch_sum = None
        self.steps = 0
        self.lost = 0
        self.state = None
        self.observation = None

    # ---- helpers ---------------------------------------------------------
    def _height(self, pose):
        """Calibrated gripper height, in the plane-fit's terms."""
        return kinematics.gripper_height_mm(pose) + self.bias_mm

    def _lowest(self):
        return max(pick.HEIGHT_FLOOR_MM,
                   self.session.stop_height_mm - BELOW_GRASP_LIMIT_MM)

    def _violations(self, pose):
        """(too_low, too_far) for a candidate pose."""
        position = kinematics.gripper_mm(pose)
        drift = math.hypot(position[0] - self.start_xy[0],
                           position[1] - self.start_xy[1])
        return self._height(pose) < self._lowest(), drift > BOX_HALF_WIDTH_MM

    def _safe_step(self, request_mm):
        """cartesian.step with the box applied as clipping, as HIL-SERL does.

        A step that would go too low loses its downward part; one that would
        go too far loses its horizontal part. If what is left still breaks
        the box, the arm stays put.
        """
        pose = self.session.pose
        candidate, achieved = cartesian.step(pose, request_mm, self.pitch_sum)
        too_low, too_far = self._violations(candidate)
        if not (too_low or too_far):
            return candidate, achieved, False
        fallback = np.asarray(request_mm, dtype=float)
        if too_low:
            fallback[2] = max(0.0, fallback[2])
        if too_far:
            fallback[:2] = 0.0
        candidate, achieved = cartesian.step(pose, fallback, self.pitch_sum)
        if not any(self._violations(candidate)):
            return candidate, achieved, True
        return list(pose), np.zeros(3), True

    def _target(self, state):
        return None if state is None else (state['cx'], state['cy'], state['area'])

    def _observe(self):
        self.observation = encode(self._target(self.state),
                                  self._height(self.session.pose)
                                  - self.session.stop_height_mm,
                                  self.session.pose, self.steps)
        return self.observation

    def _go_home(self):
        self.session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                          arm_control.HOMING_TIME_MS / 1000.0 + 1.0)

    def _retreat(self):
        """Jaws are open: rise along the known lift path, then HOME."""
        lift, _ = pick.low_release_pose(self.session.pose, clearance_mm=35.0)
        self.session.send(lift, 1100, 1.8)
        self._go_home()

    def _info(self, **extra):
        state = self.state
        info = {
            'pose': [int(v) for v in self.session.pose],
            'height_mm': round(self._height(self.session.pose), 2),
            'stop_height_mm': round(float(self.session.stop_height_mm), 2),
            'target_px': ([round(state['cx'], 1), round(state['cy'], 1)]
                          if state is not None else None),
            'step': self.steps,
        }
        info.update(extra)
        return info

    # ---- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        session = self.session
        _fresh_episode(session)
        self.steps = 0
        self.lost = 0
        self._go_home()
        try:
            if not pick.preflight(session):
                raise RuntimeError('Preflight failed; the scene needs a person')
            state = collect.align_teacher(session)
            if state is None:
                raise RuntimeError('Rule alignment failed')
            state = pick.descend(session, state)
            if state is None:
                raise RuntimeError('Rule descent lost the target')
            if state.get('height') is None:
                raise RuntimeError('No depth height at the end of the descent; '
                                   'cannot calibrate forward kinematics')
            self.bias_mm = state['height'] - kinematics.gripper_height_mm(session.pose)
            self.pitch_sum = sum(session.pose[1:4])
            request = [self.rng.uniform(*START_OFFSET_MM[key])
                       for key in ('radial', 'lateral', 'up')]
            start = cartesian.offset(session.pose, request, self.pitch_sum)
            delta = tuple(start[j] - session.pose[j] for j in range(4))
            predicted = reach_candy.predict_target(session.last_px, delta)
            session.send(start, 900, 1.5)
            self.state = session.look(predicted)
            if self.state is None:
                raise RuntimeError('Target lost after the start offset')
        except BaseException:
            self._go_home()
            raise
        self.start_xy = kinematics.gripper_mm(session.pose)[:2].tolist()
        name = self.audit.start({
            'object_top_mm': session.object_top_mm,
            'stop_height_mm': session.stop_height_mm,
            'bias_mm': self.bias_mm,
            'start_offset_request_mm': [round(v, 2) for v in request],
            'start_pose': list(session.pose),
            'pose_is_commanded_not_measured': True,
        })
        self.audit._image(session.streams.colour(), 'start.jpg')
        return self._observe(), self._info(episode=name)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        session = self.session
        self.steps += 1
        if action[3] > CLOSE_THRESHOLD:
            return self._close(action)

        request = action[:3] * cartesian.MAX_STEP_MM
        pose, achieved, clipped = self._safe_step(request)
        delta = tuple(pose[j] - session.pose[j] for j in range(4))
        if any(delta):
            predicted = reach_candy.predict_target(session.last_px, delta)
            session.send(pose, STEP_MS, STEP_SETTLE_S)
            self.state = session.look(predicted)
        else:
            self.state = session.look(session.last_px)
        self.lost = 0 if self.state is not None else self.lost + 1
        observation = self._observe()
        info = self._info(achieved_mm=[round(v, 2) for v in achieved],
                          clipped=bool(clipped))
        self.audit.step(dict(action=action.tolist(), **info))
        if self.lost >= LOST_LIMIT:
            self._retreat()
            self.audit.finish(outcome='target_lost', reward=0.0)
            return observation, 0.0, True, False, dict(info, outcome='target_lost')
        if self.steps >= MAX_STEPS:
            self._retreat()
            self.audit.finish(outcome='step_limit', reward=0.0)
            return observation, 0.0, False, True, dict(info, outcome='step_limit')
        return observation, 0.0, False, False, info

    def _close(self, action):
        session = self.session
        state = pick.grip(session, self.state or {})
        grasp_pose = list(session.pose)
        grasp_info = self._info()
        _, evidence, relocation = collect.short_lift_and_release(
            session, grasp_pose, state, self.audit, auto=True, rng=self.rng)
        reward = 1.0 if evidence['success'] is True else 0.0
        outcome = {True: 'held', False: 'not_held', None: 'inconclusive'}[
            evidence['success']]
        info = dict(grasp_info, outcome=outcome, reason=evidence['reason'],
                    relocated=relocation is not None)
        self.audit.step(dict(action=action.tolist(), **info))
        self.audit.finish(outcome=outcome, reward=reward, auto_evidence=evidence,
                          relocation=relocation)
        return self.observation, reward, True, False, info

    def close(self):
        self.session.close()


class FakeRedSweetEnv(gym.Env):
    """Same spaces and episode shape, no robot: for wiring up the learner.

    A point gripper above a point sweet. Actions move it by up to
    cartesian.MAX_STEP_MM with a little noise; closing succeeds when it is
    within 4 mm horizontally and 3 mm of the grasp height. Pixel offsets use
    the real arm's scale near the table, about 0.18 mm per pixel.
    """

    metadata = dict(RedSweetApproachEnv.metadata, name='red_sweet_last_cm_fake')
    MM_PER_PX = 0.18

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.rng = np.random.default_rng(seed)
        self.offset = None
        self.steps = 0

    def _observation(self):
        # offset = gripper minus sweet: (radial, lateral, height above grasp).
        # On the real arm a gripper beyond the sweet sees it lower in the
        # image (+y), and one to the sweet's left sees it to the left (-x).
        radial, lateral, height = self.offset
        target = (reach_candy.GRIPPER_PX[0] - lateral / self.MM_PER_PX,
                  reach_candy.GRIPPER_PX[1] + radial / self.MM_PER_PX, 4000.0)
        pose = [90, 12 + height / 4.0, 56, 32]
        return encode(target, height, pose, self.steps)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.offset = np.asarray([
            self.rng.uniform(*START_OFFSET_MM['radial']),
            self.rng.uniform(*START_OFFSET_MM['lateral']),
            25.0 + self.rng.uniform(*START_OFFSET_MM['up'])])
        return self._observation(), {'fake': True}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.steps += 1
        if action[3] > CLOSE_THRESHOLD:
            held = (math.hypot(self.offset[0], self.offset[1]) <= 4.0
                    and abs(self.offset[2]) <= 3.0)
            return (self._observation(), float(held), True, False,
                    {'fake': True, 'outcome': 'held' if held else 'not_held'})
        move = action[:3] * cartesian.MAX_STEP_MM + self.rng.normal(0.0, 0.5, 3)
        self.offset = self.offset + move
        self.offset[2] = max(self.offset[2], -BELOW_GRASP_LIMIT_MM)
        truncated = self.steps >= MAX_STEPS
        return (self._observation(), 0.0, False, truncated,
                {'fake': True, 'outcome': 'step_limit' if truncated else None})
