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
# The gripper pixel sits 40 px above the bottom of the frame, so an outward
# offset pushes the sweet toward the edge: +10.9 and +9.8 mm started it at
# y = 702 and 710 of 720, and a third outward start lost it altogether.
# Inward has the whole frame to spare.
START_OFFSET_MM = dict(radial=(-12.0, 6.0), lateral=(-12.0, 12.0), up=(0.0, 10.0))
START_ATTEMPTS = 3
# Where the sweet may be, as the gripper's distance from the base at the end
# of the rule descent (it predicts the grasp within about 4 mm, sd 3.6 over 107
# episodes). 190-245 mm kept 83 of 84 held episodes and would have stopped 5
# of 23 failures, among them three in the first S2 run with the sweet rolled
# out to 247-274 mm, further than any demonstration reached.
REACH_RADIUS_MM = (190.0, 245.0)
# A borderline check can flip between looks: S2 stopped on a colour margin of
# +3 against a pass mark of >3, the sweet lying logo-up with little red showing.
# Looking again costs a second; a scene that keeps failing still stops.
PREFLIGHT_ATTEMPTS = 3
MM_PER_PX = 0.17                 # near the table: about range / focal length
BOX_HALF_WIDTH_MM = 30.0         # horizontal distance allowed from the start
BELOW_GRASP_LIMIT_MM = 3.0       # lowest point: this far under the grasp height
STEP_MS = 600
STEP_SETTLE_S = 1.0
CLOSE_THRESHOLD = 0.0
# The grasp region: the rule expert closes here, and with reset option
# auto_close the environment closes here by itself, the policy only moving.
# Like the expert, it also tries on the last step if the gripper is low enough
# however far off: in the first auto-close evaluation (2026-10-20) three of
# twenty episodes ran out of steps at the grasp height, 45-65 px off and
# bouncing either side of the sweet, and grasps that far off have held.
GRASP_REGION_HEIGHT_MM = 1.5     # at most this far above the grasp height
GRASP_REGION_PX = 25.0           # and the sweet this close to the gripper pixel

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


def in_grasp_region(observation):
    """Sweet in view, gripper low enough and aligned: where closing is tried."""
    o = np.asarray(observation, dtype=float)
    error = math.hypot(o[0] * 640.0, o[1] * 360.0)
    return bool(o[2] > 0.5 and o[4] * 50.0 <= GRASP_REGION_HEIGHT_MM
                and error <= GRASP_REGION_PX)


def low_enough(observation):
    o = np.asarray(observation, dtype=float)
    return bool(o[2] > 0.5 and o[4] * 50.0 <= GRASP_REGION_HEIGHT_MM)


def auto_close_now(observation, last_step):
    """Under auto_close: in the grasp region, or low enough on the last step."""
    return in_grasp_region(observation) or (last_step and low_enough(observation))


def episode_step_limit(options):
    """Steps before truncation for one episode: options['max_steps'] or MAX_STEPS.

    Training may allow more than evaluation so that exploring does not run
    out of steps; the observation's step fraction stays normalised by
    MAX_STEPS either way, so the policy sees the same scale in both.
    """
    limit = int((options or {}).get('max_steps', MAX_STEPS))
    if not 1 <= limit <= 60:
        raise ValueError('max_steps must be 1-60, got {}'.format(limit))
    return limit


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


def safe_step(pose, request_mm, pitch_sum, height_of, lowest_mm, start_xy):
    """cartesian.step with the safety box applied as clipping, as HIL-SERL does.

    A step that would go below `lowest_mm` loses its downward part; one that
    would drift more than BOX_HALF_WIDTH_MM from `start_xy` loses its
    horizontal part. If what is left still breaks the box, the arm stays put.
    Returns (pose, achieved_local_mm, clipped).
    """
    def violations(candidate):
        position = kinematics.gripper_mm(candidate)
        drift = math.hypot(position[0] - start_xy[0], position[1] - start_xy[1])
        return height_of(candidate) < lowest_mm, drift > BOX_HALF_WIDTH_MM

    candidate, achieved = cartesian.step(pose, request_mm, pitch_sum)
    too_low, too_far = violations(candidate)
    if not (too_low or too_far):
        return candidate, achieved, False
    fallback = np.asarray(request_mm, dtype=float)
    if too_low:
        fallback[2] = max(0.0, fallback[2])
    if too_far:
        fallback[:2] = 0.0
    candidate, achieved = cartesian.step(pose, fallback, pitch_sum)
    if not any(violations(candidate)):
        return candidate, achieved, True
    return list(pose), np.zeros(3), True


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
        self.max_steps = MAX_STEPS
        self.auto_close = False
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

    def _safe_step(self, request_mm):
        return safe_step(self.session.pose, request_mm, self.pitch_sum,
                         self._height, self._lowest(), self.start_xy)

    def _predict_px(self, before, after):
        """Where a still sweet should appear after a move of several mm.

        The image Jacobian was measured high above the table and underestimates
        shifts down here; a metric scale does better for the start offset. A
        gripper moved outward sees the sweet lower (+y), moved left sees it to
        the left (-x).
        """
        moved = cartesian.base_to_local(
            before, kinematics.gripper_mm(after) - kinematics.gripper_mm(before))
        x, y = self.session.last_px
        return (x - moved[1] / MM_PER_PX, y + moved[0] / MM_PER_PX)

    def _start_offset(self):
        """Offset the gripper at random; retry from the descent pose if lost."""
        session = self.session
        descent_pose = list(session.pose)
        anchor = session.last_px
        for attempt in range(1, START_ATTEMPTS + 1):
            request = [self.rng.uniform(*START_OFFSET_MM[key])
                       for key in ('radial', 'lateral', 'up')]
            start = cartesian.offset(descent_pose, request, self.pitch_sum)
            predicted = self._predict_px(descent_pose, start)
            session.send(start, 900, 1.5)
            self.state = session.look(predicted)
            if self.state is not None:
                return request, attempt
            print('  start offset {} lost the sweet; back to the descent pose'.format(
                [round(v, 1) for v in request]))
            session.send(descent_pose, 900, 1.5)
            session.last_px = anchor
            if session.look(anchor) is None:
                break
        raise RuntimeError('Target lost after the start offset')

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
        self.max_steps = episode_step_limit(options)
        self.auto_close = bool((options or {}).get('auto_close', False))
        self.lost = 0
        self._go_home()
        try:
            for attempt in range(1, PREFLIGHT_ATTEMPTS + 1):
                if pick.preflight(session):
                    break
                if attempt == PREFLIGHT_ATTEMPTS:
                    raise RuntimeError('Preflight failed; the scene needs a person')
                print('  preflight failed, looking again ({}/{})'.format(
                    attempt, PREFLIGHT_ATTEMPTS))
                _fresh_episode(session)
                time.sleep(1.0)
            state = collect.align_teacher(session)
            if state is None:
                raise RuntimeError('Rule alignment failed')
            state = pick.descend(session, state)
            if state is None:
                raise RuntimeError('Rule descent lost the target')
            if state.get('height') is None:
                raise RuntimeError('No depth height at the end of the descent; '
                                   'cannot calibrate forward kinematics')
            reach = math.hypot(*kinematics.gripper_mm(session.pose)[:2])
            if not REACH_RADIUS_MM[0] <= reach <= REACH_RADIUS_MM[1]:
                raise RuntimeError(
                    'The sweet is {:.0f} mm from the base; episodes are only run '
                    'between {:.0f} and {:.0f} mm. Move it nearer the middle of the '
                    'view.'.format(reach, *REACH_RADIUS_MM))
            self.bias_mm = state['height'] - kinematics.gripper_height_mm(session.pose)
            self.pitch_sum = sum(session.pose[1:4])
            request, attempts = self._start_offset()
        except BaseException:
            self._go_home()
            raise
        self.start_xy = kinematics.gripper_mm(session.pose)[:2].tolist()
        name = self.audit.start({
            'object_top_mm': session.object_top_mm,
            'stop_height_mm': session.stop_height_mm,
            'bias_mm': self.bias_mm,
            'start_offset_request_mm': [round(v, 2) for v in request],
            'start_offset_attempts': attempts,
            'start_pose': list(session.pose),
            'pose_is_commanded_not_measured': True,
        })
        self.audit._image(session.streams.colour(), 'start.jpg')
        return self._observe(), self._info(episode=name, auto_close=self.auto_close)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        session = self.session
        self.steps += 1
        if action[3] > CLOSE_THRESHOLD and not self.auto_close:
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
        if self.auto_close and auto_close_now(observation, self.steps >= self.max_steps):
            return self._close(action)
        if self.steps >= self.max_steps:
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
        if evidence['reason'] == 'dropped_while_carrying':
            outcome = 'dropped'
        info = dict(grasp_info, outcome=outcome, reason=evidence['reason'],
                    relocated=relocation is not None and not relocation.get('dropped'))
        self.audit.step(dict(action=action.tolist(), **info))
        self.audit.finish(outcome=outcome, reward=reward, auto_evidence=evidence,
                          relocation=relocation)
        return self.observation, reward, True, False, info

    def close(self):
        self.session.close()


class FakeRedSweetEnv(gym.Env):
    """Same spaces and episode shape, no robot: for the learner and for tests.

    Kinematic, not a toy: actions go through the same cartesian.step, forward
    kinematics and safety clipping as the real arm, so joint limits, folding
    and a sweet pushed out of the frame show up here first. The sweet sits
    under the gripper at the end of a typical rule descent, a few millimetres
    off, 25-30 mm below; the start is offset with START_OFFSET_MM as on the
    real arm. The image uses 0.18 mm per pixel, deliberately not the 0.17 the
    expert assumes, with a little pixel and height noise. Closing holds the
    sweet within 4 mm horizontally and 3 mm of its grasp height.
    """

    metadata = dict(RedSweetApproachEnv.metadata, name='red_sweet_last_cm_fake')
    MM_PER_PX = 0.18
    FRAME = (1280.0, 720.0)

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.rng = np.random.default_rng(seed)
        self.pose = None
        self.sweet = None
        self.steps = 0
        self.max_steps = MAX_STEPS
        self.auto_close = False
        self.lost = 0

    def _relative(self):
        """(radial, lateral, height above grasp) of the gripper over the sweet."""
        local = cartesian.base_to_local(self.pose,
                                        kinematics.gripper_mm(self.pose) - self.sweet)
        return local

    def _height(self, pose):
        return float(kinematics.gripper_mm(pose)[2] - self.sweet[2])

    def _observation(self):
        radial, lateral, height = self._relative()
        x = (reach_candy.GRIPPER_PX[0] - lateral / self.MM_PER_PX
             + self.rng.normal(0.0, 2.0))
        y = (reach_candy.GRIPPER_PX[1] + radial / self.MM_PER_PX
             + self.rng.normal(0.0, 2.0))
        visible = 0.0 <= x < self.FRAME[0] and 0.0 <= y < self.FRAME[1]
        self.lost = 0 if visible else self.lost + 1
        return encode((x, y, 4000.0) if visible else None,
                      height + self.rng.normal(0.0, 0.3), self.pose, self.steps)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.max_steps = episode_step_limit(options)
        self.auto_close = bool((options or {}).get('auto_close', False))
        self.lost = 0
        descent = [int(self.rng.integers(80, 101)), 18, 62,
                   int(self.rng.integers(18, 31)), 90, 90]
        self.pitch_sum = sum(descent[1:4])
        below = kinematics.gripper_mm(descent)
        misalign = cartesian.local_to_base(
            descent, [*self.rng.normal(0.0, 3.0, 2), 0.0])
        self.sweet = below + misalign - np.asarray(
            [0.0, 0.0, self.rng.uniform(25.0, 30.0)])
        # Like the real reset: redraw a start that loses the sweet.
        for _ in range(START_ATTEMPTS):
            request = [self.rng.uniform(*START_OFFSET_MM[key])
                       for key in ('radial', 'lateral', 'up')]
            self.pose = cartesian.offset(descent, request, self.pitch_sum)
            if self._in_frame():
                break
        self.start_xy = kinematics.gripper_mm(self.pose)[:2].tolist()
        return self._observation(), {'fake': True, 'auto_close': self.auto_close}

    def _grasp(self, observation):
        radial, lateral, height = self._relative()
        held = math.hypot(radial, lateral) <= 4.0 and abs(height) <= 3.0
        return (observation, float(held), True, False,
                {'fake': True, 'outcome': 'held' if held else 'not_held'})

    def _in_frame(self):
        radial, lateral, _ = self._relative()
        x = reach_candy.GRIPPER_PX[0] - lateral / self.MM_PER_PX
        y = reach_candy.GRIPPER_PX[1] + radial / self.MM_PER_PX
        return 0.0 <= x < self.FRAME[0] and 0.0 <= y < self.FRAME[1]

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.steps += 1
        if action[3] > CLOSE_THRESHOLD and not self.auto_close:
            return self._grasp(self._observation())
        self.pose, _, clipped = safe_step(
            self.pose, action[:3] * cartesian.MAX_STEP_MM, self.pitch_sum,
            self._height, -BELOW_GRASP_LIMIT_MM, self.start_xy)
        observation = self._observation()
        info = {'fake': True, 'clipped': bool(clipped), 'outcome': None}
        if self.lost >= LOST_LIMIT:
            return observation, 0.0, True, False, dict(info, outcome='target_lost')
        if self.auto_close and auto_close_now(observation, self.steps >= self.max_steps):
            return self._grasp(observation)
        truncated = self.steps >= self.max_steps
        if truncated:
            info['outcome'] = 'step_limit'
        return observation, 0.0, False, truncated, info
