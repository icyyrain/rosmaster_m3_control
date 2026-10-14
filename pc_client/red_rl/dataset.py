"""Small, inspectable vision-to-joint demonstration format."""

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import pick_and_place
import reach_candy


PHASES = ('align', 'descend', 'creep')

# Which grasp-height rule the rule teacher used. Before 2026-10-14 the final
# approach stopped at top + 2 mm below a 26 mm switch and top + 8 mm above it;
# since then it stops at mid-height + 10 mm. Episodes recorded before the field
# existed are all legacy. Only the creep phase depends on the rule: align and
# descend run the same code under both, so their steps can be pooled.
LEGACY_GRASP_RULE = 'top_switch_26mm'
CURRENT_GRASP_RULE = 'mid_height_v1'
RULE_INDEPENDENT_PHASES = ('align', 'descend')
FEATURE_NAMES = (
    'target_dx', 'target_dy', 'target_range', 'gripper_height',
    'height_valid', 'target_visible',
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    'object_top', 'stop_height', 'target_area', 'object_aspect',
    'phase_align', 'phase_descend', 'phase_creep',
)
_STOP_HEIGHT = FEATURE_NAMES.index('stop_height')


def grasp_rule(meta):
    return meta.get('grasp_rule', LEGACY_GRASP_RULE)


def current_stop_height_mm(object_top_mm):
    """Where today's pick_and_place would stop for an object this tall.

    Asks pick_and_place itself rather than restating its formula, so a later
    rule change cannot silently disagree with this conversion.
    """
    probe = SimpleNamespace(args=SimpleNamespace(grasp_offset_mm=None))
    pick_and_place.Session.set_object_top(probe, object_top_mm)
    return probe.stop_height_mm


def features(state, pose, session, phase):
    """Use only the camera-derived state and last *commanded* joint pose.

    The pose is not encoder feedback. Missing vision/depth is explicit rather
    than silently imputed as a real measurement.
    """
    visible = state is not None
    height = state.get('height') if visible else None
    cx = state['cx'] if visible else reach_candy.GRIPPER_PX[0]
    cy = state['cy'] if visible else reach_candy.GRIPPER_PX[1]
    return np.asarray([
        (cx - reach_candy.GRIPPER_PX[0]) / 640.0,
        (cy - reach_candy.GRIPPER_PX[1]) / 360.0,
        ((state.get('range_mm') or 0.0) if visible else 0.0) / 300.0,
        (height or 0.0) / 200.0,
        float(height is not None), float(visible),
        *[float(j) / 180.0 for j in pose],
        (session.object_top_mm or 0.0) / 50.0,
        session.stop_height_mm / 100.0,
        ((state.get('area') or 0.0) if visible else 0.0) / 50000.0,
        (session.object_aspect or 0.0) / 5.0,
        *[float(phase == p) for p in PHASES],
    ], dtype=np.float32)


class EpisodeRecorder:
    """Records each pre-grasp command and its subsequent observation."""

    def __init__(self, root, metadata):
        name = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        self.path = Path(root) / name
        (self.path / 'images').mkdir(parents=True)
        self.log = (self.path / 'steps.jsonl').open('w', encoding='utf-8')
        self.metadata = dict(metadata, format_version=1, episode=name,
                             feature_names=FEATURE_NAMES)
        self.count = 0
        self.pending = None

    def _image(self, frame, name):
        if frame is None:
            return None
        target = self.path / 'images' / name
        if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError('Could not save {}'.format(target))
        return str(target.relative_to(self.path)).replace('\\', '/')

    def before_send(self, state, pose, target, phase, session, frame):
        if self.pending is not None:
            # A command with no subsequent camera read cannot be an RL transition.
            self.after_look(None, target, session, None)
        index = self.count
        self.count += 1
        self.pending = {
            'step': index, 'phase': phase, 'time_s': time.time(),
            'observation': features(state, pose, session, phase).tolist(),
            'image': self._image(frame, '{:04d}-before.jpg'.format(index)),
            'commanded_pose_before': list(pose),
            'commanded_pose_after': list(target),
            'action_joint_delta_deg': [target[i] - pose[i] for i in range(4)],
            'duration_ms': None,
        }

    def after_look(self, state, pose, session, frame):
        if self.pending is None:
            return
        item = self.pending
        item['next_observation'] = features(state, pose, session, item['phase']).tolist()
        item['next_image'] = self._image(frame, '{:04d}-after.jpg'.format(item['step']))
        item['next_target_visible'] = state is not None
        self.log.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.log.flush()
        self.pending = None

    def finish(self, outcome, human_success=None, notes=None,
               auto_success=None, auto_evidence=None):
        if self.pending is not None:
            raise RuntimeError('Last action has no following observation')
        self.metadata.update(outcome=outcome, human_success=human_success,
                             auto_success=auto_success,
                             auto_evidence=auto_evidence,
                             notes=notes, steps=self.count, finished_s=time.time())
        (self.path / 'episode.json').write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        self.log.close()


def _harmonise_stop_height(step, meta):
    """Rewrite a legacy episode's stop-height feature in today's rule.

    The policy sees the stop height as an input. Left alone, the same sweet
    reads 35 mm in a legacy episode and 23.5 mm in a current one, which is a
    difference in the teacher, not in the scene.
    """
    top = meta.get('object_top_mm')
    if top is None:
        return step
    value = current_stop_height_mm(top) / 100.0
    step = dict(step)
    for key in ('observation', 'next_observation'):
        vector = list(step[key])
        vector[_STOP_HEIGHT] = value
        step[key] = vector
    return step


def load_examples(root, success_only=True, creep_rule=CURRENT_GRASP_RULE,
                  harmonise=True):
    """Load valid transitions, grouped by episode for honest splits.

    Failed/unknown episodes remain on disk for future RL analysis but must not
    teach the imitation policy that an unsuccessful grasp is the goal.

    Align and descend steps come from every grasp rule. Creep steps come only
    from `creep_rule` (None keeps every rule), because the old and new rules
    stop the final approach about 6 mm apart and mixing them would teach the
    average of two teachers. With `harmonise`, legacy episodes have their
    stop-height feature restated in the current rule.
    """
    episodes = []
    for meta_path in sorted(Path(root).glob('*/episode.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('format_version') != 1:
            continue
        if success_only and not (meta.get('outcome') == 'completed'
                                 and meta.get('human_success') is True):
            continue
        rule = grasp_rule(meta)
        rows = []
        for line in (meta_path.parent / 'steps.jsonl').read_text(
                encoding='utf-8').splitlines():
            step = json.loads(line)
            if step['observation'][FEATURE_NAMES.index('target_visible')] != 1.0:
                continue
            if not step.get('next_target_visible'):
                continue
            if (creep_rule is not None and rule != creep_rule
                    and step['phase'] not in RULE_INDEPENDENT_PHASES):
                continue
            if harmonise and rule != CURRENT_GRASP_RULE:
                step = _harmonise_stop_height(step, meta)
            rows.append(step)
        if rows:
            episodes.append((meta, rows))
    return episodes
