"""Small, inspectable vision-to-joint demonstration format."""

import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import reach_candy


PHASES = ('align', 'descend', 'creep')
FEATURE_NAMES = (
    'target_dx', 'target_dy', 'target_range', 'gripper_height',
    'height_valid', 'target_visible',
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    'object_top', 'stop_height', 'target_area', 'object_aspect',
    'phase_align', 'phase_descend', 'phase_creep',
)


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


def load_examples(root, success_only=True):
    """Load valid transitions, grouped by episode for honest splits.

    Failed/unknown episodes remain on disk for future RL analysis but must not
    teach the imitation policy that an unsuccessful grasp is the goal.
    """
    episodes = []
    for meta_path in sorted(Path(root).glob('*/episode.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('format_version') != 1:
            continue
        if success_only and not (meta.get('outcome') == 'completed'
                                 and meta.get('human_success') is True):
            continue
        rows = []
        for line in (meta_path.parent / 'steps.jsonl').read_text(
                encoding='utf-8').splitlines():
            step = json.loads(line)
            if step['observation'][FEATURE_NAMES.index('target_visible')] != 1.0:
                continue
            if not step.get('next_target_visible'):
                continue
            rows.append(step)
        if rows:
            episodes.append((meta, rows))
    return episodes
