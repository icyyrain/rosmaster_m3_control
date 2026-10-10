"""Read-only student action predictions alongside rule-teacher commands."""

from pathlib import Path

import numpy as np

from pc_client.red_rl.dataset import FEATURE_NAMES
from pc_client.red_rl.train_bc import predict


class ShadowPolicy:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as model:
            if tuple(model['feature_names'].tolist()) != FEATURE_NAMES:
                raise ValueError('Shadow model feature schema does not match the recorder')
            self.weights = model['weights'].copy()
            self.mean = model['feature_mean'].copy()
            self.scale = model['feature_scale'].copy()
            self.training_episodes = model['training_episodes'].tolist()
        if self.weights.shape != (len(FEATURE_NAMES) + 1, 4):
            raise ValueError('Shadow model must predict four joint increments')

    def action(self, observation):
        value = predict(np.asarray(observation, dtype=np.float64)[None, :],
                        self.weights, self.mean, self.scale)[0]
        if not np.all(np.isfinite(value)):
            raise ValueError('Shadow model produced a non-finite action')
        return value.tolist()


def episode_report(folder):
    """Summarise only steps with visible targets; never issues robot commands."""
    import json

    rows = [json.loads(line) for line in (Path(folder) / 'steps.jsonl').read_text(
        encoding='utf-8').splitlines()]
    rows = [row for row in rows if 'shadow_action_joint_delta_deg' in row
            and row['observation'][FEATURE_NAMES.index('target_visible')] == 1.0]
    if not rows:
        return None
    error = np.asarray([row['shadow_abs_error_deg'] for row in rows])
    by_phase = {}
    for phase in ('align', 'descend', 'creep'):
        indices = [i for i, row in enumerate(rows) if row['phase'] == phase]
        if indices:
            by_phase[phase] = dict(steps=len(indices),
                                   mae_deg=error[indices].mean(axis=0).round(2).tolist())
    return dict(steps=len(rows), mae_deg=error.mean(axis=0).round(2).tolist(),
                by_phase=by_phase)
