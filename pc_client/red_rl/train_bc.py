"""Train a conservative, offline behaviour-cloning baseline from demonstrations.

This is *not* yet online reinforcement learning. It checks that collected
camera-derived observations predict the teacher's joint-angle increments.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pc_client.red_rl.dataset import (CURRENT_GRASP_RULE, FEATURE_NAMES,
                                      grasp_rule, load_examples)


def arrays(episodes):
    x = np.asarray([step['observation'] for _, rows in episodes for step in rows],
                   dtype=np.float64)
    y = np.asarray([step['action_joint_delta_deg']
                    for _, rows in episodes for step in rows], dtype=np.float64)
    return x, y


def fit(x, y, ridge=1.0):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.05)
    normal = (x - mean) / scale
    design = np.column_stack([normal, np.ones(len(normal))])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return weights, mean, scale


def predict(x, weights, mean, scale):
    design = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    return design @ weights


def report(name, x, y, weights, mean, scale):
    if len(x) == 0:
        return None
    pred = predict(x, weights, mean, scale)
    mae = np.abs(pred - y).mean(axis=0)
    print('{}: {} steps, joint 1-4 MAE [deg]: {}'.format(
        name, len(x), ', '.join('{:.2f}'.format(v) for v in mae)))
    return mae.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='datasets/red_sweet')
    parser.add_argument('--output', default='models/red_sweet_bc.npz')
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--creep-from', choices=('current', 'all'), default='current',
                        help='current: creep steps only from the current grasp '
                             'rule ({}); align/descend always use every rule. '
                             'all: also pool legacy creep steps.'.format(
                                 CURRENT_GRASP_RULE))
    args = parser.parse_args()
    if args.ridge <= 0:
        parser.error('--ridge must be positive')
    episodes = load_examples(
        args.dataset,
        creep_rule=CURRENT_GRASP_RULE if args.creep_from == 'current' else None)
    if not episodes:
        parser.error('No complete episodes with visible before/after targets')
    composition = {}
    for meta, rows in episodes:
        for row in rows:
            key = (grasp_rule(meta), row['phase'])
            composition[key] = composition.get(key, 0) + 1
    print('training data (human-confirmed successes, steps by rule and phase):')
    for (rule, phase), count in sorted(composition.items()):
        print('  {:<16} {:<8} {}'.format(rule, phase, count))
    rng = np.random.default_rng(20261010)
    order = rng.permutation(len(episodes))
    split = max(1, int(round(0.8 * len(episodes))))
    if len(episodes) > 1:
        split = min(split, len(episodes) - 1)
    train = [episodes[i] for i in order[:split]]
    validation = [episodes[i] for i in order[split:]]
    x_train, y_train = arrays(train)
    if len(x_train) < 5:
        parser.error('Need at least five valid pre-grasp transitions')
    weights, mean, scale = fit(x_train, y_train, args.ridge)
    train_mae = report('train', x_train, y_train, weights, mean, scale)
    validation_mae = None
    if validation:
        x_val, y_val = arrays(validation)
        validation_mae = report('held-out episodes', x_val, y_val,
                                weights, mean, scale)
    else:
        print('Only one episode: no held-out validation; do not deploy this model.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, weights=weights, feature_mean=mean, feature_scale=scale,
             feature_names=np.asarray(FEATURE_NAMES),
             train_mae=np.asarray(train_mae),
             validation_mae=np.asarray(validation_mae or []),
             creep_from=np.asarray(args.creep_from),
             training_episodes=np.asarray([m['episode'] for m, _ in train]),
             validation_episodes=np.asarray([m['episode'] for m, _ in validation]))
    print('Saved {}'.format(output))
    print('This model is offline only; there is intentionally no robot rollout command.')


if __name__ == '__main__':
    main()
