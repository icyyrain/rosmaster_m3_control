"""Combine demonstration files into the single dataset the learner loads.

    python -m pc_client.red_rl.demos merge OUT.npz IN1.success.npz IN2.success.npz ...
    python -m pc_client.red_rl.demos summary FILE.npz

Inputs and output use the simulation project's transition schema (version 1,
see run_expert.save_transitions). Episodes keep their order and are renumbered
from 0, as the schema requires; a JSON file next to the output records where
every episode came from and the SHA-256 of each input.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from pc_client.red_rl import run_expert

FIELDS = ('observations', 'policy_actions', 'executed_actions', 'next_observations',
          'rewards', 'terminated', 'truncated', 'sources', 'interventions',
          'episode_ids', 'step_ids', 'successes')


def load(path):
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data['schema_version'])
        if version != run_expert.SCHEMA_VERSION:
            raise ValueError('{}: schema version {}'.format(path, version))
        return {field: data[field].copy() for field in FIELDS}


def episodes(arrays):
    """Split one file into per-episode index arrays, checking the boundaries."""
    ends = np.flatnonzero(arrays['terminated'] | arrays['truncated'])
    if len(ends) == 0 or ends[-1] != len(arrays['rewards']) - 1:
        raise ValueError('the last transition does not end an episode')
    starts = np.concatenate([[0], ends[:-1] + 1])
    split = []
    for start, end in zip(starts, ends):
        index = np.arange(start, end + 1)
        if not (arrays['step_ids'][index] == np.arange(len(index))).all():
            raise ValueError('step ids are not 0..n within an episode')
        if len(set(arrays['episode_ids'][index].tolist())) != 1:
            raise ValueError('one episode carries several episode ids')
        split.append(index)
    return split


def merge(paths, out):
    transitions, origin = [], []
    for path in paths:
        arrays = load(path)
        for local_id, index in enumerate(episodes(arrays)):
            origin.append(dict(file=Path(path).name, episode=local_id,
                               steps=int(len(index)),
                               success=bool(arrays['successes'][index[-1]])))
            for i in index:
                transitions.append(dict(
                    observation=arrays['observations'][i],
                    action=arrays['executed_actions'][i],
                    next_observation=arrays['next_observations'][i],
                    reward=float(arrays['rewards'][i]),
                    terminated=bool(arrays['terminated'][i]),
                    truncated=bool(arrays['truncated'][i]),
                    step_id=int(arrays['step_ids'][i]),
                    episode_id=len(origin) - 1))
    out = run_expert.save_transitions(out, transitions)
    summary = dict(
        episodes=len(origin), transitions=len(transitions),
        successful_episodes=sum(1 for o in origin if o['success']),
        inputs=[dict(path=str(p), sha256=run_expert.sha256(p)) for p in paths],
        output_sha256=run_expert.sha256(out), origin=origin)
    Path(out).with_suffix('.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    merge_parser = commands.add_parser('merge')
    merge_parser.add_argument('out')
    merge_parser.add_argument('inputs', nargs='+')
    summary_parser = commands.add_parser('summary')
    summary_parser.add_argument('path')
    args = parser.parse_args()
    if args.command == 'merge':
        summary = merge(args.inputs, args.out)
        print('{episodes} episodes ({successful_episodes} successful), '
              '{transitions} transitions -> {0}'.format(args.out, **summary))
        return 0
    arrays = load(args.path)
    split = episodes(arrays)
    lengths = [len(index) for index in split]
    print('{} episodes, {} transitions, {} successful, steps {}-{} (median {})'.format(
        len(split), len(arrays['rewards']), int(arrays['successes'].sum()),
        min(lengths), max(lengths), int(np.median(lengths))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
