"""Combine demonstration files into the single dataset the learner loads.

    python -m pc_client.red_rl.demos merge OUT.npz IN1.success.npz IN2.success.npz ...
    python -m pc_client.red_rl.demos build OUT.npz RUN1.json RUN2.json ...
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

from pc_client.red_rl import auto_reward
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


def judge(audit_meta):
    """Would today's rules keep this episode? Returns (keep, reason).

    A person's "not held" always excludes. Otherwise the stored short-lift
    evidence is judged again with the current auto_reward thresholds, so a
    rule tightened after collection applies to old episodes too, and a sweet
    dropped while being carried is excluded.
    """
    if audit_meta.get('human_success') is False:
        return False, 'person: not held'
    evidence = audit_meta.get('auto_evidence') or {}
    if (audit_meta.get('relocation') or {}).get('dropped'):
        return False, 'dropped while carrying'
    turn = evidence.get('turn_check') or {}
    if turn.get('success') is not None:
        # The base turn decided at collection time; its thresholds are applied
        # there, to the photographs, so the stored verdict stands.
        if turn['success'] is False and audit_meta.get('human_success') is not True:
            return False, 'camera: not held ({})'.format(turn['reason'])
        return True, turn['reason']
    if None in (evidence.get('before_px'), evidence.get('after_px'),
                evidence.get('stationary_prediction_px')):
        return audit_meta.get('human_success') is True, 'no lift evidence'
    verdict = auto_reward.assess_short_lift(
        evidence['before_px'], evidence['after_px'],
        evidence['stationary_prediction_px'], evidence['commanded_rise_mm'])
    if verdict['success'] is not True and audit_meta.get('human_success') is not True:
        return False, 'camera now: {} ({})'.format(
            {False: 'not held', None: 'cannot tell'}[verdict['success']],
            verdict['reason'])
    return True, verdict['reason']


def audit_names(run, audit_root):
    """Audit folder of each outcome of a run, in order.

    Runs recorded before run_expert stored the name are matched by time: the
    folders created from the run's start onward, one per outcome, each of
    whose recorded outcome must agree with the run's.
    """
    names = [outcome.get('episode') for outcome in run['outcomes']]
    if all(names):
        return names
    start = run['name'][:15]
    folders = sorted(path.name for path in Path(audit_root).iterdir()
                     if path.is_dir() and path.name[:15] >= start)
    names = folders[:len(run['outcomes'])]
    for name, outcome in zip(names, run['outcomes']):
        meta = json.loads((Path(audit_root) / name / 'episode.json').read_text(
            encoding='utf-8'))
        if meta.get('outcome') != outcome['outcome']:
            raise ValueError('{}: audit {} says {}, run says {}'.format(
                run['name'], name, meta.get('outcome'), outcome['outcome']))
    if len(names) != len(run['outcomes']):
        raise ValueError('{}: too few audit folders'.format(run['name']))
    return names


def build(run_summaries, out, audit_root='datasets/red_sweet_rl'):
    """Demonstration set from run_expert runs, under the current rules."""
    kept, excluded = [], []
    for summary_path in run_summaries:
        run = json.loads(Path(summary_path).read_text(encoding='utf-8'))
        arrays = load(Path(run['files']['all']['path']))
        split = episodes(arrays)
        if len(split) != len(run['outcomes']):
            raise ValueError('{}: {} episodes in the data, {} outcomes'.format(
                summary_path, len(split), len(run['outcomes'])))
        names = audit_names(run, audit_root)
        for index, outcome, name in zip(split, run['outcomes'], names):
            if outcome['reward'] != 1.0:
                continue
            audit = Path(audit_root) / name / 'episode.json'
            keep, reason = judge(json.loads(audit.read_text(encoding='utf-8')))
            outcome = dict(outcome, episode=name)
            record = dict(run=run['name'], episode=outcome['episode'], reason=reason)
            (kept if keep else excluded).append(record)
            if keep:
                record['transitions'] = [dict(
                    observation=arrays['observations'][i],
                    action=arrays['executed_actions'][i],
                    next_observation=arrays['next_observations'][i],
                    reward=float(arrays['rewards'][i]),
                    terminated=bool(arrays['terminated'][i]),
                    truncated=bool(arrays['truncated'][i]),
                    step_id=int(arrays['step_ids'][i])) for i in index]
    flat = []
    for episode_id, record in enumerate(kept):
        flat += [dict(t, episode_id=episode_id) for t in record.pop('transitions')]
    out = run_expert.save_transitions(out, flat)
    summary = dict(
        rule=dict(method=auto_reward.METHOD,
                  held_motion_ratio=auto_reward.HELD_MOTION_RATIO),
        episodes=len(kept), transitions=len(flat), kept=kept, excluded=excluded,
        runs=[str(p) for p in run_summaries], output_sha256=run_expert.sha256(out))
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
    build_parser = commands.add_parser('build')
    build_parser.add_argument('out')
    build_parser.add_argument('runs', nargs='+')
    build_parser.add_argument('--audit-root', default='datasets/red_sweet_rl')
    summary_parser = commands.add_parser('summary')
    summary_parser.add_argument('path')
    args = parser.parse_args()
    if args.command == 'merge':
        summary = merge(args.inputs, args.out)
        print('{episodes} episodes ({successful_episodes} successful), '
              '{transitions} transitions -> {0}'.format(args.out, **summary))
        return 0
    if args.command == 'build':
        summary = build(args.runs, args.out, args.audit_root)
        print('{} episodes, {} transitions -> {}'.format(
            summary['episodes'], summary['transitions'], args.out))
        for record in summary['excluded']:
            print('  excluded {episode} ({run}): {reason}'.format(**record))
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
