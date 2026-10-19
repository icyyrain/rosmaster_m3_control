"""Deliberate misses, to see the camera judge say "not held".

    python -m pc_client.red_rl.judge_probe --execute --episodes 4

The rule expert runs in the environment as usual, except that it aims
--short-mm short of the sweet, toward the base, so the jaws close on nothing
at the grasp height with the sweet on the table just ahead of them. Every
episode should end "not held"; a "held" is a false positive of the judge.
Short rather than sideways: the open jaws reach far to either side and would
push the sweet, while ahead of them there is clear table.

Nothing is saved for training; the audit folders are marked as probes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import arm_control
from pc_client.red_rl import env as red_env
from pc_client.red_rl import expert as red_expert
from pc_client.red_rl.run_expert import run_episode

PROBE_NOTE = 'judge probe: deliberate miss, aimed {:.0f} mm short of the sweet'


def short_of_the_sweet(short_mm):
    """The expert, made to centre a point short_mm beyond the sweet's pixel.

    It drives target_dy to zero; adding the offset makes it stop with the
    sweet that far above the gripper pixel, i.e. ahead of the jaws.
    """
    offset = short_mm / red_expert.MM_PER_PX / 360.0
    index = red_expert._index('target_dy')

    def policy(observation):
        biased = np.array(observation, dtype=np.float32)
        if biased[red_expert._index('target_visible')] > 0.5:
            biased[index] += offset
        return red_expert.expert(biased)
    return policy


def mark(audit_root, name, short_mm):
    path = Path(audit_root) / name / 'episode.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta['human_success'] = False
    meta['probe'] = dict(kind='deliberate_miss', short_mm=short_mm)
    meta['notes'] = ((meta.get('notes') or '') + ' ' + PROBE_NOTE.format(short_mm)).strip()
    path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fake', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--short-mm', type=float, default=25.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--audit-root', default='datasets/red_sweet_rl')
    args = parser.parse_args()

    if args.fake:
        environment = red_env.FakeRedSweetEnv(seed=args.seed)
    else:
        print('The REAL arm will close {:.0f} mm short of the sweet, {} times.'.format(
            args.short_mm, args.episodes))
        if not arm_control.confirm_execute():
            print('Cancelled.')
            return 0
        environment = red_env.RedSweetApproachEnv(audit_root=args.audit_root,
                                                  seed=args.seed)
    policy = short_of_the_sweet(args.short_mm)
    results = []
    try:
        for number in range(1, args.episodes + 1):
            print('\n=== probe {}/{} ==='.format(number, args.episodes))
            try:
                transitions, info = run_episode(environment, policy)
            except RuntimeError as error:
                print('Stopping, reset failed: {}'.format(error))
                break
            closed = bool(transitions[-1]['action'][3] > red_env.CLOSE_THRESHOLD)
            result = dict(episode=info.get('episode'), outcome=info.get('outcome'),
                          closed=closed, steps=len(transitions))
            if not args.fake and closed:
                meta = mark(args.audit_root, info['episode'], args.short_mm)
                evidence = meta.get('auto_evidence') or {}
                result.update(short_lift=(evidence.get('short_lift') or {}).get('success'),
                              turn=(evidence.get('turn_check') or {}).get('success'),
                              turn_reason=(evidence.get('turn_check') or {}).get('reason'),
                              shift_px=(evidence.get('turn_check') or {}).get('shift_px'))
            results.append(result)
            print('  {}'.format(result))
    finally:
        if not args.fake:
            environment.close()
    wrong = [r for r in results if r['outcome'] == 'held']
    print('\n{} probes, {} judged held (false positives), {} not held, {} cannot tell'.format(
        len(results), len(wrong), sum(r['outcome'] == 'not_held' for r in results),
        sum(r['outcome'] == 'inconclusive' for r in results)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
