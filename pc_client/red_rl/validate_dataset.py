"""Check recorded episodes before using them for imitation or RL."""

import argparse
import json
import math
from pathlib import Path

import cv2

from pc_client.red_rl.auto_reward import KNOWN_METHODS
from pc_client.red_rl.dataset import (AUTO_LABEL_SOURCE, CURRENT_GRASP_RULE,
                                      FEATURE_NAMES, HUMAN_LABEL_SOURCE,
                                      LEGACY_GRASP_RULE, episode_success,
                                      grasp_rule, label_source)

KNOWN_GRASP_RULES = (LEGACY_GRASP_RULE, CURRENT_GRASP_RULE)
KNOWN_LABEL_SOURCES = (HUMAN_LABEL_SOURCE, AUTO_LABEL_SOURCE)


def validate_episode(folder):
    folder = Path(folder)
    problems = []
    meta_path = folder / 'episode.json'
    steps_path = folder / 'steps.jsonl'
    if not meta_path.is_file() or not steps_path.is_file():
        return ['missing episode.json or steps.jsonl']
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if tuple(meta.get('feature_names', ())) != FEATURE_NAMES:
        problems.append('feature schema differs from the current recorder')
    if meta.get('task') != 'red_sweet_approach':
        problems.append('not a red-sweet approach episode')
    if grasp_rule(meta) not in KNOWN_GRASP_RULES:
        problems.append('unknown grasp rule {!r}'.format(meta.get('grasp_rule')))
    if label_source(meta) not in KNOWN_LABEL_SOURCES:
        problems.append('unknown label source {!r}'.format(meta.get('label_source')))
    elif (label_source(meta) == AUTO_LABEL_SOURCE
          and meta.get('outcome') == 'completed'
          and meta.get('auto_evidence') is None):
        problems.append('camera-labelled episode has no lift evidence')
    if 'auto_evidence' in meta and meta['auto_evidence'] is not None:
        evidence = meta['auto_evidence']
        if not isinstance(evidence, dict) or evidence.get('method') not in KNOWN_METHODS:
            problems.append('invalid automatic lift evidence')
        elif meta.get('auto_success') is not evidence.get('success'):
            problems.append('automatic verdict differs from saved evidence')
        else:
            for key in ('image_before', 'image_after'):
                image = evidence.get(key)
                if image is None:
                    continue
                path = (folder / image).resolve()
                if folder.resolve() not in path.parents or not path.is_file():
                    problems.append('automatic evidence {} unavailable'.format(key))
                elif cv2.imread(str(path)) is None:
                    problems.append('automatic evidence {} cannot be decoded'.format(key))
    rows = [json.loads(line) for line in steps_path.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != meta.get('steps'):
        problems.append('step count differs from episode.json')
    previous_time = -float('inf')
    for index, step in enumerate(rows):
        tag = 'step {}'.format(index)
        if step.get('step') != index:
            problems.append('{}: non-sequential index'.format(tag))
        if step.get('time_s', 0) < previous_time:
            problems.append('{}: timestamp moved backward'.format(tag))
        previous_time = step.get('time_s', previous_time)
        for key in ('observation', 'next_observation'):
            vector = step.get(key, [])
            if len(vector) != len(FEATURE_NAMES) or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in vector):
                problems.append('{}: invalid {}'.format(tag, key))
        before = step.get('commanded_pose_before', [])
        after = step.get('commanded_pose_after', [])
        delta = step.get('action_joint_delta_deg', [])
        if len(before) != 6 or len(after) != 6 or len(delta) != 4:
            problems.append('{}: invalid pose/action shape'.format(tag))
        elif any(abs(after[j] - before[j] - delta[j]) > 1e-5
                 for j in range(4)):
            problems.append('{}: joint delta does not match commands'.format(tag))
        for key in ('image', 'next_image'):
            image = step.get(key)
            if image is None:
                problems.append('{}: missing {}'.format(tag, key))
                continue
            path = (folder / image).resolve()
            if folder.resolve() not in path.parents or not path.is_file():
                problems.append('{}: {} path unavailable'.format(tag, key))
            elif cv2.imread(str(path)) is None:
                problems.append('{}: {} cannot be decoded'.format(tag, key))
    return problems


def summarise_by_rule(meta_paths):
    """Outcome counts per grasp rule; success rates are never pooled across rules.

    A grasp-rule change moves where the jaws close by several millimetres, so
    a rate over both rules describes neither teacher. Override experiments
    (grasp offset, pregrasp shifts, a forced creep controller) and episodes
    from before joint 5 was pinned are counted separately inside each rule
    for the same reason.
    """
    summary = {}
    for meta_path in meta_paths:
        meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        experiment = any([meta.get('grasp_offset_override_mm') is not None,
                          meta.get('pregrasp_retract_j4_deg'),
                          meta.get('pregrasp_base_mm'),
                          meta.get('creep_controller', 'auto') != 'auto'])
        if experiment:
            setting = 'experiment'
        elif 'grasp_profile' not in meta:
            # Recorded before the collector pinned joint 5; the wrist could
            # still rotate on a wrinkled wrapper.
            setting = 'wrist_free'
        else:
            setting = 'default'
        counts = summary.setdefault(grasp_rule(meta), {}).setdefault(
            setting, {'held': 0, 'not_held': 0, 'unlabelled': 0, 'aborted': 0,
                      'camera_labelled': 0})
        success = episode_success(meta)
        if meta.get('outcome') != 'completed':
            counts['aborted'] += 1
            continue
        if meta.get('human_success') is None and success is not None:
            counts['camera_labelled'] += 1
        if success is True:
            counts['held'] += 1
        elif success is False:
            counts['not_held'] += 1
        else:
            counts['unlabelled'] += 1
    return summary


def print_summary(summary):
    print('\nBy grasp rule (rates are per rule and setting, never pooled):')
    for rule in sorted(summary):
        for setting in ('default', 'wrist_free', 'experiment'):
            counts = summary[rule].get(setting)
            if counts is None:
                continue
            # Divide by every completed grasp, not only the labelled ones: the
            # camera mostly abstains on misses, so leaving its abstentions out
            # would flatter unattended runs.
            completed = counts['held'] + counts['not_held'] + counts['unlabelled']
            rate = ('{}/{} held'.format(counts['held'], completed)
                    if completed else 'no completed grasps')
            print('  {:<16} {:<10} {:<14} not held {}, unlabelled {}, '
                  'aborted before closing {}, labelled by camera only {}'.format(
                      rule, setting, rate, counts['not_held'],
                      counts['unlabelled'], counts['aborted'],
                      counts['camera_labelled']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='datasets/red_sweet')
    args = parser.parse_args()
    folders = sorted(Path(args.dataset).glob('*/episode.json'))
    if not folders:
        parser.error('No recorded episodes in {}'.format(args.dataset))
    failed = 0
    for meta_path in folders:
        issues = validate_episode(meta_path.parent)
        print('{}: {}'.format(meta_path.parent.name,
                              'OK' if not issues else '{} issue(s)'.format(len(issues))))
        for issue in issues:
            print('  - {}'.format(issue))
        failed += bool(issues)
    print('{} episode(s) checked; {} failed'.format(len(folders), failed))
    print_summary(summarise_by_rule(folders))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
