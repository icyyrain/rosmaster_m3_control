"""Conservative camera-only grasp suggestion and human-label comparison.

This is a weak visual signal, not a force/encoder measurement or an RL reward.
The operator's label remains authoritative for imitation learning.
"""

import argparse
import json
import math
from pathlib import Path

import reach_candy


METHOD = 'short_lift_motion_v2'
# The short lift followed by hold_check's base turn, which decides when it can.
TURN_METHOD = METHOD + '+base_turn_v1'
KNOWN_METHODS = ('short_lift_motion_v1', METHOD, TURN_METHOD)
MIN_EXPECTED_MOTION_PX = 45.0
# v1 accepted up to 0.30. An expert grasp at 0.22 was carried off and dropped
# on the way, seen by the operator; another at 0.20 was a loose hold. Of 14
# person-confirmed holds, 0.20 would have turned 2 (0.25, 0.29) into "cannot
# tell" and 0.15 would have turned 3. A hold that slips more than this on a
# 35 mm lift is not one to carry; the carry check in relocate.py catches what
# still gets through.
HELD_MOTION_RATIO = 0.20
NOT_HELD_MOTION_RATIO = 0.70
STATIONARY_ERROR_RATIO = 0.30
MAX_GRIPPER_OFFSET_PX = 150.0


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assess_short_lift(before_px, after_px, stationary_px, rise_mm):
    """Suggest True/False/None from target motion relative to the camera.

    An object attached to the gripper stays near its old camera pixel. A
    stationary table object shifts toward the Jacobian-predicted pixel as the
    eye-in-hand camera lifts. Weak geometry or tracking loss means abstain.
    """
    evidence = {
        'method': METHOD,
        'success': None,
        'reason': None,
        'commanded_rise_mm': float(rise_mm),
        'before_px': list(before_px) if before_px is not None else None,
        'after_px': list(after_px) if after_px is not None else None,
        'stationary_prediction_px': (
            list(stationary_px) if stationary_px is not None else None),
        'expected_stationary_motion_px': None,
        'observed_motion_px': None,
        'stationary_prediction_error_px': None,
        'gripper_offset_px': None,
    }
    if before_px is None or after_px is None or stationary_px is None:
        evidence['reason'] = 'target_not_tracked_on_both_sides_of_lift'
        return evidence
    if not all(math.isfinite(float(v)) for point in (before_px, after_px, stationary_px)
               for v in point):
        evidence['reason'] = 'nonfinite_target_coordinate'
        return evidence

    expected = _distance(before_px, stationary_px)
    moved = _distance(before_px, after_px)
    stationary_error = _distance(after_px, stationary_px)
    gripper_offset = _distance(after_px, reach_candy.GRIPPER_PX)
    evidence.update(expected_stationary_motion_px=round(expected, 2),
                    observed_motion_px=round(moved, 2),
                    stationary_prediction_error_px=round(stationary_error, 2),
                    gripper_offset_px=round(gripper_offset, 2))
    if expected < MIN_EXPECTED_MOTION_PX:
        evidence['reason'] = 'lift_motion_too_small_to_separate_hypotheses'
    elif (moved <= HELD_MOTION_RATIO * expected
          and gripper_offset <= MAX_GRIPPER_OFFSET_PX):
        evidence['success'] = True
        evidence['reason'] = 'target_stayed_near_gripper_pixel'
    elif (moved >= NOT_HELD_MOTION_RATIO * expected
          and stationary_error <= STATIONARY_ERROR_RATIO * expected):
        evidence['success'] = False
        evidence['reason'] = 'target_followed_stationary_scene_prediction'
    else:
        evidence['reason'] = 'ambiguous_target_motion'
    return evidence


def compare_labels(dataset):
    """Counts only completed episodes with both a human label and new evidence."""
    counts = {'human_labeled': 0, 'agree': 0, 'disagree': 0,
              'auto_abstained': 0, 'human_unknown': 0, 'without_auto': 0,
              'auto_only': 0}
    for path in sorted(Path(dataset).glob('*/episode.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('outcome') != 'completed':
            continue
        if meta.get('auto_evidence') is None:
            counts['without_auto'] += 1
            continue
        human = meta.get('human_success')
        automatic = meta.get('auto_success')
        if human is None and meta.get('label_source') == 'auto_short_lift':
            # Unattended episodes carry no person's label to compare with.
            counts['auto_only'] += 1
        elif human is None:
            counts['human_unknown'] += 1
        else:
            counts['human_labeled'] += 1
            if automatic is None:
                counts['auto_abstained'] += 1
            elif automatic == human:
                counts['agree'] += 1
            else:
                counts['disagree'] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='datasets/red_sweet')
    args = parser.parse_args()
    counts = compare_labels(args.dataset)
    print('Human-labeled episodes with auto evidence: {}'.format(counts['human_labeled']))
    print('  agree: {}, disagree: {}, auto abstained: {}'.format(
        counts['agree'], counts['disagree'], counts['auto_abstained']))
    print('Human unknown: {}; old episodes without auto evidence: {}'.format(
        counts['human_unknown'], counts['without_auto']))
    print('Unattended episodes labelled by the camera alone: {}'.format(
        counts['auto_only']))
    print('A human label, where present, always overrides the camera verdict.')


if __name__ == '__main__':
    main()
