"""Run the rule expert in the RL environment and save its transitions.

    python -m pc_client.red_rl.run_expert --fake --episodes 50
    python -m pc_client.red_rl.run_expert --execute --episodes 20

M1 checks that the episode is solvable from the random starts: the expert's
success rate should be close to the rule teacher's. The same run is M2's
demonstration collection: transitions are written in the simulation
project's schema (isaaclab_hil_serl/data/schema.py, version 1), so its WSL
learner loads them with load_transition_dataset unchanged:

    <name>.all.npz        every episode, for analysis
    <name>.success.npz    successful episodes only, source="demonstration",
                          for the demonstration replay buffer
    <name>.json           counts, settings and per-episode outcomes

Unattended like the collector: a miss is retried, MAX_MISSES in a row stop
the run, and a reset failure (preflight, alignment, descent) stops it at once.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import arm_control
from pc_client.red_rl import env as red_env
from pc_client.red_rl.expert import expert

SCHEMA_VERSION = 1               # isaaclab_hil_serl.data.schema.SCHEMA_VERSION
MAX_MISSES = 2


def save_transitions(path, transitions):
    """np.savez_compressed with the keys and dtypes of the simulation schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        observations=np.stack([t['observation'] for t in transitions]).astype(np.float32),
        policy_actions=np.stack([t['action'] for t in transitions]).astype(np.float32),
        executed_actions=np.stack([t['action'] for t in transitions]).astype(np.float32),
        next_observations=np.stack([t['next_observation'] for t in transitions]).astype(np.float32),
        rewards=np.asarray([t['reward'] for t in transitions], dtype=np.float32),
        terminated=np.asarray([t['terminated'] for t in transitions], dtype=np.bool_),
        truncated=np.asarray([t['truncated'] for t in transitions], dtype=np.bool_),
        sources=np.asarray(['demonstration'] * len(transitions), dtype='U16'),
        interventions=np.zeros(len(transitions), dtype=np.bool_),
        episode_ids=np.asarray([t['episode_id'] for t in transitions], dtype=np.int64),
        step_ids=np.asarray([t['step_id'] for t in transitions], dtype=np.int64),
        successes=np.asarray([t['reward'] == 1.0 for t in transitions], dtype=np.bool_),
    )
    return path


def renumber(episodes):
    """Flatten episodes with contiguous ids from 0, as the schema requires."""
    flat = []
    for new_id, episode in enumerate(episodes):
        for transition in episode:
            flat.append(dict(transition, episode_id=new_id))
    return flat


def run_episode(environment, policy=expert):
    observation, info = environment.reset()
    episode_name = info.get('episode')      # the audit folder, named at reset
    transitions = []
    step_id = 0
    while True:
        action = np.asarray(policy(observation), dtype=np.float32)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        transitions.append(dict(
            observation=np.asarray(observation, dtype=np.float32), action=action,
            next_observation=np.asarray(next_observation, dtype=np.float32),
            reward=float(reward), terminated=bool(terminated),
            truncated=bool(truncated), step_id=step_id))
        step_id += 1
        observation = next_observation
        if terminated or truncated:
            return transitions, dict(info, episode=info.get('episode', episode_name))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fake', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='datasets/red_sweet_rl_demos')
    parser.add_argument('--audit-root', default='datasets/red_sweet_rl')
    args = parser.parse_args()

    if args.fake:
        environment = red_env.FakeRedSweetEnv(seed=args.seed)
    else:
        print('The REAL arm will run {} expert episodes unattended.'.format(args.episodes))
        if not arm_control.confirm_execute():
            print('Cancelled.')
            return 0
        environment = red_env.RedSweetApproachEnv(audit_root=args.audit_root,
                                                  seed=args.seed)
    name = time.strftime('%Y%m%d-%H%M%S') + ('-fake' if args.fake else '')
    episodes, outcomes = [], []
    misses = 0
    stop_reason = None
    try:
        for number in range(1, args.episodes + 1):
            print('\n=== expert episode {}/{} ==='.format(number, args.episodes))
            try:
                transitions, info = run_episode(environment)
            except RuntimeError as error:
                stop_reason = 'reset failed: {}'.format(error)
                print('Stopping: {}'.format(stop_reason))
                break
            reward = transitions[-1]['reward']
            outcome = info.get('outcome')
            outcomes.append(dict(
                outcome=outcome, reward=reward, steps=len(transitions),
                closed=bool(transitions[-1]['action'][3] > red_env.CLOSE_THRESHOLD),
                episode=info.get('episode'), reason=info.get('reason'),
                height_mm=info.get('height_mm'), stop_height_mm=info.get('stop_height_mm'),
                target_px=info.get('target_px')))
            episodes.append(transitions)
            print('  {} after {} steps'.format(outcome, len(transitions)))
            misses = 0 if reward == 1.0 else misses + 1
            if misses >= MAX_MISSES:
                stop_reason = '{} misses in a row'.format(misses)
                print('Stopping: {}'.format(stop_reason))
                break
    finally:
        if not args.fake:
            environment.close()

    held = sum(1 for o in outcomes if o['reward'] == 1.0)
    print('\nexpert: {}/{} held'.format(held, len(outcomes)))
    if not episodes:
        return 1
    out = Path(args.out)
    files = {'all': save_transitions(out / (name + '.all.npz'), renumber(episodes))}
    successes = [e for e in episodes if e[-1]['reward'] == 1.0]
    if successes:
        files['success'] = save_transitions(out / (name + '.success.npz'),
                                            renumber(successes))
    summary = dict(
        name=name, fake=args.fake, seed=args.seed, schema_version=SCHEMA_VERSION,
        observation_names=list(red_env.OBSERVATION_NAMES),
        action_names=list(red_env.ACTION_NAMES), max_steps=red_env.MAX_STEPS,
        episodes=len(outcomes), held=held, stop_reason=stop_reason,
        transitions=sum(len(e) for e in episodes),
        success_transitions=sum(len(e) for e in successes),
        files={key: dict(path=str(path), sha256=sha256(path)) for key, path in files.items()},
        outcomes=outcomes)
    (out / (name + '.json')).write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for key, path in files.items():
        print('saved {} transitions: {}'.format(key, path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
