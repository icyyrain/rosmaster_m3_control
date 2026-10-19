"""Run a trained policy on the served environment, deterministically (WSL).

    ./pc_client/red_rl/wsl/run_wsl.sh evaluate.py MODEL_DIR --episodes 20

Connects with the simulation project's RemoteEnv to env_server on Windows
(--fake for the kinematic fake, --execute for the arm; the Windows address
defaults to WSL's gateway). No correction, no exploration: the policy's own
mode action every step, as an autonomous evaluation must be.

Writes, under MODEL_DIR/evaluations/<time>/, every transition in the
simulation project's schema (source "policy", usable as online replay later)
and a summary with each episode's outcome. Stops early after --max-misses
misses in a row, as unattended collection does.
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from isaaclab_hil_serl.data.schema import Transition, save_transition_dataset
from isaaclab_hil_serl.protocol import RemoteEnv

import common


def windows_host():
    route = subprocess.run(['ip', 'route'], capture_output=True, text=True).stdout
    for line in route.splitlines():
        if line.startswith('default'):
            return line.split()[2]
    raise RuntimeError('no default route; pass --host')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--host', default=None)
    parser.add_argument('--port', type=int, default=18766)
    parser.add_argument('--max-misses', type=int, default=3)
    args = parser.parse_args()

    agent, checkpoint = common.restore(Path(args.model) / 'checkpoints')
    common.act(agent, np.zeros(common.STATE_DIM, dtype=np.float32))   # compile first
    env = RemoteEnv(args.host or windows_host(), args.port, timeout=600.0)
    assert env.observation_space.shape == (common.STATE_DIM,)
    assert env.action_space.shape == (common.ACTION_DIM,)
    out = Path(args.model) / 'evaluations' / time.strftime('%Y%m%d-%H%M%S')
    transitions, outcomes = [], []
    misses = 0
    stop_reason = None
    try:
        for episode in range(args.episodes):
            try:
                observation, info = env.reset()
            except RuntimeError as error:
                stop_reason = 'reset failed: {}'.format(error)
                print(stop_reason)
                break
            name = info.get('episode')
            step = 0
            while True:
                action = common.act(agent, np.asarray(observation, dtype=np.float32))
                next_observation, reward, terminated, truncated, info = env.step(action)
                transitions.append(Transition(
                    observation=observation, policy_action=action, executed_action=action,
                    next_observation=next_observation, reward=reward,
                    terminated=terminated, truncated=truncated, source='policy',
                    intervention=False, episode_id=len(outcomes), step_id=step,
                    success=reward > 0.0))
                observation = next_observation
                step += 1
                if terminated or truncated:
                    break
            outcome = info.get('outcome')
            outcomes.append(dict(episode=name, outcome=outcome, reward=float(reward),
                                 steps=step, height_mm=info.get('height_mm'),
                                 stop_height_mm=info.get('stop_height_mm'),
                                 target_px=info.get('target_px')))
            held = sum(o['reward'] for o in outcomes)
            print('episode {:>2}: {:<13} after {:>2} steps   ({:.0f}/{} held)'.format(
                len(outcomes), str(outcome), step, held, len(outcomes)), flush=True)
            misses = 0 if reward > 0 else misses + 1
            if misses >= args.max_misses:
                stop_reason = '{} misses in a row'.format(misses)
                print('stopping: ' + stop_reason)
                break
    finally:
        env.close()
    if transitions:
        save_transition_dataset(out / 'transitions.npz', transitions)
    held = int(sum(o['reward'] for o in outcomes))
    common.write_json(out / 'summary.json', dict(
        checkpoint=checkpoint, deterministic=True, interventions=False,
        environment=env.metadata.get('name'), episodes=len(outcomes), held=held,
        stop_reason=stop_reason, outcomes=outcomes))
    print('{}/{} held; saved {}'.format(held, len(outcomes), out))


if __name__ == '__main__':
    main()
