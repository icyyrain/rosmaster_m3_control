"""Shared pieces of the WSL learner side (JAX; not importable on Windows).

Uses the simulation project's state SAC (official HIL-SERL SACAgent) and its
transition schema, so what is learned here is what the plug-insertion work
will use. Run with PYTHONPATH set as in run_wsl.sh.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from isaaclab_hil_serl.data.schema import load_transition_dataset
from isaaclab_hil_serl.learning import make_state_sac_agent

STATE_DIM = 10                   # pc_client/red_rl/env.py OBSERVATION_NAMES
ACTION_DIM = 4                   # (radial, lateral, up, close)
CLOSE = 3
# pc_client/red_rl/env.py; that module needs the robot stack, so mirrored here.
GRASP_REGION_HEIGHT_MM = 1.5
GRASP_REGION_PX = 25.0
OPEN = -1.0                      # the close value sent and stored under auto-close
AGENT_KWARGS = dict(hidden_dims=(256, 256), discount=0.97,
                    target_entropy=-ACTION_DIM / 2.0)


def make_agent(seed=0):
    return make_state_sac_agent(
        seed=seed,
        sample_observation={'state': jnp.zeros(STATE_DIM, dtype=jnp.float32)},
        sample_action=jnp.zeros(ACTION_DIM, dtype=jnp.float32),
        **AGENT_KWARGS)


def in_grasp_region(observation):
    o = np.asarray(observation, dtype=float)
    return bool(o[2] > 0.5 and o[4] * 50.0 <= GRASP_REGION_HEIGHT_MM
                and math.hypot(o[0] * 640.0, o[1] * 360.0) <= GRASP_REGION_PX)


def without_close(action):
    """The action with its close value fixed open: under auto-close it does nothing."""
    action = np.array(action, dtype=np.float32)
    action[CLOSE] = OPEN
    return action


def for_auto_close(transitions):
    """Demonstrations recast for an environment that closes in the grasp region.

    The expert closed with a stationary close action once in the region; the
    environment now closes as the move into the region lands. So each episode
    ends at its first transition that arrives in the region, carrying the
    episode's outcome, and every close value is set open. Episodes the expert
    closed outside the region (its last-step attempt) cannot be recast and are
    dropped. Returns (transitions, episodes kept, episodes dropped).
    """
    episodes = {}
    for transition in transitions:
        episodes.setdefault(transition.episode_id, []).append(transition)
    kept, dropped = [], 0
    for steps in episodes.values():
        steps = sorted(steps, key=lambda t: t.step_id)
        outcome = steps[-1]
        arrive = next((i for i, t in enumerate(steps)
                       if in_grasp_region(t.next_observation)), None)
        if arrive is None or arrive == len(steps) - 1:
            dropped += 1
            continue
        episode_id = len(set(t.episode_id for t in kept))
        for i, t in enumerate(steps[:arrive + 1]):
            last = i == arrive
            kept.append(dataclasses.replace(
                t, policy_action=without_close(t.policy_action),
                executed_action=without_close(t.executed_action),
                reward=outcome.reward if last else 0.0,
                terminated=True if last else False, truncated=False,
                success=bool(outcome.success) if last else False,
                episode_id=episode_id))
    return kept, len(episodes) - dropped, dropped


def load_demos(path, auto_close=False):
    """Arrays from a demonstration file: observations, actions, episode ids."""
    transitions = load_transition_dataset(path)
    if auto_close:
        transitions = for_auto_close(transitions)[0]
    observations = np.stack([t.observation for t in transitions]).astype(np.float32)
    actions = np.stack([t.executed_action for t in transitions]).astype(np.float32)
    episodes = np.asarray([t.episode_id for t in transitions])
    assert observations.shape[1] == STATE_DIM and actions.shape[1] == ACTION_DIM
    return observations, actions, episodes


def act(agent, observations):
    """Deterministic actions (the tanh-squashed mode), batched or single."""
    single = observations.ndim == 1
    batch = observations[None] if single else observations
    actions = np.asarray(agent.sample_actions({'state': jnp.asarray(batch)},
                                              argmax=True), dtype=np.float32)
    return actions[0] if single else actions


def save(agent, directory, step):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints.save_checkpoint(str(directory), agent.state, step=step,
                                keep=3, overwrite=True)


def restore(directory, seed=0):
    agent = make_agent(seed)
    latest = checkpoints.latest_checkpoint(str(Path(directory).resolve()))
    assert latest is not None, 'no checkpoint in {}'.format(directory)
    return agent.replace(state=checkpoints.restore_checkpoint(latest, target=agent.state)), latest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
