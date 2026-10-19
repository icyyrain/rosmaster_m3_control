"""Shared pieces of the WSL learner side (JAX; not importable on Windows).

Uses the simulation project's state SAC (official HIL-SERL SACAgent) and its
transition schema, so what is learned here is what the plug-insertion work
will use. Run with PYTHONPATH set as in run_wsl.sh.
"""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from isaaclab_hil_serl.data.schema import load_transition_dataset
from isaaclab_hil_serl.learning import make_state_sac_agent

STATE_DIM = 10                   # pc_client/red_rl/env.py OBSERVATION_NAMES
ACTION_DIM = 4                   # (radial, lateral, up, close)
CLOSE = 3
AGENT_KWARGS = dict(hidden_dims=(256, 256), discount=0.97,
                    target_entropy=-ACTION_DIM / 2.0)


def make_agent(seed=0):
    return make_state_sac_agent(
        seed=seed,
        sample_observation={'state': jnp.zeros(STATE_DIM, dtype=jnp.float32)},
        sample_action=jnp.zeros(ACTION_DIM, dtype=jnp.float32),
        **AGENT_KWARGS)


def load_demos(path):
    """Arrays from a demonstration file: observations, actions, episode ids."""
    transitions = load_transition_dataset(path)
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
