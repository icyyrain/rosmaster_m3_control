"""How does a policy's close decision look on the states it actually visited? (WSL)

    JAX_PLATFORMS=cpu ./pc_client/red_rl/wsl/run_wsl.sh inspect_close_noise.py RUN_DIR

For each saved checkpoint of an online run, on every online state that was
not a close: the deterministic close value (the mode, >0 closes) and the
chance that a sampled action closes. A close dimension whose mode stays
negative but whose samples cross 0 is exploration noise, not a learned
decision.
"""

import sys
from pathlib import Path

import jax
import numpy as np
from flax.training import checkpoints

from isaaclab_hil_serl.data.schema import load_transition_dataset

import common


def main():
    run = Path(sys.argv[1])
    transitions = [t for path in sorted((run / 'transitions').glob('episode_*.npz'))
                   for t in load_transition_dataset(path)]
    observations = np.stack([t.observation for t in transitions]).astype(np.float32)
    actions = np.stack([t.executed_action for t in transitions])
    height = observations[:, 4] * 50.0
    moving = actions[:, common.CLOSE] <= 0          # states where no close was taken
    high = moving & (height > 5.0)                  # well above the grasp height
    print('{} online states, {} of them more than 5 mm above the grasp height'.format(
        len(observations), int(high.sum())))
    base = common.make_agent()
    for path in sorted((run / 'checkpoints').glob('checkpoint_*'),
                       key=lambda p: int(p.name.split('_')[-1])):
        agent = base.replace(state=checkpoints.restore_checkpoint(
            str(path.resolve()), target=base.state))
        mode = common.act(agent, observations)[:, common.CLOSE]
        samples = np.stack([np.asarray(agent.sample_actions(
            {'state': observations}, seed=jax.random.PRNGKey(k), argmax=False))
            [:, common.CLOSE] for k in range(50)])
        close_chance = (samples > 0).mean(axis=0)
        print('{:<18} high states: mode closes {:5.1%}, mean mode {:+.2f}, '
              'sample closes {:5.1%}; spread of close {:.2f}'.format(
                  path.name, (mode[high] > 0).mean(), mode[high].mean(),
                  close_chance[high].mean(), samples[:, high].std(axis=0).mean()))


if __name__ == '__main__':
    main()
