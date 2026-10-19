"""How noisy would a policy's sampled actions be on the demonstration states? (WSL)

    ./pc_client/red_rl/wsl/run_wsl.sh inspect_noise.py MODEL_DIR DEMOS.npz

SAC explores by sampling its action distribution. BC with MSE alone trains
only the distribution's mode, so its spread is whatever the network starts
with; this reports that spread in millimetres and how often a sampled action
would close the jaws where the expert did not.
"""

import sys

import jax
import numpy as np

import common


def main():
    model, demos = sys.argv[1], sys.argv[2]
    agent, _ = common.restore(model + '/checkpoints')
    observations, actions, _ = common.load_demos(demos)
    mode = common.act(agent, observations)
    samples = np.stack([np.asarray(agent.sample_actions(
        {'state': observations}, seed=jax.random.PRNGKey(k), argmax=False))
        for k in range(20)])
    spread = samples.std(axis=0).mean(axis=0)
    expert_close = actions[:, common.CLOSE] > 0
    sampled_close = (samples[:, ~expert_close, common.CLOSE] > 0).mean()
    print('sampled-action spread per dim: {} (x4 mm for the first three: {})'.format(
        spread.round(3).tolist(), (spread[:3] * 4).round(2).tolist()))
    print('mean |sample - mode| {}'.format(np.abs(samples - mode).mean(axis=(0, 1)).round(3).tolist()))
    print('a sampled action closes on {:.1%} of steps where the expert did not close'.format(
        sampled_close))


if __name__ == '__main__':
    main()
