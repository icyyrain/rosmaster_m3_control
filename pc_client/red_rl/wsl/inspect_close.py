"""Where does a policy decide to close, compared with the expert? (WSL)

    ./pc_client/red_rl/wsl/run_wsl.sh inspect_close.py MODEL_DIR DEMOS.npz

Lists every step where the policy would close and the expert did not, with the
height still to go and the image offset, and every expert close the policy
would miss. A close a step early at 2 mm is harmless; one at 15 mm is a miss.
"""

import sys

import numpy as np

import common


def main():
    model, demos = sys.argv[1], sys.argv[2]
    agent, checkpoint = common.restore(model + '/checkpoints')
    observations, actions, episodes = common.load_demos(demos)
    predicted = common.act(agent, observations)
    policy_close = predicted[:, common.CLOSE] > 0
    expert_close = actions[:, common.CLOSE] > 0
    height = observations[:, 4] * 50.0
    offset = np.hypot(observations[:, 0] * 640.0, observations[:, 1] * 360.0)
    print('checkpoint', checkpoint)
    print('policy closes where the expert does not:')
    for i in np.flatnonzero(policy_close & ~expert_close):
        print('  episode {:>2} step {:>2}: {:5.1f} mm above the grasp height, '
              '{:4.0f} px off'.format(episodes[i], i - np.flatnonzero(episodes == episodes[i])[0],
                                      height[i], offset[i]))
    print('expert closes the policy would miss: {}'.format(
        int((expert_close & ~policy_close).sum())))
    print('expert closes at: {:.1f} to {:.1f} mm, {:.0f} to {:.0f} px'.format(
        height[expert_close].min(), height[expert_close].max(),
        offset[expert_close].min(), offset[expert_close].max()))


if __name__ == '__main__':
    main()
