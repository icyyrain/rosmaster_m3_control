"""M3: behaviour cloning of the expert demonstrations (WSL).

    ./pc_client/red_rl/wsl/run_wsl.sh bc.py --demos datasets/red_sweet_rl_demos/m2_expert.npz

Pretrains the actor of the state SAC agent with the simulation project's
pretrain_state_sac_actor_with_bc, as its BC-pretrain config does: mean
squared error to the expert action only, batch 256, 5000 updates.

First, as a check, it holds out whole episodes, trains on the rest and
reports how well the held-out expert actions are reproduced, including the
one decision that ends an episode: when to close. Then it trains on every
episode and saves that checkpoint for the arm.
"""

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from isaaclab_hil_serl.learning import pretrain_state_sac_actor_with_bc

import common


def train(observations, actions, updates, batch_size, seed, log_every=1000):
    agent = common.make_agent(seed)
    rng = np.random.default_rng(seed)
    for update in range(1, updates + 1):
        index = rng.integers(0, len(observations), batch_size)
        batch = {'observations': {'state': jax.device_put(observations[index])},
                 'actions': jax.device_put(actions[index])}
        agent, info = pretrain_state_sac_actor_with_bc(agent, batch, 0.0, 1.0)
        if update % log_every == 0 or update == updates:
            print('  update {:>5}: bc mse {:.4f}'.format(update, float(info['actor']['bc_mse'])),
                  flush=True)
    return agent


def score(agent, observations, actions):
    """How a policy's deterministic actions compare with the expert's."""
    predicted = common.act(agent, observations)
    error = predicted - actions
    expert_close = actions[:, common.CLOSE] > 0
    policy_close = predicted[:, common.CLOSE] > 0
    return dict(
        steps=int(len(actions)),
        mae_per_dim=np.abs(error[:, :3]).mean(axis=0).round(3).tolist(),
        mae_mm_per_dim=(np.abs(error[:, :3]).mean(axis=0) * 4.0).round(2).tolist(),
        close_steps=int(expert_close.sum()),
        close_recall=float((policy_close & expert_close).sum() / max(1, expert_close.sum())),
        early_close_rate=float((policy_close & ~expert_close).sum()
                               / max(1, (~expert_close).sum())),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demos', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--updates', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--holdout', type=int, default=5,
                        help='Whole episodes held out for the check.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out or 'models/red_sweet_rl/bc-' + time.strftime('%Y%m%d-%H%M%S'))
    observations, actions, episodes = common.load_demos(args.demos)
    ids = np.unique(episodes)
    print('{} demonstration steps in {} episodes'.format(len(actions), len(ids)))

    held = np.random.default_rng(args.seed).permutation(ids)[:args.holdout]
    train_rows = ~np.isin(episodes, held)
    print('check: training on {} episodes, holding out {}'.format(
        len(ids) - len(held), sorted(held.tolist())))
    check_agent = train(observations[train_rows], actions[train_rows],
                        args.updates, args.batch_size, args.seed)
    check = dict(holdout_episodes=sorted(held.tolist()),
                 train=score(check_agent, observations[train_rows], actions[train_rows]),
                 holdout=score(check_agent, observations[~train_rows], actions[~train_rows]))
    print('held-out: mean error {} mm (radial, lateral, up), closes on {:.0%} of '
          'the expert closes, closes early on {:.1%} of other steps'.format(
              check['holdout']['mae_mm_per_dim'], check['holdout']['close_recall'],
              check['holdout']['early_close_rate']))

    print('final: training on all {} episodes'.format(len(ids)))
    agent = train(observations, actions, args.updates, args.batch_size, args.seed)
    common.save(agent, out / 'checkpoints', args.updates)
    final = score(agent, observations, actions)
    common.write_json(out / 'summary.json', dict(
        stage='M3 behaviour cloning', demos=str(args.demos),
        demos_sha256=common.sha256(args.demos), updates=args.updates,
        batch_size=args.batch_size, seed=args.seed, loss='bc mse (log-prob weight 0)',
        agent=common.AGENT_KWARGS, check=check, final_train=final))
    print('saved {}'.format(out))


if __name__ == '__main__':
    main()
