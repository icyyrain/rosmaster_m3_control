"""S2: BC-initialised SAC trained online on the served environment (WSL).

    ./pc_client/red_rl/wsl/run_wsl.sh online.py --init MODEL_DIR \
        --demos datasets/red_sweet_rl_demos/m2_expert.npz --episodes 40 --out RUN_DIR

The recipe follows the simulation project's learner and HIL-SERL/RLPD:
  * every update draws half its batch from online experience and half from the
    demonstrations, which stay in their own buffer for the whole run;
  * the actor is updated with update_state_sac_with_bc, SAC's loss plus a
    pull toward the demonstrated actions (bc_weight), so the critic, which
    starts from nothing, cannot unlearn the BC policy in the first updates;
  * before the first episode the critic alone is warmed up on demonstrations
    and on --seed transitions (e.g. the M3 evaluation, failures included).

Synchronous: one process, the arm steps (about a second each), then
--utd gradient updates. Exploration is the policy's own sampled action; the
environment's safety box and episode endings are unchanged. Checkpoints,
online transitions (schema version 1) and per-episode metrics go to the run
directory; --resume continues from its latest checkpoint and transitions.

S4 (--from-scratch --bc-weight 0 --intervene): HIL-SERL as published, with
the rule corrector for the person: a freshly initialised agent, plain SAC
updates (no pull toward the demonstrations), demonstrations only as half of
every batch.

S3 (--intervene): each reset asks the served environment for the rule
corrector (intervene.py). When it takes over, info["intervene_action"] is
the action actually executed; as in HIL-SERL's train_rlpd.py, that
transition is stored with the executed action, marked as an intervention,
and also added to the demonstration buffer.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from isaaclab_hil_serl.data.schema import (Transition, load_transition_dataset,
                                           save_transition_dataset)
from isaaclab_hil_serl.learning import update_state_sac_with_bc
from isaaclab_hil_serl.protocol import RemoteEnv
from serl_launcher.data.data_store import ReplayBufferDataStore

import common
from evaluate import windows_host


def spaces():
    observation = gym.spaces.Dict({'state': gym.spaces.Box(
        -np.inf, np.inf, (common.STATE_DIM,), np.float32)})
    action = gym.spaces.Box(-1.0, 1.0, (common.ACTION_DIM,), np.float32)
    return observation, action


def serl(transition):
    value = transition.to_serl_dict()
    value['observations'] = {'state': value['observations']}
    value['next_observations'] = {'state': value['next_observations']}
    return value


def buffer_of(transitions, capacity):
    buffer = ReplayBufferDataStore(*spaces(), capacity=capacity)
    for transition in transitions:
        buffer.insert(serl(transition))
    return buffer


def mixed_batch(online, demos, size):
    half = size // 2
    return jax.device_put(jax.tree_util.tree_map(
        lambda a, b: np.concatenate([a, b], axis=0),
        online.sample(half), demos.sample(size - half)))


def explore(sample, mode, args):
    """The executed action: the policy's mode with a measured amount of its noise.

    Moves: mode + scale x (sample - mode). Close: the sample only near the
    decision (mode within the band below 0), so it can learn when closing
    pays without closing at random in mid-air; elsewhere the mode.
    """
    action = np.array(mode, dtype=np.float32)
    action[:common.CLOSE] = np.clip(
        mode[:common.CLOSE] + args.move_noise_scale * (sample[:common.CLOSE]
                                                       - mode[:common.CLOSE]),
        -1.0, 1.0)
    near = -args.close_explore_band < mode[common.CLOSE] <= 0.0
    if args.close_noise or near:
        action[common.CLOSE] = sample[common.CLOSE]
    return action


def update(agent, batch, demo_buffer, args):
    """One update: SAC plus the pull toward demonstrations, or plain SAC."""
    if args.bc_weight <= 0.0:
        return agent.update(batch)
    expert = jax.device_put(demo_buffer.sample(args.batch_size))
    return update_state_sac_with_bc(agent, batch, expert, args.bc_weight,
                                    args.bc_log_prob_weight, args.bc_mse_weight)


def scalars(info):
    flat = {}
    for group, values in info.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if np.ndim(value) == 0:
                    flat['{}/{}'.format(group, key)] = float(value)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--init', help='Model directory to start from (BC, or an S2 run).')
    start.add_argument('--from-scratch', action='store_true',
                       help='A freshly initialised agent, as HIL-SERL starts (S4).')
    parser.add_argument('--demos', required=True)
    parser.add_argument('--seed-transitions', nargs='*', default=[],
                        help='Earlier policy transitions to start the online buffer with.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--utd', type=int, default=8, help='Updates per arm step.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--critic-warmup', type=int, default=2000)
    parser.add_argument('--bc-weight', type=float, default=1.0,
                        help='0: plain SAC updates, as HIL-SERL (S4).')
    parser.add_argument('--bc-log-prob-weight', type=float, default=0.0)
    parser.add_argument('--bc-mse-weight', type=float, default=1.0)
    parser.add_argument('--actor-delay-steps', type=int, default=0,
                        help='Arm steps at the start of the run with critic-only updates, '
                             'so a critic that has only seen demonstrations does not '
                             'drag the BC actor off course in the first episode.')
    parser.add_argument('--close-noise', action='store_true',
                        help='Sample the close value everywhere, as the first S2 run did. '
                             'It closed in mid-air on 14.5%% of steps above 5 mm at one '
                             'point, while the mode never did there.')
    parser.add_argument('--move-noise-scale', type=float, default=0.3,
                        help='Scale of the sampled deviation from the mode on the three '
                             'moves. 1.0 is plain SAC; on the arm that wandered 3-4 mm a '
                             'step and ran out of steps. 0.3 is about 1 mm.')
    parser.add_argument('--close-explore-band', type=float, default=0.3,
                        help='Sample the close only where the mode lies within this of 0, '
                             'i.e. where the policy is already close to closing; elsewhere '
                             'use the mode. 0 never explores the close.')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Episode step limit during training (evaluation keeps 20).')
    parser.add_argument('--max-failures', type=int, default=6,
                        help='Stop after this many unrewarded episodes in a row.')
    parser.add_argument('--auto-close', action='store_true',
                        help='The environment closes in the grasp region and the policy '
                             'only moves; demonstrations are recast to match.')
    parser.add_argument('--intervene-progress', type=float, default=0.0,
                        help='With --intervene, also take over whenever a move advances less '
                             'than this fraction of the expert move (0 = clear mistakes only).')
    parser.add_argument('--intervene-cooldown', type=int, default=0,
                        help='Steps the policy keeps after each takeover before the '
                             'attentive rule may take over again.')
    parser.add_argument('--intervene', action='store_true',
                        help='S3: let the rule corrector take over when the policy is '
                             'clearly going wrong.')
    parser.add_argument('--host', default=None)
    parser.add_argument('--port', type=int, default=18766)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    out = Path(args.out)
    (out / 'transitions').mkdir(parents=True, exist_ok=True)
    demos = load_transition_dataset(args.demos)
    if args.auto_close:
        demos, kept, dropped = common.for_auto_close(demos)
        print('demonstrations recast for auto-close: {} episodes, {} transitions '
              '({} dropped: closed outside the grasp region)'.format(kept, len(demos), dropped))
    online_files = sorted((out / 'transitions').glob('episode_*.npz'))
    previous = [t for path in online_files for t in load_transition_dataset(path)]
    corrected = [t for t in previous if t.intervention] if args.resume else []
    # Room for the corrections to come: a full buffer would overwrite demos.
    demo_buffer = buffer_of(demos + corrected, len(demos) + 20000)
    seed = [t for path in args.seed_transitions for t in load_transition_dataset(path)]
    online_buffer = buffer_of(seed + (previous if args.resume else []), 50000)

    latest = checkpoints.latest_checkpoint(str((out / 'checkpoints').resolve()))
    if args.resume and latest:
        agent, checkpoint = common.restore(out / 'checkpoints', args.seed)
        episode_index = len(online_files)
        update_step = int(Path(checkpoint).name.split('_')[-1])
        print('resumed {} at episode {}'.format(checkpoint, episode_index))
    else:
        if args.from_scratch:
            agent, checkpoint = common.make_agent(args.seed), 'scratch'
        else:
            agent, checkpoint = common.restore(Path(args.init) / 'checkpoints', args.seed)
        episode_index, update_step = 0, 0
        print('initialised from {}; warming up the critic for {} updates on {} '
              'demonstration and {} seed transitions'.format(
                  checkpoint, args.critic_warmup, len(demos), len(seed)))
        source = online_buffer if len(seed) else demo_buffer
        for k in range(args.critic_warmup):
            batch = mixed_batch(source, demo_buffer, args.batch_size)
            agent, info = agent.update(batch, networks_to_update=frozenset({'critic'}))
        if args.critic_warmup:
            print('  critic loss after warm-up: {:.4f}'.format(
                scalars(info).get('critic/critic_loss', float('nan'))))
        common.save(agent, out / 'checkpoints', 0)
        common.write_json(out / 'config.json', dict(vars(args), init_checkpoint=checkpoint))

    # Compile the per-step functions now, with the arm still at HOME. Their
    # first call otherwise compiles in the middle of the first episode, and
    # the arm sat still for about 20 s right where the policy takes over.
    rng = jax.random.PRNGKey(args.seed + 1000 * episode_index)
    probe = np.zeros(common.STATE_DIM, dtype=np.float32)
    agent.sample_actions({'state': jnp.asarray(probe)}, seed=rng, argmax=False)
    batch = mixed_batch(online_buffer if len(online_buffer) else demo_buffer,
                        demo_buffer, args.batch_size)
    jax.block_until_ready(update(agent, batch, demo_buffer, args))   # result discarded
    print('compiled the action and update functions', flush=True)

    env = RemoteEnv(args.host or windows_host(), args.port, timeout=600.0)
    metrics_path = out / 'episodes.jsonl'
    failures = 0
    run_steps = 0 if not args.resume else 10 ** 9
    try:
        while episode_index < args.episodes:
            try:
                observation, info = env.reset(options={'max_steps': args.max_steps,
                                                       'intervene': args.intervene,
                                                       'intervene_progress': args.intervene_progress,
                                                       'intervene_cooldown': args.intervene_cooldown,
                                                       'auto_close': args.auto_close})
                if args.auto_close and not info.get('auto_close'):
                    raise SystemExit('the served environment cannot auto-close; '
                                     'restart env_server from this commit')
                if args.intervene and not info.get('intervention_enabled'):
                    raise SystemExit('the served environment has no rule corrector; '
                                     'restart env_server from this commit')
            except RuntimeError as error:
                print('stopping, reset failed: {}'.format(error))
                break
            name, step, transitions, started = info.get('episode'), 0, [], time.time()
            losses = []
            while True:
                rng, key = jax.random.split(rng)
                sample = np.array(agent.sample_actions(
                    {'state': jnp.asarray(observation)}, seed=key, argmax=False),
                    dtype=np.float32)
                mode = common.act(agent, np.asarray(observation, dtype=np.float32))
                action = explore(sample, mode, args)
                if args.auto_close:
                    action = common.without_close(action)
                next_observation, reward, terminated, truncated, info = env.step(action)
                corrected = 'intervene_action' in info
                executed = (np.asarray(info['intervene_action'], dtype=np.float32)
                            if corrected else action)
                if args.auto_close:
                    executed = common.without_close(executed)
                transition = Transition(
                    observation=observation, policy_action=action, executed_action=executed,
                    next_observation=next_observation, reward=reward,
                    terminated=terminated, truncated=truncated,
                    source='intervention' if corrected else 'policy',
                    intervention=corrected, episode_id=0, step_id=step,
                    success=reward > 0.0)
                transitions.append(transition)
                online_buffer.insert(serl(transition))
                if corrected:
                    demo_buffer.insert(serl(transition))
                critic_only = run_steps < args.actor_delay_steps
                for _ in range(args.utd):
                    batch = mixed_batch(online_buffer, demo_buffer, args.batch_size)
                    if critic_only:
                        agent, update_info = agent.update(
                            batch, networks_to_update=frozenset({'critic'}))
                    else:
                        agent, update_info = update(agent, batch, demo_buffer, args)
                    update_step += 1
                run_steps += 1
                losses.append(scalars(update_info))
                observation = next_observation
                step += 1
                if terminated or truncated:
                    break
            episode_index += 1
            save_transition_dataset(
                out / 'transitions' / 'episode_{:04d}.npz'.format(episode_index), transitions)
            failures = 0 if reward > 0 else failures + 1
            mean = {key: float(np.mean([l[key] for l in losses if key in l]))
                    for key in losses[-1]}
            record = dict(episode=episode_index, audit=name, outcome=info.get('outcome'),
                          reward=float(reward), steps=step, seconds=round(time.time() - started, 1),
                          update_step=update_step,
                          interventions=int(info.get('intervention_count', 0)),
                          intervention_steps=int(info.get('intervention_steps', 0)),
                          policy_closed=bool(transitions[-1].policy_action[common.CLOSE] > 0.0
                                             and not transitions[-1].intervention),
                          **{k: round(v, 5) for k, v in mean.items()})
            with metrics_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            recent = [json.loads(line)['reward'] for line in
                      metrics_path.read_text(encoding='utf-8').splitlines()][-10:]
            print('episode {:>3}: {:<13} {:>2} steps  corrected {}x/{} steps  '
                  'last-10 success {:.0%}  critic {:.4f}  bc_mse {:.4f}  temp {:.4f}'.format(
                      episode_index, str(info.get('outcome')), step,
                      record['interventions'], record['intervention_steps'], np.mean(recent),
                      mean.get('critic/critic_loss', float('nan')),
                      mean.get('actor/bc_mse', float('nan')),
                      mean.get('actor/temperature', float('nan'))), flush=True)
            if episode_index % 5 == 0 or episode_index == args.episodes:
                common.save(agent, out / 'checkpoints', update_step)
            if failures >= args.max_failures:
                print('stopping: {} unrewarded episodes in a row'.format(failures))
                break
    finally:
        common.save(agent, out / 'checkpoints', update_step)
        env.close()
    print('done: {} episodes, {} updates; {}'.format(episode_index, update_step, out))


if __name__ == '__main__':
    main()
