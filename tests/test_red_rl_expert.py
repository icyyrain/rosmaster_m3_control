import tempfile
import unittest
from pathlib import Path

import numpy as np

from pc_client.red_rl import env as red_env
from pc_client.red_rl import run_expert
from pc_client.red_rl.expert import ALIGN_PX, expert, read


def observation(dx_px=0.0, dy_px=0.0, visible=True, height=20.0, step=0):
    target = ((707.0 + dx_px, 680.0 + dy_px, 3000.0) if visible else None)
    return red_env.encode(target, height, [90, 12, 56, 32], step)


class ExpertTests(unittest.TestCase):
    def test_reads_back_what_was_encoded(self):
        dx, dy, visible, height, step = read(observation(30.0, -40.0, True, 12.5, 7))
        self.assertAlmostEqual(dx, 30.0, places=3)
        self.assertAlmostEqual(dy, -40.0, places=3)
        self.assertTrue(visible)
        self.assertAlmostEqual(height, 12.5, places=3)
        self.assertEqual(step, 7)

    def test_directions_match_the_image(self):
        low = expert(observation(dy_px=50.0))         # sweet low: gripper beyond it
        self.assertLess(low[0], 0)
        left = expert(observation(dx_px=-50.0))       # sweet left: gripper left of it
        self.assertLess(left[1], 0)
        self.assertLess(expert(observation())[2], 0)  # above the grasp height: descend

    def test_descends_slowly_while_badly_aligned(self):
        far = expert(observation(dx_px=120.0, height=20.0))
        near = expert(observation(dx_px=10.0, height=20.0))
        self.assertGreater(far[2], near[2])

    def test_closes_only_when_low_and_aligned(self):
        self.assertGreater(expert(observation(10.0, 10.0, height=1.0))[3], 0)
        self.assertLess(expert(observation(ALIGN_PX + 20, 0.0, height=1.0))[3], 0)
        self.assertLess(expert(observation(0.0, 0.0, height=6.0))[3], 0)
        last = red_env.MAX_STEPS - 1
        self.assertGreater(expert(observation(ALIGN_PX + 20, 0.0, height=1.0,
                                              step=last))[3], 0)

    def test_climbs_when_the_sweet_is_out_of_view(self):
        action = expert(observation(visible=False))
        self.assertGreater(action[2], 0)
        self.assertLess(action[3], 0)

    def test_solves_the_fake_episode(self):
        environment = red_env.FakeRedSweetEnv(seed=0)
        held = 0
        for episode in range(50):
            transitions, info = run_expert.run_episode(environment)
            held += transitions[-1]['reward']
        self.assertGreaterEqual(held, 48)


class DemoFileTests(unittest.TestCase):
    def test_schema_matches_the_simulation_project(self):
        environment = red_env.FakeRedSweetEnv(seed=1)
        episodes = [run_expert.run_episode(environment)[0] for _ in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            path = run_expert.save_transitions(Path(folder) / 'd.npz',
                                               run_expert.renumber(episodes))
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(int(data['schema_version']), 1)
                self.assertEqual(set(data.files), {
                    'schema_version', 'observations', 'policy_actions',
                    'executed_actions', 'next_observations', 'rewards',
                    'terminated', 'truncated', 'sources', 'interventions',
                    'episode_ids', 'step_ids', 'successes'})
                self.assertEqual(data['observations'].shape[1],
                                 len(red_env.OBSERVATION_NAMES))
                self.assertEqual(list(np.unique(data['episode_ids'])), [0, 1, 2])
                ends = np.flatnonzero(data['terminated'] | data['truncated'])
                self.assertEqual(len(ends), 3)
                np.testing.assert_array_equal(data['successes'],
                                              data['rewards'] == 1.0)
                self.assertTrue((data['sources'] == 'demonstration').all())
                np.testing.assert_array_equal(data['policy_actions'],
                                              data['executed_actions'])

    def test_merge_renumbers_and_keeps_every_episode(self):
        from pc_client.red_rl import demos
        environment = red_env.FakeRedSweetEnv(seed=2)
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, count in (('a', 2), ('b', 3)):
                episodes = [run_expert.run_episode(environment)[0] for _ in range(count)]
                paths.append(run_expert.save_transitions(
                    Path(folder) / (name + '.npz'), run_expert.renumber(episodes)))
            summary = demos.merge(paths, Path(folder) / 'merged.npz')
            self.assertEqual(summary['episodes'], 5)
            merged = demos.load(Path(folder) / 'merged.npz')
            split = demos.episodes(merged)
            self.assertEqual(len(split), 5)
            self.assertEqual([int(merged['episode_ids'][i[0]]) for i in split],
                             [0, 1, 2, 3, 4])
            originals = sum(len(demos.load(p)['rewards']) for p in paths)
            self.assertEqual(len(merged['rewards']), originals)
            self.assertEqual([o['file'] for o in summary['origin']],
                             ['a.npz', 'a.npz', 'b.npz', 'b.npz', 'b.npz'])
            self.assertTrue((Path(folder) / 'merged.json').is_file())


if __name__ == '__main__':
    unittest.main()
