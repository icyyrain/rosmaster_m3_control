import unittest

import numpy as np

import reach_candy
from pc_client.red_rl import env as red_env
from pc_client.red_rl import intervene
from pc_client.red_rl.expert import expert

POSE = [90, 20, 60, 30, 90, 160]
GX, GY = reach_candy.GRIPPER_PX


def observation(dx=0.0, dy=0.0, height=1.0, visible=True):
    return red_env.encode((GX + dx, GY + dy, 4000.0) if visible else None, height, POSE, 5)


def run(policy, auto_close, seed=0):
    env = red_env.FakeRedSweetEnv(seed=seed)
    obs, info = env.reset(options={'auto_close': auto_close})
    assert info['auto_close'] is auto_close
    for step in range(1, 100):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            return step, reward, info, obs


def moves_only(obs):
    action = np.array(expert(obs), dtype=np.float32)
    action[3] = -1.0
    return action


class GraspRegionTests(unittest.TestCase):
    def test_region_is_the_experts_close_rule(self):
        self.assertTrue(red_env.in_grasp_region(observation(dx=15, dy=-15, height=1.4)))
        self.assertFalse(red_env.in_grasp_region(observation(height=2.0)))
        self.assertFalse(red_env.in_grasp_region(observation(dx=20, dy=20)))
        self.assertFalse(red_env.in_grasp_region(observation(visible=False)))


class AutoCloseTests(unittest.TestCase):
    def test_closes_by_itself_on_arrival(self):
        for seed in range(5):
            steps, reward, info, obs = run(moves_only, auto_close=True, seed=seed)
            self.assertIn(info['outcome'], ('held', 'not_held'))
            self.assertTrue(red_env.in_grasp_region(obs))
            self.assertEqual(reward, 1.0)

    def test_the_policys_close_is_ignored(self):
        hover_and_close = lambda obs: np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        steps, reward, info, _ = run(hover_and_close, auto_close=True)
        self.assertEqual(info['outcome'], 'step_limit')
        self.assertEqual(steps, red_env.MAX_STEPS)
        # without auto-close the same action closes at once
        steps, _, info, _ = run(hover_and_close, auto_close=False)
        self.assertEqual(steps, 1)

    def test_low_but_off_tries_on_the_last_step_only(self):
        off = observation(dx=55, height=0.5)
        self.assertFalse(red_env.auto_close_now(off, last_step=False))
        self.assertTrue(red_env.auto_close_now(off, last_step=True))
        self.assertFalse(red_env.auto_close_now(observation(dx=55, height=3.0), last_step=True))
        self.assertFalse(red_env.auto_close_now(observation(visible=False, height=0.5),
                                                last_step=True))

    def test_a_policy_that_gets_low_but_drifts_off_still_gets_its_try(self):
        def descend_then_sidestep(obs):
            low = obs[4] * 50.0 <= 4.0
            return np.asarray([0.0, 0.6 if low else 0.0, -1.0, -1.0], dtype=np.float32)
        tried = 0
        for seed in range(6):
            steps, _, info, _ = run(descend_then_sidestep, auto_close=True, seed=seed)
            if steps == red_env.MAX_STEPS and info['outcome'] in ('held', 'not_held'):
                tried += 1
        self.assertGreater(tried, 0)

    def test_off_by_default(self):
        steps, _, info, _ = run(moves_only, auto_close=False)
        self.assertEqual(info['outcome'], 'step_limit')

    def test_the_corrector_brings_a_hovering_policy_into_the_region(self):
        env = intervene.RuleIntervention(red_env.FakeRedSweetEnv(seed=2))
        env.reset(options={'intervene': True, 'auto_close': True})
        hover = np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(red_env.MAX_STEPS):
            _, reward, terminated, truncated, info = env.step(hover)
            if terminated or truncated:
                break
        self.assertEqual(info['outcome'], 'held')
        self.assertGreater(info['intervention_count'], 0)


if __name__ == '__main__':
    unittest.main()
