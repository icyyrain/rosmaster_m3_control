import unittest

import numpy as np

import reach_candy
from pc_client.red_rl import env as red_env
from pc_client.red_rl import intervene
from pc_client.red_rl.expert import expert

POSE = [90, 20, 60, 30, 90, 160]
GX, GY = reach_candy.GRIPPER_PX
DESCEND = np.asarray([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
HOVER = np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
CLOSE = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def observation(dx=0.0, dy=0.0, height=10.0, step=3, visible=True):
    target = (GX + dx, GY + dy, 4000.0) if visible else None
    return red_env.encode(target, height, POSE, step)


class ReasonTests(unittest.TestCase):
    def reason(self, obs, action=DESCEND, history=None, steps_left=10):
        return intervene.reason_to_intervene(obs, action, history or [(10.0, 0.0)], steps_left)

    def test_a_policy_on_course_is_left_alone(self):
        self.assertIsNone(self.reason(observation(dx=30, dy=20)))

    def test_clear_mistakes_are_corrected(self):
        self.assertEqual(self.reason(observation(visible=False)), 'target_not_visible')
        self.assertEqual(self.reason(observation(dx=90)), 'off_target')
        self.assertEqual(self.reason(observation(height=-1.0)), 'below_grasp_and_descending')
        self.assertEqual(self.reason(observation(), steps_left=3), 'running_out_of_steps')

    def test_closing_is_never_overridden_for_time_or_height(self):
        self.assertIsNone(self.reason(observation(height=-1.0), action=CLOSE, steps_left=1))

    def test_no_progress_over_three_steps(self):
        stuck = [(1.0, 20.0)] * 4
        self.assertEqual(self.reason(observation(height=1.0, dx=20), HOVER, stuck),
                         'no_progress')
        moving = [(4.0, 20.0), (3.0, 20.0), (2.0, 20.0), (1.0, 20.0)]
        self.assertIsNone(self.reason(observation(height=1.0, dx=20), HOVER, moving))
        self.assertIsNone(self.reason(observation(height=1.0, dx=20), HOVER, stuck[:3]))


class WrapperTests(unittest.TestCase):
    def run_episode(self, policy, intervene_on, seed=3):
        env = intervene.RuleIntervention(red_env.FakeRedSweetEnv(seed=seed))
        obs, info = env.reset(options={'intervene': intervene_on, 'max_steps': 20})
        steps = []
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            steps.append(info)
            if terminated or truncated:
                return steps, reward

    def test_off_unless_the_reset_asks(self):
        steps, _ = self.run_episode(lambda obs: HOVER, intervene_on=False)
        self.assertFalse(any('intervene_action' in info for info in steps))
        self.assertEqual(steps[-1]['outcome'], 'step_limit')
        self.assertEqual(steps[-1]['intervention_count'], 0)

    def test_a_hovering_policy_is_taken_over_and_the_expert_finishes(self):
        steps, reward = self.run_episode(lambda obs: HOVER, intervene_on=True)
        corrected = [info for info in steps if 'intervene_action' in info]
        self.assertTrue(corrected)
        self.assertEqual(corrected[0]['intervention_reason'], 'no_progress')
        self.assertEqual(len(corrected[0]['intervene_action']), 4)
        self.assertGreater(steps[-1]['intervention_count'], 0)
        self.assertEqual(steps[-1]['intervention_steps'], len(corrected))
        self.assertEqual(reward, 1.0)

    def test_the_expert_alone_is_never_corrected(self):
        for seed in range(5):
            steps, _ = self.run_episode(expert, intervene_on=True, seed=seed)
            self.assertEqual(steps[-1]['intervention_count'], 0)

    def test_takeover_is_short(self):
        steps, _ = self.run_episode(lambda obs: HOVER, intervene_on=True)
        run = 0
        for info in steps:
            run = run + 1 if 'intervene_action' in info else 0
            self.assertLessEqual(run, intervene.TAKEOVER_STEPS)


if __name__ == '__main__':
    unittest.main()
