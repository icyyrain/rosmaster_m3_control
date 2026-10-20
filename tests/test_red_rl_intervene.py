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

    def test_far_off_but_heading_back_is_left_alone(self):
        # The offset start: the sweet 127 px above the gripper pixel (the
        # gripper is short of it) and the policy moving out toward it, as it
        # did in the first S3 run when the old rule took over anyway.
        obs = observation(dx=5, dy=-127, height=35.0, step=0)
        heading_back = np.asarray([0.96, 0.12, -0.39, -0.93], dtype=np.float32)
        self.assertGreater(intervene.toward_mm(obs, heading_back), 3.0)
        self.assertIsNone(self.reason(obs, heading_back))
        moving_away = np.asarray([-0.6, 0.0, -0.39, -0.93], dtype=np.float32)
        self.assertEqual(self.reason(obs, moving_away), 'off_target')
        sideways = np.asarray([0.0, 0.9, -0.39, -0.93], dtype=np.float32)
        self.assertEqual(self.reason(obs, sideways), 'off_target')

    def test_a_close_in_mid_air_or_far_off_is_held_back(self):
        self.assertEqual(self.reason(observation(height=12.0), CLOSE), 'premature_close')
        self.assertEqual(self.reason(observation(dx=75, height=1.0), CLOSE), 'premature_close')
        # as high and as far off as real grasps that held
        self.assertIsNone(self.reason(observation(dx=60, height=4.5), CLOSE))

    def test_near_the_bottom_edge_and_not_coming_back(self):
        obs = observation(dy=30, height=8.0)
        self.assertEqual(self.reason(obs, DESCEND), 'off_target')
        coming_back = np.asarray([-0.5, 0.0, -1.0, -1.0], dtype=np.float32)
        self.assertIsNone(self.reason(obs, coming_back))

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

    def test_a_policy_that_always_closes_is_held_back_until_it_is_low(self):
        steps, reward = self.run_episode(lambda obs: CLOSE, intervene_on=True)
        self.assertEqual(steps[0]['intervention_reason'], 'premature_close')
        self.assertGreater(len(steps), 3)          # not over at the first step
        self.assertEqual(reward, 1.0)

    def test_out_of_view_it_goes_back_toward_where_the_sweet_was(self):
        env = intervene.RuleIntervention(red_env.FakeRedSweetEnv(seed=1))
        env.reset(options={'intervene': True})
        env.last_seen = observation(dy=35, height=6.0)       # last seen low in the frame
        env.observation = observation(visible=False)
        back = env.correction()
        self.assertLess(back[0], 0.0)                        # radial in, bringing it up
        self.assertGreaterEqual(back[intervene.UP], 0.0)     # never descending blind
        self.assertLess(back[intervene.CLOSE], 0.0)

    def test_takeover_is_short(self):
        steps, _ = self.run_episode(lambda obs: HOVER, intervene_on=True)
        run = 0
        for info in steps:
            run = run + 1 if 'intervene_action' in info else 0
            self.assertLessEqual(run, intervene.TAKEOVER_STEPS)


if __name__ == '__main__':
    unittest.main()
