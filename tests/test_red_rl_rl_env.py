import json
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import kinematics
from pc_client.red_rl import cartesian
from pc_client.red_rl import env as red_env
from pc_client.red_rl.rpc import EnvironmentRpcServer, RemoteEnv
from pc_client.red_rl.rpc.codec import _from_wire, _to_wire

POSES = ([90, 18, 62, 24, 90, 90], [88, 12, 56, 32, 90, 90],
         [88, 10, 55, 37, 90, 90], [106, 10, 56, 38, 90, 90])


class CartesianTests(unittest.TestCase):
    def test_steps_land_close_to_the_request(self):
        rng = np.random.default_rng(0)
        for pose in POSES:
            pitch = sum(pose[1:4])
            for _ in range(25):
                request = rng.uniform(-cartesian.MAX_STEP_MM, cartesian.MAX_STEP_MM, 3)
                new, achieved = cartesian.step(pose, request, pitch)
                self.assertLess(np.linalg.norm(achieved - request), 2.2)
                self.assertLessEqual(abs(sum(new[1:4]) - pitch),
                                     cartesian.PITCH_TOLERANCE_DEG)
                self.assertEqual(new[4:], pose[4:])
                self.assertTrue(cartesian.in_bounds(new))
                actual = kinematics.gripper_mm(new) - kinematics.gripper_mm(pose)
                np.testing.assert_allclose(
                    cartesian.base_to_local(pose, actual), achieved, atol=1e-9)

    def test_no_request_no_motion(self):
        pose = POSES[1]
        self.assertEqual(cartesian.step(pose, [0, 0, 0], sum(pose[1:4]))[0], pose)

    def test_axes_follow_the_heading(self):
        for pose in (POSES[1], POSES[3]):      # base turned two ways
            _, achieved = cartesian.step(pose, [4, 0, 0], sum(pose[1:4]))
            self.assertGreater(achieved[0], 2.5)
            self.assertLess(abs(achieved[1]), 1.0)

    def test_offset_walks_further_than_one_step(self):
        pose = POSES[0]
        moved = cartesian.offset(pose, [10, -8, 6], sum(pose[1:4]))
        actual = cartesian.base_to_local(
            pose, kinematics.gripper_mm(moved) - kinematics.gripper_mm(pose))
        self.assertLess(np.linalg.norm(actual - [10, -8, 6]), 3.5)


def scripted_session():
    """A Session double: commands move the pose, looks see a fixed target."""
    session = mock.Mock()
    session.args = red_env.teacher_args(execute=False)
    session.pose = list(red_env.arm_control.HOME_POSE)
    session.last_px = (707.0, 680.0)
    session.stop_height_mm = 23.0
    session.object_top_mm = 26.0
    session.streams.colour.return_value = None
    sent = []

    def send(pose, duration, settle):
        sent.append(list(pose))
        session.pose = list(pose)
    session.send.side_effect = send
    session.sent = sent
    session.look.return_value = {'cx': 700.0, 'cy': 660.0, 'area': 3000,
                                 'height': 30.0, 'range_mm': 130.0}
    return session


class EnvTests(unittest.TestCase):
    def make_env(self, folder):
        session = scripted_session()
        environment = red_env.RedSweetApproachEnv(session=session, audit_root=folder,
                                                  seed=3)
        return environment, session

    def reset(self, environment, session, height=50.0):
        descent_pose = [90, 18, 62, 24, 90, 90]

        def descend(sess, state):
            sess.pose = list(descent_pose)
            sess.last_px = (705.0, 675.0)       # the real descent's last look
            return {'cx': 705.0, 'cy': 675.0, 'area': 2500, 'height': height}
        with mock.patch.object(red_env.pick, 'preflight', return_value=True), \
                mock.patch.object(red_env.collect, 'align_teacher',
                                  return_value={'cx': 700, 'cy': 670}), \
                mock.patch.object(red_env.pick, 'descend', side_effect=descend):
            return environment.reset()

    def test_reset_calibrates_and_offsets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            observation, info = self.reset(environment, session)
            self.assertEqual(observation.shape, (len(red_env.OBSERVATION_NAMES),))
            self.assertTrue(environment.observation_space.contains(observation))
            self.assertEqual(session.sent[0], list(red_env.arm_control.HOME_POSE))
            self.assertNotEqual(session.pose[:4], [90, 18, 62, 24])
            self.assertAlmostEqual(
                environment.bias_mm,
                50.0 - kinematics.gripper_height_mm([90, 18, 62, 24, 90, 90]))
            json.dumps(_to_wire(info))                  # survives the wire
            self.assertEqual(_from_wire(json.loads(json.dumps(_to_wire(info)))), info)

    def test_lost_start_offset_is_retried_from_the_descent_pose(self):
        import tempfile
        found = {'cx': 700.0, 'cy': 660.0, 'area': 3000, 'height': 30.0}
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            session.look.side_effect = [None, found, found]
            observation, info = self.reset(environment, session)
            self.assertEqual(observation[2], 1.0)
            self.assertIn([90, 18, 62, 24, 90, 90], session.sent)   # went back
            meta = environment.audit.meta
            self.assertEqual(meta['start_offset_attempts'], 2)
            self.assertLessEqual(meta['start_offset_request_mm'][0],
                                 red_env.START_OFFSET_MM['radial'][1])

    def test_start_offset_that_keeps_losing_the_sweet_stops(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            found = {'cx': 700.0, 'cy': 660.0, 'area': 3000, 'height': 30.0}
            session.look.side_effect = [None, found] * red_env.START_ATTEMPTS
            with self.assertRaisesRegex(RuntimeError, 'start offset'):
                self.reset(environment, session)
            self.assertEqual(session.sent[-1], list(red_env.arm_control.HOME_POSE))

    def test_offset_prediction_follows_the_image_axes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            session.last_px = (700.0, 650.0)
            pose = [90, 18, 62, 24, 90, 90]
            outward = cartesian.offset(pose, [8, 0, 0], sum(pose[1:4]))
            leftward = cartesian.offset(pose, [0, 8, 0], sum(pose[1:4]))
            self.assertGreater(environment._predict_px(pose, outward)[1], 650 + 30)
            self.assertLess(environment._predict_px(pose, leftward)[0], 700 - 30)

    def test_failed_reset_goes_home_and_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            with mock.patch.object(red_env.pick, 'preflight', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'Preflight'):
                    environment.reset()
            self.assertEqual(session.sent[-1], list(red_env.arm_control.HOME_POSE))

    def test_box_clips_instead_of_diving(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            self.reset(environment, session)
            for _ in range(15):
                _, reward, terminated, truncated, info = environment.step([0, 0, -1, -1])
                self.assertEqual(reward, 0.0)
                self.assertGreaterEqual(info['height_mm'],
                                        environment._lowest() - 1e-6)
                if terminated or truncated:
                    break
            self.assertTrue(info['clipped'])

    def test_lost_target_ends_the_episode_at_home(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            self.reset(environment, session)
            session.look.return_value = None
            outcomes = [environment.step([0.5, 0, 0, -1]) for _ in range(2)]
            self.assertFalse(outcomes[0][2])
            self.assertTrue(outcomes[1][2])
            self.assertEqual(outcomes[1][4]['outcome'], 'target_lost')
            self.assertEqual(session.sent[-1], list(red_env.arm_control.HOME_POSE))

    def test_step_limit_truncates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            environment, session = self.make_env(folder)
            self.reset(environment, session)
            result = None
            for _ in range(red_env.MAX_STEPS):
                result = environment.step([0, 0, 0, -1])
            self.assertTrue(result[3])
            self.assertFalse(result[2])
            self.assertEqual(result[4]['outcome'], 'step_limit')

    def test_close_rewards_the_camera_verdict(self):
        import tempfile
        for success, reward in ((True, 1.0), (None, 0.0), (False, 0.0)):
            with tempfile.TemporaryDirectory() as folder:
                environment, session = self.make_env(folder)
                self.reset(environment, session)
                evidence = {'success': success, 'reason': 'test'}
                with mock.patch.object(red_env.pick, 'grip', return_value={}), \
                        mock.patch.object(red_env.collect, 'short_lift_and_release',
                                          return_value=(None, evidence, None)) as lift:
                    _, got, terminated, truncated, info = environment.step([0, 0, 0, 1])
                self.assertEqual((got, terminated, truncated), (reward, True, False))
                self.assertTrue(lift.call_args.kwargs['auto'])


class FakeEnvAndWireTests(unittest.TestCase):
    def test_fake_env_rewards_a_sensible_policy(self):
        environment = red_env.FakeRedSweetEnv(seed=0)
        held = 0
        for episode in range(20):
            observation, _ = environment.reset(seed=episode)
            for _ in range(red_env.MAX_STEPS):
                dx, dy, height = (observation[0] * 640 * 0.18,
                                  observation[1] * 360 * 0.18, observation[4] * 50)
                # undo the image mapping: dy is radial, -dx is lateral
                action = np.clip([-dy / 4.0, dx / 4.0, -height / 4.0, -1], -1, 1)
                if abs(height) < 2.0 and math.hypot(dx, dy) < 3.0:
                    action[3] = 1.0
                observation, reward, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    held += reward
                    break
        self.assertGreaterEqual(held, 15)

    def test_remote_round_trip(self):
        server = EnvironmentRpcServer(red_env.FakeRedSweetEnv(seed=1),
                                      host='127.0.0.1', port=0)
        server.start_in_thread()
        try:
            remote = RemoteEnv(*server.address, timeout=10)
            self.assertEqual(remote.observation_space.shape,
                             (len(red_env.OBSERVATION_NAMES),))
            self.assertEqual(remote.action_space.shape, (4,))
            observation, _ = remote.reset(seed=5)
            local, _ = red_env.FakeRedSweetEnv(seed=1).reset(seed=5)
            np.testing.assert_array_equal(observation, local)
            result = remote.step(np.zeros(4, dtype=np.float32))
            self.assertEqual(len(result), 5)
            self.assertEqual(remote.metadata['name'], 'red_sweet_last_cm_fake')
            remote.close()
        finally:
            server.stop()


if __name__ == '__main__':
    unittest.main()
