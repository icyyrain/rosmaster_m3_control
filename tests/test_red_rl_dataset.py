import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pc_client.red_rl.dataset import EpisodeRecorder, FEATURE_NAMES, features, load_examples
from pc_client.red_rl.train_bc import fit, predict
from pc_client.red_rl.validate_dataset import validate_episode
from pc_client.red_rl.collect import align_teacher, pick, TEACHER_ALIGN_TOLERANCE_PX
from pc_client.red_rl.collect import TeacherSession
from pc_client.red_rl.shadow import ShadowPolicy, episode_report


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(object_top_mm=32.0, stop_height_mm=56.0,
                                       object_aspect=1.0)
        self.state = {'cx': 700.0, 'cy': 675.0, 'height': 120.0,
                      'range_mm': 260.0, 'area': 9000}
        self.pose = [90, 120, 0, 0, 90, 90]

    def test_features_mark_missing_depth_and_vision(self):
        observed = features(self.state, self.pose, self.session, 'align')
        missing = features(None, self.pose, self.session, 'creep')
        self.assertEqual(len(observed), len(FEATURE_NAMES))
        self.assertEqual(observed[FEATURE_NAMES.index('target_visible')], 1)
        self.assertEqual(missing[FEATURE_NAMES.index('target_visible')], 0)
        self.assertEqual(missing[FEATURE_NAMES.index('height_valid')], 0)
        self.assertEqual(missing[FEATURE_NAMES.index('phase_creep')], 1)

    def test_record_and_filter_by_human_success(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = np.zeros((12, 16, 3), dtype=np.uint8)
            good = EpisodeRecorder(folder, {'task': 'red_sweet_approach'})
            good.before_send(self.state, self.pose, [91, 120, 0, 2, 90, 90],
                             'align', self.session, frame)
            good.after_look(self.state, [91, 120, 0, 2, 90, 90],
                            self.session, frame)
            good.finish('completed', True)
            bad = EpisodeRecorder(folder, {'task': 'red_sweet_approach'})
            bad.before_send(self.state, self.pose, [91, 120, 0, 2, 90, 90],
                            'align', self.session, frame)
            bad.after_look(self.state, [91, 120, 0, 2, 90, 90],
                           self.session, frame)
            bad.finish('completed', False, auto_success=True,
                       auto_evidence={'method': 'short_lift_motion_v1',
                                      'success': True})
            self.assertEqual(len(load_examples(folder)), 1)
            self.assertEqual(len(load_examples(folder, success_only=False)), 2)
            meta, rows = load_examples(folder)[0]
            self.assertTrue(meta['human_success'])
            self.assertEqual(rows[0]['action_joint_delta_deg'], [1, 0, 0, 2])
            self.assertTrue((good.path / rows[0]['image']).exists())
            self.assertEqual(validate_episode(good.path), [])
            self.assertEqual(validate_episode(bad.path), [])

    def test_ridge_baseline_recovers_simple_teacher(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(50, len(FEATURE_NAMES)))
        y = np.column_stack([x[:, 0] * 2, x[:, 1] * -3,
                             x[:, 2], np.ones(len(x))])
        weights, mean, scale = fit(x, y, ridge=0.001)
        estimated = predict(x, weights, mean, scale)
        self.assertLess(np.abs(estimated - y).mean(), 0.001)

    def test_teacher_tolerance_is_local_to_alignment(self):
        original = pick.ALIGN_TOLERANCE_PX
        with patch.object(pick, 'align', side_effect=lambda _:
                          pick.ALIGN_TOLERANCE_PX):
            self.assertEqual(align_teacher(None), TEACHER_ALIGN_TOLERANCE_PX)
        self.assertEqual(pick.ALIGN_TOLERANCE_PX, original)

    def test_shadow_prediction_cannot_replace_teacher_command(self):
        class FakeDemo:
            def before_send(self, *args):
                self.pending = {}

        session = TeacherSession.__new__(TeacherSession)
        session.demo = FakeDemo()
        session.phase = 'align'
        session.last_state = self.state
        session.pose = list(self.pose)
        session.object_top_mm = 32.0
        session.stop_height_mm = 56.0
        session.object_aspect = 1.0
        session.streams = SimpleNamespace(colour=lambda: None)
        session.shadow = SimpleNamespace(action=lambda _: [100, 100, 100, 100])
        target = [91, 120, 0, 2, 90, 90]
        with patch.object(pick.Session, 'send') as real_send:
            session.send(target, 600, 1.0)
        real_send.assert_called_once_with(target, 600, 1.0)
        self.assertEqual(session.demo.pending['shadow_action_joint_delta_deg'],
                         [100, 100, 100, 100])
        session.shadow = SimpleNamespace(action=lambda _:
                                         (_ for _ in ()).throw(ValueError('bad model')))
        with patch.object(pick.Session, 'send') as real_send:
            session.send(target, 600, 1.0)
        real_send.assert_called_once_with(target, 600, 1.0)
        self.assertEqual(session.demo.pending['shadow_error'], 'bad model')

    def test_shadow_model_load_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'policy.npz'
            weights = np.zeros((len(FEATURE_NAMES) + 1, 4))
            weights[-1, :] = [1, 0, 0, 2]
            np.savez(model, weights=weights,
                     feature_mean=np.zeros(len(FEATURE_NAMES)),
                     feature_scale=np.ones(len(FEATURE_NAMES)),
                     feature_names=np.asarray(FEATURE_NAMES),
                     training_episodes=np.asarray(['teacher-1']))
            shadow = ShadowPolicy(model)
            self.assertEqual(shadow.action(np.zeros(len(FEATURE_NAMES))),
                             [1.0, 0.0, 0.0, 2.0])
            recorder = EpisodeRecorder(folder, {'task': 'red_sweet_approach'})
            recorder.before_send(self.state, self.pose,
                                 [91, 120, 0, 2, 90, 90],
                                 'align', self.session,
                                 np.zeros((12, 16, 3), dtype=np.uint8))
            recorder.pending['shadow_action_joint_delta_deg'] = [1, 0, 0, 2]
            recorder.pending['shadow_abs_error_deg'] = [0, 0, 0, 0]
            recorder.after_look(self.state, [91, 120, 0, 2, 90, 90],
                                self.session,
                                np.zeros((12, 16, 3), dtype=np.uint8))
            recorder.finish('completed', True)
            self.assertEqual(episode_report(recorder.path)['mae_deg'],
                             [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
