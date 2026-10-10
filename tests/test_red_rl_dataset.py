import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pc_client.red_rl.dataset import EpisodeRecorder, FEATURE_NAMES, features, load_examples
from pc_client.red_rl.train_bc import fit, predict


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
            bad.finish('completed', False)
            self.assertEqual(len(load_examples(folder)), 1)
            self.assertEqual(len(load_examples(folder, success_only=False)), 2)
            meta, rows = load_examples(folder)[0]
            self.assertTrue(meta['human_success'])
            self.assertEqual(rows[0]['action_joint_delta_deg'], [1, 0, 0, 2])
            self.assertTrue((good.path / rows[0]['image']).exists())

    def test_ridge_baseline_recovers_simple_teacher(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(50, len(FEATURE_NAMES)))
        y = np.column_stack([x[:, 0] * 2, x[:, 1] * -3,
                             x[:, 2], np.ones(len(x))])
        weights, mean, scale = fit(x, y, ridge=0.001)
        estimated = predict(x, weights, mean, scale)
        self.assertLess(np.abs(estimated - y).mean(), 0.001)


if __name__ == '__main__':
    unittest.main()
