import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pc_client.red_rl.dataset import (CURRENT_GRASP_RULE, FEATURE_NAMES,
                                      LEGACY_GRASP_RULE, EpisodeRecorder,
                                      current_stop_height_mm, grasp_rule,
                                      load_examples)
from pc_client.red_rl.validate_dataset import summarise_by_rule, validate_episode

STOP = FEATURE_NAMES.index('stop_height')


class GraspRuleTests(unittest.TestCase):
    def setUp(self):
        self.state = {'cx': 700.0, 'cy': 675.0, 'height': 120.0,
                      'range_mm': 260.0, 'area': 9000}
        self.pose = [90, 120, 0, 0, 90, 90]
        self.frame = np.zeros((12, 16, 3), dtype=np.uint8)

    def record(self, folder, meta, stop_height_mm, held=True):
        session = SimpleNamespace(object_top_mm=27.0, stop_height_mm=stop_height_mm,
                                  object_aspect=1.0)
        recorder = EpisodeRecorder(folder, dict(meta, task='red_sweet_approach',
                                                object_top_mm=27.0))
        for phase in ('align', 'descend', 'creep'):
            target = list(self.pose)
            target[1] -= 1
            recorder.before_send(self.state, self.pose, target, phase, session,
                                 self.frame)
            recorder.after_look(self.state, target, session, self.frame)
        recorder.finish('completed', held)
        return recorder.path

    def test_conversion_follows_pick_and_place(self):
        self.assertAlmostEqual(current_stop_height_mm(27.0), 23.5)
        self.assertAlmostEqual(current_stop_height_mm(16.0), 18.0)

    def test_unlabelled_episodes_are_legacy(self):
        self.assertEqual(grasp_rule({}), LEGACY_GRASP_RULE)
        self.assertEqual(grasp_rule({'grasp_rule': CURRENT_GRASP_RULE}),
                         CURRENT_GRASP_RULE)

    def test_legacy_creep_is_left_out_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            self.record(folder, {}, stop_height_mm=35.0)
            self.record(folder, {'grasp_rule': CURRENT_GRASP_RULE},
                        stop_height_mm=23.5)
            by_rule = {grasp_rule(meta): [row['phase'] for row in rows]
                       for meta, rows in load_examples(folder)}
            self.assertEqual(by_rule[LEGACY_GRASP_RULE], ['align', 'descend'])
            self.assertEqual(by_rule[CURRENT_GRASP_RULE],
                             ['align', 'descend', 'creep'])
            pooled = {grasp_rule(meta): len(rows)
                      for meta, rows in load_examples(folder, creep_rule=None)}
            self.assertEqual(pooled[LEGACY_GRASP_RULE], 3)

    def test_legacy_stop_height_is_restated_in_current_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            self.record(folder, {}, stop_height_mm=35.0)
            meta, rows = load_examples(folder)[0]
            for row in rows:
                self.assertAlmostEqual(row['observation'][STOP], 0.235)
                self.assertAlmostEqual(row['next_observation'][STOP], 0.235)
            meta, rows = load_examples(folder, harmonise=False)[0]
            self.assertAlmostEqual(rows[0]['observation'][STOP], 0.35)
            # The files on disk are never rewritten.
            first = json.loads((Path(folder) / meta['episode'] / 'steps.jsonl')
                               .read_text(encoding='utf-8').splitlines()[0])
            self.assertAlmostEqual(first['observation'][STOP], 0.35)

    def test_summary_never_pools_rules_or_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            self.record(folder, {}, 35.0, held=False)
            self.record(folder, {'grasp_profile': 'red_fixed_wrist'}, 35.0, held=True)
            self.record(folder, {'grasp_profile': 'red_fixed_wrist',
                                 'grasp_offset_override_mm': 5.0}, 32.0, held=False)
            path = self.record(folder, {'grasp_rule': CURRENT_GRASP_RULE,
                                        'grasp_profile': 'red_fixed_wrist'}, 23.5)
            summary = summarise_by_rule(sorted(Path(folder).glob('*/episode.json')))
            legacy = summary[LEGACY_GRASP_RULE]
            self.assertEqual(legacy['wrist_free']['not_held'], 1)
            self.assertEqual(legacy['default']['held'], 1)
            self.assertEqual(legacy['experiment']['not_held'], 1)
            self.assertEqual(summary[CURRENT_GRASP_RULE]['default']['held'], 1)
            self.assertEqual(validate_episode(path), [])

    def test_unknown_rule_fails_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.record(folder, {'grasp_rule': 'made_up'}, 30.0)
            self.assertIn("unknown grasp rule 'made_up'", validate_episode(path))


if __name__ == '__main__':
    unittest.main()
