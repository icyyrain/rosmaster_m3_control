import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import arm_control
import numpy as np

from pc_client.red_rl.auto_reward import assess_short_lift, compare_labels
from pc_client.red_rl.collect import short_lift_and_release
import pick_and_place as pick


class ShortLiftRewardTests(unittest.TestCase):
    def test_held_target_stays_near_gripper(self):
        result = assess_short_lift((670, 623), (669, 623), (669, 685), 35.6)
        self.assertIs(result['success'], True)
        self.assertGreater(result['expected_stationary_motion_px'], 60)

    def test_a_hold_that_slips_a_fifth_is_not_counted(self):
        # 20261015-155642: moved 21 of 97 px on the lift and fell out while
        # being carried. v1 called that HELD; v2 cannot tell.
        before, stationary = (707.0, 680.0), (707.0, 777.0)
        slipped = assess_short_lift(before, (707.0, 701.0), stationary, 35.0)
        firm = assess_short_lift(before, (707.0, 688.0), stationary, 35.0)
        self.assertIsNone(slipped['success'])
        self.assertIs(firm['success'], True)
        self.assertEqual(firm['method'], 'short_lift_motion_v2')

    def test_table_target_follows_stationary_prediction(self):
        result = assess_short_lift((670, 623), (670, 684), (669, 685), 35.6)
        self.assertIs(result['success'], False)

    def test_middle_motion_abstains(self):
        result = assess_short_lift((670, 623), (670, 650), (669, 685), 35.6)
        self.assertIsNone(result['success'])

    def test_lost_target_or_small_motion_abstains(self):
        self.assertIsNone(assess_short_lift((670, 623), None, (669, 685), 35.6)['success'])
        self.assertIsNone(assess_short_lift((670, 623), (670, 623), (670, 650), 20)['success'])

    def test_comparison_keeps_unknown_and_old_episodes_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = [
                {'outcome': 'completed', 'human_success': True,
                 'auto_success': True, 'auto_evidence': {'method': 'test'}},
                {'outcome': 'completed', 'human_success': False,
                 'auto_success': True, 'auto_evidence': {'method': 'test'}},
                {'outcome': 'completed', 'human_success': True,
                 'auto_success': None, 'auto_evidence': {'method': 'test'}},
                {'outcome': 'completed', 'human_success': None,
                 'auto_success': False, 'auto_evidence': {'method': 'test'}},
                {'outcome': 'completed', 'human_success': True},
            ]
            for index, metadata in enumerate(examples):
                folder = root / str(index)
                folder.mkdir()
                (folder / 'episode.json').write_text(
                    json.dumps(metadata), encoding='utf-8')
            counts = compare_labels(root)
            self.assertEqual(counts, {
                'human_labeled': 3, 'agree': 1, 'disagree': 1,
                'auto_abstained': 1, 'human_unknown': 1, 'without_auto': 1,
                'auto_only': 0,
            })

    def test_short_lift_still_releases_and_homes_after_label(self):
        class FakeSession:
            def __init__(self):
                self.pose = [70, 6, 70, 31, 90, 164]
                self.last_px = (670, 623)
                self.commands = []
                self.streams = mock.Mock()
                self.streams.colour.return_value = np.zeros((8, 8, 3), dtype=np.uint8)

            def send(self, pose, _duration, _settle):
                self.pose = list(pose)
                self.commands.append(list(pose))

            def look(self, _predicted):
                return {'cx': 669, 'cy': 623, 'area': 3906}

        session = FakeSession()
        demo = mock.Mock()
        demo._image.return_value = 'images/audit.jpg'
        lift = ([70, 21, 61, 31, 90, 164], 35.6)
        release = ([70, 9, 68, 30, 90, 164], 10.4)
        with mock.patch.object(pick, 'low_release_pose', side_effect=[lift, release]):
            with mock.patch('pc_client.red_rl.collect.ask_success', return_value=True):
                human, automatic, relocation = short_lift_and_release(
                    session, [70, 6, 70, 31, 90, 164], None, demo)
        self.assertIs(human, True)
        self.assertIs(automatic['success'], True)
        self.assertIsNone(relocation)       # attended runs put it back in place
        # lift, base turn and back, release, open, HOME
        self.assertEqual(len(session.commands), 6)
        self.assertEqual(session.commands[1][1:], lift[0][1:])
        self.assertEqual(session.commands[2], lift[0])
        self.assertEqual(automatic['decided_by'], 'lift')      # no red in a blank frame
        self.assertEqual(session.commands[-1], list(arm_control.HOME_POSE))


if __name__ == '__main__':
    unittest.main()
