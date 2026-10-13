import pathlib
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pc_client'))

import pick_and_place


class PickTargetHandoffTests(unittest.TestCase):
    def test_colour_handoff_is_anchored_to_depth_target(self):
        session = pick_and_place.Session.__new__(pick_and_place.Session)
        session.args = SimpleNamespace(detector='table')
        session.acquired = True
        session.last_px = (713.0, 488.0)
        session.streams = mock.Mock()
        session.rng = mock.Mock()
        found = {'cx': 718.0, 'cy': 489.0}

        with mock.patch.object(session, '_look_red', return_value=found) as tracker, \
                mock.patch.object(pick_and_place.table_plane, 'gripper_height_retry',
                                  return_value=(174.0, 11000)):
            result = session.look()

        tracker.assert_called_once_with((713.0, 488.0))
        self.assertEqual(result['cx'], 718.0)


if __name__ == '__main__':
    unittest.main()
