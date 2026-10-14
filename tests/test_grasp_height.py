import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pc_client'))

import pick_and_place


def session_for(top_mm, offset=None, controller='auto'):
    session = pick_and_place.Session.__new__(pick_and_place.Session)
    session.args = SimpleNamespace(grasp_offset_mm=offset,
                                   creep_controller=controller)
    session.object_top_mm = None
    session.stop_height_mm = pick_and_place.HEIGHT_FLOOR_MM
    if top_mm is not None:
        session.set_object_top(top_mm)
    return session


class GraspHeightTests(unittest.TestCase):

    def test_biscuit_keeps_the_height_that_held_it(self):
        self.assertAlmostEqual(session_for(16.0).stop_height_mm, 18.0)

    def test_sweet_closes_below_its_top_on_both_sides_of_old_switch(self):
        # The sweet measured 24.7-28.8 mm. Each of these used to land on one
        # side or the other of a 26 mm switch, 6 mm apart in grasp height.
        for top in (24.7, 25.8, 26.2, 28.8):
            stop = session_for(top).stop_height_mm
            self.assertLess(stop, top - 1.2, top)
            self.assertGreater(stop, top - 6.0, top)

    def test_height_is_continuous_in_object_height(self):
        a = session_for(25.9).stop_height_mm
        b = session_for(26.1).stop_height_mm
        self.assertAlmostEqual(a, b, delta=0.2)

    def test_floor_still_applies(self):
        self.assertEqual(session_for(10.0).stop_height_mm,
                         pick_and_place.HEIGHT_FLOOR_MM)

    def test_explicit_offset_is_relative_to_top(self):
        self.assertAlmostEqual(session_for(27.0, offset=-3.0).stop_height_mm, 24.0)
        self.assertAlmostEqual(session_for(27.0, offset=8.0).stop_height_mm, 35.0)

    def test_auto_controller_choice(self):
        calls = []
        original = (pick_and_place.creep_adaptive, pick_and_place.creep_legacy)
        pick_and_place.creep_adaptive = lambda s, st: calls.append('adaptive')
        pick_and_place.creep_legacy = lambda s, st: calls.append('legacy')
        try:
            for top in (16.0, 25.8, 26.2, 29.6):
                pick_and_place.creep(session_for(top), {})
            pick_and_place.creep(session_for(None), {})
            pick_and_place.creep(session_for(27.0, controller='legacy'), {})
        finally:
            pick_and_place.creep_adaptive, pick_and_place.creep_legacy = original
        self.assertEqual(calls, ['adaptive'] * 4 + ['legacy', 'legacy'])


if __name__ == '__main__':
    unittest.main()
