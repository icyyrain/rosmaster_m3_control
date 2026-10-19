import random
import unittest
from unittest import mock

import cv2
import numpy as np

import pick_and_place as pick
import reach_candy
from pc_client.red_rl import auto_reward, collect, demos, hold_check, relocate

GRASP = [88, 10, 54, 43, 90, 167]
GX, GY = (int(v) for v in reach_candy.GRIPPER_PX)
TABLE_SHIFT_PX = 90             # about what 4 degrees of base turn moves the table


def frame(*discs):
    """Grey 1280x720 frame with red discs (x, y, radius); the bottom edge clips."""
    image = np.full((720, 1280, 3), 128, np.uint8)
    for x, y, radius in discs:
        cv2.circle(image, (int(x), int(y)), int(radius), (30, 30, 200), -1)
    return image


HELD = (GX - 10, GY + 20, 45)               # half under the jaws, clipped below
EMPTY = frame()


class TurnJudgeTests(unittest.TestCase):
    def test_red_that_turns_with_the_camera_is_held(self):
        evidence = hold_check.judge_turn(frame(HELD), frame(HELD))
        self.assertIs(evidence['success'], True)
        self.assertEqual(evidence['shift_px'], 0)
        self.assertEqual(evidence['method'], hold_check.METHOD)

    def test_red_that_stays_with_the_table_is_not_held(self):
        for direction in (1, -1):
            moved = (HELD[0] + direction * TABLE_SHIFT_PX,) + HELD[1:]
            evidence = hold_check.judge_turn(frame(HELD), frame(moved))
            self.assertIs(evidence['success'], False)
            self.assertEqual(evidence['reason'], 'red_stayed_with_the_table')
            self.assertAlmostEqual(evidence['shift_px'], direction * TABLE_SHIFT_PX, delta=2)

    def test_a_sliver_of_red_is_enough(self):
        sliver = (GX, 718 + 12, 30)                 # only 20 px rows show at the edge
        evidence = hold_check.judge_turn(frame(sliver), frame(sliver))
        self.assertGreaterEqual(evidence['red_before_px'], hold_check.MIN_RED_PX)
        self.assertIs(evidence['success'], True)

    def test_no_red_after_the_lift_leaves_it_to_the_lift(self):
        evidence = hold_check.judge_turn(EMPTY, frame(HELD))
        self.assertIsNone(evidence['success'])

    def test_red_gone_on_the_turn_is_not_held(self):
        evidence = hold_check.judge_turn(frame(HELD), EMPTY)
        self.assertIs(evidence['success'], False)

    def test_a_shift_between_held_and_table_cannot_tell(self):
        moved = (HELD[0] + 32,) + HELD[1:]
        evidence = hold_check.judge_turn(frame(HELD), frame(moved))
        self.assertIsNone(evidence['success'])

    def test_red_far_from_the_jaws_is_ignored(self):
        far = (GX + hold_check.ROI_HALF_WIDTH_PX + 60, 200, 40)
        evidence = hold_check.judge_turn(frame(HELD, far), frame(HELD, (far[0] - 90,) + far[1:]))
        self.assertIs(evidence['success'], True)


class CarryJudgeTests(unittest.TestCase):
    def test_still_in_the_jaws(self):
        held, evidence = hold_check.judge_carry(frame(HELD), frame((HELD[0] + 8,) + HELD[1:]))
        self.assertTrue(held)
        self.assertEqual(evidence['reason'], 'red_still_in_the_jaws')

    def test_gone_or_moved_away(self):
        self.assertFalse(hold_check.judge_carry(frame(HELD), EMPTY)[0])
        away = (HELD[0] + 2 * hold_check.CARRY_SHIFT_PX + 40,) + HELD[1:]
        self.assertFalse(hold_check.judge_carry(frame(HELD), frame(away))[0])

    def test_no_red_at_the_lift_does_not_abort(self):
        held, evidence = hold_check.judge_carry(EMPTY, EMPTY)
        self.assertTrue(held)
        self.assertEqual(evidence['reason'], 'no_red_after_the_lift_to_compare')


class CombineTests(unittest.TestCase):
    lift = dict(method=auto_reward.METHOD, success=None,
                reason='ambiguous_target_motion', before_px=[700, 650])

    def test_the_turn_decides_when_it_can(self):
        evidence = hold_check.combine(dict(self.lift), dict(success=True, reason='r'))
        self.assertIs(evidence['success'], True)
        self.assertEqual(evidence['decided_by'], 'turn')
        self.assertEqual(evidence['method'], auto_reward.TURN_METHOD)
        self.assertIn(evidence['method'], auto_reward.KNOWN_METHODS)
        self.assertEqual(evidence['short_lift']['reason'], 'ambiguous_target_motion')
        self.assertEqual(evidence['before_px'], [700, 650])

    def test_otherwise_the_lift_does(self):
        lift = dict(self.lift, success=False, reason='target_followed_stationary_scene_prediction')
        evidence = hold_check.combine(lift, dict(success=None, reason='no_red'))
        self.assertIs(evidence['success'], False)
        self.assertEqual(evidence['decided_by'], 'lift')

    def test_demos_keep_the_turn_verdict(self):
        meta = dict(auto_evidence=hold_check.combine(
            dict(self.lift), dict(success=True, reason='red_turned_with_the_camera')))
        self.assertEqual(demos.judge(meta), (True, 'red_turned_with_the_camera'))
        meta['auto_evidence']['turn_check']['success'] = False
        self.assertFalse(demos.judge(meta)[0])
        meta['human_success'] = True
        self.assertTrue(demos.judge(meta)[0])


class TurnCheckTests(unittest.TestCase):
    def test_turns_inward_and_back(self):
        band = pick.J1_BAND
        self.assertEqual(hold_check.turned([100, 1, 2, 3, 90, 160], (40, 140))[0],
                         100 + hold_check.TURN_DEG)
        self.assertEqual(hold_check.turned([band[1], 1, 2, 3, 90, 160], band)[0],
                         band[1] - hold_check.TURN_DEG)

    def test_photographs_before_and_after_and_returns_to_the_lift(self):
        lift = [88, 20, 50, 40, 90, 167]
        session = mock.Mock()
        session.pose = list(lift)
        sent = []
        session.send.side_effect = lambda pose, ms, settle: sent.append(list(pose))
        moved = (HELD[0] + TABLE_SHIFT_PX,) + HELD[1:]
        session.streams.colour.side_effect = [frame(HELD), frame(moved)]
        evidence, before, after = hold_check.turn_check(session)
        self.assertIs(evidence['success'], False)
        self.assertEqual(len(sent), 2)
        self.assertEqual(abs(sent[0][0] - lift[0]), hold_check.TURN_DEG)
        self.assertEqual(sent[0][1:], lift[1:])
        self.assertEqual(sent[1], lift)
        self.assertIsNotNone(before)


class CarryOnTheArmTests(unittest.TestCase):
    def carry(self, carry_frame):
        lift, _ = pick.low_release_pose(GRASP, clearance_mm=35.0)
        route = relocate.plan(GRASP, lift, random.Random(5))
        session = mock.Mock()
        session.pose = list(lift)
        sent = []

        def send(pose, duration, settle):
            sent.append(list(pose))
            session.pose = list(pose)
        session.send.side_effect = send
        session.streams.colour.return_value = carry_frame
        session.look.side_effect = AssertionError('centroid check used')
        result = relocate.execute(session, route, held_px=(700.0, 690.0),
                                  held_frame=frame(HELD))
        return result, route, sent

    def test_held_red_is_put_down(self):
        result, route, sent = self.carry(frame(HELD))
        self.assertFalse(result['dropped'])
        self.assertIn(route['release'], sent)
        self.assertEqual(result['carry_check']['reason'], 'red_still_in_the_jaws')

    def test_red_gone_is_not_put_down(self):
        result, route, sent = self.carry(EMPTY)
        self.assertTrue(result['dropped'])
        self.assertNotIn(route['release'], sent)


class ShortLiftTests(unittest.TestCase):
    def run_lift(self, lift_verdict, turn_verdict):
        session = mock.Mock()
        session.pose = list(GRASP)
        session.last_px = (707.0, 680.0)
        session.look.return_value = {'cx': 709.0, 'cy': 682.0, 'area': 4000}
        lift_frame = frame(HELD)
        with mock.patch.object(collect, 'assess_short_lift', return_value=dict(lift_verdict)), \
                mock.patch.object(collect, '_audit_image', return_value=None), \
                mock.patch.object(collect, '_audit_frame', return_value='images/base-turn.jpg'), \
                mock.patch.object(collect.hold_check, 'turn_check',
                                  return_value=(dict(turn_verdict), lift_frame, EMPTY)), \
                mock.patch.object(collect.relocate, 'execute', return_value=dict(
                    dropped=False, carry_px=None, image=None, carry_check={})) as execute:
            _, evidence, relocation = collect.short_lift_and_release(
                session, GRASP, {}, mock.Mock(), auto=True, rng=random.Random(2))
        return evidence, relocation, execute, lift_frame

    def test_a_turn_verdict_overrides_an_unsure_lift(self):
        evidence, relocation, execute, lift_frame = self.run_lift(
            dict(method=auto_reward.METHOD, success=None, reason='ambiguous_target_motion'),
            dict(method=hold_check.METHOD, success=True, reason='red_turned_with_the_camera'))
        self.assertIs(evidence['success'], True)
        self.assertEqual(evidence['turn_check']['image'], 'images/base-turn.jpg')
        self.assertIsNone(evidence['short_lift']['success'])
        execute.assert_called_once()
        self.assertIs(execute.call_args.kwargs['held_frame'], lift_frame)
        self.assertIsNotNone(relocation)

    def test_a_turn_that_sees_the_table_is_not_held(self):
        evidence, relocation, execute, _ = self.run_lift(
            dict(method=auto_reward.METHOD, success=True, reason='target_stayed_near_gripper_pixel'),
            dict(method=hold_check.METHOD, success=False, reason='red_stayed_with_the_table'))
        self.assertIs(evidence['success'], False)
        execute.assert_not_called()
        self.assertIsNone(relocation)


if __name__ == '__main__':
    unittest.main()
