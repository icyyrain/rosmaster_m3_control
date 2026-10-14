import sys
import unittest
import math
from unittest import mock

from pc_client.red_rl import collect


class CollectArgumentsTests(unittest.TestCase):
    def test_compensation_keeps_wrist_rotation_disabled(self):
        with mock.patch.object(sys, 'argv', ['collect', '--centre-compensation',
                                             '--grasp-offset-mm', '5',
                                             '--creep-controller', 'adaptive',
                                             '--pregrasp-retract-j4-deg', '1']):
            args = collect.arguments()
        self.assertTrue(args.centre_compensation)
        self.assertTrue(args.no_orient_gripper)
        self.assertEqual(args.grasp_offset_mm, 5.0)
        self.assertEqual(args.creep_controller, 'adaptive')
        self.assertEqual(args.pregrasp_retract_j4_deg, 1)

    def test_default_profile_is_unchanged(self):
        with mock.patch.object(sys, 'argv', ['collect']):
            args = collect.arguments()
        self.assertFalse(args.centre_compensation)
        self.assertIsNone(args.grasp_offset_mm)
        self.assertEqual(args.creep_controller, 'auto')
        self.assertEqual(args.pregrasp_retract_j4_deg, 0)

    def test_zero_grasp_offset_is_explicit_experiment_only(self):
        with mock.patch.object(sys, 'argv', ['collect', '--grasp-offset-mm', '0']):
            args = collect.arguments()
        self.assertEqual(args.grasp_offset_mm, 0.0)

    def test_two_degree_pregrasp_move_is_baseward(self):
        session = mock.Mock()
        session.pose = [82, 13, 57, 34, 90, 90]
        session.last_px = (707.0, 666.0)
        session.look.return_value = {'cx': 707.0, 'cy': 652.0}
        state = {'cx': 707.0, 'cy': 666.0}

        self.assertEqual(collect.retract_pregrasp(session, state, 2),
                         session.look.return_value)
        sent_pose = session.send.call_args.args[0]
        self.assertEqual(sent_pose, [82, 13, 57, 32, 90, 90])

    def test_base_down_pregrasp_move_guards_both_axes(self):
        session = mock.Mock()
        session.pose = [84, 11, 57, 37, 90, 90]
        session.last_px = (707.0, 666.0)
        session.stop_height_mm = 26.7
        session.object_top_mm = 24.7
        session.look.return_value = {'cx': 707.0, 'cy': 630.0}
        state = {'cx': 707.0, 'cy': 666.0}

        self.assertIs(collect.offset_pregrasp(session, state, 8.0, 2.0),
                      session.look.return_value)
        after = session.send.call_args.args[0]
        first = collect.kinematics.gripper_mm(session.pose)
        last = collect.kinematics.gripper_mm(after)
        import math
        retreat = math.hypot(first[0], first[1]) - math.hypot(last[0], last[1])
        self.assertAlmostEqual(retreat, 8.0, delta=1.25)
        self.assertAlmostEqual(first[2] - last[2], 2.0, delta=0.4)

    def test_base_down_pregrasp_refuses_floor_violation(self):
        session = mock.Mock()
        session.pose = [84, 11, 57, 37, 90, 90]
        session.stop_height_mm = 18.0
        session.object_top_mm = 16.0
        with self.assertRaisesRegex(RuntimeError, 'No guarded'):
            collect.offset_pregrasp(session, {}, 8.0, 2.0)
        session.send.assert_not_called()

    def test_base_down_quantized_fallback_for_aborted_real_pose(self):
        session = mock.Mock()
        session.pose = [95, 11, 56, 42, 90, 90]
        session.last_px = (700.0, 650.0)
        session.stop_height_mm = 29.6655
        session.object_top_mm = 27.6655
        session.look.return_value = {'cx': 700.0, 'cy': 650.0}

        collect.offset_pregrasp(session, session.look.return_value, 8.0, 2.0)
        after = session.send.call_args.args[0]
        before_fk = collect.kinematics.gripper_mm(session.pose)
        after_fk = collect.kinematics.gripper_mm(after)
        retreat = (math.hypot(before_fk[0], before_fk[1])
                   - math.hypot(after_fk[0], after_fk[1]))
        descent = before_fk[2] - after_fk[2]
        self.assertAlmostEqual(retreat, 8.0, delta=1.25)
        self.assertGreaterEqual(descent, 1.25)
        self.assertLessEqual(descent, 2.4)
        self.assertGreaterEqual(session.stop_height_mm - descent,
                                session.object_top_mm - 0.5)

    def test_lower_grasp_offset_keeps_object_and_table_guards(self):
        session = mock.Mock()
        session.pose = [95, 11, 57, 43, 90, 90]
        session.last_px = (700.0, 650.0)
        session.stop_height_mm = 27.5
        session.object_top_mm = 27.5
        session.look.return_value = {'cx': 700.0, 'cy': 650.0}

        collect.offset_pregrasp(session, session.look.return_value, 8.0, 2.0)
        after = session.send.call_args.args[0]
        before_fk = collect.kinematics.gripper_mm(session.pose)
        after_fk = collect.kinematics.gripper_mm(after)
        descent = before_fk[2] - after_fk[2]
        self.assertGreaterEqual(session.stop_height_mm - descent,
                                session.object_top_mm - 3.0)
        self.assertGreaterEqual(session.stop_height_mm - descent,
                                collect.pick.HEIGHT_FLOOR_MM)

    def test_unreachable_pregrasp_returns_home_without_closing_gripper(self):
        with mock.patch.object(sys, 'argv', ['collect', '--execute',
                                             '--pregrasp-base-mm', '8',
                                             '--pregrasp-down-mm', '2']):
            args = collect.arguments()
        session = mock.Mock()
        session.streams.wait.return_value = True
        session.target_geometry_initial = None
        session.object_top_mm = 27.0
        session.width_mm = 28.0
        session.stop_height_mm = 29.0
        session.pose = [95, 11, 56, 42, 90, 90]
        demo = mock.Mock()
        demo.pending = None
        demo.path = 'test-episode'
        state = {'cx': 700, 'cy': 650}
        with (mock.patch.object(collect, 'arguments', return_value=args),
              mock.patch.object(collect.arm_control, 'confirm_execute',
                                return_value=True),
              mock.patch.object(collect, 'TeacherSession', return_value=session),
              mock.patch.object(collect.pick, 'preflight', return_value=True),
              mock.patch.object(collect, 'align_teacher', return_value=state),
              mock.patch.object(collect.pick, 'descend', return_value=state),
              mock.patch.object(collect.pick, 'creep', return_value=state),
              mock.patch.object(collect, 'EpisodeRecorder',
                                return_value=demo) as recorder,
              mock.patch.object(collect, 'offset_pregrasp', side_effect=
                                collect.NoGuardedPregraspMove('No guarded')),
              mock.patch.object(collect.pick, 'grip') as grip):
            self.assertEqual(collect.main(), 1)
        self.assertEqual(session.send.call_args.args[0],
                         list(collect.arm_control.HOME_POSE))
        grip.assert_not_called()
        self.assertEqual(demo.finish.call_args.args[0],
                         'pregrasp_offset_unreachable')
        self.assertEqual(recorder.call_args.args[1]['grasp_rule'],
                         collect.CURRENT_GRASP_RULE)


if __name__ == '__main__':
    unittest.main()
