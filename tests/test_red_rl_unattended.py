import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import kinematics
import pick_and_place as pick
from pc_client.red_rl import collect, relocate, run_session
from pc_client.red_rl.dataset import (AUTO_LABEL_SOURCE, EpisodeRecorder,
                                      episode_success, load_examples)
from pc_client.red_rl.validate_dataset import summarise_by_rule, validate_episode

# Grasp poses from real red-sweet runs under the mid-height rule and before it.
GRASPS = ([88, 10, 54, 43, 90, 167], [77, 9, 53, 44, 90, 168],
          [106, 10, 56, 38, 90, 166], [88, 11, 55, 35, 90, 163],
          [107, 3, 72, 29, 90, 163])


def polar(pose):
    position = kinematics.gripper_mm(pose)
    return (math.hypot(position[0], position[1]),
            math.degrees(math.atan2(position[1], position[0])), position[:2])


class RelocationPlanTests(unittest.TestCase):
    def test_reach_table_matches_forward_kinematics(self):
        table = relocate._reach_table()
        for a, b, c in ((0, 0, 0), (11, 30, 35), (45, 65, 70)):
            pose = [90, int(relocate.SEARCH_J2[a]), int(relocate.SEARCH_J3[b]),
                    int(relocate.SEARCH_J4[c]), 90, 90]
            local = relocate._ARM1_ROTATION @ table[a, b, c] + relocate._ARM1_ORIGIN_MM
            self.assertLess(np.abs(local - kinematics.gripper_mm(pose)).max(), 1e-9)

    def test_plans_land_in_reach_and_put_the_sweet_down_gently(self):
        rng = random.Random(7)
        for grasp in GRASPS:
            lift, _ = pick.low_release_pose(grasp, clearance_mm=35.0)
            height = kinematics.gripper_height_mm(grasp)
            for _ in range(10):
                route = relocate.plan(grasp, lift, rng)
                self.assertIsNotNone(route, grasp)
                radius, azimuth, xy = polar(route['release'])
                self.assertTrue(relocate.PLACE_RADIUS_MM[0] - 4 <= radius
                                <= relocate.PLACE_RADIUS_MM[1] + 4, radius)
                self.assertTrue(relocate.PLACE_AZIMUTH_DEG[0] - 1.5 <= azimuth
                                <= relocate.PLACE_AZIMUTH_DEG[1] + 1.5, azimuth)
                self.assertGreaterEqual(route['moved_mm'], relocate.MIN_MOVE_MM)
                self.assertAlmostEqual(
                    kinematics.gripper_height_mm(route['release']) - height,
                    relocate.RELEASE_CLEARANCE_MM, delta=1.5)
                self.assertAlmostEqual(
                    kinematics.gripper_height_mm(route['carry']) - height,
                    relocate.CARRY_CLEARANCE_MM, delta=1.5)
                for pose in (route['turn'], route['carry'], route['release']):
                    self.assertEqual(pose[4], 90)
                    self.assertEqual(pose[5], grasp[5])     # still gripping
                    self.assertTrue(pick.J1_BAND[0] <= pose[0] <= pick.J1_BAND[1])
                self.assertEqual(route['clear'][5], relocate.gripper_state.OPEN_CMD)
                self.assertGreaterEqual(
                    relocate.path_clearance(route['turn'], route['carry'], height),
                    relocate.MIN_TRAVEL_CLEARANCE_MM)
                self.assertLessEqual(
                    relocate.lowering_drift_mm(route['carry'], route['release']),
                    relocate.MAX_LOWERING_DRIFT_MM)

    def test_turn_is_a_pure_base_rotation_at_lift_height(self):
        grasp = GRASPS[0]
        lift, _ = pick.low_release_pose(grasp, clearance_mm=35.0)
        route = relocate.plan(grasp, lift, random.Random(3))
        self.assertEqual(route['turn'][1:], lift[1:])
        self.assertAlmostEqual(kinematics.gripper_height_mm(route['turn']),
                               kinematics.gripper_height_mm(lift), delta=0.1)

    def test_same_seed_same_spot(self):
        grasp = GRASPS[1]
        lift, _ = pick.low_release_pose(grasp, clearance_mm=35.0)
        self.assertEqual(relocate.plan(grasp, lift, random.Random(11)),
                         relocate.plan(grasp, lift, random.Random(11)))

    def test_execute_order_keeps_grip_until_lowered(self):
        grasp = GRASPS[0]
        lift, _ = pick.low_release_pose(grasp, clearance_mm=35.0)
        route = relocate.plan(grasp, lift, random.Random(5))
        session = mock.Mock()
        session.pose = list(lift)
        sent = []

        def send(pose, duration, settle):
            sent.append(list(pose))
            session.pose = list(pose)
        session.send.side_effect = send
        snapshot = mock.Mock(return_value='images/relocated.jpg')
        result = relocate.execute(session, route, snapshot)
        self.assertEqual(result['image'], 'images/relocated.jpg')
        self.assertFalse(result['dropped'])
        self.assertEqual(sent[:3], [route['turn'], route['carry'], route['release']])
        self.assertEqual(sent[3][:5], route['release'][:5])
        self.assertEqual(sent[3][5], relocate.gripper_state.OPEN_CMD)
        self.assertEqual(sent[4], route['clear'])
        self.assertEqual(sent[5], list(relocate.arm_control.HOME_POSE))
        self.assertTrue(all(pose[4] == 90 for pose in sent))
        snapshot.assert_called_once_with()

    def carry_session(self, look):
        grasp = GRASPS[0]
        lift, _ = pick.low_release_pose(grasp, clearance_mm=35.0)
        route = relocate.plan(grasp, lift, random.Random(5))
        session = mock.Mock()
        session.pose = list(lift)
        sent = []

        def send(pose, duration, settle):
            sent.append(list(pose))
            session.pose = list(pose)
        session.send.side_effect = send
        session.look.return_value = look
        return session, route, sent

    def test_a_sweet_still_in_the_jaws_is_put_down(self):
        session, route, sent = self.carry_session({'cx': 712.0, 'cy': 655.0})
        result = relocate.execute(session, route, held_px=(705.0, 650.0))
        self.assertFalse(result['dropped'])
        self.assertEqual(result['carry_px'], (712.0, 655.0))
        self.assertIn(route['release'], sent)

    def test_a_dropped_sweet_is_not_put_down(self):
        for look in (None, {'cx': 705.0, 'cy': 650.0 + relocate.CARRY_DRIFT_PX + 40}):
            session, route, sent = self.carry_session(look)
            snapshot = mock.Mock()
            result = relocate.execute(session, route, snapshot, held_px=(705.0, 650.0))
            self.assertTrue(result['dropped'])
            self.assertNotIn(route['release'], sent)       # never lowered
            self.assertEqual(sent[-2][:5], route['carry'][:5])
            self.assertEqual(sent[-2][5], relocate.gripper_state.OPEN_CMD)
            self.assertEqual(sent[-1], list(relocate.arm_control.HOME_POSE))
            snapshot.assert_not_called()


class UnattendedCollectorTests(unittest.TestCase):
    def lift_session(self, grasp):
        session = mock.Mock()
        session.pose = list(grasp)
        session.last_px = (707.0, 680.0)
        session.look.return_value = {'cx': 709.0, 'cy': 682.0, 'area': 4000}
        return session

    def test_auto_never_asks_and_relocates_a_confirmed_hold(self):
        grasp = GRASPS[0]
        session = self.lift_session(grasp)
        held = {'success': True, 'reason': 'target_stayed_near_gripper_pixel'}
        with mock.patch.object(collect, 'assess_short_lift', return_value=dict(held)), \
                mock.patch.object(collect, '_audit_image', return_value=None), \
                mock.patch.object(collect.relocate, 'execute', return_value=dict(
                    dropped=False, carry_px=(709.0, 682.0),
                    image='images/relocated.jpg')) as execute, \
                mock.patch('builtins.input', side_effect=AssertionError('asked')):
            human, evidence, relocation = collect.short_lift_and_release(
                session, grasp, {}, mock.Mock(), auto=True, rng=random.Random(2))
        self.assertIsNone(human)
        self.assertTrue(evidence['success'])
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs['held_px'], (709.0, 682.0))
        self.assertEqual(relocation['image'], 'images/relocated.jpg')
        self.assertFalse(relocation['dropped'])
        self.assertGreaterEqual(relocation['moved_mm'], relocate.MIN_MOVE_MM)

    def test_a_drop_while_carrying_turns_the_label_into_a_failure(self):
        grasp = GRASPS[0]
        session = self.lift_session(grasp)
        held = {'success': True, 'reason': 'target_stayed_near_gripper_pixel'}
        with mock.patch.object(collect, 'assess_short_lift', return_value=dict(held)), \
                mock.patch.object(collect, '_audit_image', return_value=None), \
                mock.patch.object(collect.relocate, 'execute', return_value=dict(
                    dropped=True, carry_px=None, image=None)):
            _, evidence, relocation = collect.short_lift_and_release(
                session, grasp, {}, mock.Mock(), auto=True, rng=random.Random(2))
        self.assertIs(evidence['success'], False)
        self.assertIs(evidence['lift_success'], True)
        self.assertEqual(evidence['reason'], 'dropped_while_carrying')
        self.assertTrue(relocation['dropped'])

    def test_auto_puts_back_in_place_when_not_confirmed(self):
        grasp = GRASPS[0]
        session = self.lift_session(grasp)
        unsure = {'success': None, 'reason': 'ambiguous_target_motion'}
        with mock.patch.object(collect, 'assess_short_lift', return_value=dict(unsure)), \
                mock.patch.object(collect, '_audit_image', return_value=None), \
                mock.patch.object(collect.relocate, 'execute') as execute, \
                mock.patch('builtins.input', side_effect=AssertionError('asked')):
            _, evidence, relocation = collect.short_lift_and_release(
                session, grasp, {}, mock.Mock(), auto=True, rng=random.Random(2))
        execute.assert_not_called()
        self.assertIsNone(relocation)
        self.assertEqual(session.send.call_args.args[0],
                         list(collect.arm_control.HOME_POSE))

    def run_main(self, evidence):
        with mock.patch.object(sys, 'argv', ['collect', '--execute', '--auto',
                                             '--seed', '4']):
            args = collect.arguments()
        session = mock.Mock()
        session.streams.wait.return_value = True
        session.target_geometry_initial = None
        session.object_top_mm = 25.0
        session.width_mm = 25.0
        session.stop_height_mm = 22.5
        session.pose = list(GRASPS[0])
        demo = mock.Mock()
        demo.pending = None
        demo.metadata = {}
        state = {'cx': 700, 'cy': 650}
        with mock.patch.object(collect, 'arguments', return_value=args), \
                mock.patch.object(collect.arm_control, 'confirm_execute',
                                  return_value=True), \
                mock.patch.object(collect, 'TeacherSession', return_value=session), \
                mock.patch.object(collect.pick, 'preflight', return_value=True), \
                mock.patch.object(collect, 'align_teacher', return_value=state), \
                mock.patch.object(collect.pick, 'descend', return_value=state), \
                mock.patch.object(collect.pick, 'creep', return_value=state), \
                mock.patch.object(collect.pick, 'grip', return_value=state), \
                mock.patch.object(collect, 'EpisodeRecorder',
                                  return_value=demo) as recorder, \
                mock.patch.object(collect, 'short_lift_and_release',
                                  return_value=(None, evidence, None)):
            status = collect.main()
        return status, recorder.call_args.args[1], demo

    def test_auto_exit_status_follows_the_camera(self):
        status, meta, demo = self.run_main({'success': True})
        self.assertEqual(status, 0)
        self.assertEqual(meta['label_source'], AUTO_LABEL_SOURCE)
        self.assertEqual(meta['relocation_seed'], 4)
        self.assertEqual(demo.finish.call_args.kwargs['auto_success'], True)
        self.assertEqual(self.run_main({'success': False})[0], collect.EXIT_NOT_HELD)
        self.assertEqual(self.run_main({'success': None})[0], collect.EXIT_NOT_HELD)


class CameraLabelTests(unittest.TestCase):
    def record(self, folder, meta, human, auto, outcome='completed'):
        session = SimpleNamespace(object_top_mm=25.0, stop_height_mm=22.5,
                                  object_aspect=1.0)
        state = {'cx': 700.0, 'cy': 675.0, 'height': 120.0, 'range_mm': 260.0,
                 'area': 9000}
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        recorder = EpisodeRecorder(folder, dict(meta, task='red_sweet_approach',
                                                grasp_rule='mid_height_v1',
                                                grasp_profile='red_fixed_wrist'))
        recorder.before_send(state, [90, 120, 0, 0, 90, 90], [91, 120, 0, 0, 90, 90],
                             'align', session, frame)
        recorder.after_look(state, [91, 120, 0, 0, 90, 90], session, frame)
        evidence = {'method': 'short_lift_motion_v1', 'success': auto}
        recorder.finish(outcome, human, auto_success=auto, auto_evidence=evidence)
        return recorder.path

    def test_person_wins_and_camera_fills_in_only_for_unattended_runs(self):
        auto = {'label_source': AUTO_LABEL_SOURCE}
        self.assertTrue(episode_success(dict(auto, human_success=None, auto_success=True)))
        self.assertFalse(episode_success(dict(auto, human_success=False, auto_success=True)))
        self.assertIsNone(episode_success({'human_success': None, 'auto_success': True}))

    def test_camera_labels_train_and_count_against_every_completed_grasp(self):
        with tempfile.TemporaryDirectory() as folder:
            auto = {'label_source': AUTO_LABEL_SOURCE}
            held = self.record(folder, auto, None, True)
            self.record(folder, auto, None, None)       # camera abstained
            self.record(folder, {}, None, True)         # attended, unlabelled
            self.assertEqual([meta['episode'] for meta, _ in load_examples(folder)],
                             [held.name])
            counts = summarise_by_rule(sorted(Path(folder).glob('*/episode.json')))
            default = counts['mid_height_v1']['default']
            self.assertEqual((default['held'], default['unlabelled'],
                              default['camera_labelled']), (1, 2, 1))
            self.assertEqual(validate_episode(held), [])

    def test_camera_label_needs_its_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.record(folder, {'label_source': AUTO_LABEL_SOURCE}, None, True)
            meta = json.loads((path / 'episode.json').read_text(encoding='utf-8'))
            meta['auto_evidence'] = None
            meta['auto_success'] = None
            (path / 'episode.json').write_text(json.dumps(meta), encoding='utf-8')
            self.assertIn('camera-labelled episode has no lift evidence',
                          validate_episode(path))


class UnattendedSessionTests(unittest.TestCase):
    def fake_trial(self, folder, status, metadata):
        def run(command, cwd, input, text):
            self.assertIn('--auto', command)
            self.assertIn('--execute', command)
            self.assertEqual(input, 'MOVE\n')
            episode = Path(folder) / 'ep'
            episode.mkdir()
            (episode / 'episode.json').write_text(json.dumps(metadata),
                                                  encoding='utf-8')
            return mock.Mock(returncode=status)
        return run

    def test_trial_outcomes(self):
        cases = (
            (0, {'steps': 30, 'relocation': {'moved_mm': 41.0, 'radius_mm': 220.0,
                                             'azimuth_deg': 5.0}}, ('held', True)),
            (collect.EXIT_NOT_HELD, {'outcome': 'completed', 'auto_success': None,
                                     'auto_evidence': {'reason': 'x'}}, ('missed', None)),
            (collect.EXIT_NOT_HELD, {'outcome': 'completed', 'auto_success': False,
                                     'auto_evidence': {'reason': 'x'}}, ('missed', False)),
            (1, {'outcome': 'align_failed'}, ('stop', None)),
        )
        for status, meta, expected in cases:
            with tempfile.TemporaryDirectory() as folder:
                with mock.patch.object(run_session.subprocess, 'run',
                                       side_effect=self.fake_trial(folder, status, meta)):
                    result, name, label = run_session.collect_auto(Path(folder))
            self.assertEqual((result, label), expected)
            self.assertEqual(name, 'ep')

    def run_loop(self, outcomes, count=10, max_misses=2):
        calls = iter(outcomes)
        with mock.patch.object(run_session, 'collect_auto',
                               side_effect=lambda *_: next(calls)) as trial:
            completed = run_session.unattended(Path('.'), count, (), max_misses)
        return completed, trial.call_count

    def test_a_held_trial_resets_the_miss_count(self):
        outcomes = [('missed', 'a', None), ('held', 'b', True),
                    ('missed', 'c', False), ('held', 'd', True)]
        self.assertEqual(self.run_loop(outcomes, count=4), (4, 4))

    def test_two_misses_in_a_row_stop(self):
        outcomes = [('held', 'a', True), ('missed', 'b', None),
                    ('missed', 'c', False), ('held', 'never', True)]
        self.assertEqual(self.run_loop(outcomes), (3, 3))
        self.assertEqual(self.run_loop(outcomes[:2] + outcomes[3:], count=3,
                                       max_misses=1), (2, 2))

    def test_any_other_failure_stops_at_once(self):
        outcomes = [('held', 'a', True), ('stop', None, None), ('held', 'never', True)]
        self.assertEqual(self.run_loop(outcomes), (1, 2))


if __name__ == '__main__':
    unittest.main()
