"""Collect a red-sweet approach demonstration with the proven rule controller.

Run as ``python -m pc_client.red_rl.collect`` from the repository root. Only
align/descend/creep actions are learning data. Wrist orientation, closing,
short verification lift and release remain deterministic.
"""

import argparse
from types import SimpleNamespace

import arm_control
import gripper_state
import reach_candy
import pick_and_place as pick
from pc_client.red_rl.dataset import EpisodeRecorder, features
from pc_client.red_rl.auto_reward import assess_short_lift
from pc_client.red_rl.shadow import ShadowPolicy, episode_report


# At high HOME height the integer-degree arm can stop improving around 26 px.
# The descent controller continues to re-centre on every waypoint. Keeping
# this local avoids changing the proven standalone pick script.
TEACHER_ALIGN_TOLERANCE_PX = 30.0


def align_teacher(session):
    original = pick.ALIGN_TOLERANCE_PX
    try:
        pick.ALIGN_TOLERANCE_PX = TEACHER_ALIGN_TOLERANCE_PX
        return pick.align(session)
    finally:
        pick.ALIGN_TOLERANCE_PX = original


class TeacherSession(pick.Session):
    def __init__(self, args):
        super().__init__(args)
        self.demo = None
        self.phase = None
        self.last_state = None
        self.shadow = None

    def send(self, pose, duration, settle):
        target = arm_control.clamp_pose(pose)
        if target[4] != arm_control.HOME_POSE[4]:
            raise RuntimeError('Red-sweet collector refuses wrist rotation: '
                               'joint5 {} instead of {}'.format(
                                   target[4], arm_control.HOME_POSE[4]))
        if self.demo is not None and self.phase is not None:
            self.demo.before_send(self.last_state, self.pose, target, self.phase,
                                  self, self.streams.colour())
            self.demo.pending['duration_ms'] = duration
            if self.shadow is not None:
                try:
                    observation = features(self.last_state, self.pose, self,
                                           self.phase)
                    proposal = self.shadow.action(observation)
                    teacher = [target[i] - self.pose[i] for i in range(4)]
                    self.demo.pending['shadow_action_joint_delta_deg'] = proposal
                    self.demo.pending['shadow_abs_error_deg'] = [
                        abs(proposal[i] - teacher[i]) for i in range(4)]
                except Exception as error:
                    # Shadow evaluation must never veto or replace a teacher
                    # command after the physical sequence has started.
                    self.demo.pending['shadow_error'] = str(error)
        super().send(target, duration, settle)

    def look(self, predicted=None):
        state = super().look(predicted)
        self.last_state = state
        if self.demo is not None and self.demo.pending is not None:
            self.demo.after_look(state, self.pose, self, self.streams.colour())
        return state


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--dataset', default='datasets/red_sweet')
    parser.add_argument('--fast', action='store_true',
                        help='Use the same 10-waypoint fast descent as the rule script.')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move the arm; otherwise only run camera preflight.')
    parser.add_argument('--shadow-model', metavar='PATH',
                        help='Evaluate this model alongside teacher actions; '
                             'model output is NEVER sent to the arm.')
    cli = parser.parse_args()
    return SimpleNamespace(
        host=cli.host, port=cli.port, detector='table', colour='red-sat',
        a_star=140, min_area=150, width_mm=28.0, auto_width=True,
        descent_steps=10 if cli.fast else 20,
        settle=0.6 if cli.fast else 1.0, fast=cli.fast,
        creep_controller='auto', no_orient_gripper=True,
        skip_preflight=False, grasp_offset_mm=None, squeeze_mm=5.0,
        keep_holding=False, record=None, execute=cli.execute,
        dataset=cli.dataset,
        shadow_model=cli.shadow_model,
    )


def ask_success():
    while True:
        try:
            answer = input('Is the RED SWEET held after the short lift? '
                           '[y]es / [n]o / [u]nknown: ').strip().lower()
        except EOFError:
            return None
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        if answer in ('u', 'unknown', ''):
            return None


def _audit_image(session, demo, name):
    """Best-effort still frame; failure must not strand an arm holding an object."""
    try:
        return demo._image(session.streams.colour(), name)
    except (OSError, ValueError) as error:
        print('  audit snapshot unavailable: {}'.format(error))
        return None


def short_lift_and_release(session, grasp_pose, state, demo):
    """Fixed post-policy routine; never part of the learned action target."""
    lift, actual_rise = pick.low_release_pose(grasp_pose, clearance_mm=35.0)
    if actual_rise < 20.0 or actual_rise > 50.0:
        raise RuntimeError('No safe short lift on the known path: {:.1f} mm'.format(
            actual_rise))
    print('=== fixed short lift ===')
    print('  lift {:.1f} mm to {}'.format(actual_rise, lift[:4]))
    previous = session.last_px
    before_image = _audit_image(session, demo, 'grasp-closed.jpg')
    delta = tuple(lift[i] - session.pose[i] for i in range(4))
    predicted = (reach_candy.predict_target(previous, delta)
                 if previous is not None else None)
    session.send(lift, 1100, 2.0)
    fresh = session.look(predicted)
    if fresh is None:
        print('  camera cannot verify the held target at this height')
    else:
        print('  camera target {:.0f},{:.0f}, area {}'.format(
            fresh['cx'], fresh['cy'], fresh['area']))
    after_image = _audit_image(session, demo, 'short-lift.jpg')
    after_px = (fresh['cx'], fresh['cy']) if fresh is not None else None
    auto_evidence = assess_short_lift(previous, after_px, predicted, actual_rise)
    auto_evidence['image_before'] = before_image
    auto_evidence['image_after'] = after_image
    human_success = ask_success()
    auto_label = {True: 'HELD', False: 'NOT HELD', None: 'INCONCLUSIVE'}[
        auto_evidence['success']]
    print('  automatic camera suggestion: {} ({})'.format(
        auto_label, auto_evidence['reason']))
    if human_success is not None and auto_evidence['success'] is not None:
        print('  comparison with your label: {}'.format(
            'agrees' if human_success == auto_evidence['success'] else 'DISAGREES'))

    print('=== fixed low release ===')
    release, clearance = pick.low_release_pose(grasp_pose, clearance_mm=10.0)
    print('  lower to {:.1f} mm above grasp height'.format(clearance))
    session.send(release, 1100, 1.8)
    opened = list(session.pose)
    opened[5] = gripper_state.OPEN_CMD
    session.send(opened, 1200, 2.5)
    session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                 arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
    return human_success, auto_evidence


def main():
    args = arguments()
    shadow = ShadowPolicy(args.shadow_model) if args.shadow_model else None
    if args.execute:
        print('The REAL arm will return HOME, approach the red sweet, close, '
              'lift, and release near the table.')
        print('Clear the work area and be ready to cut power.')
        if not arm_control.confirm_execute():
            raise SystemExit('Cancelled.')
    session = TeacherSession(args)
    session.shadow = shadow
    demo = None
    outcome = 'interrupted'
    human_success = None
    auto_evidence = None
    try:
        if not session.streams.wait():
            raise RuntimeError('No camera frames; check the camera/rosbridge nodes')
        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.5 if args.execute else 0.1)
        if not pick.preflight(session):
            print('Preflight failed; no approach will run.')
            return 1
        if not args.execute:
            print('Dry run passed. Add --execute to collect a demonstration.')
            return 0
        # This collector is RED SWEET ONLY. The generic pick script's
        # elongated-target wrist rotation and clipped-centre compensation
        # were introduced for the blue biscuit. Wrapping folds can make the
        # sweet's depth mask exceed the 1.8:1 threshold and accidentally turn
        # both features on. Keep the proven sweet grip geometry instead.
        session.lock_clipped_target = False
        print('Red-sweet grip profile: fixed joint5, no elongated-target '
              'centre compensation.')
        demo = EpisodeRecorder(args.dataset, {
            'source': 'pick_and_place rule teacher', 'task': 'red_sweet_approach',
            'grasp_profile': 'red_fixed_wrist_no_clipped_compensation',
            'host': args.host, 'fast': args.fast,
            'object_top_mm': session.object_top_mm,
            'object_width_mm': session.width_mm,
            'stop_height_mm': session.stop_height_mm,
            'pose_is_commanded_not_measured': True,
            'shadow_model': str(shadow.path) if shadow is not None else None,
            'shadow_training_episodes': (
                shadow.training_episodes if shadow is not None else None),
        })
        session.demo = demo
        print('Recording episode at {}'.format(demo.path))
        session.phase = 'align'
        state = align_teacher(session)
        if state is None:
            outcome = 'align_failed'
            session.phase = None
            print('High-level alignment failed; returning HOME with jaws open.')
            session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                         arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
            return 1
        session.phase = None
        session.phase = 'descend'
        state = pick.descend(session, state)
        if state is None:
            outcome = 'descent_failed'
            return 1
        session.phase = 'creep'
        state = pick.creep(session, state)
        if state is None:
            outcome = 'creep_failed'
            return 1
        session.phase = None
        state = pick.grip(session, state)
        grasp_pose = list(session.pose)
        human_success, auto_evidence = short_lift_and_release(
            session, grasp_pose, state, demo)
        outcome = 'completed'
        return 0
    finally:
        if demo is not None:
            session.phase = None
            if demo.pending is not None:
                demo.after_look(None, session.pose, session, None)
            demo.finish(outcome, human_success,
                        auto_success=(auto_evidence['success']
                                      if auto_evidence is not None else None),
                        auto_evidence=auto_evidence)
            print('Saved {} approach steps to {}'.format(demo.count, demo.path))
            if shadow is not None:
                print('Read-only shadow report: {}'.format(episode_report(demo.path)))
        session.close()


if __name__ == '__main__':
    raise SystemExit(main())
