"""Collect a red-sweet approach demonstration with the proven rule controller.

Run as ``python -m pc_client.red_rl.collect`` from the repository root. Only
align/descend/creep actions are learning data. Wrist orientation, closing,
short verification lift and release remain deterministic.

With ``--auto`` nobody is asked for a label: the camera's short-lift verdict
is recorded as the label, a held sweet is put down gently at a random spot
the next trial can reach, and anything but a confirmed hold exits with
EXIT_NOT_HELD so an unattended session stops.
"""

import argparse
import math
import random
import time
from types import SimpleNamespace

import arm_control
import gripper_state
import kinematics
import reach_candy
import pick_and_place as pick
from pc_client.red_rl import hold_check, relocate
from pc_client.red_rl.dataset import (AUTO_LABEL_SOURCE, CURRENT_GRASP_RULE,
                                      HUMAN_LABEL_SOURCE, EpisodeRecorder, features)
from pc_client.red_rl.auto_reward import assess_short_lift
from pc_client.red_rl.shadow import ShadowPolicy, episode_report

# Exit status of an --auto trial that completed but did not confirm a hold:
# the camera said NOT HELD or could not tell. The session stops on it.
EXIT_NOT_HELD = 3


# At high HOME height the integer-degree arm can stop improving around 26 px.
# The descent controller continues to re-centre on every waypoint. Keeping
# this local avoids changing the proven standalone pick script.
TEACHER_ALIGN_TOLERANCE_PX = 30.0
# Integer servo targets cannot always realize a 2 mm descent at the end of a
# small radial correction. Accept up to 0.75 mm *less* descent, never more than
# 0.4 mm extra, while keeping the existing floor and swept-path guards.
PREGRASP_DOWN_SHORTFALL_MM = 0.75
PREGRASP_DOWN_OVERSHOOT_MM = 0.4
# The closed jaws must overlap the object body, not just touch its top. This
# experimental allowance is still bounded by the absolute table-height floor.
PREGRASP_MAX_BELOW_OBJECT_TOP_MM = 3.0


class NoGuardedPregraspMove(RuntimeError):
    """The requested correction cannot be made within the motion guards."""


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
    parser.add_argument('--centre-compensation', action='store_true',
                        help='Compensate for a clipped target centre during approach; '
                             'joint 5 still stays fixed at 90 degrees.')
    parser.add_argument('--grasp-offset-mm', type=float,
                        help='Experimental gripper height relative to measured object top '
                             '(0-10 mm; 0 is level with the top before correction). '
                             'Default keeps the existing automatic rule.')
    parser.add_argument('--creep-controller', choices=('auto', 'legacy', 'adaptive'),
                        default='auto', help='Experimental final-approach controller.')
    parser.add_argument('--pregrasp-retract-j4-deg', type=int, choices=(0, 1, 2),
                        default=0, help='Experimentally move joint 4 back by 0-2 '
                                        'degrees just before closing (about 3 mm '
                                        'toward the base per degree).')
    parser.add_argument('--pregrasp-base-mm', type=float, default=0.0,
                        help='Experimental baseward gripper shift before closing (0-10 mm).')
    parser.add_argument('--pregrasp-down-mm', type=float, default=0.0,
                        help='Experimental downward gripper shift before closing (0-2.3 mm).')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move the arm; otherwise only run camera preflight.')
    parser.add_argument('--shadow-model', metavar='PATH',
                        help='Evaluate this model alongside teacher actions; '
                             'model output is NEVER sent to the arm.')
    parser.add_argument('--auto', action='store_true',
                        help='Unattended: record the camera verdict as the label, put '
                             'a held sweet down at a random reachable spot, and exit '
                             'with status {} unless the hold was confirmed.'.format(
                                 EXIT_NOT_HELD))
    parser.add_argument('--seed', type=int,
                        help='Seed for the --auto placement; default from the clock. '
                             'Recorded in the episode either way.')
    cli = parser.parse_args()
    if cli.grasp_offset_mm is not None and not 0.0 <= cli.grasp_offset_mm <= 10.0:
        parser.error('--grasp-offset-mm must be between 0 and 10')
    if not 0.0 <= cli.pregrasp_base_mm <= 10.0:
        parser.error('--pregrasp-base-mm must be between 0 and 10')
    if not 0.0 <= cli.pregrasp_down_mm <= 2.3:
        parser.error('--pregrasp-down-mm must be between 0 and 2.3')
    if bool(cli.pregrasp_base_mm) != bool(cli.pregrasp_down_mm):
        parser.error('set both --pregrasp-base-mm and --pregrasp-down-mm')
    if cli.pregrasp_retract_j4_deg and cli.pregrasp_base_mm:
        parser.error('choose either the joint4 correction or the base/down correction')
    return SimpleNamespace(
        host=cli.host, port=cli.port, detector='table', colour='red-sat',
        a_star=140, min_area=150, width_mm=28.0, auto_width=True,
        descent_steps=10 if cli.fast else 20,
        settle=0.6 if cli.fast else 1.0, fast=cli.fast,
        creep_controller=cli.creep_controller, no_orient_gripper=True,
        skip_preflight=False, grasp_offset_mm=cli.grasp_offset_mm, squeeze_mm=5.0,
        keep_holding=False, record=None, execute=cli.execute,
        dataset=cli.dataset,
        shadow_model=cli.shadow_model,
        centre_compensation=cli.centre_compensation,
        pregrasp_retract_j4_deg=cli.pregrasp_retract_j4_deg,
        pregrasp_base_mm=cli.pregrasp_base_mm,
        pregrasp_down_mm=cli.pregrasp_down_mm,
        auto=cli.auto,
        seed=cli.seed if cli.seed is not None else int(time.time() * 1000) % 2 ** 32,
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
    return _audit_frame(demo, session.streams.colour(), name)


def _audit_frame(demo, frame, name):
    try:
        return demo._image(frame, name)
    except (OSError, ValueError) as error:
        print('  audit snapshot unavailable: {}'.format(error))
        return None


def retract_pregrasp(session, state, degrees):
    """A small baseward correction at grasp height, with a height-drop guard."""
    if not degrees:
        return state
    before = list(session.pose)
    after = list(before)
    after[3] -= degrees
    if after[3] < pick.J4_BAND[0]:
        raise RuntimeError('Pregrasp retract would exceed the joint 4 safety band')
    first = kinematics.gripper_mm(before)
    second = kinematics.gripper_mm(after)
    first_radius = (first[0] ** 2 + first[1] ** 2) ** 0.5
    second_radius = (second[0] ** 2 + second[1] ** 2) ** 0.5
    retreat_mm = first_radius - second_radius
    height_drop_mm = first[2] - second[2]
    if not 0.5 <= retreat_mm <= 7.0 or height_drop_mm > 1.5:
        raise RuntimeError('Pregrasp correction is not a small baseward move')
    print('=== pregrasp baseward correction ===')
    print('  joint4 -{} deg: predicted {:.1f} mm toward base, {:.1f} mm lower'.format(
        degrees, retreat_mm, height_drop_mm))
    predicted = reach_candy.predict_target(
        session.last_px, (0, 0, 0, -degrees))
    session.send(after, 700, 1.4)
    fresh = session.look(predicted)
    if fresh is None:
        session.send(before, 700, 1.4)
        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
        raise RuntimeError('Target lost after pregrasp correction; do not close')
    return fresh


def offset_pregrasp(session, state, base_mm, down_mm):
    """Small FK-guarded radial/downward move, decoupling position from height."""
    if not base_mm:
        return state
    before = list(session.pose)
    origin = kinematics.gripper_mm(before)
    radius = math.hypot(origin[0], origin[1])
    options = []
    for delta2 in range(-3, 4):
        for delta3 in range(-3, 4):
            for delta4 in range(-5, 1):
                if delta2 == delta3 == delta4 == 0:
                    continue
                after = list(before)
                after[1] += delta2
                after[2] += delta3
                after[3] += delta4
                if not (0 <= after[1] <= 180 and 0 <= after[2] <= 180
                        and pick.J4_BAND[0] <= after[3] <= pick.J4_BAND[1]):
                    continue
                end = kinematics.gripper_mm(after)
                retreat = radius - math.hypot(end[0], end[1])
                descent = origin[2] - end[2]
                if not (abs(retreat - base_mm) <= 1.25
                        and down_mm - PREGRASP_DOWN_SHORTFALL_MM <= descent
                        <= down_mm + PREGRASP_DOWN_OVERSHOOT_MM
                        and 0.0 <= descent <= 2.5):
                    continue
                if (session.stop_height_mm - descent
                        < max(pick.HEIGHT_FLOOR_MM,
                              session.object_top_mm
                              - PREGRASP_MAX_BELOW_OBJECT_TOP_MM)):
                    continue
                # Servo motion interpolates joint commands; check the path, too.
                safe = True
                for fraction in (0.25, 0.5, 0.75):
                    intermediate = list(before)
                    for joint in (1, 2, 3):
                        intermediate[joint] += (after[joint] - before[joint]) * fraction
                    z = kinematics.gripper_mm(intermediate)[2]
                    if origin[2] - z > 2.5:
                        safe = False
                        break
                if not safe:
                    continue
                cost = (retreat - base_mm) ** 2 + 3 * (descent - down_mm) ** 2
                cost += 0.005 * (delta2 ** 2 + delta3 ** 2 + delta4 ** 2)
                options.append((cost, after, retreat, descent))
    if not options:
        raise NoGuardedPregraspMove(
            'No guarded base/down pregrasp move for this pose')
    _, after, retreat, descent = min(options, key=lambda option: option[0])
    print('=== pregrasp base/down correction ===')
    print('  {} -> {}: predicted {:.1f} mm toward base, {:.1f} mm lower'.format(
        before[:4], after[:4], retreat, descent))
    if down_mm - descent > PREGRASP_DOWN_OVERSHOOT_MM:
        print('  quantized servo fallback: requested {:.1f} mm lower, '
              'using {:.1f} mm (no downward overshoot)'.format(down_mm, descent))
    delta = tuple(after[joint] - before[joint] for joint in range(4))
    predicted = reach_candy.predict_target(session.last_px, delta)
    session.send(after, 800, 1.6)
    fresh = session.look(predicted)
    if fresh is None:
        session.send(before, 800, 1.6)
        session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                     arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
        raise RuntimeError('Target lost after base/down correction; do not close')
    return fresh


def short_lift_and_release(session, grasp_pose, state, demo, auto=False, rng=None):
    """Fixed post-policy routine; never part of the learned action target.

    Returns (human_success, auto_evidence, relocation). With `auto` there is
    no question: a confirmed hold is carried to a random reachable spot and
    `relocation` describes where; anything else is put down where it was.
    """
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
    lift_evidence = assess_short_lift(previous, after_px, predicted, actual_rise)
    lift_evidence['image_before'] = before_image
    lift_evidence['image_after'] = after_image
    print('  short lift: {} ({})'.format(
        {True: 'held', False: 'not held', None: 'cannot tell'}[lift_evidence['success']],
        lift_evidence['reason']))
    turn, lift_frame, turned_frame = hold_check.turn_check(session)
    turn['image'] = _audit_frame(demo, turned_frame, 'base-turn.jpg')
    print('  base turn {} deg: {} ({}; red shift {} px, overlap {})'.format(
        hold_check.TURN_DEG,
        {True: 'held', False: 'not held', None: 'cannot tell'}[turn['success']],
        turn['reason'], turn.get('shift_px'), turn.get('overlap')))
    auto_evidence = hold_check.combine(lift_evidence, turn)
    human_success = None if auto else ask_success()
    auto_label = {True: 'HELD', False: 'NOT HELD', None: 'INCONCLUSIVE'}[
        auto_evidence['success']]
    print('  automatic camera {}: {} ({})'.format(
        'label' if auto else 'suggestion', auto_label, auto_evidence['reason']))
    if human_success is not None and auto_evidence['success'] is not None:
        print('  comparison with your label: {}'.format(
            'agrees' if human_success == auto_evidence['success'] else 'DISAGREES'))

    if auto and auto_evidence['success'] is True:
        route = relocate.plan(grasp_pose, lift, rng)
        if route is not None:
            result = relocate.execute(
                session, route,
                snapshot=lambda: _audit_image(session, demo, 'relocated.jpg'),
                held_px=after_px, held_frame=lift_frame,
                audit=lambda frame, name: _audit_frame(demo, frame, name))
            relocation = {key: route[key] for key in (
                'radius_mm', 'azimuth_deg', 'xy_mm', 'moved_mm',
                'turn', 'carry', 'release', 'clear')}
            relocation.update(image=result['image'], dropped=result['dropped'],
                              carry_px=(list(result['carry_px'])
                                        if result['carry_px'] is not None else None),
                              carry_check=result.get('carry_check'))
            if result['dropped']:
                # Held on the lift, gone by the carry pose: a grasp that does
                # not survive being carried is not a success.
                auto_evidence['lift_success'] = auto_evidence['success']
                auto_evidence['success'] = False
                auto_evidence['reason'] = 'dropped_while_carrying'
                print('  automatic camera label: NOT HELD (dropped while carrying)')
            return human_success, auto_evidence, relocation
        print('  no checked route to a new spot; putting it back where it was')

    print('=== fixed low release ===')
    release, clearance = pick.low_release_pose(
        grasp_pose, clearance_mm=relocate.RELEASE_CLEARANCE_MM)
    print('  lower to {:.1f} mm above grasp height'.format(clearance))
    session.send(release, 1100, 1.8)
    opened = list(session.pose)
    opened[5] = gripper_state.OPEN_CMD
    session.send(opened, 1200, 2.5)
    session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                 arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
    return human_success, auto_evidence, None


def main():
    args = arguments()
    shadow = ShadowPolicy(args.shadow_model) if args.shadow_model else None
    if args.execute:
        print('The REAL arm will return HOME, approach the red sweet, close, '
              'lift, and release near the table{}.'.format(
                  ' at a random reachable spot' if args.auto else ''))
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
        geometry = session.target_geometry_initial
        long_axis_ratio = (geometry['long_px'] / geometry['short_px']
                           if geometry is not None and geometry['short_px'] > 1.0
                           else None)
        compensation_ready = (long_axis_ratio is not None
                              and long_axis_ratio >= 1.5)
        if args.centre_compensation:
            print('Clipped-centre compensation: {} (preflight colour long axis {}).'.format(
                'ready' if compensation_ready else 'DISABLED: direction unreliable',
                '{:.2f}:1'.format(long_axis_ratio) if long_axis_ratio else 'not found'))
        if not args.execute:
            print('Dry run passed. Add --execute to collect a demonstration.')
            return 0
        # Centre compensation is a separate experiment from wrist rotation.
        # The red-sweet collector always keeps joint 5 at HOME.
        session.lock_clipped_target = args.centre_compensation and compensation_ready
        print('Red-sweet grip profile: fixed joint5, clipped-centre '
              'compensation {}.'.format('on' if args.centre_compensation else 'off'))
        demo = EpisodeRecorder(args.dataset, {
            'source': 'pick_and_place rule teacher', 'task': 'red_sweet_approach',
            'grasp_rule': CURRENT_GRASP_RULE,
            'label_source': AUTO_LABEL_SOURCE if args.auto else HUMAN_LABEL_SOURCE,
            'relocation_seed': args.seed if args.auto else None,
            'grasp_profile': ('red_fixed_wrist_clipped_compensation'
                              if session.lock_clipped_target else
                              'red_fixed_wrist_no_clipped_compensation'),
            'host': args.host, 'fast': args.fast,
            'centre_compensation': args.centre_compensation,
            'centre_compensation_active': session.lock_clipped_target,
            'grasp_offset_override_mm': args.grasp_offset_mm,
            'creep_controller': args.creep_controller,
            'pregrasp_retract_j4_deg': args.pregrasp_retract_j4_deg,
            'pregrasp_base_mm': args.pregrasp_base_mm,
            'pregrasp_down_mm': args.pregrasp_down_mm,
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
        state = retract_pregrasp(session, state, args.pregrasp_retract_j4_deg)
        pregrasp_pose = list(session.pose)
        try:
            state = offset_pregrasp(session, state, args.pregrasp_base_mm,
                                   args.pregrasp_down_mm)
        except NoGuardedPregraspMove as error:
            outcome = 'pregrasp_offset_unreachable'
            print('{}; returning HOME with jaws open.'.format(error))
            session.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS,
                         arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
            return 1
        corrected_pose = list(session.pose)
        start_fk = kinematics.gripper_mm(pregrasp_pose)
        end_fk = kinematics.gripper_mm(corrected_pose)
        demo.metadata.update(
            pregrasp_pose_before=pregrasp_pose,
            pregrasp_pose_after=corrected_pose,
            pregrasp_baseward_fk_mm=(
                math.hypot(start_fk[0], start_fk[1])
                - math.hypot(end_fk[0], end_fk[1])),
            pregrasp_down_fk_mm=start_fk[2] - end_fk[2])
        state = pick.grip(session, state)
        grasp_pose = list(session.pose)
        human_success, auto_evidence, relocation = short_lift_and_release(
            session, grasp_pose, state, demo, auto=args.auto,
            rng=random.Random(args.seed))
        demo.metadata['relocation'] = relocation
        outcome = 'completed'
        if args.auto and auto_evidence['success'] is not True:
            return EXIT_NOT_HELD
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
