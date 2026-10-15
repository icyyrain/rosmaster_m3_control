"""Operator-guided red-sweet data collection on a Windows PC.

Starts only missing ROS services over SSH, runs the existing rule-teacher
collector once per operator-approved placement, and stops services it started.
No student-policy action is ever sent to the robot.

With --auto the operator places the sweet once and confirms once. Each trial
then labels itself from the camera and puts the sweet down at a random
reachable spot for the next one. A trial the camera does not confirm leaves
the sweet where it was and is retried; the session stops after --max-misses
such trials in a row, and at once on a preflight, tracking or other error.
"""

import argparse
from collections import deque
import getpass
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import paramiko

import arm_control
import reach_candy
from pc_client.red_rl.auto_reward import compare_labels
from pc_client.red_rl.collect import EXIT_NOT_HELD


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_COMMAND = (
    'bash -lc "source /opt/ros/humble/setup.bash && '
    'source /home/jetson/yahboomcar_ws/install/setup.bash && '
    'source /home/jetson/m3pro_ext_ws/install/setup.bash && '
    'export ROS_DOMAIN_ID=30 && '
    'exec ros2 launch m3pro_arm_bringup rosbridge.launch.py"'
)
CAMERA_COMMAND = (
    'bash -lc "source /opt/ros/humble/setup.bash && '
    'source /home/jetson/yahboomcar_ws/install/setup.bash && '
    'export ROS_DOMAIN_ID=30 && '
    'exec ros2 launch orbbec_camera dabai_dcw2.launch.py"'
)


class RemoteLaunch:
    """Keep a launch process tied to an SSH PTY and drain its log output."""

    def __init__(self, ssh, name, command):
        self.name = name
        self.log = deque(maxlen=30)
        self.channel = ssh.get_transport().open_session()
        self.channel.get_pty(term='dumb', width=120, height=30)
        self.channel.exec_command(command)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        pending = ''
        while not self.channel.closed:
            if self.channel.recv_ready():
                pending += self.channel.recv(8192).decode('utf-8', 'replace')
                *lines, pending = pending.split('\n')
                self.log.extend(line.strip() for line in lines if line.strip())
            elif self.channel.exit_status_ready():
                break
            else:
                time.sleep(0.1)
        if pending.strip():
            self.log.append(pending.strip())

    def tail(self):
        return '\n'.join(self.log)

    def stop(self):
        if not self.channel.exit_status_ready():
            self.channel.send('\x03')
            deadline = time.monotonic() + 12
            while not self.channel.exit_status_ready() and time.monotonic() < deadline:
                time.sleep(0.2)
        self.channel.close()
        self.reader.join(timeout=2)


def physical_ping(host, pc_ip):
    """Bind ICMP to the robot Ethernet address; proxy TCP is not proof."""
    command = ['ping', '-n', '1', '-w', '1500', '-S', pc_ip, host]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    return result.returncode == 0


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def wait_for_port(host, port, launch, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(host, port):
            return
        if launch.channel.exit_status_ready():
            raise RuntimeError('{} exited during startup:\n{}'.format(
                launch.name, launch.tail()))
        time.sleep(0.5)
    raise RuntimeError('{} did not become ready:\n{}'.format(
        launch.name, launch.tail()))


def wait_for_camera(streams, timeout=8):
    if not streams.wait(timeout):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if streams.depth() is not None:
            return True
        time.sleep(0.1)
    return False


def home_arm(client):
    """Establish the collector's commanded HOME before the first preflight."""
    print('\nStartup homing: move the arm to the collector HOME {}.'.format(
        arm_control.HOME_POSE))
    print('This is an open-loop move from the arm\'s current pose. Clear its full '
          'swept area before confirming.')
    if not arm_control.confirm_execute():
        print('Startup homing cancelled; no collection will run.')
        return False
    link = arm_control.ArmLink(client, True)
    try:
        link.send(list(arm_control.HOME_POSE), arm_control.HOMING_TIME_MS)
        time.sleep(arm_control.HOMING_TIME_MS / 1000.0 + 1.0)
    finally:
        link.close()
    print('HOME target sent and motion time elapsed (joint position is not measured).')
    return True


def connect_ssh(host, username, password):
    known_hosts = Path.home() / '.ssh' / 'known_hosts'
    if not known_hosts.is_file():
        raise RuntimeError('No {}. First verify the robot host key using '
                           'ssh {}@{}.'.format(known_hosts, username, host))
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(str(known_hosts))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        ssh.connect(hostname=host, username=username, password=password,
                    look_for_keys=False, allow_agent=False, timeout=8,
                    auth_timeout=8, banner_timeout=8)
    except Exception:
        ssh.close()
        raise
    return ssh


def prompt_placement(number, count):
    print('\nEpisode {}/{}'.format(number, count))
    print('The arm should be HOME. Place the red sweet at a new, clear, reachable table position.')
    print('Remove hands/obstacles and keep the gamepad down.')
    answer = input('Press Enter to preflight, or type q to finish: ').strip().lower()
    return answer not in ('q', 'quit')


def newest_episode(dataset, previous):
    folders = set(dataset.glob('*/episode.json'))
    created = folders - previous
    if len(created) != 1:
        return None
    return next(iter(created))


def collect_one(dataset, shadow_model, experiment_args=()):
    base = [sys.executable, '-m', 'pc_client.red_rl.collect', '--fast',
            '--dataset', str(dataset)]
    base += list(experiment_args)
    if shadow_model:
        base += ['--shadow-model', str(shadow_model)]
    while True:
        print('\nCamera/depth safety preflight (no arm motion)...')
        if subprocess.run(base, cwd=ROOT).returncode == 0:
            break
        answer = input('Preflight failed. Fix placement, Enter to retry or q to stop: ')
        if answer.strip().lower() in ('q', 'quit'):
            return False
    previous = set(dataset.glob('*/episode.json'))
    print('\nReal rule-teacher trial. You must type MOVE at its safety prompt.')
    result = subprocess.run(base + ['--execute'], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError('Trial stopped (exit {}). Inspect the arm before '
                           'another run; it may not be HOME.'.format(result.returncode))
    episode = newest_episode(dataset, previous)
    if episode is None:
        raise RuntimeError('Trial ended but exactly one new episode was not found.')
    metadata = json.loads(episode.read_text(encoding='utf-8'))
    label = metadata.get('human_success')
    label_text = {True: 'SUCCESS', False: 'FAILURE', None: 'UNKNOWN'}[label]
    print('Saved {}: {} ({} approach steps)'.format(
        episode.parent.name, label_text, metadata.get('steps')))
    return True


# A missed grasp leaves the sweet where it was, low-released, so the next trial
# can simply try again. Two misses in a row suggest something a retry will not
# fix (the sweet knocked over or out of reach, the light changed), so stop.
MAX_CONSECUTIVE_MISSES = 2


def confirm_auto(count, max_misses=MAX_CONSECUTIVE_MISSES):
    print('\nUNATTENDED collection: up to {} trials without further questions.'.format(count))
    print('Each held sweet is carried to a random reachable spot and put down. A miss '
          'is put back where it was and retried; the session stops after {} misses in '
          'a row, or at once on a preflight, tracking or other error.'.format(max_misses))
    print('Place the red sweet in reach, clear the table, and stay near the power switch.')
    return input('Type AUTO to start: ').strip() == 'AUTO'


def collect_auto(dataset, experiment_args=()):
    """One unattended trial.

    Returns (status, episode_name, auto_success) where status is 'held',
    'missed' (completed, camera said NOT HELD or could not tell) or 'stop'.
    """
    previous = set(dataset.glob('*/episode.json'))
    command = [sys.executable, '-m', 'pc_client.red_rl.collect', '--fast',
               '--dataset', str(dataset), '--execute', '--auto']
    command += list(experiment_args)
    # The collector's own MOVE prompt was answered once, by confirm_auto.
    result = subprocess.run(command, cwd=ROOT, input='MOVE\n', text=True)
    episode = newest_episode(dataset, previous)
    metadata = (json.loads(episode.read_text(encoding='utf-8'))
                if episode is not None else None)
    name = episode.parent.name if episode is not None else None
    if result.returncode == 0 and metadata is not None:
        where = metadata.get('relocation')
        print('Saved {}: HELD (camera), {} approach steps; {}'.format(
            name, metadata.get('steps'),
            'put down {moved_mm} mm away at radius {radius_mm} mm, azimuth '
            '{azimuth_deg} deg'.format(**where) if where else
            'put back where it was (no checked route to a new spot)'))
        return 'held', name, True
    if (result.returncode == EXIT_NOT_HELD and metadata is not None
            and metadata.get('outcome') == 'completed'):
        evidence = metadata.get('auto_evidence') or {}
        print('Saved {} as {}: the camera did not confirm a hold ({}); the sweet was '
              'put back where it was.'.format(
                  name, {False: 'NOT HELD', None: 'UNLABELLED'}[
                      metadata.get('auto_success')], evidence.get('reason')))
        return 'missed', name, metadata.get('auto_success')
    print('Stopping: trial exited with status {}{}. Inspect the arm before another '
          'run; it may not be HOME.'.format(
              result.returncode,
              '' if episode is None else ' (episode {}, outcome {})'.format(
                  name, metadata.get('outcome'))))
    return 'stop', name, None


def unattended(dataset, count, experiment_args=(),
               max_misses=MAX_CONSECUTIVE_MISSES):
    """Run up to `count` trials; returns how many completed a grasp attempt."""
    completed = held = misses = 0
    needs_label = []
    for number in range(1, count + 1):
        print('\n=== unattended trial {}/{} ==='.format(number, count))
        status, name, auto_success = collect_auto(dataset, experiment_args)
        if status == 'stop':
            break
        completed += 1
        if status == 'held':
            held += 1
            misses = 0
            continue
        misses += 1
        if auto_success is None:
            needs_label.append(name)
        if misses >= max_misses:
            print('Stopping: {} misses in a row. Check the sweet and the table '
                  'before another session.'.format(misses))
            break
        print('Retrying from where it lies (miss {} of {} allowed in a row).'.format(
            misses, max_misses))
    print('\nUnattended session: {} grasp attempt(s), {} held.'.format(completed, held))
    if needs_label:
        print('The camera could not tell on these; label them by hand if you saw '
              'the result, otherwise they stay out of training and success rates:')
        for name in needs_label:
            print('  {}'.format(name))
    return completed


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=5,
                        help='Maximum trials this session; default 5. Quit earlier with q.')
    parser.add_argument('--auto', action='store_true',
                        help='Unattended: camera labels, random reachable re-placement, '
                             'retry a miss in place, stop after --max-misses in a row '
                             'or at once on any other failure. Confirm once with AUTO.')
    parser.add_argument('--max-misses', type=int, default=MAX_CONSECUTIVE_MISSES,
                        help='With --auto, misses in a row that end the session '
                             '(default {}; 1 stops at the first).'.format(
                                 MAX_CONSECUTIVE_MISSES))
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--pc-ip', default='192.168.2.10')
    parser.add_argument('--user', default='jetson')
    parser.add_argument('--dataset', type=Path, default=Path('datasets/red_sweet'))
    parser.add_argument('--shadow-model', type=Path,
                        help='Optional read-only model; NEVER controls the arm.')
    parser.add_argument('--centre-compensation', action='store_true',
                        help='Experimental red-sweet clipped-centre compensation; '
                             'wrist rotation remains disabled.')
    parser.add_argument('--grasp-offset-mm', type=float,
                        help='Experimental gripper height relative to measured object top '
                             '(0-10 mm).')
    parser.add_argument('--creep-controller', choices=('auto', 'legacy', 'adaptive'),
                        default='auto', help='Experimental final-approach controller.')
    parser.add_argument('--pregrasp-retract-j4-deg', type=int, choices=(0, 1, 2),
                        default=0, help='Small baseward correction before closing.')
    parser.add_argument('--pregrasp-base-mm', type=float, default=0.0,
                        help='Experimental baseward shift before closing (0-10 mm).')
    parser.add_argument('--pregrasp-down-mm', type=float, default=0.0,
                        help='Experimental downward shift before closing (0-2.3 mm).')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be at least 1')
    if args.max_misses < 1:
        parser.error('--max-misses must be at least 1')
    if args.grasp_offset_mm is not None and not 0.0 <= args.grasp_offset_mm <= 10.0:
        parser.error('--grasp-offset-mm must be between 0 and 10')
    if not 0.0 <= args.pregrasp_base_mm <= 10.0:
        parser.error('--pregrasp-base-mm must be between 0 and 10')
    if not 0.0 <= args.pregrasp_down_mm <= 2.3:
        parser.error('--pregrasp-down-mm must be between 0 and 2.3')
    if bool(args.pregrasp_base_mm) != bool(args.pregrasp_down_mm):
        parser.error('set both --pregrasp-base-mm and --pregrasp-down-mm')
    if args.pregrasp_retract_j4_deg and args.pregrasp_base_mm:
        parser.error('choose either the joint4 correction or the base/down correction')
    return args


def main():
    args = arguments()
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    dataset.mkdir(parents=True, exist_ok=True)
    if args.shadow_model:
        model = (args.shadow_model if args.shadow_model.is_absolute()
                 else ROOT / args.shadow_model)
        if not model.is_file():
            raise RuntimeError('Shadow model does not exist: {}'.format(model))
    else:
        model = None
    experiment_args = []
    if args.centre_compensation:
        experiment_args.append('--centre-compensation')
    if args.grasp_offset_mm is not None:
        experiment_args += ['--grasp-offset-mm', str(args.grasp_offset_mm)]
    if args.creep_controller != 'auto':
        experiment_args += ['--creep-controller', args.creep_controller]
    if args.pregrasp_retract_j4_deg:
        experiment_args += ['--pregrasp-retract-j4-deg',
                            str(args.pregrasp_retract_j4_deg)]
    if args.pregrasp_base_mm:
        experiment_args += ['--pregrasp-base-mm', str(args.pregrasp_base_mm),
                            '--pregrasp-down-mm', str(args.pregrasp_down_mm)]

    print('Rule-teacher data collection only; no model will control the arm.')
    print('No continuous video recording. Each real trial saves per-step images.')
    print('Keep a hand near the robot power switch. Do not use the gamepad.')
    if not physical_ping(args.host, args.pc_ip):
        raise RuntimeError('Robot did not answer ping bound to {}. Run car-mode.cmd '
                           'and check the cable/power first.'.format(args.pc_ip))
    password = getpass.getpass('SSH password for {}@{}: '.format(args.user, args.host))
    ssh = connect_ssh(args.host, args.user, password)
    del password
    launched = []
    client = None
    streams = None
    try:
        if not port_open(args.host, 9090):
            print('Starting rosbridge on the robot...')
            bridge = RemoteLaunch(ssh, 'rosbridge', BRIDGE_COMMAND)
            launched.append(bridge)
            wait_for_port(args.host, 9090, bridge)
        else:
            print('Reusing existing rosbridge.')

        client = arm_control.connect(args.host, 9090)
        streams = reach_candy.Streams(client)
        if not wait_for_camera(streams, timeout=4):
            print('Starting Orbbec camera on the robot...')
            camera = RemoteLaunch(ssh, 'camera', CAMERA_COMMAND)
            launched.append(camera)
            if not wait_for_camera(streams, timeout=25):
                raise RuntimeError('Camera did not provide colour and depth:\n{}'.format(
                    camera.tail()))
        else:
            print('Reusing existing camera.')
        if not home_arm(client):
            return 0
        streams.close()
        streams = None
        client.terminate()
        client = None

        completed = 0
        if args.auto:
            if model:
                experiment_args += ['--shadow-model', str(model)]
            if not confirm_auto(args.count, args.max_misses):
                print('Unattended collection cancelled.')
                return 0
            completed = unattended(dataset, args.count, experiment_args,
                                   args.max_misses)
        else:
            for number in range(1, args.count + 1):
                if not prompt_placement(number, args.count):
                    break
                if not collect_one(dataset, model, experiment_args):
                    break
                completed += 1
        print('\nSession finished: {} new completed trial(s).'.format(completed))
        comparison = compare_labels(dataset)
        print('Auto-vs-human labels so far: {} agree, {} disagree, {} auto '
              'inconclusive ({} human-labeled new-format episodes).'.format(
                  comparison['agree'], comparison['disagree'],
                  comparison['auto_abstained'], comparison['human_labeled']))
        print('Validate with: python -m pc_client.red_rl.validate_dataset')
        return 0
    finally:
        if streams is not None:
            streams.close()
        if client is not None:
            client.terminate()
        for launch in reversed(launched):
            print('Stopping {} started by this session...'.format(launch.name))
            launch.stop()
        ssh.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired,
            paramiko.SSHException) as error:
        print('Collection stopped: {}'.format(error), file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nInterrupted. Inspect the arm before another run.', file=sys.stderr)
        raise SystemExit(130)
