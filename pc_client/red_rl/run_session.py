"""Operator-guided red-sweet data collection on a Windows PC.

Starts only missing ROS services over SSH, runs the existing rule-teacher
collector once per operator-approved placement, and stops services it started.
No student-policy action is ever sent to the robot.
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


def collect_one(dataset, shadow_model):
    base = [sys.executable, '-m', 'pc_client.red_rl.collect', '--fast',
            '--dataset', str(dataset)]
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=5,
                        help='Maximum trials this session; default 5. Quit earlier with q.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--pc-ip', default='192.168.2.10')
    parser.add_argument('--user', default='jetson')
    parser.add_argument('--dataset', type=Path, default=Path('datasets/red_sweet'))
    parser.add_argument('--shadow-model', type=Path,
                        help='Optional read-only model; NEVER controls the arm.')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be at least 1')
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
        streams.close()
        streams = None
        client.terminate()
        client = None

        completed = 0
        for number in range(1, args.count + 1):
            if not prompt_placement(number, args.count):
                break
            if not collect_one(dataset, model):
                break
            completed += 1
        print('\nSession finished: {} new completed trial(s).'.format(completed))
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
