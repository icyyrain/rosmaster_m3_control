"""Serve the red-sweet RL environment from Windows to the WSL learner.

    python -m pc_client.red_rl.env_server --fake            # no robot at all
    python -m pc_client.red_rl.env_server --check           # camera only, no motion
    python -m pc_client.red_rl.env_server --execute         # the real arm

The learner connects with RemoteEnv from isaac-hil-serl (or this package's
copy, pc_client.red_rl.rpc) to the Windows address of the WSL adapter. Every
reset and step runs on this process's main thread, which owns the rosbridge
connection.
"""

import argparse
import subprocess
import sys
import time


import arm_control
import pick_and_place as pick
from pc_client.red_rl import env as red_env
from pc_client.red_rl.intervene import RuleIntervention
from pc_client.red_rl.rpc import EnvironmentRpcServer

DEFAULT_PORT = 18766             # the simulation project uses 18765


def wsl_adapter_address():
    """Windows address of the vEthernet (WSL) adapter, as the learner sees it."""
    command = ['powershell', '-NoProfile', '-Command',
               "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object "
               "{ $_.InterfaceAlias -like 'vEthernet (WSL*' -and "
               "$_.AddressState -eq 'Preferred' } | Select-Object -First 1)"
               ".IPAddress"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def check(args):
    """Camera, depth and preflight from where the arm already is; never moves."""
    session = red_env.collect.TeacherSession(red_env.teacher_args(
        args.robot, args.robot_port, execute=False))
    try:
        if not session.streams.wait():
            print('No camera frames; check the camera/rosbridge nodes')
            return 1
        ok = pick.preflight(session)
        print('preflight {}; stop height {:.1f} mm'.format(
            'passed' if ok else 'FAILED', session.stop_height_mm))
        print('observation names: {}'.format(', '.join(red_env.OBSERVATION_NAMES)))
        return 0 if ok else 1
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fake', action='store_true', help='Serve the robot-free environment.')
    mode.add_argument('--check', action='store_true',
                      help='Run camera preflight once without moving; no server.')
    mode.add_argument('--execute', action='store_true', help='Serve the real arm.')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Address to listen on; the learner is on the WSL adapter.')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--robot', default='192.168.2.4')
    parser.add_argument('--robot-port', type=int, default=9090)
    parser.add_argument('--audit-root', default='datasets/red_sweet_rl')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.check:
        return check(args)
    if args.fake:
        environment = red_env.FakeRedSweetEnv(seed=args.seed)
    else:
        print('The REAL arm will run unattended episodes driven by the learner: '
              'rule approach, policy steps inside a safety box, grasp, put-down.')
        if not arm_control.confirm_execute():
            print('Cancelled.')
            return 0
        environment = red_env.RedSweetApproachEnv(
            args=red_env.teacher_args(args.robot, args.robot_port, execute=True),
            audit_root=args.audit_root, seed=args.seed)

    # The rule corrector is inert unless a reset asks for it (S3 training);
    # evaluation never does.
    environment = RuleIntervention(environment)
    server = EnvironmentRpcServer(environment, host=args.host, port=args.port)
    server.start_in_thread(dispatch_on_owner_thread=True)
    host, port = server.address
    print('serving {} on {}:{}'.format(environment.metadata['name'], host, port))
    print('observation {}, action {}'.format(environment.observation_space.shape,
                                             environment.action_space.shape))
    wsl_host = wsl_adapter_address()
    if wsl_host:
        print('from WSL: RemoteEnv("{}", {})'.format(wsl_host, port))
    print('Ctrl+C to stop.')
    try:
        while True:
            server.process_pending_requests()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        server.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
