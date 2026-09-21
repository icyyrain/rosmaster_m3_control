#!/usr/bin/env python3

import argparse
import time

import roslibpy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Publish one guarded arm command through rosbridge. Dry-run by default.'
    )
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--source', choices=('test', 'rl'), default='test')
    parser.add_argument('--joints', type=int, nargs=6, required=True, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'))
    parser.add_argument('--time-ms', type=int, default=200)
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually connect and publish. Without this flag, only print the payload.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if any(value < 0 or value > 180 for value in args.joints):
        raise ValueError('Every joint target must initially remain in [0, 180] degrees.')
    if not 100 <= args.time_ms <= 2000:
        raise ValueError('--time-ms must be in [100, 2000].')

    payload = {
        f'joint{index + 1}': value
        for index, value in enumerate(args.joints)
    }
    payload['time'] = args.time_ms
    print(f'Command source: {args.source}')
    print(f'Command payload: {payload}')

    if not args.execute:
        print('Dry run only. Add --execute to publish.')
        return

    confirmation = input('Type MOVE to publish this command: ')
    if confirmation != 'MOVE':
        print('Cancelled.')
        return

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run(timeout=10)
    if not client.is_connected:
        raise RuntimeError(f'Unable to connect to ws://{args.host}:{args.port}')

    mode_topic = roslibpy.Topic(client, '/arm/control_mode', 'std_msgs/msg/String')
    command_topic = roslibpy.Topic(
        client,
        f'/arm/command/{args.source}',
        'arm_msgs/msg/ArmJoints',
    )
    mode_topic.publish(roslibpy.Message({'data': args.source}))
    time.sleep(0.25)
    command_topic.publish(roslibpy.Message(payload))
    time.sleep(0.5)
    client.terminate()
    print('Command published. The Jetson safety node still decides whether to forward it.')


if __name__ == '__main__':
    main()
