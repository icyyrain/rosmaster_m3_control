#!/usr/bin/env python3

import argparse
import json
import signal
import threading
import time

import roslibpy


def parse_args():
    parser = argparse.ArgumentParser(description='Monitor M3 Pro arm feedback through rosbridge.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    return parser.parse_args()


def main():
    args = parse_args()
    client = roslibpy.Ros(host=args.host, port=args.port)
    stopped = threading.Event()

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    topics = [
        ('/joint_states', roslibpy.Topic(client, '/joint_states', 'sensor_msgs/msg/JointState')),
        (
            '/arm/feedback_status',
            roslibpy.Topic(client, '/arm/feedback_status', 'std_msgs/msg/String'),
        ),
        (
            '/arm/safety_status',
            roslibpy.Topic(client, '/arm/safety_status', 'std_msgs/msg/String'),
        ),
        ('/battery', roslibpy.Topic(client, '/battery', 'std_msgs/msg/Float32')),
    ]

    def print_message(topic_name):
        def callback(message):
            if topic_name.endswith('status') and 'data' in message:
                try:
                    message = json.loads(message['data'])
                except json.JSONDecodeError:
                    pass
            print(f'{time.strftime("%H:%M:%S")} {topic_name}: {message}')
        return callback

    client.run(timeout=10)
    if not client.is_connected:
        raise RuntimeError(f'Unable to connect to ws://{args.host}:{args.port}')
    print(f'Connected to ws://{args.host}:{args.port}. Press Ctrl+C to stop.')

    for topic_name, topic in topics:
        topic.subscribe(print_message(topic_name))

    try:
        while not stopped.wait(0.25):
            pass
    finally:
        for _, topic in topics:
            topic.unsubscribe()
        client.terminate()


if __name__ == '__main__':
    main()
