#!/usr/bin/env python3

"""Live camera viewer for the M3 Pro Orbbec DaBai DCW2 over rosbridge.

Colour and IR arrive as JPEG (sensor_msgs/CompressedImage) because a raw
1280x720 colour frame is about 2.7 MB, which the WebSocket cannot carry at
frame rate.

Depth is different. The driver advertises
/high_camera/depth/image_raw/compressed but publishes an empty payload there
(88-byte messages, header only), so that topic is unusable. This viewer reads
the raw 16UC1 topic instead, which is genuinely metric: pixel values are
millimetres. It costs about 0.46 MB per frame at 10 Hz.
"""

import argparse
import base64
import collections
import pathlib
import time

import cv2
import numpy
import roslibpy

STREAMS = {
    'color': {
        'topic': '/high_camera/color/image_raw/compressed',
        'type': 'sensor_msgs/msg/CompressedImage',
        'kind': 'compressed',
    },
    'ir': {
        'topic': '/high_camera/ir/image_raw/compressed',
        'type': 'sensor_msgs/msg/CompressedImage',
        'kind': 'compressed',
    },
    'depth': {
        'topic': '/high_camera/depth/image_raw',
        'type': 'sensor_msgs/msg/Image',
        'kind': 'raw16',
    },
}

# Display range for the depth colour map, in millimetres.
DEPTH_NEAR_MM = 200
DEPTH_FAR_MM = 4000


def parse_args():
    parser = argparse.ArgumentParser(description='Live view of the M3 Pro camera over rosbridge.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--stream', choices=sorted(STREAMS), default='color')
    parser.add_argument('--fps', type=float, default=15.0, help='Requested rate; rosbridge throttles to this.')
    parser.add_argument('--scale', type=float, default=1.0, help='Display scale factor.')
    parser.add_argument('--save-dir', default='.', help='Where the s key writes snapshots.')
    return parser.parse_args()


def decode(message, kind):
    """Return (bgr_image, raw_bytes, depth_mm_or_None)."""
    payload = base64.b64decode(message['data'])
    if not payload:
        return None, 0, None

    if kind == 'raw16':
        height = message['height']
        width = message['width']
        expected = height * width * 2
        if len(payload) < expected:
            return None, len(payload), None
        depth_mm = numpy.frombuffer(payload[:expected], dtype=numpy.uint16).reshape(height, width)
        clipped = numpy.clip(depth_mm, DEPTH_NEAR_MM, DEPTH_FAR_MM)
        span = float(DEPTH_FAR_MM - DEPTH_NEAR_MM)
        normalised = ((clipped - DEPTH_NEAR_MM) * (255.0 / span)).astype(numpy.uint8)
        image = cv2.applyColorMap(normalised, cv2.COLORMAP_TURBO)
        # Pixels the sensor could not resolve come back as 0; show them black.
        image[depth_mm == 0] = 0
        return image, len(payload), depth_mm

    buffer = numpy.frombuffer(payload, dtype=numpy.uint8)
    if buffer.size == 0:
        return None, 0, None
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, len(payload), None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, len(payload), None


def annotate(image, lines):
    """Draw HUD lines top-left, sized so the longest one fits the frame."""
    width = image.shape[1]
    longest = max(len(text) for text in lines) if lines else 1
    # FONT_HERSHEY_SIMPLEX advances roughly 19 px per character at scale 1.0.
    font_scale = min(0.7, (width - 24) / (longest * 19.0))
    font_scale = max(0.3, font_scale)
    line_height = int(38 * font_scale) + 8
    for index, text in enumerate(lines):
        origin = (12, line_height + index * line_height)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    spec = STREAMS[args.stream]
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run(timeout=10)
    if not client.is_connected:
        raise RuntimeError('Unable to connect to ws://{}:{}'.format(args.host, args.port))
    print('Connected to ws://{}:{}'.format(args.host, args.port))
    print('Stream: {}'.format(spec['topic']))
    print('Keys: q quit, s snapshot.')

    # queue_length=1 keeps only the newest frame on the bridge, so a slow PC
    # falls behind in frame rate rather than in latency.
    topic = roslibpy.Topic(
        client,
        spec['topic'],
        spec['type'],
        throttle_rate=int(1000.0 / args.fps),
        queue_length=1,
        queue_size=1,
    )

    latest = {}
    arrivals = collections.deque(maxlen=30)

    def on_frame(message):
        latest['message'] = message
        arrivals.append(time.time())

    topic.subscribe(on_frame)

    window = 'M3 Pro {}'.format(args.stream)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    shown = 0
    dropped = 0
    last_seen = None

    try:
        while True:
            message = latest.get('message')
            if message is not None and message is not last_seen:
                last_seen = message
                image, byte_count, depth_mm = decode(message, spec['kind'])
                if image is None:
                    dropped += 1
                else:
                    rate = 0.0
                    if len(arrivals) > 1:
                        span = arrivals[-1] - arrivals[0]
                        if span > 0:
                            rate = (len(arrivals) - 1) / span
                    lines = ['{:.1f} fps  {} KB  {}x{}'.format(
                        rate, byte_count // 1024, image.shape[1], image.shape[0]
                    )]
                    if depth_mm is not None:
                        centre = int(depth_mm[depth_mm.shape[0] // 2, depth_mm.shape[1] // 2])
                        lines.append('centre: {} mm'.format(centre if centre else 'no return'))
                    if dropped:
                        lines.append('dropped {}'.format(dropped))

                    if args.scale != 1.0:
                        image = cv2.resize(image, None, fx=args.scale, fy=args.scale)
                    if depth_mm is not None:
                        height, width = image.shape[:2]
                        cv2.drawMarker(image, (width // 2, height // 2), (255, 255, 255),
                                       cv2.MARKER_CROSS, 18, 1)
                    annotate(image, lines)
                    cv2.imshow(window, image)
                    shown += 1

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and last_seen is not None:
                image, _, _ = decode(last_seen, spec['kind'])
                if image is not None:
                    path = save_dir / 'm3pro_{}_{}.png'.format(args.stream, int(time.time()))
                    cv2.imwrite(str(path), image)
                    print('Saved {}'.format(path))
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        topic.unsubscribe()
        client.terminate()
        cv2.destroyAllWindows()
        print('Displayed {} frames, dropped {}.'.format(shown, dropped))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
