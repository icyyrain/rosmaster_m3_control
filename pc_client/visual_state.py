#!/usr/bin/env python3

"""Estimate the camera 6-DoF pose from a fixed AprilTag, over rosbridge.

This is the "B-lite" state channel. The depth camera is bolted to arm link
arm4 (see the URDF: DCW2_Joint is a fixed joint whose parent is arm4), so the
camera pose relative to a stationary tag is a deterministic function of joints
1-4. That makes these six numbers usable directly as proprioception, with no
hand-eye calibration and no learning: only the factory intrinsics, which the
driver already publishes on camera_info.

What it cannot see: joints 5 and 6 are downstream of arm4, so wrist rotation
and the gripper do not move the camera and are invisible here.

Run it with the arm held still to read the noise floor. That figure decides
whether converting these poses into joint angles is worth doing at all: if the
position noise is already centimetres, the derived joint angles will be junk.
"""

import argparse
import base64
import collections
import math
import time

import cv2
import numpy
import roslibpy

FAMILY = cv2.aruco.DICT_APRILTAG_36h11
COLOR_TOPIC = '/high_camera/color/image_raw/compressed'
INFO_TOPIC = '/high_camera/color/camera_info'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera 6-DoF pose from a fixed AprilTag, via rosbridge.')
    parser.add_argument('--host', default='192.168.2.4')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--tag-id', type=int, default=0)
    parser.add_argument('--tag-mm', type=float, default=100.0,
                        help='Side of the outer black square, in mm. Measure the print.')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--seconds', type=float, default=0.0,
                        help='Stop after this long. 0 means run until Ctrl+C.')
    parser.add_argument('--window', action='store_true',
                        help='Show the video with the detected tag outlined.')
    return parser.parse_args()


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY)
    params = cv2.aruco.DetectorParameters()
    # Subpixel corner refinement; without it the pose jitters far more.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


def object_points(tag_mm):
    """Tag corners in the tag frame, in metres, matching the aruco corner order.

    aruco returns corners clockwise from top-left. The tag frame has x right,
    y up, z out of the tag face.
    """
    half = tag_mm / 2000.0
    return numpy.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=numpy.float64)


def rotation_to_euler_deg(matrix):
    """Intrinsic X-Y-Z (roll, pitch, yaw) in degrees."""
    sy = max(-1.0, min(1.0, -matrix[2, 0]))
    pitch = math.asin(sy)
    if abs(sy) < 0.99999:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return [math.degrees(value) for value in (roll, pitch, yaw)]


Pose = collections.namedtuple('Pose', 'position_mm euler_deg reproj_px margin_px')

# Both solvers are tried and every candidate scored, rather than trusting one
# flag's own pick. Two reasons:
#   - A square seen exactly face-on is a singular configuration for
#     SOLVEPNP_IPPE_SQUARE: at 0.0 degrees of tilt both of its solutions come
#     back about 93 degrees from the truth with ~68 px of reprojection error,
#     while SQPNP is exact. Three degrees of tilt is enough to fix IPPE, but
#     the tool must not fall over in the symmetric case.
#   - Four coplanar points admit two poses in general. Picking by
#     reprojection error, and reporting how far ahead the winner was, makes
#     the classic AprilTag pose flip visible instead of silent.
SOLVERS = (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_SQPNP)


def solve_camera_pose(corners, tag_mm, camera_matrix, dist_coeffs):
    """Camera pose in the tag frame, or None if no candidate is usable."""
    objp = object_points(tag_mm)
    imgp = numpy.asarray(corners, dtype=numpy.float64).reshape(4, 2)

    scored = []
    for flag in SOLVERS:
        try:
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                objp, imgp, camera_matrix, dist_coeffs, flags=flag)
        except cv2.error:
            continue
        for index in range(count):
            rvec = rvecs[index]
            tvec = tvecs[index]
            if tvec[2] <= 0:
                continue  # Tag behind the camera; not a real view.
            projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            error = float(numpy.linalg.norm(projected.reshape(4, 2) - imgp, axis=1).mean())
            scored.append((error, rvec, tvec))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    error, rvec, tvec = scored[0]

    # How much better the winner was than the best genuinely different pose.
    # A small margin means the estimate may flip between frames.
    margin = float('inf')
    best_rotation, _ = cv2.Rodrigues(rvec)
    for other_error, other_rvec, _ in scored[1:]:
        other_rotation, _ = cv2.Rodrigues(other_rvec)
        delta = numpy.linalg.norm(cv2.Rodrigues(best_rotation.T @ other_rotation)[0])
        if math.degrees(delta) > 1.0:
            margin = other_error - error
            break

    # rvec/tvec put the tag in the camera frame. Invert for camera-in-tag.
    cam_rotation = best_rotation.T
    cam_position = (-cam_rotation @ tvec).reshape(3)
    return Pose(cam_position * 1000.0, rotation_to_euler_deg(cam_rotation), error, margin)


def summarise(samples):
    """Mean and standard deviation per axis for a window of 6-vectors."""
    array = numpy.asarray(samples, dtype=numpy.float64)
    return array.mean(axis=0), array.std(axis=0)


def fetch_intrinsics(client, timeout=10.0):
    info = {}
    topic = roslibpy.Topic(client, INFO_TOPIC, 'sensor_msgs/msg/CameraInfo')
    topic.subscribe(lambda message: info.setdefault('message', message))
    deadline = time.time() + timeout
    while 'message' not in info and time.time() < deadline:
        time.sleep(0.05)
    topic.unsubscribe()
    if 'message' not in info:
        raise RuntimeError(
            'No camera_info on {}. Is the camera driver running?'.format(INFO_TOPIC))
    message = info['message']
    camera_matrix = numpy.asarray(message['k'], dtype=numpy.float64).reshape(3, 3)
    dist_coeffs = numpy.asarray(message['d'], dtype=numpy.float64).reshape(1, -1)
    return camera_matrix, dist_coeffs, message['width'], message['height']


def main():
    args = parse_args()
    detector = make_detector()

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run(timeout=10)
    if not client.is_connected:
        raise RuntimeError('Unable to connect to ws://{}:{}'.format(args.host, args.port))

    try:
        camera_matrix, dist_coeffs, width, height = fetch_intrinsics(client)
    except RuntimeError:
        client.terminate()
        raise

    print('Intrinsics {}x{}  fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}  {} dist coeffs'.format(
        width, height, camera_matrix[0, 0], camera_matrix[1, 1],
        camera_matrix[0, 2], camera_matrix[1, 2], dist_coeffs.size))
    print('Looking for AprilTag 36h11 id {} at {:.1f} mm.'.format(args.tag_id, args.tag_mm))
    print('Hold the arm still to read the noise floor. Ctrl+C to stop.\n')

    latest = {}
    image_topic = roslibpy.Topic(
        client, COLOR_TOPIC, 'sensor_msgs/msg/CompressedImage',
        throttle_rate=int(1000.0 / args.fps), queue_length=1, queue_size=1)
    image_topic.subscribe(lambda message: latest.__setitem__('message', message))

    recent = collections.deque(maxlen=60)
    all_poses = []
    errors = []
    margins = []
    frames = 0
    detections = 0
    last_print = 0.0
    last_seen = None
    started = time.time()

    if args.window:
        cv2.namedWindow('visual_state', cv2.WINDOW_NORMAL)

    try:
        while True:
            if args.seconds and time.time() - started >= args.seconds:
                break

            message = latest.get('message')
            if message is None or message is last_seen:
                if args.window and cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                if not args.window:
                    time.sleep(0.01)
                continue
            last_seen = message

            payload = base64.b64decode(message['data'])
            if not payload:
                continue
            image = cv2.imdecode(numpy.frombuffer(payload, dtype=numpy.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            frames += 1

            corners, ids, _ = detector.detectMarkers(image)
            pose = None
            if ids is not None:
                flat = ids.flatten().tolist()
                if args.tag_id in flat:
                    pose = solve_camera_pose(
                        corners[flat.index(args.tag_id)], args.tag_mm,
                        camera_matrix, dist_coeffs)

            if pose is not None:
                detections += 1
                sample = list(pose.position_mm) + list(pose.euler_deg)
                recent.append(sample)
                all_poses.append(sample)
                errors.append(pose.reproj_px)
                margins.append(pose.margin_px)

            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                elapsed = max(now - started, 1e-6)
                if recent:
                    mean, std = summarise(recent)
                    distance = float(numpy.linalg.norm(mean[:3]))
                    recent_margins = [v for v in margins[-60:] if math.isfinite(v)]
                    margin_text = ('{:.2f}'.format(min(recent_margins))
                                   if recent_margins else '  inf')
                    print('\rxyz {:7.1f} {:7.1f} {:7.1f} mm | rpy {:7.2f} {:7.2f} {:7.2f} deg'
                          ' | dist {:6.1f} | jitter {:.2f}mm {:.3f}deg | reproj {:.3f}px'
                          ' | margin {}px | det {:3.0f}% {:.1f} fps'.format(
                              mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], distance,
                              float(std[:3].max()), float(std[3:].max()),
                              float(numpy.mean(errors[-60:])), margin_text,
                              100.0 * detections / max(frames, 1), frames / elapsed),
                          end='', flush=True)
                else:
                    print('\rno tag yet | frames {} | {:.1f} fps'.format(
                        frames, frames / elapsed).ljust(78), end='', flush=True)

            if args.window:
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.imshow('visual_state', cv2.resize(image, None, fx=0.6, fy=0.6))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        image_topic.unsubscribe()
        client.terminate()
        if args.window:
            cv2.destroyAllWindows()

    print('\n')
    if not all_poses:
        print('No detections over {} frames. Check that the tag is lit, unoccluded,'.format(frames))
        print('roughly face-on, and that --tag-id matches the print.')
        return 1

    mean, std = summarise(all_poses)
    print('Samples {} of {} frames ({:.0f}% detected) over {:.1f}s'.format(
        len(all_poses), frames, 100.0 * detections / max(frames, 1), time.time() - started))
    print('Mean camera pose in tag frame:')
    print('  position  x {:9.2f}  y {:9.2f}  z {:9.2f}   mm'.format(*mean[:3]))
    print('  rotation  r {:9.3f}  p {:9.3f}  y {:9.3f}   deg'.format(*mean[3:]))
    print('Noise floor (standard deviation):')
    print('  position  x {:9.3f}  y {:9.3f}  z {:9.3f}   mm'.format(*std[:3]))
    print('  rotation  r {:9.4f}  p {:9.4f}  y {:9.4f}   deg'.format(*std[3:]))
    print('Mean reprojection error: {:.4f} px'.format(float(numpy.mean(errors))))

    finite = [value for value in margins if math.isfinite(value)]
    if finite:
        worst = min(finite)
        print('Pose ambiguity margin: min {:.4f} px over {} frames with a rival pose.'.format(
            worst, len(finite)))
        if worst < 0.5:
            print('  WARNING: a rival pose came within 0.5 px. Expect flipping.')
            print('  Tilt the tag further off face-on, move closer, or print it larger.')
    else:
        print('Pose ambiguity margin: no rival pose in any frame.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
