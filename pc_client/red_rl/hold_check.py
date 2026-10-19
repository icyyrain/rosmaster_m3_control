"""Is the sweet in the jaws? Turn the base a little and see whether it follows.

The short-lift check watches the sweet's centroid move down the image as the
camera rises, but the gripper pixel is only 40 px above the bottom edge: a
sweet left on the table slides out of the frame, and one that is held shows
mostly wrapper, a sliver of red half behind the jaws. On 2026-10-19 that
centroid moved 63-68 px for sweets the operator saw held, and two runs were
scored INCONCLUSIVE where the photographs show a firm grasp.

This check moves sideways instead. At the lift pose the base turns
TURN_DEG: a sweet in the jaws turns with the camera and its red stays where
it was in the image; one left on the table shifts across the image by about
the camera's motion (roughly 90 px for 4 degrees near the table), and
sideways it stays in the frame. The comparison is between the red masks near
the jaws before and after the turn, best-overlap shift against no shift, so
a sliver of red is enough and no centroid is needed.

The same mask comparison checks the carry: a held sweet is rigid with the
camera, so at the carry pose its red is where it was after the lift.
"""

import cv2
import numpy as np

import reach_candy
from pc_client.red_rl import auto_reward

METHOD = 'base_turn_v1'

TURN_DEG = 4
TURN_MS = 500
TURN_SETTLE_S = 1.5
MIN_RED_PX = 120                 # too little red near the jaws to compare
HELD_SHIFT_PX = 20               # the red stayed put
MOVED_SHIFT_PX = 45              # the red moved like the table
MIN_OVERLAP = 0.35               # the best shift must explain the red this well
MAX_SHIFT_PX = 160
CARRY_SHIFT_PX = 40              # a held sweet may swing this much while carried

ROI_ABOVE_PX = 260               # rows from gripper_y - this to the bottom edge
ROI_HALF_WIDTH_PX = 320


def red_near_jaws(bgr):
    """Boolean red mask of the region around and above the jaws."""
    mask, _ = reach_candy.chroma_mask(bgr, 'red-sat', reach_candy.DEFAULT_CHROMA)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gx, gy = reach_candy.GRIPPER_PX
    top = max(0, int(gy - ROI_ABOVE_PX))
    left = max(0, int(gx - ROI_HALF_WIDTH_PX))
    right = min(mask.shape[1], int(gx + ROI_HALF_WIDTH_PX))
    return mask[top:, left:right] > 0


def horizontal_shift(before, after, max_shift=MAX_SHIFT_PX, step=2):
    """Shift of `after` relative to `before` that best overlays their red.

    Returns (best_dx, overlap_at_best, overlap_at_zero); overlap is the shared
    red over the smaller of the two red areas, 0..1.
    """
    a, b = before.astype(bool), after.astype(bool)
    norm = float(min(a.sum(), b.sum()))
    if norm <= 0:
        return None, 0.0, 0.0
    width = a.shape[1]
    best = (0, -1.0)
    zero = 0.0
    for dx in range(-max_shift, max_shift + 1, step):
        # after(x) == before(x - dx) when the content moved by +dx
        if dx >= 0:
            shared = np.logical_and(a[:, :width - dx], b[:, dx:]).sum()
        else:
            shared = np.logical_and(a[:, -dx:], b[:, :width + dx]).sum()
        score = shared / norm
        if dx == 0:
            zero = score
        if score > best[1]:
            best = (dx, score)
    return best[0], float(best[1]), float(zero)


def judge_turn(before_bgr, after_bgr):
    """Held (True), on the table (False) or cannot tell (None), with evidence."""
    before, after = red_near_jaws(before_bgr), red_near_jaws(after_bgr)
    evidence = dict(method=METHOD, turn_deg=TURN_DEG,
                    red_before_px=int(before.sum()), red_after_px=int(after.sum()),
                    shift_px=None, overlap=None, overlap_at_zero=None,
                    success=None, reason=None)
    if before.sum() < MIN_RED_PX:
        # A held sweet can hide behind the jaws; leave it to the lift check.
        evidence['reason'] = 'no_red_near_the_jaws_after_the_lift'
        return evidence
    if after.sum() < MIN_RED_PX:
        evidence['reason'] = 'red_left_the_view_on_the_turn'
        evidence['success'] = False
        return evidence
    dx, overlap, zero = horizontal_shift(before, after)
    evidence.update(shift_px=dx, overlap=round(overlap, 3), overlap_at_zero=round(zero, 3))
    if overlap < MIN_OVERLAP:
        evidence['reason'] = 'red_does_not_match_across_the_turn'
    elif abs(dx) <= HELD_SHIFT_PX:
        evidence.update(success=True, reason='red_turned_with_the_camera')
    elif abs(dx) >= MOVED_SHIFT_PX:
        evidence.update(success=False, reason='red_stayed_with_the_table')
    else:
        evidence['reason'] = 'shift_between_held_and_table'
    return evidence


def judge_carry(lift_bgr, carry_bgr):
    """Still held at the carry pose? (True/False, evidence)."""
    before, after = red_near_jaws(lift_bgr), red_near_jaws(carry_bgr)
    evidence = dict(red_lift_px=int(before.sum()), red_carry_px=int(after.sum()),
                    shift_px=None, overlap=None, reason=None)
    if before.sum() < MIN_RED_PX:
        evidence['reason'] = 'no_red_after_the_lift_to_compare'
        return True, evidence            # nothing to judge by; do not abort
    if after.sum() < MIN_RED_PX:
        evidence['reason'] = 'red_gone_from_the_jaws'
        return False, evidence
    dx, overlap, _ = horizontal_shift(before, after, max_shift=CARRY_SHIFT_PX * 2)
    evidence.update(shift_px=dx, overlap=round(overlap, 3))
    held = bool(overlap >= MIN_OVERLAP and abs(dx) <= CARRY_SHIFT_PX)
    evidence['reason'] = 'red_still_in_the_jaws' if held else 'red_moved_off_the_jaws'
    return held, evidence


def combine(lift, turn):
    """One verdict from the short-lift evidence and the turn evidence.

    The turn decides when it can; otherwise the short lift does. Both stay in
    the record: the lift fields at the top level as before (demos.judge
    re-judges them), the turn under 'turn_check', and the lift's own verdict
    under 'short_lift'.
    """
    evidence = dict(lift, method=auto_reward.TURN_METHOD, turn_check=turn,
                    short_lift=dict(method=lift.get('method'), success=lift.get('success'),
                                    reason=lift.get('reason')))
    if turn.get('success') is not None:
        evidence.update(success=turn['success'], reason=turn['reason'],
                        decided_by='turn')
    else:
        evidence['decided_by'] = 'lift'
    return evidence


def turned(pose, band):
    """The lift pose with joint 1 turned TURN_DEG, inward from the band edge."""
    target = list(pose)
    target[0] = pose[0] + TURN_DEG if pose[0] + TURN_DEG <= band[1] else pose[0] - TURN_DEG
    return target


def turn_check(session):
    """At the lift pose: photograph, turn, photograph, turn back. Returns evidence."""
    lift = list(session.pose)
    before = session.streams.colour()
    session.send(turned(lift, _band()), TURN_MS, TURN_SETTLE_S)
    after = session.streams.colour()
    session.send(lift, TURN_MS, TURN_SETTLE_S)
    if before is None or after is None:
        return dict(method=METHOD, success=None, reason='no_frame'), before, after
    return judge_turn(before, after), before, after


def _band():
    import pick_and_place
    return pick_and_place.J1_BAND
