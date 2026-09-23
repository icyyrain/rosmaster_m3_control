# M3 Pro arm and camera measurements

Measured 2026-09-23 with `pc_client/arm_diagnostics.py` and
`pc_client/visual_state.py` against the real robot at `192.168.2.4`.

None of these numbers can be read from ROS. The control board publishes no
joint feedback, so they were measured optically, using the camera as the
sensor. Everything below is a differential measurement against the scene, so
it needs no AprilTag and no calibration.

## Why the camera can measure the arm

`yahboom_M3Pro_description/urdf/M3Pro.urdf` attaches the Orbbec DaBai DCW2
through a **fixed** joint `DCW2_Joint` whose parent is **`arm4`**:

```text
base_link -> arm1 -> arm2 -> arm3 -> arm4 -> DCW2
```

So the camera is eye-in-hand. Its view is a function of joints 1-4. Joint 5
(`arm5_Joint`, wrist roll) and the gripper (`rlink1_Joint`) are downstream of
arm4 and do not move it.

Two things that sound the same but are not. Joint 5 and the gripper do not
*move the camera*, so their angles cannot be inferred from the image
transform. They are nevertheless *inside the field of view*: the gripper hangs
in front of an arm4 mounted camera and appears along the bottom edge of the
frame. Measured by holding joints 1-4 fixed and toggling each:

| toggled | pixels changed | largest region |
| --- | --- | --- |
| nothing (control) | 1020 px | noise |
| gripper open vs closed | 12409 px (1.35%) | 157x91 at 55% across, 94% down |
| wrist roll 90 vs 150 | 71753 px (7.79%) | 359x244 at 42% across, 87% down |

So the gripper is directly observable as pixels even though its angle is not
recoverable from the global transform, and grasp state is likely checkable
optically without a second camera. It sits at the very bottom of the frame and
is partly cropped at the home pose, so whether it stays in view through a
reaching motion needs checking per trajectory.

Joint 5 also barely moves the gripper's *position*. Its axis is Z and the
gripper offset from `arm5` is almost purely along that same Z
(`0.093235` m in z against `0.000625`/`0.00024` in x/y), so a full revolution
of joint 5 displaces the gripper origin by at most **0.88 mm**, with z
unchanged. Gripper position is therefore set by joints 1-4 alone; only its roll
depends on joint 5. For a reaching task, which is positional, the blind spot
does not matter. The camera and gripper origins are 111.9 mm apart in the arm4
frame.

The URDF also carries a `Camera` link on `base_link` with zero limits. No
matching USB device was present, so treat it as an unused mount.

URDF joint limits for arm1..arm5 are -1.571..1.571 rad, that is +-90 degrees,
matching the 0..180 servo command range with 90 as centre. The gripper
`rlink1_Joint` is -0.95..0 rad.

## Kinematic coupling, measured

Joints predicted invisible were given six times the command of the others, so
the experiment was biased against its own hypothesis.

| joint | command | image motion | verdict |
| --- | --- | --- | --- |
| 1 base yaw | +10 deg | 118.15 px | moves camera |
| 2 shoulder | +10 deg | 110.50 px | moves camera |
| 3 elbow | +10 deg | 127.35 px | moves camera |
| 4 wrist pitch | +10 deg | 138.56 px | moves camera |
| 5 wrist roll | **+60 deg** | **0.00 px** | invisible |
| 5 wrist roll | **-60 deg** | 0.85 px | invisible |
| 6 gripper | **+60 deg** | 1.43 px | invisible |

Static noise floor over the same window was 0.03-0.05 px. The URDF is
confirmed, including the blind spot.

## Joint 1 sweep: deadband, linearity, backlash

One degree is the finest `ArmJoints` command, so whether a one-degree command
does anything decides whether a one-degree action space is meaningful.

- **No deadband.** All 40 one-degree steps produced motion. Median step
  17.84 px, minimum 2.56 px, maximum 26.01 px.
- **Sensitivity 18.81 px per degree** near home, worst linearity residual
  4.61 px, that is 0.245 degrees.
- **Backlash 0.996 degrees.** The same commanded angle lands 17.77 px apart on
  average depending on whether it was approached rising or falling.
- The **first step after a direction reversal delivers only 14-30%** of normal
  travel: 2.56 px and 5.34 px against a 17.84 px median. This is the clearest
  signature of the backlash.

Same-direction repeatability is excellent. A direction reversal costs about a
full degree.

## Image Jacobian

Measured at base pose `[90, 120, 10, 10, 90, 90]`, which is home with joints 3
and 4 lifted to 10 so they have room for a symmetric step. Joints 3 and 4 sit
at 0 at home and cannot go negative.

Displacement is evaluated at the image centre, not taken from the affine
translation. The affine translation refers to the top-left corner and is
inflated by any rotation, which is why joint 1 reads 18.8 px per degree that
way but 11.8 px per degree at the centre. The centre figure is the one a
controller needs.

Each sample point was approached while travelling in the same direction, with a
central difference about the base pose, so the 1 degree of backlash falls
outside the measurement.

Units are px per degree of joint command.

| | joint1 | joint2 | joint3 | joint4 |
| --- | --- | --- | --- | --- |
| dx | -11.70 | +0.42 | +0.86 | +0.63 |
| dy | +1.55 | +13.80 | +16.09 | +14.34 |
| roll deg | -0.421 | +0.091 | +0.135 | +0.096 |

The structure follows the kinematics exactly. Joint 1 turns about Z and pans
the camera horizontally, also rolling the image by -0.42 degrees per degree.
Joints 2, 3 and 4 turn about three parallel pitch axes and all tilt the camera
vertically by a similar amount.

### The matrix is rank 2, so do not drive four joints with a pixel error

A 2x4 matrix of rank 2 leaves a two-dimensional null space. These joint
combinations produce no image motion at all:

```text
[+0.03  -0.85  +1.00  -0.31]  ->  0.0000 px/deg
[+0.01  -0.70  -0.29  +1.00]  ->  0.0000 px/deg
```

A pseudo-inverse will return the minimum-norm solution, but posture then
drifts inside that null space over repeated iterations, uncontrolled.

Conditioning of every two-joint subset:

| subset | condition number | |
| --- | --- | --- |
| j1+j2 | 1.2 | good |
| j1+j4 | 1.2 | good |
| j1+j3 | 1.4 | good |
| j2+j3 | 88.1 | degenerate |
| j2+j4 | 148.5 | degenerate |
| j3+j4 | 212.1 | degenerate |

Joint 1 must be paired with exactly one of joints 2, 3 and 4. Any pair drawn
from within that group is degenerate, because all three do nearly the same
thing in the image.

**j1 + j4** is the suggested pair: condition 1.2, and joint 4 is the wrist
pitch, so it carries the least inertia and moves fastest for fine positioning.

Since a pixel error cannot observe distance, and reaching is a
three-dimensional task, the better arrangement uses the depth stream as a third
error term and three joints, giving a square well-posed system: error
`(ex, ey, e_depth)` against `(j1, j4, j2)`, with joint 2 setting reach.

### Backlash, confirmed a second time

The one-sided differences were markedly asymmetric. A constant offset between
the anchor's true position and the nominal base pose inflates the low-side
estimate and deflates the high-side estimate by the same factor, which
recovers the offset:

| | low side | high side | implied offset |
| --- | --- | --- | --- |
| j1 dx | 14.59 | 8.81 | 0.74 deg |
| j2 dy | 19.22 | 8.39 | 1.18 deg |
| j3 dy | 20.51 | 11.66 | 0.83 deg |
| j4 dy | 20.17 | 8.51 | 1.22 deg |
| | | mean | **0.99 deg** |

The sweep test measured backlash as **0.996 deg** by a completely different
route. Two independent experiments agreeing to two decimal places validates
both the figure and the central-difference design, since the matrix entries are
exactly the averages of the two one-sided estimates and are therefore immune to
this offset.

## Step response

Six-degree steps on joint 1, several `ArmJoints.time` values.

| commanded time | travel | onset | 95% complete |
| --- | --- | --- | --- |
| 200 ms | 6.26 deg | 241 ms | 416 ms |
| 500 ms | 6.29 deg | 264 ms | 704 ms |
| 1000 ms | 6.01 deg | 329 ms | 1143 ms |
| 1500 ms | 6.15 deg | 285 ms | 1538 ms |

- **Amplitude is accurate**: 6 degrees commanded gives 6.01-6.29 measured.
- **`ArmJoints.time` really sets trajectory duration.** 95% completion tracks
  the commanded value.
- **Onset latency 240-330 ms**, end to end: Windows, rosbridge, DDS,
  micro-ROS, servo, then back through JPEG encode, the WebSocket and decode.
  This is the dead time of any vision-in-the-loop controller built this way.

## Holding position

| condition | std dev | range | inliers |
| --- | --- | --- | --- |
| idle, long after last command | 0.046 px | 0.215 px | 942 |
| 1.5-4.5 s after a 6 deg step | 0.124 px | 0.618 px | 828 |
| 10-13 s after the same step | 0.031 px | 0.125 px | 924 |
| while re-sent every 300 ms | 0.032 px | 0.154 px | 929 |

**The arm does not hunt.** Residual motion after a move is at most 0.124 px,
roughly 0.007 degrees, and continuously re-sending the target does not excite
it. Static poses are reliable; all the uncertainty is in getting there.

## Two measurement artifacts that produced wrong numbers

Both came from putting the reference frame too far from what was being
measured. They are recorded because both looked like real hardware findings.

1. **Accumulated return error read as joint coupling.** Comparing every trial
   against one reference captured at the start made joints 5 and 6 report 9.67
   and 9.44 px, enough to look like real motion. The values were suspiciously
   equal to each other despite very different commands, and the inlier count
   had risen from ~230 to ~840, meaning the view was nearly identical. The
   cause was that the arm does not return exactly to a commanded pose, so late
   trials inherited the accumulated error. Taking a fresh reference immediately
   before each trial gave 0.00 and 0.85 px.

2. **Reduced view overlap read as servo hunting.** Measuring the post-move tail
   against a pre-move reference gave a 2.04 px standard deviation, about
   twentyfold the idle figure, which looked like the arm oscillating. But the
   inlier count had collapsed from ~940 to ~400. Taking the reference at the
   new pose gave 0.124 px.

**Rule:** take the reference close to the measurement, and read the inlier
count. If it falls by half, the numbers are not comparable.

## AprilTag pose pipeline

`visual_state.py` was validated offline against the robot's real intrinsics
(`fx=720.501, fy=720.4833, cx=649.1655, cy=359.6946`, 8-coefficient rational
polynomial). Seven poses from 350 mm to 1200 mm, face-on to steeply tilted, all
recovered to 0.000000 mm and 0.000000 deg with zero reprojection error.

**`SOLVEPNP_IPPE_SQUARE` fails on an exactly face-on square.** At 0.0 degrees
of tilt *both* of its solutions come back about 93 degrees from the truth with
68 px of reprojection error. Three degrees of tilt is enough to fix it, and
`SOLVEPNP_SQPNP` is exact throughout. `visual_state.py` therefore enumerates
candidates from both solvers, rejects any with the tag behind the camera, picks
the lowest reprojection error, and reports how far ahead the winner was so that
the classic AprilTag pose flip is visible rather than silent.

Predicted noise floor against subpixel corner noise, from the same validation:

| corner noise | distance | lateral | along normal | rotation |
| --- | --- | --- | --- | --- |
| 0.05 px | 400 mm | 0.88 mm | 0.08 mm | 0.12 deg |
| 0.10 px | 400 mm | 1.79 mm | 0.15 mm | 0.26 deg |
| 0.10 px | 800 mm | 15.0 mm | 0.71 mm | 1.08 deg |
| 0.30 px | 800 mm | 43.6 mm | 3.12 mm | 3.15 deg |

Lateral noise exceeds noise along the normal because it is dominated by
rotation error: 400 mm times 0.26 degrees is 0.87 mm, matching the table.
Distance along the tag normal is fixed by apparent tag size and is the
best-determined axis.

Tag scale error passes straight through to distance: a tag printed 1% small
puts every distance 1% off, while angles stay correct. Hence the 100 mm
reference line on the generated sheet. Absolute tag-based pose is roughly an
order of magnitude less precise than the differential scene measurement, so
prefer differential measurement wherever the task allows it.

The camera itself: colour 1280x720 at 29.5 fps over rosbridge, 3.1 MB/s; depth
640x360 at 9.7 fps; IR 640x400 at 10 fps.

## Visual servoing, closed on hardware

`pc_client/reach_candy.py` drives a red target under the gripper with vision
only: no learning, no joint feedback, no inverse kinematics.

### The gripper projects to a fixed pixel

This is what makes the loop cheap. The camera is on arm4 and the gripper on
arm5, and joint 5 turns about an axis nearly collinear with the offset to the
gripper, so the gripper origin moves at most 0.88 mm through a full revolution.
It should therefore land on the same pixel whatever joints 1-4 do.

Measured by toggling the gripper through three open/close cycles at four very
different poses and keeping only the pixels that changed in *every* cycle:

| pose | consistent px | single-cycle px | centroid | bbox |
| --- | --- | --- | --- | --- |
| home | 11940 | 13248, 13159, 27187 | 706.3, 679.1 | 156x91 |
| joint1 +12 | 11671 | 77529, 12024, 12044 | 706.2, 679.0 | 154x92 |
| joint2 +12, joint3 +15 | 12152 | 98741, 21314, 46610 | 708.7, 678.6 | 158x91 |
| joint4 +20 | 11228 | 20871, 26203, 26438 | 706.4, 681.4 | 157x92 |

**`GRIPPER_PX = (707, 680)`, spread 2.6 x 2.8 px across all four poses.** The
loop never has to detect the gripper.

The single-cycle column is why lock-in was necessary. A first attempt
differenced one open frame against one closed frame and returned regions of
13k-75k px on the right of the image, where a person sits; only one of its four
poses found the gripper. Requiring agreement across cycles rejected 85-90% of
that contamination, because motion in the room is uncorrelated with the
command while the gripper is perfectly correlated with it.

### Detecting the target

Threshold the Lab **a\*** channel, not HSV hue. The wrapper is dark (L about
47) and hue is unstable at low lightness: the target's median hue fell three
degrees outside a 170-179 red band and only 105 px passed.

| method | target px | rival regions | largest rival |
| --- | --- | --- | --- |
| HSV 170-179 | 105 | 0 | 0 |
| HSV 160-179 widened | 1048 | 4 | 504 |
| **Lab a\* > 140** | **938** | **0** | **0** |
| Lab a\* > 145 | 305 | 0 | 0 |

The target's a\* is 141 while the whole frame's median is 129 and its 99.5th
percentile is 138, so a threshold of 140 sits in a clean gap.

Largest-blob selection is still not enough. Skin also sits high on a\*, so a
hand in frame produces rivals, and on one run a rival briefly grew larger than
the target: the fix jumped 539 px across the image and the loop issued a 3
degree command from it before recovering. The detector now accepts only blobs
within 120 px of the previous fix, and returns nothing rather than steering
from a bad one.

### The gripper cannot be found in depth

Checked because locating both target and gripper in one image space would
avoid the unregistered depth stream. At the gripper's known colour position the
height above the fitted table plane is 0.6, -0.7 and 0.6 mm at its centre and
two jaws: the depth sensor returns the **table** there, not the gripper.
A first version of this test toggled the jaws and looked for depth changes over
8 mm, but the jaws move across the line of sight so their distance barely
changes; it reported a region in the top-right corner at 1102 mm and wrongly
declared success.

So the gripper is located in colour, as a constant, and depth is used only for
range.

### Result

From home, gain 0.35, step limit 3 degrees, tolerance 25 px:

| step | target px | error px | range |
| --- | --- | --- | --- |
| 1 | 713, 323 | 6, -357 | 303 mm |
| 4 | 716, 445 | 9, -235 | 309 mm |
| 7 | 714, 582 | 7, -98 | 306 mm |
| 10 | 718, 659 | aligned | 299 mm |

**Converged in 10 steps**, monotonically, with rivals present throughout and
correctly rejected. Final pose `[90, 120, 0, 23, 90, 90]`: joint 4 finished at
23 degrees where the Jacobian predicted 24, and joint 1 never moved. Two runs
ended at the same joint 4 value.

Alignment is not contact. The target ends on the line through the gripper but
still 299 mm from the camera, while the gripper sits about 112 mm from it.
Closing that gap is a third degree of freedom, and the range floor
(`--min-range`, default 180 mm) exists because aligning tilts the wrist down
and carries the gripper toward the table. The approach stage is not
implemented.

## Reading the gripper from the camera

The gripper's angle cannot be recovered the way joints 1-4 can, since it does
not move the camera. It is nonetheless plainly visible: the two jaws are dark
shapes along the bottom of the frame near x 490 and x 930, either side of the
fixed gripper pixel, and that position does not shift when joints 1-4 move.

So it can be measured directly. `pc_client/gripper_state.py` does this.

### Which feature works

The gap between the outermost dark regions in the strip does **not** work: it
correlates only +0.46 with the command, because the strip also contains a
keyboard edge, cables and shadows, giving three to seven dark blobs whose
extremes jump around.

Dark **area** in the strip does work:

| | |
| --- | --- |
| correlation with command, 30-120 band | **+0.962** |
| sensitivity | **+97 px per degree** |
| above 120 | saturates, jaws fully in view |
| hysteresis, closing against opening | 8-14% |

The hysteresis matches the backlash the arm joints show, so a reading is always
approached from the same direction.

### Grasp detection

Absolute area depends on the strip's background and therefore shifts whenever
the arm moves. The robust quantity is the difference at one pose, where the
background cancels:

```text
grasp_signal = area(closed) - area(open)
```

On air, six cycles at one pose:

| state | mean | sd | range |
| --- | --- | --- | --- |
| open at 30 | 5927 px | 16 px | 45 px |
| closed at 120 | 13714 px | 9 px | 28 px |
| signal | 7786 px | 9 px | |

**Repeatability is 9 px, which is 0.09 degrees of jaw travel.** A three-sigma
deviation is 0.28 degrees. A sweet is tens of degrees thick, so grasp
detection has an enormous margin.

A later probe at the same pose in different light gave open 5628 px and closed
13420 px, both shifted by roughly 300 px, while the signal came out at 7798 px
against the stored 7786. The background really does cancel.

## Picking the sweet up

Done on hardware. Alignment, descent, grasp, lift, release.

| stage | result |
| --- | --- |
| visual alignment | 7 steps, target at 265 mm |
| descent | **20 waypoints, target never lost**, gripper 199 -> 49 mm above the table, target range 265 -> 123 mm |
| final creep | 5 half-steps, target range inside the gripper's own 112 mm |
| clamp and lift | **held** |
| lower and release | **released** |

### The tracker needed a motion model

A first descent attempt lost the target immediately. The detector gated new
detections within 200 px of the previous one, which had fixed an earlier
rival-blob failure, but one descent waypoint moves joints 2 and 3 by 17 and 11
degrees and so shifts the image by about 412 px. Every correct detection was
rejected.

The fix is `predict_target`: apply the full four-column Jacobian to the joint
change about to be commanded, and gate around the predicted pixel rather than
the last observed one. That keeps the gate tight enough to still reject the
539 px rival jump that broke the alignment loop, while following arbitrarily
large commanded motion. Across the 20 descent waypoints, predictions landed
within about 20 px of observations throughout.

### Depth runs out before contact, and the model stops short

Two things the approach had to work around.

The FK goal, computed from the target's vision-estimated position, stops about
20 mm high, because that estimate sits that far above the real object. The
stopping rule is therefore the gripper's measured height above the fitted table
plane, which tracked 199 mm down to 49 mm and agreed with FK to 14.5 mm.

Below roughly 120 mm the depth stream stops being usable, and at the final pose
the depth frame was **0% valid** across the whole image, not merely near the
target. Most likely the wrist occludes the depth module at that posture. So the
last few millimetres are necessarily open loop on range, guided by the colour
image alone.

Ranging by blob area was tried as a substitute and **does not work**: over a
262-290 mm sweep, `area * Z^2` scattered by 15% and a free exponent fit gave
n = -0.54 where the inverse-square law needs -2.00. At nearly equal range the
area read 589 px and 881 px. The wrapper is specular, so the a* mask catches a
fraction that changes with viewing angle.

### Joint 2 cannot close the gap by itself

Measured: joint 2 changes the target's range by only about 1.4 mm per degree at
the aligned pose, so closing 187 mm would need 134 degrees and the travel does
not exist. Reaching requires a large posture change, which is also why a
Jacobian linearised at the aligned pose cannot get there. FK found the grasp
pose 102 degrees away on joint 2 alone.

### How the grasp was confirmed

Not by the dark-area signal. At the grasp pose it read 11008 px against an
empty-air reference of 7786, larger rather than smaller, because the strip's
background had changed completely. `verdict()` correctly reported "re-measure
the reference" instead of claiming anything. **The dark-area reference does not
transfer across large posture changes**; it has to be re-taken at the working
pose.

The reliable test is kinematic, and it works in both directions. Lifting moved
joint 2 by +87 degrees and joint 3 by -52, which by the Jacobian displaces a
stationary object by (-8, +364) px, taking it off the bottom of a 720-row
frame. The target moved (+9, -20) px and stayed within 79 px of the gripper
pixel: held.

Releasing inverted the test. Lowering with the sweet held moved joints 2 and 3
by -72 and +42 degrees, predicting -318 px for a stationary object; the target
moved 1 px. Opening the jaws then jumped it from (699, 600) to (864, 689) with
its area falling from 7135 to 2535, and on retraction it finally drifted as
predicted, 58 px from the prediction at the first step and out of frame after
that. Exactly what a sweet sitting on the desk should do.

## Grip force, and what the readout can really see

There is no force or current feedback on this arm. `ArmJoints` carries angles
only, so a servo told to reach 180 on an object that stops it at 120 keeps
pushing with its stall torque. That is what flattened the chocolate on the
first successful grasp, where the clamp was commanded straight to 180.

Grip force is therefore set by how far past contact the command goes, which
means contact has to be found. `gripper_state.py --mode grip` closes in
increments and watches the dark-area readout: free jaws move it at a measurable
rate, blocked jaws stop moving while the command keeps rising.

### Every constant describing the readout is pose dependent

This was the surprise. The earlier characterisation was taken at pose
`[90, 120, 0, 23]` and gave a usable band of 30-120 at about 97 px per degree.
Measured again at the home pose:

| joint6 | 30-50 | 55 | 65 | 90 | 105 | 110 | 115 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| area px | **0** | 122 | 996 | 5803 | 7714 | 7938 | 7956 | 7934 |

The jaws do not register at all below 55, the rate is about 161-197 px per
degree rather than 97, and the reading saturates at 110 rather than 120. So the
band, the slope and the saturation angle are all pose and lighting dependent,
which is broader than the earlier note that only the absolute reference fails
to transfer.

`--mode calibrate` now measures all three on empty jaws at the working pose and
prints the flags to hand to `--mode grip`. At the home pose it reported:
visible from 60 degrees, free rate 197 px per degree, saturation at 110.

### The false grip on empty jaws

A first version of the contact test used the single global 97 px per degree and
started judging from 30 degrees. At the home pose the reading there is flat
zero, so the increments were zero, which the test read as a collapse: it
reported contact at 35 degrees and a firm grip on empty jaws.

Two fixes. Nothing is judged until the area clears a visibility floor, because
an increment of zero means "cannot see", not "blocked". And saturation is
distinguished from contact using the calibrated ceiling, since the two look
identical in this signal; a collapse at or above the ceiling is reported as
closed on nothing.

The negative control now passes. On empty jaws at the home pose the sweep
labels 35-55 as not visible, 60-105 as free at 40-109% of the free rate, sees
the first weak increment at 110, which is the ceiling, and concludes that the
jaws closed on nothing, leaving them open.

Still to test: the positive case, which needs an object between the jaws.

## Jaw aperture, and how to set grip force

The first successful pick flattened the sweet, because the clamp was commanded
to 180. The second attempt commanded 120 instead and failed to hold anything.
Both failures have the same root cause, which is that 120 was being treated as
"closed" when it is not.

The aperture was therefore measured against the command, by tracking the two
jaws as the two largest dark regions across the full frame width:

| joint6 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap mm | 71.7 | 66.0 | 60.0 | **53.8** | 46.8 | 39.3 | 31.3 | **24.1** | 15.9 |

Linear, **0.701 mm per degree**. At 180 the two jaws merge and cannot be told
apart optically, so the fit is not extrapolated past 175.

So at 120 the jaws are still **54 mm apart** and never touch a 25 mm sweet,
while 180 closes past 14 mm and compresses it by nearly half. Inverting the
fit:

| aperture wanted | joint6 |
| --- | --- |
| 30 mm | 152 |
| 25 mm | 159 |
| 20 mm | 166 |
| 15 mm | 173 |

`gripper_state.grip_command(width_mm, squeeze_mm)` does this. Force is set by
`squeeze_mm`, since there is nothing else to set it with.

### This also rules the optical contact search out

A 25 mm object first meets the jaws near 159 degrees. The dark-area readout
saturates near 110, because by then the jaws are simply fully in view. Contact
happens where the signal has already stopped responding, so no amount of
tuning makes that search work for objects in this size range. It is not merely
fragile; it is looking in the wrong half of the travel.

The readout had already failed in five other ways: a pose-dependent visible
band, a pose-dependent rate, saturation being indistinguishable from a
blockage, contamination by whatever is held between the jaws, and the jaws
leaving the frame entirely below about 70 degrees. A sixth: two static dark
blobs at the frame edges, the monitor bezel at x=73 and a chair at x=1194, are
larger than the jaws and win any "two largest regions" contest unless excluded
by position.

### Second pick, gently

With the clamp at 162, an aperture of 23 mm, the same sequence ran again and
the wrapper came back intact.

| lift step | target moved | if stationary |
| --- | --- | --- |
| [93, 33, 53] | 7 px | 108 px |
| [93, 54, 40] | 2 px | 81 px |
| [93, 90, 20] | 3 px | 175 px |

And the set-down, which is the same test inverted:

| stage | target moved | if stationary |
| --- | --- | --- |
| lowering, still clamped | 2, 3, 3, 4 px | 175, 81, 41, 21 px |
| jaws opened | jumped (680, 662) to (631, 643) | - |
| retracting, released | 14 px from prediction | then off-frame as predicted |

The kinematic test is the one to trust for confirmation. It needs no
calibration that can drift, only the Jacobian, and it gave a margin of one to
two orders of magnitude in both directions.

## The whole sequence, as one tool

`pc_client/pick_and_place.py` runs align, descend, creep, grip, verify, place
and home. Verified end to end on the sweet, with the grip set from the aperture
calibration rather than slammed shut.

| phase | steps | outcome |
| --- | --- | --- |
| align | 1 | target already within tolerance |
| descend | 20 | gripper 178 -> 47.9 mm above the table, range 238 -> 120 mm, target never lost |
| creep | 5 | depth dropped out after step 2, as expected below 120 mm |
| grip | 1 | joint6 162, aperture 23.0 mm, for a 28 mm object squeezed 5 mm |
| lift | 3 | moved 13, 2, 2 px where a stationary object would move 108, 81, 175 |
| lower | 3 | moved 1, 1 px against 175, 81: still held |
| release | 3 | 91 px against 81 expected, then out of frame: released |

The two verdicts are the same test run in opposite directions, and both gave
one to two orders of magnitude of margin.

Object size is reported during align. Height above the fitted plane is
trustworthy, reading 26.1-26.6 mm for a sweet measured at 28 mm. The colour
blob's apparent width is **a lower bound only**: the a* mask covers just the
red enough part, and the same sweet came out at 18.2 mm, which fed to
`grip_command` would ask for a 13 mm aperture and crush it. It is printed to
check the detection, not to set the grip.

Two rough edges left in the tool, both noted in the code. The FK grasp pose
`GOAL_J2, GOAL_J3 = 18, 62` was computed for a target at roughly 265 mm from
the aligned pose, so a target at a very different distance needs it recomputed.
And the descent ladder assumes the object sits on a flat table that the depth
plane fit can find.

## Generalising beyond the red sweet, and checking before moving

Three things the first version of `pick_and_place.py` could not do: work on
anything but a red object, tell you up front that a run would fail, or run any
faster than one cautious waypoint at a time.

### Detection is split, because geometry forces it

`--detector table` fits the table plane and takes whatever stands proud of it.
Colour-agnostic, and it measures real size, which the colour blob cannot: on the
sweet it returned 29-30 mm against a true 28, where the a* mask read 11-19 mm
because it only covers the sufficiently red part. `--auto-width` feeds that
measurement to the grip calibration.

Live comparison on the same sweet:

| | position | measured width |
| --- | --- | --- |
| colour, a* > 140 | (723, 538) | 18.9 mm, a lower bound |
| depth, table plane | (728, 545) | 29 mm |

The two agree to within 10 px, so the depth-to-colour mapping is far better
than the 33 px of parallax a 12 mm baseline would imply. Depth must be
partly registered despite `depth_registration` reading false.

**But the table detector cannot track.** Depth stops being usable below about
120 mm and the gripper sits a fixed 111.9 mm from the camera, so *a held object
is permanently inside the depth dead zone*. A run that tried to track with
depth lost the target at the second creep step, gripped nothing, and could not
verify anything:

```text
=== creep ===
   1 [97, 15, 64, 17]   height   45.8 range   122 target  700,647
   2 [97, 12, 66, 18]   target lost; stopping here
=== lift and verify ===
  [97, 33, 53, 18]      not found     697,728                --
  -> not held during the lift
```

So `table` acquires and measures, then hands off to colour tracking, and
preflight refuses the run when colour cannot see the object, because then
nothing can track it through the grasp. A genuinely non-red object needs an
appearance tracker, which is not implemented and is the honest gap here.

### Does the keyboard interfere?

No, and with a large margin: it measures 201 x 188 mm against a 60 mm size
limit. But the size filter is thinner than that makes it sound, and it was not
doing as much of the work as it appeared to.

Measured on the desk, the size filter alone left **six** candidates:

| area | size mm | top mm | colour px | verdict |
| --- | --- | --- | --- | --- |
| 39012 | 200.7 x 188.3 | 35.2 | (288, 200) | too big, the keyboard |
| 4064 | 46.6 x **61.3** | 29.4 | (58, 646) | too big, by 1.3 mm |
| 771 | 20.0 x 53.6 | 20.3 | (18, 225) | passes |
| **699** | **18.5 x 34.6** | 27.9 | **(696, 438)** | **passes, the sweet** |
| 197 | 22.0 x 13.2 | 10.8 | (26, 312) | passes |
| 125 | 8.1 x 14.5 | 17.0 | (9, 415) | passes |

Note where the spurious passers are: x = 9, 18, 25, 26, 64. All at the extreme
left edge, so they are the monitor and the frame border, not objects on the
table. The sweet was winning only on "nearest the frame centre", which is a
convention rather than a measurement: a mug placed centrally would have won
instead. And the 61.3 mm object missed the size cut by 1.3 mm.

So `WORKING_AREA` was added, accepting only the central 60% of the frame:

```text
objects on the plane      : 8
after the size filter     : 6  at [(64,661), (19,228), (25,567), (695,438), (10,299), (7,415)]
after the working area too: 1  at [(695,438)]
```

Six candidates down to one, excluded by construction rather than by a tie
break, and the one left is the sweet: the independent colour detector put it at
(704, 427) against the depth detector's (695, 438), nine pixels apart.

### What actually generalised

Only acquisition and sizing. Tracking did not, and that bounds the whole
feature: a genuinely non-red object can be found and measured but not followed
through the grasp, so preflight refuses the run. The gap is an appearance
tracker, and it is not implemented.

### What changed for the original red-sweet task

Very little, deliberately. With default flags the detection call is the same
`reach_candy.find_target`, and align, creep, grip, travel and place are
untouched. `descend` differs only in reading the waypoint count from an
argument whose default is the previous constant, and every settling wait is
multiplied by a factor that defaults to 1.0, so the timing is identical. The
one behavioural change is that preflight now runs first and can refuse to
start.

### Preflight

Eleven checks before anything moves, each printed with its evidence:

```text
  [pass] colour stream                frames arriving
  [pass] depth stream                 frames arriving
  [pass] target detected              at (728, 544), area 856
  [warn] no rival detections          7 other candidate(s)
  [pass] table plane found            gripper 175 mm above the table, 8496 inliers
  [pass] target range known           257 mm
  [pass] range within the solved band 257 mm against 200-340 mm
  [pass] object fits the jaws         29 mm measured from depth, jaws take 8-60 mm
  [pass] object stands off the table  27.8 mm tall
  [pass] detectors agree              independent colour fix differs by (-5, +10) px
  [pass] grip aperture sane           joint6 161 for a 29 mm object, aperture 23.7 mm
```

Two notes on what these do and do not prove. The range band is **not** a
workspace check: a real one needs forward kinematics at run time, which this
tool does not carry, so the band is simply the range the FK descent path was
solved over. And the detector-agreement check deliberately uses an
*independent* colour fix; an earlier version compared the chosen detector
against itself and passed by construction, which is worse than no check.

`--width-mm 200` fails it cleanly:

```text
  [FAIL] grip aperture sane           joint6 30 for a 200 mm object, aperture 115.5 mm
  1 check(s) failed: grip aperture sane
not moving. Fix the failures, or pass --skip-preflight.
```

### Speed

`--descent-steps` and `--settle` are now parameters, with `--fast` as
shorthand for 10 steps at 0.6x settling, which roughly halves the run. Verified
end to end at that setting with the table detector and `--auto-width`: aligned
in 5 steps, descended 10 waypoints from 184 to 55.4 mm, gripped at joint6 161
for a 29 mm object, and both verdicts came out clean.

| stage | target moved | if stationary |
| --- | --- | --- |
| lift | 16, 2, 4 px | 108, 81, 175 px |
| lower, still held | 3, 4 px | 175, 81 px |
| release | 109 px | 81 px, then out of frame |

Fewer waypoints costs monitoring quality rather than accuracy: each step moves
the image further, so the tracker has more to predict and the plane fit sees
more motion blur. Below about 0.5x settling the arm is still moving when it is
measured.

## Colour as a parameter, and five bugs it exposed

`--colour` takes a name, a literal Lab direction, or `@x,y` to sample the
colour off the object. A colour becomes a direction in the Lab a*-b* plane and
the threshold is the projection onto it, so one number means the same thing
whichever colour is asked for.

### Getting the directions right

The first set was guessed and wrong. Pure blue is a* = +79, b* = -108, so it
leans towards red on the a* axis rather than sitting at (0, -1), and a
best-of-eight-directions test mismatched five of the eight colours. Taking the
directions from the actual Lab chroma of saturated sRGB fixed all eight.

`red` is kept on the bare a* axis because that is what was validated on the
sweet, and the true red direction would break it: the wrapper's chroma is
(13, 0), which projects to 13 on the a* axis but only 10.0 on (0.767, 0.642),
under the threshold of 12. The cost is that the a* axis is really a "not green"
test, and with a biscuit bar on the table it grabbed the bar's orange printing
instead of the sweet, 7697 px against the sweet's 814. `red-sat` is the true
direction and, on that scene, picked the sweet with a margin of +15 where plain
`red` had +2.

Measured on one frame with both objects present:

| colour | target | area | chroma | background | margin |
| --- | --- | --- | --- | --- | --- |
| red | (630, 504) | 10504 | 29 | 9 | +20, the biscuit print |
| red-sat | (793, 477) | 774 | 21 | 6 | +15, the sweet |
| blue | (629, 499) | 15307 | 69 | 19 | +50, the biscuit |
| sampled @610,470 | (629, 499) | 15554 | - | - | the biscuit |

### Five things this broke, all now fixed

**The margin was measured at the centroid.** On a concave blob, such as
printing wrapped round a bar, the centroid lands off the object: it reported a
chroma of 0 for a blob that had clearly passed the threshold. Now measured over
the blob's own pixels.

**The background baseline included the target.** A blue biscuit filling
15000 px gave a target chroma of 69 against a "background" of 68, a margin of
+1, because the biscuit was most of what the 99.5th percentile measured.
Excluding the chosen blob turns that into +50.

**Both jaw axes had to fit.** That rejected a 33 x 83 mm bar the jaws can hold
across its width. The jaws close on the narrower axis, so that is the dimension
that must fit, with a generous ceiling on the long axis that still keeps the
keyboard out at 201 x 188 mm.

**Preflight and the tracker disagreed about which object they meant.**
Preflight chose by geometry alone, so with both objects present it measured the
biscuit at 16 mm while `--colour red-sat` tracked the 28 mm sweet, and the stop
height and `--auto-width` were set from the wrong one. Both now share one
selection path: colour first, then size and position.

**`--auto-width` asked for a crushing grip.** The object mask is everything
more than 6 mm above the table, which on a sphere captures only the upper cap,
so the narrow axis underestimates: the sweet measured 21 mm across while
standing 27 mm tall, and gripping it as 21 mm asks for a 15.3 mm aperture,
about what flattened one at 180. An object resting on a table has its height as
an independent estimate of its diameter, so the larger of the two is used. That
gives joint6 163 and a 22.3 mm aperture, the grip proven to hold the sweet
unmarked.

### The biscuit still cannot be picked up, and why

Two attempts failed the same way: the creep ran joint 2 down to its 0 travel
limit and the jaws closed roughly 20 mm above a 15.9 mm bar. Making the stop
height follow the object, at its top plus 8 mm, did not help. The FK descent
path `GOAL_J2, GOAL_J3 = 18, 62` was solved for a 28 mm object at 265 mm and
its endpoint is simply too high for a flat one. Reaching flatter objects needs
the goal re-solved per target, which means carrying forward kinematics at run
time.

A check built on this got it wrong first, and that is worth recording. It
claimed a "reachable floor" of 38 mm, taken from the lowest gripper height any
descent ever reported, and then refused the sweet that had already been picked
up twice. The error: depth stops returning anything below about 120 mm of
range, so the height reading disappears while the arm is still descending.
38 mm was where measurement stopped, not where motion stopped. The check is now
an honest non-fatal warning carrying the two data points it actually has:
28 mm grasped, 15.9 mm failed twice.

### Where it stands

| object | preflight | run |
| --- | --- | --- |
| 28 mm sweet, `--colour red-sat --auto-width` | all clear | held and replaced, target moving 1 px per lift step against 108, 81 and 175 expected |
| 15.9 mm biscuit bar, `--colour blue --auto-width` | two warnings | jaws close above it |

## What this means for reinforcement learning

Usable today, without touching the firmware, for **visual servoing**: the
camera sees the target, the policy commands joints, the arm executes
accurately, the camera sees the result. The loop closes without joint angles.

Not usable for:

- **Joint 5's angle**, which nothing here observes.
  The gripper's opening, however, now *is* measured, from dark area in a fixed
  strip at the bottom of the frame; see the section above. An earlier revision
  of this file called grasp confirmation impossible, which conflated "does not
  move the camera" with "cannot be seen".
- **Stall and collision detection**, which needs servo feedback. Safety stays
  a human-in-the-loop matter, and `m3pro_arm_safety` should keep
  `require_feedback: true`.
- **Control above about 3 Hz** over rosbridge, given 416 ms to 95% at
  `time=200`. Run the policy on the Jetson to beat that.

Backlash is the same size as the action resolution, so a naive one-degree
action space carries as much unmodelled error as its own step whenever the
policy reverses direction. Prefer 2-3 degree steps, or keep the previous action
in the observation so the policy can learn the hysteresis, or approach from one
direction only.

For training, real-hardware pixel RL from scratch is impractical at 2-3 Hz.
The figures above are exactly what a simulator needs: the URDF and a MoveIt
config already exist, so inject 1.0 degree of backlash, 300 ms of latency,
300-500 ms control periods and 0.3 degrees of amplitude error, train there, and
transfer. Alternatively collect demonstrations with `teleop_view.py` and clone
them before fine-tuning on hardware.

## Reproducing

```powershell
python .\pc_client\arm_diagnostics.py --test coupling
python .\pc_client\arm_diagnostics.py --test sweep
python .\pc_client\arm_diagnostics.py --test step
python .\pc_client\arm_diagnostics.py --test hold
python .\pc_client\arm_diagnostics.py --test jacobian
```

These move the arm. Re-run them after any firmware change or mechanical work.
