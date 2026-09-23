# M3 Pro control-board arm feedback experiment

This directory contains a reviewable patch against Yahboom's official
`Microros_Samples/Subscriber_uart_servo` STM32H743 example. It is deliberately
kept separate from the downloaded vendor tree.

## What the patch changes

- reads the present-position register (`0x38`, two bytes) from one bus servo per
  ROS task iteration;
- publishes a six-joint sweep as `arm_msgs/msg/ArmJoints` on
  `/arm6_joints_feedback`;
- uses `ArmJoints.time` as a feedback sweep sequence number (`0..32767`), not as
  a movement duration on the feedback topic;
- retains `/arm_joint` and adds `/arm6_joints` command subscriptions;
- fixes the official example's RX-header validation (`&&` to `||`);
- selects blocking UART receive because the example enables interrupt receive
  without implementing `HAL_UART_RxCpltCallback`;
- fixes `Arm_Set_Snyc_Buffer`, which otherwise ignores all six input values.

A failed or malformed servo read is published as `-1` for that joint. The
Jetson bridge rejects invalid/range-violating readings and publishes `NaN` in
the corresponding `sensor_msgs/msg/JointState.position` entry.

## Base artifacts and integrity

The vendor downloads are intentionally excluded from Git. Their expected
SHA-256 values are in [SOURCE_HASHES.txt](SOURCE_HASHES.txt). The patch base is:

`Board_Samples.zip/Microros_Samples/Subscriber_uart_servo`

Apply from the extracted sample root:

```bash
git apply --check /path/to/0001-add-arm-position-feedback.patch
git apply /path/to/0001-add-arm-position-feedback.patch
```

## Important deployment boundary

This is based on Yahboom's *servo subscriber sample*, whose micro-ROS node is
`YB_Example_Node`. It is not the complete factory `YB_Node` source. Flashing it
would temporarily replace the factory control-board program and may remove
chassis, odometry, IMU, battery and other factory functions.

Do not flash this image merely because it compiles. First verify all of the
following:

1. The factory rollback HEX is present and its checksum matches
   `SOURCE_HASHES.txt`.
2. A working UART or SWD recovery path is physically available.
3. The robot is supported so the arm cannot collide with the chassis or desk.
4. The sample is first tested with torque/motion risk controlled.

Yahboom's M3 Pro control-board course documents a serial recovery route: connect
the control board to Windows using its Type-C data port, select UART in
STM32CubeProgrammer, hold **BOOT**, press **RESET**, then release **BOOT** before
connecting/programming. The serial port must not be held by micro-ROS or another
program. Official instructions:

<https://www.yahboom.net/public/upload/upload-html/1755253798/4.Burning%20STM32%20firmware%20using%20serial%20port.html>

The official Yahboom Drive copy of STM32CubeProgrammer 2.19 for Windows has
also been downloaded locally as
`vendor/yahboom_m3pro_official/tools/en.stm32cubeprg-win64-v2-19-0.zip`; its
checksum is recorded in `SOURCE_HASHES.txt`. The `vendor` directory is excluded
from Git because these third-party packages are large.

STM32CubeProgrammer 2.19 is installed per-user at:

```text
%LOCALAPPDATA%\STMicroelectronics\STM32CubeProgrammer
```

The CLI reports 2.19.0 and runs. The optional ST-LINK driver was not installed;
it is not required for the documented CP2104/UART route. Use the repository
wrapper:

```powershell
.\stm32-programmer.cmd -l uart
```

The running robot identifies the control-board link as Silicon Labs CP2104
USB-to-UART (`10c4:ea60`, serial `02C4DDB5`) and maps it to
`/dev/myserial -> /dev/ttyUSB0`. The factory micro-ROS agent normally owns this
port at 2,000,000 baud, so it must be stopped before entering the ROM bootloader
or attempting any programmer connection.

On the assembled robot inspected on 2026-09-23, the only externally visible
Type-C ports were the Jetson device-mode port and the `YB-MAE02-V1.0` voice
module. The latter enumerates as CH340K (`1a86:7522`) and is **not** the motion
control board. The CP2104 control-board connection remains internal through the
Jetson USB hub.

`tools/probe_stm32_bootloader.py` was run on the Jetson after temporarily
stopping the agent. Four common DTR/RTS auto-reset/BOOT0 polarities produced no
STM32 ROM bootloader ACK; the original MCU application and ROS nodes recovered
normally. This board therefore cannot currently be put into the ROM bootloader
through CP2104 modem-control signals alone. Physical access to the control
board's BOOT0 and RESET buttons is still required unless a documented software
bootloader entry mechanism is found.

Before the first write, read and save the currently installed flash if the
programmer permits it; then verify that the official rollback HEX can at least
be opened and that the board's BOOT and RESET buttons are accessible. The
preferred production route is still to obtain the factory `YB_Node` STM32 source
from Yahboom and port the same small feedback loop into it. The sample build is
useful as a hardware/protocol proof, not yet as the final robot firmware.

## Expected ROS 2 interface

With the patched test firmware and the serial micro-ROS agent running in domain
30:

```bash
ros2 topic echo /arm6_joints_feedback arm_msgs/msg/ArmJoints
ros2 topic hz /arm6_joints_feedback
```

The current one-servo-per-loop design should produce approximately 10–16 full
six-joint sweeps per second. Measure the real rate before using it in a control
loop. This is suitable for initial state-observation experiments, but a serious
real-time RL controller will likely need higher-rate firmware and an on-robot
policy process rather than a Windows/rosbridge round trip.
