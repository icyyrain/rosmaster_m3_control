# ROSMASTER M3 Pro arm extension

This extension is deliberately separate from Yahboom's workspaces. The current
stage provides message definitions, a state bridge, a fail-closed command mux,
rosbridge launch files, Windows clients, and a separately documented
experimental MCU patch. Nothing has been flashed.

## Safety defaults

- The command mux starts in `disabled` mode.
- Real calibrated feedback is required before a control mode can be enabled.
- Raw-count fallback calibration starts with `calibration_valid: false`; the
  default degree-feedback path still needs physical zero/direction validation.
- The PC command utility is a dry run unless `--execute` is supplied and the
  operator types `MOVE`.
- The existing joystick still publishes directly to `/arm6_joints`; do not run
  algorithmic control until the joystick publisher has been stopped or remapped.

## Jetson build

```bash
cd /home/jetson/m3pro_ext_ws
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Do not add this workspace to `.bashrc` or autostart until feedback calibration
and command arbitration have been verified.

## Run without motion

Start rosbridge only:

```bash
source /home/jetson/m3pro_ext_ws/install/setup.bash
ros2 launch m3pro_arm_bringup rosbridge.launch.py
```

The arm stack can also be launched safely before MCU feedback exists. The
command mux remains disabled; the bridge will wait for
`/arm6_joints_feedback` and therefore publishes no `/joint_states` until real
data arrives:

```bash
ros2 launch m3pro_arm_bringup arm_stack.launch.py
```

## Windows client

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\pc_client\requirements.txt
python .\pc_client\monitor.py
```

`safe_command.py` is intentionally dry-run by default:

```powershell
python .\pc_client\safe_command.py --joints 90 90 90 90 90 90 --time-ms 200
```

## MCU feedback prototype

`firmware/m3pro_arm_feedback` now contains a compile-checked patch against
Yahboom's official `Subscriber_uart_servo` STM32H743 example. It:

1. Queries UART3 present position for servo IDs 1 through 6.
2. Checks response header, ID and checksum, with a two-millisecond timeout.
3. Polls one joint per ROS task iteration.
4. Publishes a completed sweep as `arm_msgs/msg/ArmJoints` on
   `/arm6_joints_feedback`.

Do not treat the compiled sample as production firmware: it is not the complete
factory `YB_Node`. Obtain the factory source and port the patch when possible.
For a temporary hardware proof, first establish an ST-LINK recovery path and
verify the downloaded factory rollback HEX checksum. After real readings arrive,
validate each joint's direction, physical zero, usable limits, update
`position_offset_deg`/`direction`, and only then enable command arbitration.
