# ROSMASTER M3 Pro arm extension

This extension is deliberately separate from Yahboom's workspaces. The current
stage provides message definitions, a state bridge, a fail-closed command mux,
rosbridge launch files, and Windows clients. It does not contain or flash MCU
firmware.

## Safety defaults

- The command mux starts in `disabled` mode.
- Real calibrated feedback is required before a control mode can be enabled.
- Calibration starts with `calibration_valid: false`.
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

The arm stack can also be launched safely before MCU feedback exists. It will
remain disabled and will not publish `/joint_states` while calibration is false:

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

## MCU work still required

Locate the STM32 factory-firmware source containing `Arm_Set_Angle()` and add:

1. A UART3 position-query function for servo IDs 1 through 6.
2. Response checksum and timeout handling.
3. A bounded round-robin poller, initially at 10 Hz for all six joints.
4. A micro-ROS publisher on `/arm/servo_states_raw` using
   `m3pro_arm_msgs/msg/ArmServoState`.

After receiving raw data, calibrate each joint independently and only then set
`calibration_valid: true` in `arm_bridge.yaml`.
