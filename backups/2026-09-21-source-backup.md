# Jetson source backup - 2026-09-21

- Robot: ROSMASTER M3 Pro, Jetson Orin NX
- Robot address: `192.168.2.4`
- Archive: `backups/remote/2026-09-21/m3pro_source_backup_20260921.tar.gz`
- SHA-256: `284a250e440c1d4fc3dea8c234158539638f692198c6c0228f75d9e24ebf7f83`
- Archive size: approximately 31 MiB

Included:

- `/home/jetson/yahboomcar_ws/src`
- `/home/jetson/M3Pro_ws/src`
- selected `/home/jetson/WORK` source and scripts
- `/home/jetson/m3pro_ext_ws/src`
- `/home/jetson/start_agent.sh`
- `/home/jetson/.config/autostart`
- `/home/jetson/oled_yahboom`
- `/etc/udev/rules.d/usb_serial.rules`
- `/etc/systemd/system/yahboom_oled.service`

Excluded to keep the source snapshot small:

- `yahboom_yolov8`
- `OrbbecSDK_ROS2`
- `yahboomcar_mediapipe`
- large camera SDK copies, images, and existing archives under `/home/jetson/WORK`
- ROS 2 build/install/log outputs

This is a Jetson-side source/configuration backup. It is not an STM32 flash
image. Obtain the matching factory STM32 project and rollback binary before
flashing the control board.

