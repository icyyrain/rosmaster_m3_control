# ROSMASTER M3 Pro 接入与控制工程

这份 README 同时作为新会话的交接说明。当前目标是让 Windows 电脑上的程序通过
ROS 2/rosbridge 与 ROSMASTER M3 Pro 交互，最终用于“读取真实机械臂状态 → 强化
学习算法计算动作 → 安全地下发动作”的闭环控制。

## 一分钟连接信息

- 小车型号：Yahboom ROSMASTER M3 Pro
- 小车主机：Jetson Orin NX，Ubuntu 22.04，ROS 2 Humble
- 目前确认可用的有线 IP：`192.168.2.4`
- 同事提供的备用地址：`192.168.2.3`
- SSH 用户名：`jetson`
- SSH 密码：`yahboom`
- ROS 域：`ROS_DOMAIN_ID=30`
- 控制板串口：`/dev/myserial -> /dev/ttyUSB0`
- USB 串口芯片：Silicon Labs CP2104，序列号 `02C4DDB5`
- 深度相机：Orbbec DaBai DCW2，话题命名空间 `/high_camera`，驱动需手动启动
- 相机**装在机械臂 `arm4` 连杆上**（eye-in-hand），画面是关节 1–4 的函数；
  关节 5 和夹爪在 `arm4` 下游，不影响画面
- 实测的机械臂特性（回差、延迟、死区等）见 `docs/MEASUREMENTS.md`
- Windows 有线网卡：`Realtek Gaming 2.5GbE Family Controller`
- Windows 直连小车时的地址：`192.168.2.10/24`

小车端主要工作区：

- `/home/jetson/yahboomcar_ws`：原厂底盘、机械臂和 micro-ROS 主工作区，不要直接修改。
- `/home/jetson/M3Pro_ws`：导航、SLAM、语音/大模型相关代码。
- `/home/jetson/WORK`：相机、目标跟踪和抓取等实验代码。
- `/home/jetson/m3pro_ext_ws`：本项目新增的独立机械臂扩展工作区。

已确认的原则：不要随意改动车内原厂代码；需要修改或烧录 STM32 固件前，先明确
说明修改内容并与设备负责人确认。

## 从 Windows 连接小车

1. 用网线把电脑与小车连接。
2. 双击 `car-mode.cmd`，以管理员权限把指定网卡设置为 `192.168.2.10/24`。
3. 验证网络：

```powershell
ping 192.168.2.4
```

4. SSH 登录：

```powershell
ssh jetson@192.168.2.4
```

出现密码提示后输入 `yahboom`。如果 `.4` 不通，可以在确认网卡配置正确后再尝试
`192.168.2.3`。

返回公司内网前，双击 `company-mode.cmd` 恢复 DHCP。`network-status.cmd` 只查看
配置，不会修改网络。

## 登录后的 ROS 2 环境

每个新的 SSH 终端建议先执行：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ws/install/setup.bash
source /home/jetson/m3pro_ext_ws/install/setup.bash
export ROS_DOMAIN_ID=30
```

只读检查命令：

```bash
ros2 node list
ros2 topic list
ros2 topic info /arm6_joints
ros2 topic echo /battery
```

目前正常运行的原厂节点是：

```text
/YB_Node
/autostart_node
/joy_ctrl
/joy_node
```

如果看到这些节点，说明小车主 ROS 2 系统基本正常。当前原厂系统中：

- `/arm6_joints` 是六关节目标角度命令，不是真实反馈。
- `/arm_joint` 也是机械臂命令接口。
- `/battery`、IMU、里程计和雷达等原厂状态话题可以读取。
- 当前没有可信的真实机械臂 `/joint_states`。

## 可以怎样交互

### 1. 只读查看小车状态

优先使用 `ros2 topic list` 找到实际话题，再使用 `ros2 topic echo <话题>` 查看。
这类操作不会让机械臂运动，适合新会话首先确认网络和 ROS 状态。

### 2. 从 Windows 通过 rosbridge 读取 ROS 2

rosbridge 已安装，但没有设置成开机启动。需要时在小车 SSH 终端手动运行：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ws/install/setup.bash
source /home/jetson/m3pro_ext_ws/install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch m3pro_arm_bringup rosbridge.launch.py
```

它默认提供 `ws://192.168.2.4:9090`。保持该终端运行；结束时按 `Ctrl+C`。在 Windows
本项目目录中另开 PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\pc_client\requirements.txt
python .\pc_client\monitor.py --host 192.168.2.4
```

当前监控程序可以看到 rosbridge 连接情况和已有的 `/battery` 数据。实验反馈固件尚未
刷入，因此现在仍不会出现有效的 `/joint_states`；刷入并验证后，Jetson 桥接会把
`/arm6_joints_feedback` 转成 `/joint_states`。

注意 `roslibpy` 底层的 twisted reactor 在一个进程里只能启动一次，因此每个客户端
进程只能连接一次 rosbridge。需要同时看两路画面时，请开两个终端各跑一个进程。

扩展机械臂状态转换和安全仲裁可在另一个 SSH 终端启动：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ws/install/setup.bash
source /home/jetson/m3pro_ext_ws/install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch m3pro_arm_bringup arm_stack.launch.py
```

这个启动文件本身不会命令机械臂运动。现阶段它会报告反馈/标定未就绪，并保持控制
模式为 `disabled`。

### 3. 摄像头

小车装有 Orbbec DaBai DCW2 深度相机（USB `2bc5:0561` 彩色 + `2bc5:06a0` 深度）。
驱动没有设置自启动，需要时在小车上运行：

```bash
ros2 launch orbbec_camera dabai_dcw2.launch.py
```

话题在 `/high_camera` 命名空间下。实测（Windows 经 rosbridge 直连千兆）：

| 流 | 分辨率 | 实测帧率 | 带宽 |
| --- | --- | --- | --- |
| `color/image_raw/compressed` | 1280x720 | 29.5 fps | 3.1 MB/s |
| `depth/image_raw` | 640x360 | 9.7 fps | 4.3 MB/s |
| `ir/image_raw/compressed` | 640x400 | 10 fps | 161 KB/s |

两个容易踩的坑：

- 话题 `/rgb` 不是摄像头。它是 `std_msgs/msg/ColorRGBA`，`YB_Node` 订阅它来控制
  车身 RGB 灯带。
- `depth/image_raw/compressed` 是空的。驱动 advertise 了该话题，但每条消息只有
  88 字节，只有消息头没有数据（16 位深度无法走 JPEG）。要深度必须订阅原始
  `depth/image_raw`，它是 16UC1，像素值即毫米，可用于真实测量。

彩色原始帧约 2.7 MB，30 fps 下远超 WebSocket 承载能力，因此彩色和红外一律使用
compressed 流。点云 `depth_registered/points` 建议在 Jetson 上处理，不要拉回
Windows。

在 Windows 上实时查看：

```powershell
python .\pc_client\camera_view.py                    # 彩色
python .\pc_client\camera_view.py --stream depth     # 深度，显示中心点毫米数
python .\pc_client\camera_view.py --stream ir        # 红外
```

窗口内 `q` 退出，`s` 保存快照。

### 4. 画面加机械臂点动的联合 demo

`pc_client/teleop_view.py` 在一个窗口里同时显示相机画面和机械臂点动控制，两者共用
同一条 rosbridge 连接。默认只显示不发送：

```powershell
python .\pc_client\teleop_view.py --home
```

确认无误后加 `--execute` 才会真实驱动机械臂，并且需要在终端手动输入 `MOVE` 确认：

```powershell
python .\pc_client\teleop_view.py --home --execute
```

窗口内按键：`1`-`6` 选关节（6 为夹爪），`j`/`k` 点动，`[`/`]` 调步长，`o`/`c` 夹爪
开合，`h` 回到初始姿态，空格保持，`s` 存图，`q` 退出。

不需要画面时可以用纯终端版本 `pc_client/arm_demo.py`，按键相同。

两者共用 `pc_client/arm_control.py` 中的运动包络，关节限位只在该文件中定义一处。

因为没有关节反馈，程序只能记住自己发出的指令，无法知道机械臂实际位置。所以每次
启动必须二选一：`--home` 用 3 秒慢速走到原厂静止位 `[90, 120, 0, 0, 90, 90]` 建立
基准，或 `--assume-pose J1..J6` 由操作者声明当前姿态且不产生运动。两个都给或都不给
会直接拒绝启动。退出时会打印最后一次指令姿态，便于下次用 `--assume-pose` 接续。

`--home` 那一次运动是从未知位置出发的，使用前先目视确认机械臂当前姿态与初始位置
相差不大。另外原厂 `/joy_ctrl` 节点也在发布 `/arm6_joints`，使用期间请放下手柄，
避免两个发布者同时下发指令。

这两个工具都绕过了 `m3pro_arm_safety` 仲裁器，直接发布 `/arm6_joints`。这是有意为
之：仲裁器要求真实反馈，当前条件下必然拒绝转发。等固件刷入且 `/joint_states` 有
真实数据后，应改为经 `/arm/command/test` 走仲裁。

### 5. 视觉状态与机械臂特性测量

固件没有关节反馈，但相机装在 `arm4` 上，所以**画面本身就是关节 1–4 的观测量**。
这条通道不需要刷固件，也不需要标定。

`pc_client/visual_state.py` 用固定的 AprilTag 解算相机 6 自由度绝对位姿。先生成
并打印标记：

```powershell
python .\pc_client\make_apriltag.py --id 0 --size-mm 150
```

按 100% 打印（不要"适应页面"），用尺子核对页面上的 100 mm 参考线 —— 标记尺寸误差
会 1:1 传递到距离。贴在固定位置后：

```powershell
python .\pc_client\visual_state.py --tag-mm 150 --seconds 30
```

`pc_client/vision_metrology.py` 提供另一条精度更高的通道：用场景自身做参考，估计
全局图像变换。它是差分测量，静态噪声 0.03–0.05 px，对应千分之几度，但不给绝对角度。

`pc_client/arm_diagnostics.py` 基于它测量机械臂特性。**这些测试会让机械臂运动：**

```powershell
python .\pc_client\arm_diagnostics.py --test coupling   # 哪些关节驱动相机
python .\pc_client\arm_diagnostics.py --test sweep      # 死区、线性度、回差
python .\pc_client\arm_diagnostics.py --test step       # 阶跃响应与延迟
python .\pc_client\arm_diagnostics.py --test hold       # 到位后是否抖动
```

2026-09-23 的实测结果全部记录在 `docs/MEASUREMENTS.md`，摘要：

| 项目 | 实测 |
| --- | --- |
| 1° 指令死区 | 无，40/40 步均产生运动 |
| 回差 | **0.996°**，反向后首步只走 14–30% |
| 幅度精度 | 6° 指令 → 6.01–6.29° |
| `ArmJoints.time` | 确实控制轨迹时长 |
| 视觉闭环起动延迟 | 240–330 ms |
| 到位后保持稳定性 | ≤0.124 px（约 0.007°），不抖动 |

结论：**视觉伺服类任务今天就可做**，不必刷固件。真正的短板是夹爪不可观测（无法
确认抓取成功）和 2–3 Hz 的控制上限，两者都有不拆机的解法。详见
`docs/MEASUREMENTS.md` 末节。

### 6. 机械臂命令

原厂命令格式如下，执行后机械臂会真实运动：

```bash
ros2 topic pub /arm6_joints arm_msgs/msg/ArmJoints \
  "{joint1: 90, joint2: 90, joint3: 90, joint4: 90, joint5: 90, joint6: 180, time: 1500}" \
  --once
```

不要把这条命令当成状态读取。它只是向六个舵机发送目标角度和运动时间。在确认机械臂
周围无障碍、急停/断电方式明确、角度安全且现场有人观察前，不要执行。

本项目的 Windows 命令工具默认只打印数据，不会发送：

```powershell
python .\pc_client\safe_command.py --joints 90 90 90 90 90 90 --time-ms 200
```

即使显式加入 `--execute`，Jetson 安全仲裁器也要求真实、及时且已校准的反馈；现阶段
条件不满足，因此设计上会拒绝转发算法命令。

## 当前机械臂反馈结论

目前无法从正在运行的原厂 ROS 2 节点读到真实关节角：`/arm6_joints` 是命令话题，
原厂节点内部保存的目标值也不能证明舵机已经到达。已逐一排除其它可能：`/YB_Node`
的机械臂话题全是 Subscriber 且没有任何 service；`arm_interface/msg/CurJoints`
（"当前关节"）的所有发布者都只是 demo 节点回显自己刚算出的目标值；MoveIt 配置
`M3Pro_config` 的 18 个 `state_interface` 用的是 `mock_components/GenericSystem`，
配置文件自己的注释写着不适用于真实硬件 —— 它会把命令原样回显，很容易被误当成反馈。

现已从 Yahboom 官方附件确认，总线舵机支持读取当前位置寄存器；官方 STM32 示例也
包含位置查询代码，但未发布为 ROS 2 反馈，并且示例接收路径有缺失。

**但关节角已不再是唯一的状态来源。** 相机装在 `arm4` 上，因此画面可以观测关节
1–4，无需刷固件；这条通道和实测精度见上文第 5 节与 `docs/MEASUREMENTS.md`。
仍然缺失的是夹爪与关节 5 的状态，以及堵转/碰撞检测 —— 后者只能靠舵机反馈，
所以 `m3pro_arm_safety` 的 `require_feedback: true` 应当保持。

计划中的数据链路是：

```text
舵机位置查询
  → STM32 固件
  → micro-ROS /arm6_joints_feedback
  → Jetson arm_state_bridge
  → 标准 /joint_states（弧度）
  → Windows 强化学习程序
```

本地 `firmware/m3pro_arm_feedback` 已包含一个可审查补丁和编译通过的实验 HEX。它基于
官方 `Subscriber_uart_servo` 示例：轮询 1–6 号舵机、校验响应、超时返回 `-1`，每完成
一次六轴扫描便发布 `/arm6_joints_feedback`。同时保留 `/arm_joint` 并补上
`/arm6_joints` 命令订阅。

重要限制：官方附件没有当前完整原厂 `YB_Node` 的 STM32 源码。实验固件的节点是
`YB_Example_Node`，刷入后会临时替换原厂控制板程序，底盘、里程计、IMU、电池等功能
可能消失。原厂回退固件 `microROS_STM32-FW_V1.1.3.hex` 已下载并校验。官方资料已确认
可以用控制板 Type-C 串口、STM32CubeProgrammer 和 BOOT/RESET 键刷写。现场整机外露的
两个 Type-C 已确认分别属于 Jetson 和 `YB-MAE02-V1.0` 语音板；真正的 CP2104 主控链路
位于机内。2026-09-23 已做只读 DTR/RTS bootloader 探测，未收到 STM32 ROM ACK，说明
不能仅靠软件控制 CP2104 进入烧录模式，仍需打开主控板所在舱位接触 BOOT0/RESET。
在找到按键并先尝试读取现有 Flash 前不会刷写。详细说明见
`firmware/m3pro_arm_feedback/README.md`。

## 新增扩展工程状态

`/home/jetson/m3pro_ext_ws` 已部署并成功编译以下四个 ROS 2 包：

- `m3pro_arm_msgs`：保留一条原始计数反馈的备用消息定义。
- `m3pro_arm_bridge`：默认把 `/arm6_joints_feedback` 的实测角度转换成
  `/joint_states`，也保留原始计数兼容模式。
- `m3pro_arm_safety`：在 PC/算法指令与 `/arm6_joints` 之间做安全仲裁。
- `m3pro_arm_bringup`：提供机械臂栈和 rosbridge 的启动文件。

2026-09-21 已把支持 `/arm6_joints_feedback` 的新版桥接代码同步到 Jetson，并重新编译
`m3pro_arm_bridge` 与 `m3pro_arm_bringup`。两个节点通过短时启动检查后已退出，没有设置
自启动。

扩展节点当前没有启动，也没有写入 `.bashrc` 或开机自启动。安全默认值如下：

- 控制模式为 `disabled`。
- 角度反馈中任一关节为 `-1` 或越界时，对应 `/joint_states` 值为 `NaN`，安全仲裁会
  拒绝使用这一帧。
- 安全仲裁要求真实反馈，反馈缺失或超时即拒绝控制。
- Windows 命令工具默认为 dry-run。
- 原厂手柄节点目前仍可直接发布 `/arm6_joints`，尚未统一接入仲裁器。

后续做强化学习前，还需要：真实反馈、逐关节标定、统一所有命令入口、物理限位和急停
测试，以及先低速低幅度验证，再逐步提高控制频率。

## 网络切换脚本

双击对应文件即可切换 `Realtek Gaming 2.5GbE Family Controller`：

- `car-mode.cmd`：设置为 `192.168.2.10/24`，用于直连小车；不设置网关和 DNS。
- `company-mode.cmd`：恢复 DHCP，自动获取公司内网 IP、网关和 DNS。
- `network-status.cmd`：只查看当前配置，不修改网络。

切换时 Windows 会弹出管理员权限确认。脚本只修改指定的 Realtek 有线网卡，不修改 Wi-Fi 或其他网络接口。

也可以在 PowerShell 中运行：

```powershell
.\switch-network.ps1 -Mode Car
.\switch-network.ps1 -Mode Company
.\switch-network.ps1 -Mode Status
```

## 本机目录结构

用于真实状态反馈、安全指令仲裁和 Windows 通信的独立工程位于：

- `jetson/m3pro_ext_ws`：部署到 Jetson 的 ROS2 工作区。
- `pc_client`：通过 rosbridge 连接小车的 Windows 客户端。
  - `monitor.py`：只读监控 `/battery`、`/joint_states` 等话题。
  - `arm_control.py`：共享的运动包络与点动逻辑，关节限位的唯一定义处。
  - `arm_demo.py`：纯终端的机械臂点动，默认 dry-run。
  - `camera_view.py`：实时查看相机彩色、深度或红外流。
  - `teleop_view.py`：画面与机械臂点动合并到一个窗口，默认 dry-run。
  - `safe_command.py`：经安全仲裁器下发单条指令，默认 dry-run。
  - `make_apriltag.py`：生成可打印的 AprilTag（SVG，物理尺寸精确）。
  - `visual_state.py`：从固定 AprilTag 解算相机绝对位姿，含噪声统计。
  - `vision_metrology.py`：以场景为参考的差分图像位移测量，精度高一个量级。
  - `arm_diagnostics.py`：机械臂特性测量（会让机械臂运动）。
- `docs/MEASUREMENTS.md`：相机与机械臂的实测数据、方法学陷阱和对强化学习的结论。
- `docs/IMPLEMENTATION_PLAN.md`：架构、构建步骤和安全约束。
- `firmware/m3pro_arm_feedback`：官方 STM32 示例的反馈补丁、校验值和实验 HEX。
- `backups/2026-09-21-source-backup.md`：已拉回本机的 Jetson 源码备份清单。
- `stm32-programmer.cmd`：调用已安装的 STM32CubeProgrammer 2.19 命令行工具。

扩展工程默认不会驱动机械臂：控制模式为 `disabled`，反馈校准为无效，PC
命令工具也默认只做 dry-run。
